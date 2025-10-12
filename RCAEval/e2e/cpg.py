"""
CPG (Causal Propagation Graph) Framework for Root Cause Analysis
因果傳播圖框架

This module implements a comprehensive 6-step adaptive CPG framework for RCA:
1. Preprocessing & Atomic Event Extraction
2. Aggregated Event Generation  
3. Global Anomaly Detection → Symptom Set S
4. Local CPG Construction Build_CPG
5. Root Cause Contribution Quantification
   - Enhanced: Shapley Value (NEW)
   - Original: PageRank + Anomaly Score + Priority
6. Fault Narrative & Output

新增功能 (Enhanced Features):
==================
1. Shapley Value 根因貢獻度量化
   - 基於合作賽局理論，公平分配每個服務的貢獻度
   - 支持精確計算（n≤10）和蒙特卡羅近似（n>10）
   - 實現高效緩存機制，提升計算效率
   - 提供數學嚴謹且可解釋的根因排名

2. 優化的融合策略
   - 主要使用 Shapley Value (90%) + 異常分數微調 (10%)
   - 簡化架構，提高穩定性和性能

使用方法:
========
# 使用 Shapley Value（默認）
result = cpg(data, dataset="online-boutique", use_shapley=True)

# 使用原始 PageRank 方法
result = cpg(data, dataset="online-boutique", use_shapley=False)

# 查看 Shapley 統計信息
if 'shapley_stats' in result:
    print(f"Cache hit rate: {result['shapley_stats']['cache_hit_rate']:.2%}")

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Scientific computing and ML
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from scipy import stats
import networkx as nx

# Deep learning (optional, with fallbacks)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, GraphSAGE
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from RCAEval.io.time_series import preprocess

# Shapley Value 計算模組
try:
    from .shapley_value import (
        ShapleyValueCalculator,
        CausalShapleyValueCalculator,
        get_service_priority,
        extract_base_service_name,
        create_system_anomaly_function
    )
except ImportError:
    # 當直接運行 cpg.py 時使用絕對導入
    from shapley_value import (
        ShapleyValueCalculator,
        CausalShapleyValueCalculator,
        get_service_priority,
        extract_base_service_name,
        create_system_anomaly_function
    )

def _load_service_topology() -> Dict[str, Dict[str, Any]]:
    """載入服務拓撲結構"""
    # 基於資料集結構的微服務依賴關係知識庫
    knowledge_base = {
        # Online Boutique 服務依賴
        "online-boutique": {
            "frontend": {"dependencies": ["productcatalogservice", "cartservice", "currencyservice", "adservice"], "critical": True},
            "productcatalogservice": {"dependencies": [], "critical": True},
            "cartservice": {"dependencies": ["redis"], "critical": True},
            "checkoutservice": {"dependencies": ["cartservice", "productcatalogservice", "currencyservice", "paymentservice", "shippingservice", "emailservice"], "critical": True},
            "currencyservice": {"dependencies": [], "critical": False},
            "adservice": {"dependencies": [], "critical": False},
            "paymentservice": {"dependencies": [], "critical": True},
            "shippingservice": {"dependencies": [], "critical": True},
            "emailservice": {"dependencies": [], "critical": False}
        },
        
        # Sock Shop 服務依賴
        "sock-shop": {
            "front-end": {"dependencies": ["catalogue", "carts", "orders", "user"], "critical": True},
            "catalogue": {"dependencies": ["catalogue-db"], "critical": True},
            "carts": {"dependencies": ["carts-db"], "critical": True},
            "orders": {"dependencies": ["orders-db", "payment", "shipping"], "critical": True},
            "user": {"dependencies": ["user-db"], "critical": True},
            "payment": {"dependencies": [], "critical": True}
        },
        
        # Train Ticket 服務依賴
        "train-ticket": {
            "ts-ui-dashboard": {"dependencies": ["ts-auth-service", "ts-route-service", "ts-order-service"], "critical": True},
            "ts-auth-service": {"dependencies": ["ts-auth-mongo"], "critical": True},
            "ts-route-service": {"dependencies": ["ts-route-mongo"], "critical": True},
            "ts-order-service": {"dependencies": ["ts-order-mongo", "ts-travel-service"], "critical": True},
            "ts-travel-service": {"dependencies": ["ts-travel-mongo", "ts-train-service"], "critical": True},
            "ts-train-service": {"dependencies": ["ts-train-mongo"], "critical": True}
        }
    }
    
    return knowledge_base


# create_system_anomaly_function 已移至 shapley_value.py 模組


@dataclass
class AtomicEvent:
    """原子事件數據結構，包含時間戳、服務名稱、事件類型和數據字典"""
    timestamp: float
    service_name: str
    event_type: str  # metrics/logs/traces
    data_dict: Dict[str, Any]


@dataclass
class AggregatedEvent:
    """聚合事件數據結構，包含時間戳、服務名稱和特徵向量"""
    timestamp: float
    service_name: str
    features: np.ndarray


class AdaptivePipeline:
    """自適應預處理管道"""
    
    def __init__(self):
        self.pipeline = None
        self.scaler = None
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """自動配置並應用預處理管道"""
        try:
            # 移除常量列
            constant_cols = data.columns[data.nunique() <= 1]
            if len(constant_cols) > 0:
                data = data.drop(columns=constant_cols)
            
            # 獲取當前的數值列（在移除常量列之後）
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # 處理缺失值
            if len(numeric_cols) > 0:
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            
            # 移除低方差特征
            if len(numeric_cols) > 0:
                variances = data[numeric_cols].var()
                low_var_cols = variances[variances < 0.01].index
                if len(low_var_cols) > 0:
                    data = data.drop(columns=low_var_cols)
                    # 重新獲取數值列（在移除低方差列之後）
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # 標準化（使用更新後的數值列）
            if len(numeric_cols) > 0:
                self.scaler = RobustScaler()
                data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
            
            return data
            
        except Exception as e:
            print(f"Warning: Preprocessing failed ({e}), using minimal preprocessing")
            # 降級到最小預處理
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # 只填充缺失值，不做其他處理
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            return data




class CorrelationCausalModel:
    """
    基於相關性的因果發現模型
    
    未來改進方向：
    - 實現帶時滯的交叉相關性計算（Time-Lagged Cross-Correlation）
    - 捕捉具有時間延遲的因果關係，減少偽相關
    """
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        
    def infer_causal(self, candidates: List[AggregatedEvent], target_features: np.ndarray) -> List[Dict[str, Any]]:
        """
        推斷因果關係。
        透過計算每個候選事件與目標症狀之間的特徵相關性來實現。
        
        注意：當前使用簡單的皮爾遜相關係數。
        未來可改進為考慮時間延遲的因果推斷（見類文檔字符串）。
        """
        edges = []
        
        for candidate in candidates:
            # 計算特徵相關性
            if len(candidate.features) > 0 and len(target_features) > 0:
                # 確保特徵向量長度相同才能計算相關性
                if len(candidate.features) == len(target_features):
                    correlation = np.corrcoef(candidate.features, target_features)[0, 1]
                    if not np.isnan(correlation):
                        confidence = abs(correlation)
                        # 自適應閾值：基於數據分布動態調整
                        adaptive_threshold = max(0.01, np.std([c.features for c in candidates if len(c.features) > 0]) * 0.3)
                        if confidence > adaptive_threshold:
                            edges.append({
                                'source': candidate,
                                'confidence': confidence,
                                'strength': confidence
                            })
        
        return edges


# ShapleyValueCalculator 類已移至 shapley_value.py 模組


# BayesianNetwork 類已移除 - 專注於 Shapley Value 優化


# get_service_priority 函數已移至 shapley_value.py 模組




class CPGFramework:
    """
    CPG框架 - 增強版
    
    專注於 Shapley Value 的根因貢獻度量化:
    - 基於合作賽局理論的公平分配
    - 支持精確和近似計算
    - 高效緩存機制
    - 數學嚴謹且可解釋
    """
    
    def __init__(self):
        self.pipeline = AdaptivePipeline()
        self.models = {}
        self.knowledge_base = _load_service_topology()
        
        # Shapley Value 計算器 - 專注於公平的根因貢獻度量化
        # Causal Shapley Value 計算器（利用因果依賴關係）
        self.shapley_calculator = CausalShapleyValueCalculator()
        
    def preprocess_and_extract_atomic_events(self, raw_data: pd.DataFrame) -> List[AtomicEvent]:
        """
        第1步: 前處理 & 原子事件提取
        """
        print("Step 1: Preprocessing & Atomic Event Extraction")
        
        # 簡化的預處理和事件提取
        try:
            # 自適應預處理
            cleaned_data = self.pipeline.fit_transform(raw_data.copy())
            if cleaned_data.empty:
                cleaned_data = raw_data.copy()
            
            # 統一的事件提取邏輯
            events = []
            sample_rate = max(1, len(cleaned_data) // 50)  # 採樣約50個點
            
            for idx, row in cleaned_data.iloc[::sample_rate].iterrows():
                # 使用時間列或索引作為時間戳
                timestamp = row.get('time', float(idx))
                service_name = self._infer_service_name_prioritized(row, cleaned_data.columns)
                
                event = AtomicEvent(
                    timestamp=timestamp,
                    service_name=service_name,
                    event_type='metrics',
                    data_dict=row.to_dict()
                )
                events.append(event)
            
            print(f"Extracted {len(events)} atomic events")
            return events
            
        except Exception as e:
            print(f"Warning: Event extraction failed ({e}), using minimal fallback")
            # 最小降級：只取前10行
            events = []
            sample_data = raw_data.head(min(10, len(raw_data)))
            for idx, row in sample_data.iterrows():
                service_name = self._infer_service_name_prioritized(row, raw_data.columns)
                event = AtomicEvent(
                    timestamp=float(idx),
                    service_name=service_name,
                    event_type='metrics',
                    data_dict=row.to_dict()
                )
                events.append(event)
            print(f"Fallback: Extracted {len(events)} atomic events")
            return events
    
    
    def _is_critical_service(self, service_name: str) -> bool:
        """檢查服務是否為關鍵服務"""
        # 基於知識庫判斷
        for dataset_name, services in self.knowledge_base.items():
            if service_name in services:
                return services[service_name].get('critical', False)
        
        # 基於命名模式判斷
        critical_patterns = ['frontend', 'ui', 'gateway', 'auth', 'order', 'payment']
        for pattern in critical_patterns:
            if pattern in service_name.lower():
                return True
        
        return False
    
    
    def _extract_service_from_column(self, col_name: str) -> str:
        """從列名中提取服務名稱 - 支持train-ticket命名模式"""
        # Train-ticket服務模式 (如 ts-admin-basic-info-service_container-cpu-system-seconds-total)
        if col_name.startswith('ts-') and '_container-' in col_name:
            return col_name.split('_')[0]  # 返回完整的ts-service-name
        
        # 節點指標 (如 192-168-25-221-9100_node-cpu-seconds-total) - 跳過這些
        if col_name.startswith('192-'):
            return 'node-' + col_name.split('_')[0]  # 標記為節點指標
        
        # 容器指標 (如 carts_container-cpu-system-seconds-total)
        if '_container-' in col_name:
            return col_name.split('_')[0]
        
        # Istio指標 (如 carts_istio-request-total)
        if '_istio-' in col_name:
            return col_name.split('_')[0]
        
        # 數據庫指標 (如 carts-db_container-cpu-system-seconds-total)
        if '-db_' in col_name:
            return col_name.split('_')[0]
        
        # 其他下劃線分隔的指標
        if '_' in col_name:
            parts = col_name.split('_')
            service_name = parts[0]
            
            # 過濾掉一些常見的非服務名前綴
            if service_name not in ['container', 'istio', 'node']:
                return service_name
        
        # 處理連字符分隔的指標
        if '-' in col_name:
            parts = col_name.split('-')
            service_name = parts[0]
            if service_name not in ['container', 'istio', 'node'] and len(service_name) > 2:
                return service_name
        
        # 如果沒有分隔符，直接使用列名
        if len(col_name) > 0:
            return col_name[:20]  # 限制長度
        
        return 'unknown'
    
    def _infer_service_name_prioritized(self, row: pd.Series, columns: pd.Index) -> str:
        """優先選擇微服務而不是節點指標"""
        # 收集所有服務候選
        service_candidates = {}
        
        for col in columns:
            if col == 'time':
                continue
            
            service_name = self._extract_service_from_column(col)
            if service_name and service_name != 'unknown':
                service_candidates[service_name] = service_candidates.get(service_name, 0) + 1
        
        if not service_candidates:
            return 'unknown'
        
        # 優先級排序：微服務 > 節點指標
        microservices = {k: v for k, v in service_candidates.items() if not k.startswith('node-')}
        node_services = {k: v for k, v in service_candidates.items() if k.startswith('node-')}
        
        # 優先返回微服務，特別是train-ticket的ts-服務
        if microservices:
            ts_services = {k: v for k, v in microservices.items() if k.startswith('ts-')}
            if ts_services:
                return max(ts_services.items(), key=lambda x: x[1])[0]
            else:
                return max(microservices.items(), key=lambda x: x[1])[0]
        
        # 如果沒有微服務，才使用節點指標
        if node_services:
            return max(node_services.items(), key=lambda x: x[1])[0]
        
        return 'unknown'
    
    def enrich_atomic_events_with_features(self, atomic_events: List[AtomicEvent]) -> List[AtomicEvent]:
        """
        步驟2: 事件特徵化與即時異常標記 (取代聚合事件生成)
        
        目標：不再是聚合，而是為每一個原子事件打上異常分數。
        保持時間精度，不丟失任何信息。
        """
        print("Step 2: Event Feature Enrichment (Replacing Aggregation)")
        
        if not atomic_events:
            return []
        
        # 按服務分組
        service_groups = {}
        for event in atomic_events:
            if event.service_name not in service_groups:
                service_groups[event.service_name] = []
            service_groups[event.service_name].append(event)
        
        enriched_events = []
        
        for service_name, events in service_groups.items():
            # 按時間排序
            events.sort(key=lambda x: x.timestamp)
            
            print(f"  Service {service_name}: Enriching {len(events)} events")
            
            for i, event in enumerate(events):
                # 計算流式特徵
                features = self._compute_streaming_features(events, i)
                
                # 計算異常分數
                anomaly_score = self._compute_streaming_anomaly_score(events, i)
                
                # 直接修改現有事件，添加新屬性
                event.features = features  # 新增流式特徵
                event.anomaly_score = anomaly_score  # 新增異常分數
                
                enriched_events.append(event)
        
        print(f"Enriched {len(enriched_events)} atomic events with streaming features")
        return enriched_events
    
    def _extract_ensemble_features(self, events: List[AtomicEvent]) -> np.ndarray:
        """提取原子事件結合特徵"""
        if not events:
            return np.array([])
        
        # 收集數值特徵
        all_values = []
        for event in events:
            for key, value in event.data_dict.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    all_values.append(value)
        
        if not all_values:
            return np.array([0.0])
        
        values = np.array(all_values)
        
        # 基本統計特徵
        features = [
            np.mean(values),
            np.max(values),
            np.std(values),
            np.median(values),
            len(values)
        ]
        
        return np.array(features)
    
    def _compute_streaming_features(self, events: List[AtomicEvent], current_idx: int) -> np.ndarray:
        """
        計算流式特徵：瞬時變化率、短期波動性等
        """
        features = []
        
        # 獲取當前事件的數值
        current_values = self._extract_numeric_values(events[current_idx])
        
        # 1. 瞬時變化率 (Instantaneous Rate of Change)
        if current_idx > 0:
            prev_values = self._extract_numeric_values(events[current_idx - 1])
            time_diff = events[current_idx].timestamp - events[current_idx - 1].timestamp
            
            if time_diff > 0 and len(current_values) > 0 and len(prev_values) > 0:
                # 計算變化率
                rate_of_change = (np.mean(current_values) - np.mean(prev_values)) / time_diff
                features.append(rate_of_change)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 2. 短期波動性 (Short-term Volatility)
        window_size = min(5, current_idx + 1)
        if window_size > 1:
            window_values = []
            for j in range(max(0, current_idx - window_size + 1), current_idx + 1):
                values = self._extract_numeric_values(events[j])
                if len(values) > 0:
                    window_values.append(np.mean(values))
            
            if len(window_values) > 1:
                volatility = np.std(window_values)
                features.append(volatility)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 3. 趨勢強度 (Trend Strength)
        if current_idx >= 2:
            recent_values = []
            for j in range(max(0, current_idx - 2), current_idx + 1):
                values = self._extract_numeric_values(events[j])
                if len(values) > 0:
                    recent_values.append(np.mean(values))
            
            if len(recent_values) >= 3:
                # 計算線性趨勢
                x = np.arange(len(recent_values))
                slope, _ = np.polyfit(x, recent_values, 1)
                features.append(abs(slope))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 4. 異常程度 (Anomaly Score)
        if len(current_values) > 0:
            # 使用當前值相對於歷史窗口的異常程度
            if current_idx > 0:
                historical_values = []
                for j in range(max(0, current_idx - 10), current_idx):
                    values = self._extract_numeric_values(events[j])
                    if len(values) > 0:
                        historical_values.extend(values)
                
                if len(historical_values) > 0:
                    current_mean = np.mean(current_values)
                    historical_mean = np.mean(historical_values)
                    historical_std = np.std(historical_values)
                    
                    if historical_std > 0:
                        z_score = abs(current_mean - historical_mean) / historical_std
                        features.append(z_score)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def _compute_streaming_anomaly_score(self, events: List[AtomicEvent], current_idx: int) -> float:
        """
        計算流式異常分數
        """
        if current_idx < 5:  # 需要足夠的歷史數據
            return 0.0
        
        # 獲取當前事件特徵
        current_features = self._compute_streaming_features(events, current_idx)
        
        # 獲取歷史窗口特徵
        window_size = min(15, current_idx)
        historical_features = []
        
        for j in range(current_idx - window_size, current_idx):
            features = self._compute_streaming_features(events, j)
            if len(features) > 0:
                historical_features.append(features)
        
        if len(historical_features) < 3:
            return 0.0
        
        # 計算異常分數
        historical_features = np.array(historical_features)
        
        # 使用Isolation Forest進行異常檢測
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(historical_features)
        
        # 計算當前事件的異常分數
        current_score = iso_forest.decision_function([current_features])[0]
        
        # 轉換為0-1範圍的分數
        anomaly_score = max(0.0, min(1.0, (1 - current_score) / 2))
        
        return anomaly_score
    
    def _extract_numeric_values(self, event: AtomicEvent) -> List[float]:
        """
        從原子事件中提取數值特徵
        """
        values = []
        for key, value in event.data_dict.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                values.append(float(value))
        return values
    
    def global_anomaly_detection(self, enriched_events: List[AtomicEvent]) -> List[AtomicEvent]:
        """
        步驟3: 全域異常檢測 → 症狀集合 S
        改進：基於數據驅動的症狀檢測，使用豐富化的原子事件
        """
        print("Step 3: Global Anomaly Detection (Data-Driven)")
        
        if not enriched_events:
            return []
        
        # 構造特徵矩陣
        feature_matrix = []
        for event in enriched_events:
            if hasattr(event, 'features') and len(event.features) > 0:
                feature_matrix.append(event.features)
            else:
                feature_matrix.append(np.array([0.0]))
        
        if not feature_matrix:
            return []
        
        # 確保所有特徵向量長度相同
        max_len = max(len(f) for f in feature_matrix)
        padded_features = []
        for features in feature_matrix:
            if len(features) < max_len:
                padded = np.zeros(max_len)
                padded[:len(features)] = features
                padded_features.append(padded)
            else:
                padded_features.append(features[:max_len])
        
        X = np.array(padded_features)
        print(f"Feature matrix shape: {X.shape}")
        
        # --- START: 增加健壯性檢查 ---
        if len(X) < 2:
            print("Warning: Insufficient data for IsolationForest. Falling back to statistical method.")
            # 簡單的統計異常檢測 - 即使只有一個樣本也能工作
            if len(X) == 1:
                # 只有一個樣本，直接將其作為異常
                combined_scores = np.array([1.0])
            else:
                # 使用特徵方差作為異常分數
                feature_vars = np.var(X, axis=1)
                if np.std(feature_vars) > 0:
                    combined_scores = (feature_vars - feature_vars.min()) / (feature_vars.max() - feature_vars.min() + 1e-8)
                else:
                    combined_scores = np.ones(len(X)) * 0.5
        else:
            # 集成異常檢測
            print("Using IsolationForest + statistical methods")
            isolation_forest = IsolationForest(contamination=min(0.3, max(0.1, 1.0/len(X))), random_state=42)
            isolation_forest.fit(X)  # 先擬合模型
            iso_scores = isolation_forest.decision_function(X)
            
            # 簡單的統計異常檢測作為第二個模型
            from scipy import stats as scipy_stats
            z_scores = np.abs(scipy_stats.zscore(X, axis=0)).mean(axis=1)
            
            # 處理可能的 NaN 值
            iso_scores = np.nan_to_num(iso_scores, nan=0.0)
            z_scores = np.nan_to_num(z_scores, nan=0.0)
            
            # 標準化分數到 [0,1] 範圍
            if np.std(iso_scores) > 0:
                iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
            else:
                iso_scores = np.ones_like(iso_scores) * 0.5
                
            if np.std(z_scores) > 0:
                z_scores = (z_scores - z_scores.min()) / (z_scores.max() - z_scores.min() + 1e-8)
            else:
                z_scores = np.ones_like(z_scores) * 0.5
            
            # 組合分數
            combined_scores = 0.6 * iso_scores + 0.4 * z_scores
        # --- END: 增加健壯性檢查 ---
        
        for event, score in zip(enriched_events, combined_scores):
            event.anomaly_score = score
        
        # 自適應閾值 - 基於數據分布動態調整
        if len(combined_scores) == 1:
            threshold = 0.3
            threshold_percentile = "fixed"
        else:
            # 動態調整分位數：數據越多，閾值越嚴格
            percentile = max(70, min(90, 80 + len(combined_scores) * 0.1))
            threshold_percentile = percentile
            threshold = np.percentile(combined_scores, threshold_percentile)
        
        print(f"Using threshold: {threshold:.4f} (percentile: {threshold_percentile})")
        
        # 選擇症狀
        symptoms = []
        for event, score in zip(enriched_events, combined_scores):
            if score >= threshold:
                symptoms.append(event)
        
        # 確保至少有3個症狀（如果可能）
        min_symptoms = min(3, len(enriched_events))
        if len(symptoms) < min_symptoms:
            print(f"Too few symptoms ({len(symptoms)}), selecting top {min_symptoms}")
            # 按分數排序，選擇前N個
            scored_events = list(zip(enriched_events, combined_scores))
            scored_events.sort(key=lambda x: x[1], reverse=True)
            symptoms = [event for event, _ in scored_events[:min_symptoms]]
        
        # 排序症狀按分數，最高分數的是主要症狀
        symptoms.sort(key=lambda x: x.anomaly_score, reverse=True)
        
        if symptoms:
            print(f"Detected {len(symptoms)} symptoms. Primary symptom: {symptoms[0].service_name} (score: {symptoms[0].anomaly_score:.3f})")
        else:
            print(f"Detected {len(symptoms)} symptoms from {len(enriched_events)} events")
        
        return symptoms
    
    def build_local_cpg(self, symptoms: List[AtomicEvent], all_events: List[AtomicEvent]) -> Tuple[set, set]:
        """
        步驟4: 局部CPG構建
        """
        print("Step 4: Local CPG Construction")
        
        vertices = set()
        edges = set()
        
        if not symptoms:
            return vertices, edges
        
        # 優化：在迴圈外實例化模型以提高效率
        feature_dim = len(all_events[0].features) if all_events and hasattr(all_events[0], 'features') and len(all_events[0].features) > 0 else 4
        correlation_model = CorrelationCausalModel(input_dim=feature_dim)
        
        for symptom in symptoms:
            # 獲取上游候選
            candidates = self._get_upstream_candidates(symptom, all_events)
            
            if not candidates:
                continue
            
            # 使用相關性模型推斷因果關係
            if hasattr(symptom, 'features') and symptom.features is not None and len(symptom.features) > 0:
                causal_edges = correlation_model.infer_causal(candidates, symptom.features)
                
                # 添加邊和頂點
                vertices.add(symptom.service_name)
                
                for edge_info in causal_edges:
                    source_event = edge_info['source']
                    confidence = edge_info['confidence']
                    
                    # 進一步降低閾值以確保能建立CPG
                    if confidence > 0.05:  # 從0.1降到0.05
                        vertices.add(source_event.service_name)
                        vertices.add(symptom.service_name)
                        edges.add((source_event.service_name, symptom.service_name, confidence))
        
        print(f"Built local CPG with {len(vertices)} vertices and {len(edges)} edges")
        return vertices, edges
    
    def _get_upstream_candidates(self, symptom: AtomicEvent, all_events: List[AtomicEvent]) -> List[AtomicEvent]:
        """
        獲取上游候選事件
        改進：考慮時間延遲，優先選擇時間差在30-60秒的候選事件（模擬因果傳播延遲）
        """
        candidates = []
        candidates_with_priority = []  # (event, priority) 用於考慮時間延遲
        
        # 自適應回溯窗口
        lookback_window = self._adaptive_lookback(symptom)
        
        # 理想的因果傳播延遲範圍（30-60秒）
        ideal_lag_min = 30
        ideal_lag_max = 60
        
        for event in all_events:
            # 時間條件：事件發生在症狀之前或同時
            if event.timestamp <= symptom.timestamp and event.timestamp >= symptom.timestamp - lookback_window:
                # 不同服務或同服務不同窗口
                if event.service_name != symptom.service_name or event != symptom:
                    # 計算時間差
                    time_lag = symptom.timestamp - event.timestamp
                    
                    # 根據時間差給予優先級權重
                    if ideal_lag_min <= time_lag <= ideal_lag_max:
                        priority = 1.0  # 理想時間窗口內的候選
                    elif time_lag < ideal_lag_min:
                        priority = 0.8  # 時間差太小
                    else:
                        priority = 0.6  # 時間差較大
                    
                    candidates_with_priority.append((event, priority))
        
        # 如果找不到候選，放寬條件包含所有其他事件
        if not candidates_with_priority:
            for event in all_events:
                if event != symptom:
                    candidates_with_priority.append((event, 0.5))
        
        # 按優先級排序（時間延遲在理想範圍內的優先）
        candidates_with_priority.sort(key=lambda x: x[1], reverse=True)
        candidates = [event for event, _ in candidates_with_priority]
        
        return candidates
    
    def _adaptive_lookback(self, symptom: AtomicEvent) -> float:
        """自適應回溯窗口"""
        # 基於症狀的特徵動態確定回溯窗口
        if hasattr(symptom, 'features') and symptom.features is not None and len(symptom.features) > 0:
            feature_magnitude = np.linalg.norm(symptom.features)
            return min(max(feature_magnitude * 10, 60), 300)  # 60秒到5分鐘
        return 120  # 默認2分鐘
    
    def quantify_root_causes_enhanced(
        self, 
        vertices: set, 
        edges: set, 
        all_events: List[AtomicEvent]
    ) -> Dict[str, float]:
        """
        步驟5: 根因貢獻度量化 - 改進版
        
        使用 Causal Shapley Value 計算
        
        主要改進:
        1. 利用 CPG 的因果依賴關係（拓撲約束）
        2. 只考慮符合時間序列的排列
        3. 改進的系統異常函數（減少優先級權重影響）
        4. 簡化的融合策略（主要使用 Shapley Value）
        5. 異常分數作為微調信息
        
        Returns:
            Dict[str, float]: 每個服務的根因貢獻度分數
        """
        print("Step 5: Causal Shapley Root Cause Contribution Quantification")
        print("  Method: Causal Shapley Value (Topological Constraint)")
        
        if not vertices:
            return {}
        
        # 1. 提取基礎服務名稱
        base_services = set()
        for vertex in vertices:
            base_service = extract_base_service_name(vertex)
            base_services.add(base_service)
        
        # 2. 創建改進的系統異常函數
        print("  Creating improved system anomaly function...")
        system_anomaly_func = create_system_anomaly_function(all_events)
        
        # 3. 計算 Causal Shapley Values（使用 CPG 邊信息）
        print("  Calculating Causal Shapley Values with topological constraints...")
        shapley_values = self.shapley_calculator.calculate_causal_shapley_values(
            list(base_services),
            edges,  # 傳遞 CPG 邊信息以構建因果約束
            system_anomaly_func
        )
        
        # 4. 歸一化 Shapley Values 到 [0, 1]
        if shapley_values:
            max_shapley = max(shapley_values.values())
            if max_shapley > 0:
                shapley_values = {
                    k: v / max_shapley 
                    for k, v in shapley_values.items()
                }
        
        # 5. 簡化的融合策略：主要使用 Shapley Values
        final_scores = {}
        
        # 獲取異常分數作為微調信息
        anomaly_scores = {}
        for event in all_events:
            base_service = extract_base_service_name(event.service_name)
            if hasattr(event, 'anomaly_score'):
                if base_service not in anomaly_scores:
                    anomaly_scores[base_service] = 0.0
                anomaly_scores[base_service] = max(
                    anomaly_scores[base_service], 
                    event.anomaly_score
                )
        
        for service in base_services:
            shapley_score = shapley_values.get(service, 0.0)
            anomaly_score = anomaly_scores.get(service, 0.0)
            
            # 簡化融合：主要使用 Shapley Value (90%)，異常分數作為微調 (10%)
            # Shapley Value 的公平貢獻分配
            final_score = 0.9 * shapley_score + 0.1 * anomaly_score
            final_scores[service] = final_score
        
        # 6. 輸出統計信息
        stats = self.shapley_calculator.get_statistics()
        print(f"  Causal Shapley statistics:")
        print(f"    - Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"    - Cache hits: {stats['cache_hits']}, misses: {stats['cache_misses']}")
        print(f"    - Valid permutations: {stats.get('valid_permutations', 'N/A')}")
        print(f"    - Constraint efficiency: {stats.get('constraint_efficiency', 0.0):.2%}")
        print(f"    - Total time: {stats['total_time']:.2f}s")
        
        print(f"  Causal Shapley quantification completed for {len(final_scores)} services")
        print(f"  Fusion: Causal Shapley(90%) + Anomaly(10%)")
        
        return final_scores
    
    def quantify_root_causes(
        self, 
        vertices: set, 
        edges: set, 
        all_events: List[AtomicEvent],
        use_shapley: bool = True
    ) -> Dict[str, float]:
        """
        步驟5: 根因貢獻度量化
        
        支持兩種方法:
        1. 增強版: Shapley Value (專注於公平貢獻分配)
        2. 原始版: PageRank + 異常分數 + 優先級
        
        Args:
            vertices: CPG 頂點集合
            edges: CPG 邊集合
            all_events: 所有事件列表
            use_shapley: 是否使用 Shapley Value（默認 True）
        
        Returns:
            Dict[str, float]: 每個服務的根因貢獻度分數
        """
        if not use_shapley:
            print("Using original PageRank method (as requested)")
            return self._quantify_root_causes_original(vertices, edges, all_events)
        
        # 嘗試使用增強版方法
        try:
            return self.quantify_root_causes_enhanced(vertices, edges, all_events)
        
        except Exception as e:
            print(f"  Enhanced method failed: {e}")
            print("  Falling back to original PageRank method")
            import traceback
            traceback.print_exc()
            
            # 降級到原始方法
            return self._quantify_root_causes_original(vertices, edges, all_events)
    
    def _quantify_root_causes_original(
        self, 
        vertices: set, 
        edges: set, 
        all_events: List[AtomicEvent]
    ) -> Dict[str, float]:
        """
        原始的根因貢獻度量化方法（PageRank）
        
        作為降級備份使用
        """
        print("Step 5: Root Cause Contribution Quantification (Original PageRank)")
        
        if not vertices:
            return {}
        
        # 權重配置
        w1, w2, w3 = 0.4, 0.4, 0.2  # 圖結構、異常程度、優先級
        
        # 1. 計算圖結構分數 (PageRank)
        G = nx.DiGraph()
        G.add_nodes_from(vertices)
        
        for edge in edges:
            if len(edge) >= 3:
                source, target, weight = edge[0], edge[1], edge[2]
                G.add_edge(source, target, weight=weight)
        
        try:
            pagerank_scores = nx.pagerank(G, alpha=0.85) if G.nodes() else {}
        except:
            nodes = list(G.nodes())
            pagerank_scores = {node: 1.0/len(nodes) for node in nodes} if nodes else {}
        
        # 2. 獲取異常分數和優先級分數
        event_scores = {}
        for event in all_events:
            if event.service_name in vertices:
                base_service_name = extract_base_service_name(event.service_name)
                
                if base_service_name not in event_scores:
                    event_scores[base_service_name] = {
                        "anomaly_score": 0.0,
                        "priority_score": get_service_priority(base_service_name)
                    }
                
                current_anomaly = getattr(event, 'anomaly_score', 0.0)
                event_scores[base_service_name]["anomaly_score"] = max(
                    current_anomaly,
                    event_scores[base_service_name]["anomaly_score"]
                )
        
        # 3. 融合分數
        final_scores = {}
        
        for service_window in vertices:
            base_service_name = extract_base_service_name(service_window)
            
            graph_score = pagerank_scores.get(service_window, 0.0)
            anomaly_score = event_scores.get(base_service_name, {}).get("anomaly_score", 0.0)
            priority_score = event_scores.get(base_service_name, {}).get("priority_score", 0.5)
            
            final_score = (w1 * graph_score) + (w2 * anomaly_score) + (w3 * priority_score)
            
            if base_service_name not in final_scores or final_score > final_scores[base_service_name]:
                final_scores[base_service_name] = final_score
        
        print(f"  Computed scores for {len(final_scores)} base services")
        print(f"  Score composition: Graph={w1}, Anomaly={w2}, Priority={w3}")
        
        return final_scores
    
    def generate_narrative(self, root_scores: Dict[str, float], symptoms: List[AggregatedEvent], edges: set) -> Dict[str, Any]:
        """
        步驟6: 故障敘事與輸出
        """
        print("Step 6: Fault Narrative Generation")
        
        if not root_scores:
            return {"narrative": [], "summary": "No root causes identified"}
        
        # 按分數排序
        sorted_services = sorted(root_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 自適應top選擇
        top_k = self._adaptive_top_selection(root_scores)
        top_services = sorted_services[:top_k]
        
        narrative = []
        for service_name, score in top_services:
            # 獲取相關症狀
            related_symptoms = [s for s in symptoms if s.service_name == service_name]
            
            # 獲取上游依賴
            upstream = [edge[0] for edge in edges if edge[1] == service_name]
            
            service_info = {
                "service": service_name,
                "importance_score": float(score),
                "related_symptoms_count": len(related_symptoms),
                "upstream_dependencies": upstream,
                "confidence": "high" if score > 0.7 else "medium" if score > 0.4 else "low"
            }
            narrative.append(service_info)
        
        summary = f"Identified {len(narrative)} potential root causes. " \
                 f"Top candidate: {narrative[0]['service']} (score: {narrative[0]['importance_score']:.3f})" \
                 if narrative else "No root causes identified"
        
        return {
            "narrative": narrative,
            "summary": summary,
            "total_candidates": len(root_scores)
        }
    
    def _adaptive_top_selection(self, scores: Dict[str, float]) -> int:
        """自適應top選擇"""
        if not scores:
            return 0
        
        score_values = list(scores.values())
        if len(score_values) <= 3:
            return len(score_values)
        
        # 使用elbow方法(肘部法)
        sorted_scores = sorted(score_values, reverse=True)
        diffs = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1)]
        
        if diffs:
            max_diff_idx = diffs.index(max(diffs))
            return min(max_diff_idx + 2, len(sorted_scores))
        
        return min(5, len(score_values))
    


def cpg(data, inject_time=None, dataset=None, sli=None, **kwargs):
    """
    CPG框架 
    
    Args:
        data: pd.DataFrame, 輸入數據
        inject_time: 故障注入時間
        dataset: 數據集名稱
        sli: Service Level Indicator
        **kwargs: 其他參數
            - use_shapley: 是否使用 Shapley Value（默認 True）
            - dk_select_useful: 是否選擇有用的特徵
    
    Returns:
        Dict: 包含排名結果的字典
            - ranks: 服務排名列表（按根因可能性降序）
            - narrative: 故障敘事
            - shapley_stats: Shapley 計算統計（如果使用）
            - method: 使用的方法（'shapley' 或 'pagerank'）
    """
    try:
        print("=== Starting CPG Framework ===")
        
        # 提取配置參數
        use_shapley = kwargs.get("use_shapley", True)  # 默認使用 Shapley
        
        print(f"Configuration: use_shapley={use_shapley}")
        
        # 預處理數據
        data = preprocess(data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
        
        # 初始化框架
        cpg_framework = CPGFramework()
        
        # 步驟1: 前處理 & 原子事件提取
        atomic_events = cpg_framework.preprocess_and_extract_atomic_events(data)
        
        if not atomic_events:
            print("No atomic events extracted, returning random ranking")
            cols = data.columns.tolist()
            return {"ranks": cols, "method": "fallback"}
        
        # 步驟2: 事件特徵化與即時異常標記 (取代聚合事件生成)
        enriched_events = cpg_framework.enrich_atomic_events_with_features(atomic_events)
        
        if not enriched_events:
            print("No enriched events generated, returning random ranking")
            cols = data.columns.tolist()
            return {"ranks": cols, "method": "fallback"}
        
        # 步驟3: 全局異常檢測
        symptoms = cpg_framework.global_anomaly_detection(enriched_events)
        
        if not symptoms:
            print("No symptoms detected, returning random ranking")
            cols = data.columns.tolist()
            return {"ranks": cols, "method": "fallback"}
        
        # 步驟4: 局部CPG建構
        vertices, edges = cpg_framework.build_local_cpg(symptoms, enriched_events)
        
        if not vertices:
            print("No CPG vertices found, returning symptom-based ranking")
            symptom_services = list(set(extract_base_service_name(s.service_name) for s in symptoms))
            all_services = set()
            for col in data.columns:
                if col != 'time':
                    service_name = cpg_framework._extract_service_from_column(col)
                    if service_name != 'unknown' and not service_name.startswith('node-'):
                        all_services.add(service_name)
            remaining = [s for s in all_services if s not in symptom_services]
            return {"ranks": symptom_services + remaining, "method": "fallback"}
        
        # 步驟5: 使用增強版或原始版方法
        root_scores = cpg_framework.quantify_root_causes(
            vertices, 
            edges, 
            enriched_events,
            use_shapley=use_shapley
        )
        
        if not root_scores:
            print("No root cause scores computed, returning vertex-based ranking")
            return {"ranks": list(vertices), "method": "fallback"}
        
        # 步驟6: 故障敘事生成
        narrative = cpg_framework.generate_narrative(root_scores, symptoms, edges)
        print(f"Narrative: {narrative['summary']}")
        
        # 生成最終排名
        sorted_services = sorted(root_scores.items(), key=lambda x: x[1], reverse=True)
        ranks = [service for service, _ in sorted_services]
        
        # 獲取數據中所有唯一的服務名稱
        all_service_names = set()
        for col in data.columns:
            if col != 'time':
                service_name = cpg_framework._extract_service_from_column(col)
                if service_name != 'unknown' and not service_name.startswith('node-'):
                    all_service_names.add(service_name)
        
        # 添加未在CPG排名中的其他服務
        ranked_services = set(ranks)
        remaining_services = [s for s in all_service_names if s not in ranked_services]
        ranks.extend(remaining_services)
        
        print(f"=== CPG Framework completed. Final ranking: {ranks[:5]}... ===")
        
        # 準備返回結果
        result = {
            "ranks": ranks,
            "narrative": narrative,
            "adj": [],  # 為兼容性保留
            "node_names": ranks,
            "method": "shapley" if use_shapley else "pagerank"
        }
        
        # 如果使用 Shapley，添加統計信息
        if use_shapley:
            result["shapley_stats"] = cpg_framework.shapley_calculator.get_statistics()
        
        return result
        
    except Exception as e:
        print(f"Error in CPG framework: {e}")
        import traceback
        traceback.print_exc()
        
        # 降級到簡單排名
        cols = data.columns.tolist()
        if 'time' in cols:
            cols.remove('time')
        
        return {
            "ranks": cols,
            "adj": [],
            "node_names": cols,
            "method": "error_fallback"
        }


# 為了兼容現有系統，創建別名

if __name__ == "__main__":
    # 測試 Shapley Value 實現
    print("=" * 60)
    print("Testing Enhanced CPG Framework with Shapley Value")
    print("=" * 60)
    
    # 創建測試數據
    np.random.seed(42)
    test_data = pd.DataFrame({
        'time': range(100),
        'serviceA_cpu': np.random.normal(50, 10, 100),
        'serviceA_memory': np.random.normal(60, 15, 100),
        'serviceB_cpu': np.random.normal(30, 5, 100),
        'serviceB_memory': np.random.normal(40, 8, 100),
        'serviceC_latency': np.random.exponential(2, 100)
    })
    
    # 注入異常（serviceA 是根因）
    test_data.loc[80:90, 'serviceA_cpu'] *= 2.5
    test_data.loc[85:95, 'serviceC_latency'] *= 3
    
    print("\n" + "=" * 60)
    print("Test 1: Using Shapley Value method")
    print("=" * 60)
    result_shapley = cpg(test_data, inject_time=80, dataset="test", use_shapley=True)
    print(f"\nResults:")
    print(f"  Method: {result_shapley['method']}")
    print(f"  Top 5 rankings: {result_shapley['ranks'][:5]}")
    if 'shapley_stats' in result_shapley:
        stats = result_shapley['shapley_stats']
        print(f"\nShapley Statistics:")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"  Total calculations: {stats['cache_hits'] + stats['cache_misses']}")
        print(f"  Computation time: {stats['total_time']:.2f}s")
    
    print("\n" + "=" * 60)
    print("Test 2: Using original PageRank method")
    print("=" * 60)
    result_pagerank = cpg(test_data, inject_time=80, dataset="test", use_shapley=False)
    print(f"\nResults:")
    print(f"  Method: {result_pagerank['method']}")
    print(f"  Top 5 rankings: {result_pagerank['ranks'][:5]}")
    
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Shapley top 3: {result_shapley['ranks'][:3]}")
    print(f"PageRank top 3: {result_pagerank['ranks'][:3]}")
    print(f"\nExpected root cause: serviceA")
    print(f"Shapley rank for serviceA: {result_shapley['ranks'].index('serviceA') + 1 if 'serviceA' in result_shapley['ranks'] else 'Not found'}")
    print(f"PageRank rank for serviceA: {result_pagerank['ranks'].index('serviceA') + 1 if 'serviceA' in result_pagerank['ranks'] else 'Not found'}")
    
    print("\n" + "=" * 60)
    print("CPG Framework test completed!")
    print("=" * 60)


# Alias for backward compatibility (cpg now uses Causal Shapley Values by default)
cpg_adaptive = cpg
