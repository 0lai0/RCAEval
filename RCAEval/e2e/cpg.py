import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import re
from dataclasses import dataclass
from queue import Queue
import json
import logging
from scipy import stats
from scipy.optimize import minimize
import networkx as nx

# 導入RCAEval現有工具
from RCAEval.io.time_series import preprocess, drop_constant
from RCAEval.e2e import rca

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AtomicEvent:
    """meta event事件 E = (t, N, T, D)"""
    t: int  # Unix ms 時間戳
    N: str  # 服務節點標識
    T: str  # 事件類型
    D: Dict[str, Any]  # Payload

@dataclass
class AggregatedEvent:
    """聚合事件 AE = (t_agg, N, F)"""
    t: int  # 窗口內最大時間戳
    N: str  # 服務節點
    F: np.ndarray  # 特徵向量
    
    def __hash__(self):
        """使AggregatedEvent可哈希，以便在集合中使用"""
        return hash((self.t, self.N, tuple(self.F.tolist())))
    
    def __eq__(self, other):
        """定義相等性比較"""
        if not isinstance(other, AggregatedEvent):
            return False
        return (self.t == other.t and 
                self.N == other.N and 
                np.array_equal(self.F, other.F))

class DrainLogParser:
    """簡化的Drain日誌解析器"""
    
    def __init__(self, max_depth=4, sim_threshold=0.4):
        self.max_depth = max_depth
        self.sim_threshold = sim_threshold
        self.templates = {}
        self.template_count = 0
        
    def parse(self, log_message: str) -> str:
        """解析日志消息，返回模板ID"""
        # 簡化實現：基於正則表達式的模板識別
        # 在實際應用中，建議使用完整的Drain3庫
        
        # 預處理：移除數字、IP地址等變量部分
        processed = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', log_message)
        processed = re.sub(r'\b\d+\b', '<NUM>', processed)
        processed = re.sub(r'\b[0-9a-fA-F]{8,}\b', '<HEX>', processed)
        
        # 簡單的模板匹配
        template_key = processed
        if template_key not in self.templates:
            self.template_count += 1
            self.templates[template_key] = f"TEMPLATE_{self.template_count}"
            
        return self.templates[template_key]

class POTAnomalyDetector:
    """POT (Peaks Over Threshold) 異常檢測器"""
    
    def __init__(self, window_size=500, alpha=0.01, min_samples=100):
        self.window_size = window_size
        self.alpha = alpha  # 降低alpha值，提高異常檢測閾值（降低以檢測更多異常）
        self.min_samples = min_samples  # 增加最小樣本數要求
        self.buffer = deque(maxlen=window_size)
        
    def _fit_gpd(self, excesses):
        """擬合廣義帕累托分布"""
        if len(excesses) < self.min_samples:
            return None, None
            
        # 使用矩估計作為初始值
        mean_exc = np.mean(excesses)
        var_exc = np.var(excesses)
        
        # MLE估計
        def neg_log_likelihood(params):
            sigma, xi = params
            if sigma <= 0:
                return np.inf
            if xi != 0:
                if np.any(1 + xi * excesses / sigma <= 0):
                    return np.inf
                return len(excesses) * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(1 + xi * excesses / sigma))
            else:
                return len(excesses) * np.log(sigma) + np.sum(excesses) / sigma
                
        # 初始估計
        sigma_init = mean_exc
        xi_init = -0.5 + mean_exc**2 / var_exc if var_exc > 0 else 0
        
        try:
            result = minimize(neg_log_likelihood, [sigma_init, xi_init], 
                            method='L-BFGS-B', bounds=[(1e-6, None), (-0.5, 0.5)])
            if result.success:
                return result.x[0], result.x[1]  # sigma, xi
        except:
            pass
            
        return sigma_init, xi_init
    
    def detect(self, score: float) -> Optional[float]:
        """檢測異常，返回閾值（如果異常）"""
        self.buffer.append(score)
        
        if len(self.buffer) < self.window_size:
            return None
            
        # 選擇合適的閾值
        percentiles = np.linspace(50, 95, 10)
        best_threshold = None
        best_ks = float('inf')
        
        for p in percentiles:
            threshold = np.percentile(list(self.buffer), p)
            excesses = [x - threshold for x in self.buffer if x > threshold]
            
            if len(excesses) < self.min_samples:
                continue
                
            sigma, xi = self._fit_gpd(excesses)
            if sigma is None:
                continue
                
            # 簡化的KS檢驗
            # 實際應用中應使用完整的統計檢驗
            ks_stat = np.random.random()  # 佔位符
            
            if ks_stat < best_ks:
                best_ks = ks_stat
                best_threshold = threshold
                
        if best_threshold is None:
            return None
            
        # 計算最終閾值
        excesses = [x - best_threshold for x in self.buffer if x > best_threshold]
        sigma, xi = self._fit_gpd(excesses)
        
        if sigma is None:
            return None
            
        n, k = len(self.buffer), len(excesses)
        if k == 0:
            return None
            
        # POT閾值公式
        threshold_final = best_threshold + sigma/xi * ((n/k*(1-self.alpha))**(-xi) - 1)
        
        return threshold_final if score > threshold_final else None

class TransferEntropyCalculator:
    """傳遞熵計算器"""
    
    def __init__(self, lag=1, bins=10):
        self.lag = lag
        self.bins = bins
        
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """計算從x到y的傳遞熵"""
        if len(x) != len(y) or len(x) < self.lag + 1:
            return 0.0
            
        # 離散化
        x_disc = pd.cut(x, bins=self.bins, labels=False, duplicates='drop')
        y_disc = pd.cut(y, bins=self.bins, labels=False, duplicates='drop')
        
        if x_disc is None or y_disc is None:
            return 0.0
            
        # 構建時間序列
        y_present = y_disc[self.lag:]
        y_past = y_disc[:-self.lag]
        x_past = x_disc[:-self.lag]
        
        # 計算聯合概率
        try:
            # P(Y_t, Y_{t-1}, X_{t-1})
            joint_xyz = pd.crosstab([y_present, y_past], x_past, normalize=True)
            # P(Y_t, Y_{t-1})
            joint_yz = pd.crosstab(y_present, y_past, normalize=True)
            # P(Y_t | Y_{t-1}, X_{t-1})
            cond_y_yz = joint_xyz.div(joint_xyz.sum(axis=1), axis=0).fillna(0)
            # P(Y_t | Y_{t-1})
            cond_y_z = joint_yz.div(joint_yz.sum(axis=1), axis=0).fillna(0)
            
            # 計算傳遞熵
            te = 0.0
            for i in joint_xyz.index:
                for j in joint_xyz.columns:
                    p_xyz = joint_xyz.loc[i, j] if j in joint_xyz.columns else 0
                    if p_xyz > 0:
                        p_y_yz = cond_y_yz.loc[i, j] if j in cond_y_yz.columns else 0
                        p_y_z = cond_y_z.loc[i[0], i[1]] if i[1] in cond_y_z.columns else 0
                        
                        if p_y_yz > 0 and p_y_z > 0:
                            te += p_xyz * np.log2(p_y_yz / p_y_z)
                            
            return max(0.0, te)
            
        except Exception as e:
            logger.warning(f"傳遞熵計算失敗: {e}")
            return 0.0

class CPGFramework:
    """CPG框架主類"""
    
    def __init__(self, 
                 agg_window=5000,  # 聚合窗口(ms)
                 anomaly_window=2000,  # 異常檢測窗口（增加以提高穩定性）
                 causal_threshold=0.9,  # 因果關係閾值（降低以檢測更多因果關係alpha）
                 lookback_window=60000,  # 回溯窗口(ms)（增加回溯範圍）
                 top_k=20):  # 因果候選數量（增加候選數）
        
        self.agg_window = agg_window
        self.anomaly_window = anomaly_window
        self.causal_threshold = causal_threshold
        self.lookback_window = lookback_window
        self.top_k = top_k
        
        # 初始化組件
        self.log_parser = DrainLogParser()
        self.anomaly_detector = POTAnomalyDetector(window_size=anomaly_window)
        self.te_calculator = TransferEntropyCalculator()
        
        # 儲存
        self.atomic_events = []
        self.aggregated_events = []
        self.metrics_keys = []
        self.log_templates = []
        
    def _extract_atomic_events_from_metrics(self, metrics_df: pd.DataFrame) -> List[AtomicEvent]:
        """從Metrics資料提取meta event"""
        events = []
        
        # 確保time列存在
        if 'time' not in metrics_df.columns:
            logger.error("Metrics資料缺少time列")
            return events
            
        # 獲取metrics列名並預計算統計量（性能優化）
        metric_cols = [col for col in metrics_df.columns if col != 'time']
        self.metrics_keys = metric_cols
        
        # 預計算所有列的均值和標準差
        col_stats = {}
        for col in metric_cols:
            col_stats[col] = {
                'mean': metrics_df[col].mean(),
                'std': metrics_df[col].std() + 1e-6
            }
        
        # 轉換時間列為毫秒（向量化操作）
        times_ms = (metrics_df['time'] * 1000).astype(int)
        
        # 批量處理事件生成（性能優化）
        for idx, (_, row) in enumerate(metrics_df.iterrows()):
            try:
                t = times_ms.iloc[idx]
                
                for col in metric_cols:
                    value = row[col]
                    if pd.notna(value):
                        # 使用預計算的統計量
                        normalized_value = (value - col_stats[col]['mean']) / col_stats[col]['std']
                        
                        # 只處理顯著變化的值（性能優化）
                        if abs(normalized_value) > 0.1:  # 閾值過濾
                            if '_container' in col or '_' in col:
                                svc = col.split('_')[0]
                                metric = col.split('_', 1)[1]
                            else:
                                svc = col
                                metric = 'value'
                            
                            event = AtomicEvent(
                                t=t,
                                N=svc,
                                T=f"METRIC_{metric}",
                                D={"value": normalized_value}
                            )
                            events.append(event)
            except Exception as e:
                logger.warning(f"處理Metrics事件時出錯: {str(e)}")
                continue
                    
        logger.info(f"性能優化：從{len(metrics_df) * len(metric_cols)}個潛在事件中提取了{len(events)}個顯著事件")
        return events
    
    def _extract_atomic_events_from_logs(self, logs_df: pd.DataFrame) -> List[AtomicEvent]:
        """從Logs資料提取meta event"""
        events = []
        
        for _, row in logs_df.iterrows():
            try:
                if 'timestamp' in row:
                    # 使用納秒時間戳的前13位作為毫秒
                    t = int(str(row['timestamp'])[:13])
                elif 'time' in row:
                    t = int(row['time'] * 1000)
                else:
                    logger.warning("Logs資料缺少時間戳")
                    continue
                    
                svc = row['container_name'].split('-')[0] if 'container_name' in row else 'unknown'
                message = row['message'] if 'message' in row else ''
            
            except Exception as e:
                logger.warning(f"處理Logs事件時出錯: {str(e)}")
                continue
            
            # 解析日志模板
            template_id = self.log_parser.parse(message)
            if template_id not in self.log_templates:
                self.log_templates.append(template_id)
                
            event = AtomicEvent(
                t=t,
                N=svc,
                T=f"LOG_{template_id}",
                D={"count": 1}
            )
            events.append(event)
            
        return events
    
    def _extract_atomic_events_from_traces(self, traces_df: pd.DataFrame) -> List[AtomicEvent]:
        """從Traces資料提取meta event"""
        events = []
        
        for _, row in traces_df.iterrows():
            try:
                svc = row.get('serviceName', 'unknown')
                operation = row.get('operationName', row.get('methodName', 'unknown'))
                
                # 檢查時間戳
                if 'startTimeMillis' in row:
                    start_time = row['startTimeMillis']
                elif 'time' in row:
                    start_time = row['time'] * 1000
                else:
                    logger.warning("Traces資料缺少時間戳")
                    continue
                    
                duration = row.get('duration', 0)
            except Exception as e:
                logger.warning(f"處理Trace事件時出錯: {str(e)}")
                continue
            
            t0 = int(start_time)
            t1 = int(start_time + duration)
            
            # 開始事件
            start_event = AtomicEvent(
                t=t0,
                N=svc,
                T=f"TRACE_START_{operation}",
                D={}
            )
            events.append(start_event)
            
            # 結束事件
            end_event = AtomicEvent(
                t=t1,
                N=svc,
                T=f"TRACE_END_{operation}",
                D={"duration": duration}
            )
            events.append(end_event)
            
        return events
    
    def _aggregate_events(self, events: List[AtomicEvent]) -> List[AggregatedEvent]:
        """聚合meta event"""
        # 按service分組
        service_events = defaultdict(list)
        for event in events:
            service_events[event.N].append(event)
            
        aggregated = []
        
        for service, svc_events in service_events.items():
            # 按時間排序
            svc_events.sort(key=lambda e: e.t)
            
            # 滑動窗口聚合
            buffer = []
            window_start = None
            
            for event in svc_events:
                if not buffer:
                    buffer = [event]
                    window_start = event.t
                elif event.t - window_start <= self.agg_window:
                    buffer.append(event)
                else:
                    # 生成聚合事件
                    agg_event = self._merge_events(buffer, service)
                    if agg_event is not None:
                        aggregated.append(agg_event)
                    
                    # 開始新窗口
                    buffer = [event]
                    window_start = event.t
                    
            # 處理最後一個窗口
            if buffer:
                agg_event = self._merge_events(buffer, service)
                if agg_event is not None:
                    aggregated.append(agg_event)
                
        return sorted(aggregated, key=lambda ae: ae.t)
    
    def _merge_events(self, events: List[AtomicEvent], service: str) -> Optional[AggregatedEvent]:
        """合併事件緩衝區為聚合事件"""
        try:
            t_agg = max(e.t for e in events)
            
            # 初始化特徵向量
            # 維度: 3*M + L + 4 (M=metrics數量（mean, max, last）, L=日誌模板數量, 4=trace特徵)
            M = len(self.metrics_keys)
            L = len(self.log_templates)
            feature_dim = 3 * M + L + 4
            
            F = np.zeros(feature_dim)
            
            # 處理metrics特徵
            metric_values = defaultdict(list)
            log_counts = defaultdict(int)
            trace_starts = 0
            trace_ends = 0
            durations = []
            
            for event in events:
                if event.T.startswith("METRIC_"):
                    metric_name = event.T.split("METRIC_")[1]
                    if f"{service}_{metric_name}" in self.metrics_keys:
                        metric_values[metric_name].append(event.D["value"])
                elif event.T.startswith("LOG_"):
                    template_id = event.T.split("LOG_")[1]
                    log_counts[template_id] += 1
                elif event.T.startswith("TRACE_START"):
                    trace_starts += 1
                elif event.T.startswith("TRACE_END"):
                    trace_ends += 1
                    if "duration" in event.D:
                        durations.append(event.D["duration"])
        except Exception as e:
            logger.warning(f"合併事件時出錯: {str(e)}")
            return None
        
        try:
            # 加入metrics特徵 (mean, max, last)     
            idx = 0
            for i, metric_key in enumerate(self.metrics_keys):
                if metric_key.startswith(service + "_"):
                    metric_name = metric_key.split("_", 1)[1]
                    values = metric_values.get(metric_name, [])
                    if values:
                        F[idx] = np.mean(values)      # mean
                        F[idx + 1] = np.max(values)   # max
                        F[idx + 2] = values[-1]       # last
                idx += 3
                
            # 加入日誌特徵
            for i, template in enumerate(self.log_templates):
                F[3 * M + i] = log_counts.get(template, 0)
                
            # 加入trace特徵
            trace_idx = 3 * M + L
            F[trace_idx] = trace_starts
            F[trace_idx + 1] = trace_ends
            F[trace_idx + 2] = sum(durations) if durations else 0
            F[trace_idx + 3] = np.mean(durations) if durations else 0

            logger.info(f"聚合事件: {t_agg}, {service}, {F}")
            return AggregatedEvent(t=t_agg, N=service, F=F)
        except Exception as e:
            logger.warning(f"加入特徵向量時出錯: {str(e)}")
            return None
    
    def _detect_anomalies(self, events: List[AggregatedEvent]) -> List[AggregatedEvent]:
        """檢測異常事件"""
        anomalies = []
        
        if not events:
            return anomalies
        
        # 計算所有事件的分數並進行預篩選（性能優化）
        scores = []
        for event in events:
            try:
                # 改進的異常分數計算：結合L2範數（衡量向量大小）和統計特徵（衡量變異程度）
                l2_norm = np.linalg.norm(event.F)
                
                # 計算特徵向量的變異係數和偏度
                non_zero_features = event.F[event.F != 0]
                if len(non_zero_features) > 1:
                    cv = np.std(non_zero_features) / (np.mean(np.abs(non_zero_features)) + 1e-6)   # 特徵變異係數
                    skewness = np.mean(((non_zero_features - np.mean(non_zero_features)) / (np.std(non_zero_features) + 1e-6)) ** 3) # 特徵偏度
                    score = l2_norm * (1 + 0.1 * cv + 0.05 * abs(skewness)) # 異常分數
                else:
                    score = l2_norm
                    
                scores.append((event, score))
            except Exception as e:
                logger.warning(f"計算異常分數時出錯: {str(e)}")
                continue
        
        # 按分數排序並只檢測前50%的事件（性能優化）
        scores.sort(key=lambda x: x[1], reverse=True)
        top_events = scores[:max(1, len(scores) // 2)]
        
        logger.info(f"性能優化：從{len(events)}個聚合事件中預選了{len(top_events)}個高分事件進行異常檢測")
        
        for event, score in top_events:
            try:
                # POT異常檢測
                threshold = self.anomaly_detector.detect(score)
                if threshold is not None:
                    anomalies.append(event)
                    logger.info(f"在服務 {event.N} 的時間點 {event.t} 檢測到異常，分數: {score:.3f}, 閾值: {threshold:.3f}")
            except Exception as e:
                logger.warning(f"POT檢測異常時出錯: {str(e)}")
                continue
                
        return anomalies
    
    def _build_causal_graph(self, symptom_event: AggregatedEvent, 
                          all_events: List[AggregatedEvent]) -> Tuple[List, List]:
        """建立因果圖"""
        try:
            nodes = {symptom_event}
            edges = []
            queue = Queue()
            queue.put(symptom_event)
            
            # 服務依賴關係（實際應用中需要配置文件）
            service_deps = self._get_service_dependencies()
            
            while not queue.empty():
                current_event = queue.get()
                
                # 查找上游服務
                upstream_services = service_deps.get(current_event.N, [])
                
                for upstream_svc in upstream_services:
                    try:
                        # 在時間窗口內查找候選事件
                        candidates = [
                            e for e in all_events 
                            if e.N == upstream_svc and 
                            current_event.t - self.lookback_window <= e.t < current_event.t
                        ]
                        
                        # 按時間排序，取最近的K個
                        candidates = sorted(candidates, key=lambda e: e.t, reverse=True)[:self.top_k]
                        
                        for candidate in candidates:
                            # 計算傳遞熵
                            te = self.te_calculator.calculate(candidate.F, current_event.F)
                            
                            if te > self.causal_threshold:
                                if candidate not in nodes:
                                    nodes.add(candidate)
                                    queue.put(candidate)
                                
                                edges.append((candidate, current_event, te))
                                logger.info(f"因果邊: {candidate.N} -> {current_event.N}, 傳遞熵: {te:.3f}")
                    except Exception as e:
                        logger.warning(f"處理上游服務 {upstream_svc} 時出錯: {str(e)}")
                        continue
            
            return list(nodes), edges
        except Exception as e:
            logger.error(f"建立因果圖時出錯: {str(e)}")
            return [], []
    
    def _get_service_dependencies(self) -> Dict[str, List[str]]:
        """獲取服務依賴關係（簡化版本）"""
        # 實際應用中應該從配置文件或服務註冊中心獲取
        try:
            return {
                'frontend': ['cartservice', 'catalogservice', 'recommendationservice'],
                'checkoutservice': ['cartservice', 'emailservice', 'paymentservice'],
                'cartservice': ['redis'],
                'catalogservice': ['catalogdb'],
                'recommendationservice': ['catalogservice'],
            }
        except Exception as e:
            logger.error(f"獲取服務依賴關係時出錯: {str(e)}")
            return {}
    
    def _calculate_root_cause_scores(self, nodes: List, edges: List) -> Dict[str, float]:
        """計算根因貢獻度"""
        try:
            # 構建NetworkX圖
            G = nx.DiGraph()
            
            # 添加節點
            for node in nodes:
                G.add_node(node.N)
                
            # 添加邊
            for source, target, weight in edges:
                G.add_edge(source.N, target.N, weight=weight)
            
            # 計算PageRank分數
            try:
                pagerank_scores = nx.pagerank(G, weight='weight')
            except:
                # 如果圖為空或有問題，返回均勻分佈
                pagerank_scores = {node.N: 1.0/len(nodes) for node in nodes}
                
            return pagerank_scores
        except Exception as e:
            logger.error(f"計算根因分數時出錯: {str(e)}")
            return {node.N: 1.0/len(nodes) if nodes else 0.0 for node in nodes}
    
    def process(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """處理多模態數據"""
        try:
            logger.info("開始CPG處理流程")
            
            # 第一步：元事件化
            logger.info("步驟1: 元事件化")
            all_events = []
            
            if 'metric' in data:
                metric_events = self._extract_atomic_events_from_metrics(data['metric'])
                all_events.extend(metric_events)
                logger.info(f"提取到 {len(metric_events)} 個metrics事件")
                
            if 'logts' in data:
                log_events = self._extract_atomic_events_from_logs(data['logts'])
                all_events.extend(log_events)
                logger.info(f"提取到 {len(log_events)} 個日誌事件")
                
            if 'traces' in data:
                trace_events = self._extract_atomic_events_from_traces(data['traces'])
                all_events.extend(trace_events)
                logger.info(f"提取到 {len(trace_events)} 個trace事件")
            
            if not all_events:
                logger.warning("未提取到任何事件")
                return self._empty_result()
            
            # 第二步：多模態滑窗聚合
            logger.info("步驟2: 多模態滑窗聚合")
            self.aggregated_events = self._aggregate_events(all_events)
            logger.info(f"生成 {len(self.aggregated_events)} 個聚合事件")
            
            if not self.aggregated_events:
                logger.warning("聚合事件為空")
                return self._empty_result()
            
            # 第三步：全局異常檢測
            logger.info("步驟3: 全局異常檢測")
            anomaly_events = self._detect_anomalies(self.aggregated_events)
            logger.info(f"檢測到 {len(anomaly_events)} 個異常事件")
            
            if not anomaly_events:
                logger.warning("未檢測到異常事件")
                return self._empty_result()
            
            # 第四步：局部因果圖構建
            logger.info("步驟4: 局部因果圖構建")
            all_nodes = set()
            all_edges = []
            
            for symptom in anomaly_events:
                nodes, edges = self._build_causal_graph(symptom, self.aggregated_events)
                all_nodes.update(nodes)
                all_edges.extend(edges)
                
            # 第五步：多因子歸因
            logger.info("步驟5: 多因子歸因")
            node_list = list(all_nodes)
            root_cause_scores = self._calculate_root_cause_scores(node_list, all_edges)
            
            # 排序根因
            ranked_services = sorted(root_cause_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            # 構建鄰接矩陣
            service_names = [node.N for node in node_list]
            adj_matrix = np.zeros((len(service_names), len(service_names)))
            
            for source, target, weight in all_edges:
                try:
                    i = service_names.index(source.N)
                    j = service_names.index(target.N)
                    adj_matrix[i][j] = weight
                except ValueError:
                    continue
            
            # 格式化輸出
            result = {
                "ranks": [service for service, _ in ranked_services],
                "adj": adj_matrix,
                "node_names": service_names,
                "causal_edges": [(e[0].N, e[1].N, e[2]) for e in all_edges],
                "anomaly_events": len(anomaly_events),
                "total_events": len(all_events)
            }
            
            logger.info(f"CPG處理完成，識別出前5個根因: {result['ranks'][:5]}")
            return result
            
        except Exception as e:
            logger.error(f"CPG分析失敗: {str(e)}")
            return self._empty_result()
            
    def _empty_result(self) -> Dict[str, Any]:
        """返回空結果"""
        return {
            "ranks": [],
            "adj": np.array([]),
            "node_names": [],
            "causal_edges": [],
            "anomaly_events": 0,
            "total_events": 0
        }

@rca
def cpg(data, inject_time=None, dataset=None, **kwargs):
    """
    CPG: 多模態事件驅動因果傳播框架
    
    Args:
        data: 多模態數據，可以是DataFrame或dict
              如果是dict，格式為 {"metric": df, "logts": df, "traces": df}
        inject_time: 異常注入時間點
        dataset: 數據集名稱
        **kwargs: 其他參數
            - agg_window: 聚合窗口大小(ms)，默認5000
            - anomaly_window: 異常檢測窗口大小，默認1000
            - causal_threshold: 因果關係閾值，默認0.1
            - lookback_window: 回溯窗口(ms)，默認30000
            - top_k: 因果候選數量，默認5
    
    Returns:
        dict: {
            "ranks": 根因排序列表,
            "adj": 鄰接矩陣,
            "node_names": 節點名稱列表,
            "causal_edges": 因果邊列表,
            "anomaly_events": 異常事件數量,
            "total_events": 總事件數量
        }
    """
    
    logger.info(f"啟動CPG方法，數據集: {dataset}")
    
    try:
        # 參數配置（優化後的默認值）
        config = {
            'agg_window': kwargs.get('agg_window', 5000),
            'anomaly_window': kwargs.get('anomaly_window', 2000),  # 增加穩定性
            'causal_threshold': kwargs.get('causal_threshold', 0.05),  # 降低閾值檢測更多因果關係
            'lookback_window': kwargs.get('lookback_window', 60000),  # 增加回溯範圍
            'top_k': kwargs.get('top_k', 10)  # 增加候選數量
        }
        
        # 初始化CPG框架
        cpg_framework = CPGFramework(**config)
        
        # 處理輸入數據
        if isinstance(data, dict):
            # 多模態數據
            processed_data = {}
            
            if 'metric' in data:
                metric_data = data['metric']
                logger.info(f"原始metrics數據列名: {list(metric_data.columns)}")
                logger.info(f"原始metrics數據形狀: {metric_data.shape}")

                # 在預處理前刪除 metrics 的重複欄位
                metric_data = metric_data.loc[:, ~metric_data.columns.duplicated(keep='first')]
                logger.info(f"清理重複列後的列名: {list(metric_data.columns)}")
                logger.info("已自動清理重複的 metrics 欄位")

                # 檢查time列是否存在
                if 'time' not in metric_data.columns:
                    logger.error(f"清理後仍然缺少time列！可用列: {list(metric_data.columns)}")
                    return {
                        "ranks": [],
                        "adj": np.array([]),
                        "node_names": [],
                        "causal_edges": [],
                        "anomaly_events": 0,
                        "total_events": 0
                    }

                # 保存原始時間列
                original_time = metric_data['time'].copy()
                logger.info(f"保存的時間列範圍: {original_time.min()} - {original_time.max()}")

                if inject_time is not None:
                    # 分離正常和異常數據
                    normal_data = metric_data[metric_data['time'] < inject_time]
                    anomal_data = metric_data[metric_data['time'] >= inject_time]
                    processed_metric = pd.concat([normal_data, anomal_data], ignore_index=True)
                    original_time = processed_metric['time'].copy()  # 更新時間列
                    logger.info(f"注入時間分離後的數據形狀: {processed_metric.shape}")
                else:
                    processed_metric = metric_data
                    
                logger.info(f"預處理前的列名: {list(processed_metric.columns)}")
                processed_metric = preprocess(
                    data=processed_metric, 
                    dataset=dataset, 
                    dk_select_useful=kwargs.get("dk_select_useful", False)
                )
                logger.info(f"預處理後的列名: {list(processed_metric.columns)}")
                logger.info(f"預處理後的數據形狀: {processed_metric.shape}")
                
                # 恢復 time 欄位以供原子事件提取
                processed_metric['time'] = original_time.reset_index(drop=True)
                logger.info(f"恢復time列後的列名: {list(processed_metric.columns)}")
                logger.info(f"恢復的時間列長度: {len(processed_metric['time'])}")
                
                processed_data['metric'] = processed_metric
                
            if 'logts' in data:
                logts_data = data['logts']
                logts_data = drop_constant(logts_data)
                processed_data['logts'] = logts_data
                
            if 'traces' in data:
                processed_data['traces'] = data['traces']
                
        else:
            # 單模態數據（metrics）
            logger.info(f"處理單模態數據，列名: {list(data.columns)}")
            
            # 檢查並清理重複列
            if data.columns.duplicated().any():
                data = data.loc[:, ~data.columns.duplicated(keep='first')]
                logger.info("清理了重複的列")
            
            # 檢查time列是否存在
            if 'time' not in data.columns:
                logger.error(f"單模態數據缺少time列,可用列: {list(data.columns)}")
                return {
                    "ranks": [],
                    "adj": np.array([]),
                    "node_names": [],
                    "causal_edges": [],
                    "anomaly_events": 0,
                    "total_events": 0
                }
            
            # 保存原始時間列
            original_time = data['time'].copy()
            logger.info(f"保存單模態數據時間列範圍: {original_time.min()} - {original_time.max()}")
            
            if inject_time is not None:
                normal_data = data[data['time'] < inject_time]
                anomal_data = data[data['time'] >= inject_time]
                processed_data = pd.concat([normal_data, anomal_data], ignore_index=True)
                original_time = processed_data['time'].copy()  # 更新時間列
            else:
                processed_data = data
                
            processed_data = preprocess(
                data=processed_data, 
                dataset=dataset, 
                dk_select_useful=kwargs.get("dk_select_useful", False)
            )
            
            # 恢復time列
            processed_data['time'] = original_time.reset_index(drop=True)
            logger.info(f"單模態數據恢復time列後的列名: {list(processed_data.columns)}")
            
            # 轉換為多模態格式
            processed_data = {'metric': processed_data}
        
        # 運行CPG分析
        result = cpg_framework.process(processed_data)
        
        logger.info(f"CPG分析完成，檢測到 {result.get('anomaly_events', 0)} 個異常事件")
        logger.info(f"建立因果圖包含 {len(result.get('causal_edges', []))} 條因果邊")
        
        return result
        
    except Exception as e:
        logger.error(f"CPG分析失敗: {str(e)}")
        # 返回空結果
        return {
            "ranks": [],
            "adj": np.array([]),
            "node_names": [],
            "causal_edges": [],
            "anomaly_events": 0,
            "total_events": 0
        }

if __name__ == "__main__":
    # 測試代碼
    print("CPG框架測試")
    
    # 創建測試數據
    test_metric_data = pd.DataFrame({
        'time': range(100),
        'service_a_cpu': np.random.normal(0.5, 0.1, 100),
        'service_a_memory': np.random.normal(0.6, 0.1, 100),
        'service_b_cpu': np.random.normal(0.4, 0.1, 100)
    })
    
    test_log_data = pd.DataFrame({
        'time': range(100),
        'timestamp': [f"{i}000000000" for i in range(100)],
        'container_name': ['service-a-pod'] * 50 + ['service-b-pod'] * 50,
        'message': ['Connection established'] * 30 + ['Error occurred'] * 20 + ['Request processed'] * 50
    })
    
    multimodal_data = {
        'metric': test_metric_data,
        'logts': test_log_data
    }
    
    try:
        result = cpg(multimodal_data, inject_time=50)
        print(f"測試結果: {result}")
    except Exception as e:
        print(f"測試失敗: {str(e)}")