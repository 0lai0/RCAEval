import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)

def custom_preprocess(df, modality_name, **kwargs):
    """
    模態特定預處理、特徵工程、時間對齊等
    """
    if modality_name == "logts":
        # 日誌數據預處理 - 假設已經轉換為數值特徵
        df = drop_constant(df)
        pass
    elif modality_name in ["tracets_err", "tracets_lat"]:
        # 追蹤數據預處理 - 處理缺失值和常數列
        df = df.fillna(method='ffill').fillna(0)
        df = drop_constant(df)
    elif modality_name == "metric":
        # 指標數據預處理
        df = preprocess(data=df, dataset=kwargs.get("dataset"), 
                       dk_select_useful=kwargs.get("dk_select_useful", False))
    
    # 處理 NaN - 時間序列插值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 清理常數/近常數列
    df = df.loc[:, df.nunique() > 1]  # 移除常數列
    
    return df

def check_stationarity(series, alpha=0.05):
    """
    檢查時間序列的平穩性
    """
    try:
        result = adfuller(series.dropna())
        return result[1] < alpha  # p-value < alpha 表示平穩
    except:
        return False

def granger_causality_test(data, f1, f2, maxlag=5, p_threshold=0.05):
    """
    執行 Granger 因果性測試
    """
    try:
        # 確保數據平穩性
        series1 = data[f1].dropna()
        series2 = data[f2].dropna()
        
        if len(series1) < 20 or len(series2) < 20:
            return False, 1.0
        
        # 檢查平穩性，如果不平穩則進行差分
        if not check_stationarity(series1):
            series1 = series1.diff().dropna()
        if not check_stationarity(series2):
            series2 = series2.diff().dropna()
        
        # 對齊數據
        aligned_data = pd.DataFrame({f2: series2, f1: series1}).dropna()
        
        if len(aligned_data) < 20:
            return False, 1.0
        
        test_result = grangercausalitytests(aligned_data[[f2, f1]], maxlag=maxlag, verbose=False)
        p_value = test_result[1][0]['ssr_ftest'][1] if len(test_result) > 0 else 1.0
        
        return p_value < p_threshold, p_value
    except Exception as e:
        return False, 1.0

def mmrca(
    data,
    inject_time=None,
    dataset=None,
    num_loop=None,
    sli=None,
    anomalies=None,
    domain_knowledge_graph=None,
    historical_data=None,
    system_metadata=None,
    **kwargs
):
    """
    多模態因果根因分析 (Multi-Modal Causal Root Cause Analysis)
    
    Args:
        data: 包含多種模態的原始時序數據 (DataFrame 或 dict)
        inject_time: 異常注入時間點
        domain_knowledge_graph: 預定義的因果關係圖
        historical_data: 歷史數據（用於模型訓練）
        system_metadata: 系統元數據
        **kwargs: 其他參數
    
    Returns:
        dict: 包含排序的根因、異常傳播路徑、因果圖快照和詳細洞察
    """
    
    # === 模組 A: 數據攝入與預處理 ===
    print("Step 1: Data Ingestion & Preprocessing...")
    
    # 處理數據格式：支持 DataFrame 和 dict 兩種格式
    if isinstance(data, pd.DataFrame):
        # 單一 DataFrame 格式 (來自 main.py)
        modality_mapping = {
            "metric": data  # 將 DataFrame 作為 metric 模態處理
        }
    elif isinstance(data, dict):
        # 多模態字典格式
        modality_mapping = {
            "metric": data.get("metric"),
            "logs": data.get("logs"),
            "logts": data.get("logts"),
            "traces": data.get("traces"),
            "tracets_err": data.get("tracets_err"),
            "tracets_lat": data.get("tracets_lat")
        }
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    # 處理異常時間範圍
    if anomalies is not None:
        # 使用 anomalies 參數定義的範圍
        pass
    
    processed_features = {}
    scaler_models = {}
    feature_mapping = {}  # 用來存特徵名稱
    
    # 統一處理各模態數據
    for modality_name, df_raw in modality_mapping.items():
        if df_raw is None or df_raw.empty:
            continue
        
        # 時間對齊與重採樣
        if modality_name == "metric":
            # 指標數據重採樣（每15秒取一個點）
            resample_rate = kwargs.get("metric_resample_rate", 15)
            df_resampled = df_raw.iloc[::resample_rate, :].copy()
        else:
            df_resampled = df_raw.copy()
        
        # 分割正常/異常數據
        if anomalies is None:
            normal_df = df_resampled[df_resampled["time"] < inject_time]
            anomal_df = df_resampled[df_resampled["time"] >= inject_time]
        else:
            normal_df = df_resampled.head(anomalies[0])
            anomal_df = df_resampled.tail(len(df_resampled) - anomalies[0])
        
        # 模態特定預處理
        normal_df = custom_preprocess(normal_df, modality_name, dataset=dataset, **kwargs)
        anomal_df = custom_preprocess(anomal_df, modality_name, dataset=dataset, **kwargs)
        
        # 確保列交集
        common_cols = [x for x in normal_df.columns if x in anomal_df.columns and x != "time"]
        if not common_cols:
            continue
            
        normal_df = normal_df[common_cols]
        anomal_df = anomal_df[common_cols]
        
        # 標準化處理
        scaler_function = kwargs.get("scaler_function", RobustScaler)
        
        for col in common_cols:
            if col == "time":
                continue
                
            # 存特徵名稱映射
            feature_key = f"{modality_name}_{col}"
            feature_mapping[feature_key] = col
            
            # 使用正常數據訓練 scaler
            normal_values = normal_df[col].to_numpy().reshape(-1, 1)
            anomal_values = anomal_df[col].to_numpy().reshape(-1, 1)
            
            scaler = scaler_function().fit(normal_values)
            scaler_models[feature_key] = scaler
            
            # 轉換數據
            processed_features[f"{feature_key}_normal"] = scaler.transform(normal_values)[:, 0]
            processed_features[f"{feature_key}_anomal"] = scaler.transform(anomal_values)[:, 0]
    
    print("Step 1: Data Preprocessing Completed.")
    
    # === 模組 B: 時序異常檢測 ===
    print("Step 2: Time Series Anomaly Detection...")
    
    final_anomaly_scores = {}
    anomaly_threshold = kwargs.get("anomaly_threshold", 3.0)
    
    # 對每個特徵進行異常檢測
    for feature_key in processed_features.keys():
        if feature_key.endswith("_anomal"):
            original_feature_name = feature_key.replace("_anomal", "")
            z_scores = processed_features[feature_key]
            
            # 基於閾值的異常檢測
            anomalous_indices = np.where(np.abs(z_scores) > anomaly_threshold)[0]
            
            if len(anomalous_indices) > 0:
                max_anomaly_score = np.max(np.abs(z_scores[anomalous_indices]))
                final_anomaly_scores[original_feature_name] = max_anomaly_score
    
    print(f"Step 2: Detected {len(final_anomaly_scores)} anomalous features.")
    
    # === 模組 C: 因果圖構建與動態學習 ===
    print("Step 3: Causal Graph Construction & Dynamic Learning...")
    
    G = domain_knowledge_graph if domain_knowledge_graph is not None else nx.DiGraph()
    
    if domain_knowledge_graph is None:
        print("Discovering causal relationships from data...")
        
        # 獲取所有特徵名稱
        all_features = list(set([key.replace("_normal", "").replace("_anomal", "") 
                               for key in processed_features.keys()]))
        
        # 構建因果發現用的數據
        causal_data = {}
        for feature in all_features:
            normal_key = f"{feature}_normal"
            anomal_key = f"{feature}_anomal"
            
            if normal_key in processed_features and anomal_key in processed_features:
                # 合併正常和異常數據
                combined_data = np.concatenate([
                    processed_features[normal_key],
                    processed_features[anomal_key]
                ])
                causal_data[feature] = combined_data
        
        causal_df = pd.DataFrame(causal_data)
        
        # 執行 Granger 因果性測試
        granger_maxlag = kwargs.get("granger_maxlag", 5)
        granger_p_threshold = kwargs.get("granger_p_threshold", 0.05)
        
        # 確保有足夠的數據進行因果性測試
        if len(causal_df) > 20 and len(all_features) > 1:
            for f1 in all_features:
                for f2 in all_features:
                    if f1 == f2:
                        continue
                    
                    is_causal, p_value = granger_causality_test(
                        causal_df, f1, f2, 
                        maxlag=granger_maxlag, 
                        p_threshold=granger_p_threshold
                    )
                    
                    if is_causal:
                        G.add_edge(f1, f2, weight=1-p_value)
        
        print(f"Discovered {G.number_of_edges()} causal relationships.")
    else:
        print("Using provided domain knowledge graph.")
    
    causal_graph_snapshot = G.copy()
    
    # === 模組 D: 根因分析與異常傳播追溯 ===
    print("Step 4: Root Cause Analysis & Anomaly Propagation Tracing...")
    
    ranked_root_causes = []
    anomaly_propagation_paths = {}
    
    all_anomalous_nodes = list(final_anomaly_scores.keys())
    
    if G.number_of_nodes() > 0:
        # 識別根因候選
        potential_root_candidates = []
        
        for node in all_anomalous_nodes:
            if node not in G:
                # 不在圖中的異常節點也作為潛在根因
                potential_root_candidates.append((node, final_anomaly_scores.get(node, 0)))
                continue
            
            # 檢查是否為根因候選（上游節點非異常）
            is_root_candidate = True
            if G.in_degree(node) > 0:
                for pred in G.predecessors(node):
                    if pred in all_anomalous_nodes:
                        is_root_candidate = False
                        break
            
            if is_root_candidate:
                # 計算因果影響力分數
                influence_score = final_anomaly_scores.get(node, 0)
                if G.out_degree(node) > 0:
                    influence_score *= (1 + G.out_degree(node) * 0.1)  # 出度加權
                
                potential_root_candidates.append((node, influence_score))
        
        # 根據影響力分數排序
        ranked_root_causes_with_scores = sorted(
            potential_root_candidates, 
            key=lambda x: x[1], 
            reverse=True
        )
        ranked_root_causes = [rc[0] for rc in ranked_root_causes_with_scores]
        
        # 生成異常傳播路徑
        num_top_rc = kwargs.get("num_top_rc", 5)
        for rc_name in ranked_root_causes[:num_top_rc]:
            if rc_name in G:
                # 找到從根因出發的所有下游節點
                try:
                    downstream_nodes = list(nx.dfs_tree(G, rc_name).nodes())
                    # 篩選實際觀測到異常的下游節點
                    impacted_anomalous_nodes = [
                        n for n in downstream_nodes 
                        if n in all_anomalous_nodes and n != rc_name
                    ]
                    
                    if impacted_anomalous_nodes:
                        # 計算傳播路徑
                        paths = []
                        for target in impacted_anomalous_nodes:
                            try:
                                path = nx.shortest_path(G, rc_name, target)
                                paths.append({
                                    'target': target,
                                    'path': path,
                                    'path_length': len(path) - 1,
                                    'anomaly_score': final_anomaly_scores.get(target, 0)
                                })
                            except nx.NetworkXNoPath:
                                continue
                        
                        anomaly_propagation_paths[rc_name] = paths
                except:
                    continue
    else:
        # 沒有因果圖時，退化為基於異常分數的排名
        ranked_root_causes = sorted(
            final_anomaly_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        ranked_root_causes = [rc[0] for rc in ranked_root_causes]
        print("Warning: No valid causal graph, results based on anomaly scores only.")
    
    print("Step 4: Root Cause Analysis Completed.")
    
    def process_feature_name(feature_name):
        """
        处理特征名称以符合评估系统的期望格式
        """
        parts = feature_name.split("_")
        if len(parts) >= 2:
            service_name = parts[0].replace("-db", "")
            metric_name = "_".join(parts[1:])
            return service_name, metric_name
        else:
            return parts[0].replace("-db", ""), "unknown"

    # 将特征名称转换回原始名称
    final_ranked_root_causes = []
    for rc in ranked_root_causes:
        if rc in feature_mapping:
            # 获取原始特征名称
            original_name = feature_mapping[rc]
            # 处理特征名称
            service_name, metric_name = process_feature_name(original_name)
            final_ranked_root_causes.append(f"{service_name}_{metric_name}")
        else:
            # 如果找不到映射，直接处理特征名称
            service_name, metric_name = process_feature_name(rc)
            final_ranked_root_causes.append(f"{service_name}_{metric_name}")

    # 生成詳細洞察
    detailed_insights = {
        "total_anomalous_features": len(final_anomaly_scores),
        "causal_edges_discovered": G.number_of_edges(),
        "top_anomaly_scores": dict(sorted(final_anomaly_scores.items(), key=lambda x: x[1], reverse=True)[:10]),
        "analysis_summary": f"Identified {len(ranked_root_causes)} potential root causes with {len(anomaly_propagation_paths)} propagation paths."
    }
    
    return {
        "ranked_root_causes": final_ranked_root_causes,
        "anomaly_propagation_paths": anomaly_propagation_paths,
        "causal_graph_snapshot": causal_graph_snapshot,
        "detailed_insights": detailed_insights,
        "node_names": list(final_anomaly_scores.keys()),
        "ranks": final_ranked_root_causes  # 為了與其他方法兼容
    }

def mmbaro_mmrca(data, inject_time=None, dataset=None, **kwargs):
    """
    使用 RobustScaler 的 MMrca 包裝器
    """
    return mmrca(
        data=data, 
        inject_time=inject_time, 
        dataset=dataset, 
        scaler_function=RobustScaler, 
        **kwargs
    )

def mmnsigma_mmrca(data, inject_time=None, dataset=None, **kwargs):
    """
    使用 StandardScaler 的 MMrca 包裝器
    """
    return mmrca(
        data=data, 
        inject_time=inject_time, 
        dataset=dataset, 
        scaler_function=StandardScaler, 
        **kwargs
    ) 