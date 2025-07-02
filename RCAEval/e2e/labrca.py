import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

from RCAEval.io.time_series import preprocess
from RCAEval.e2e.code_analyzer import StackTraceAnalyzer

class FeatureExtractor(nn.Module): 
    def __init__(self, input_dim, hidden_dim=32):
        super(FeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class MultiModalEncoder(nn.Module):
    def __init__(self, metric_dim, log_dim=None, trace_dim=None, hidden_dim=32):
        super(MultiModalEncoder, self).__init__()
        self.metric_encoder = nn.LSTM(metric_dim, hidden_dim, 
                                    num_layers=1,
                                    bidirectional=True, 
                                    batch_first=True)
        
        self.log_encoder = nn.Linear(log_dim, hidden_dim) if log_dim else None
        self.trace_encoder = nn.Linear(trace_dim, hidden_dim) if trace_dim else None
        
        # 注意力機制
        self.multihead_attention = nn.MultiheadAttention(hidden_dim*2, num_heads=8)
        
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
        # code-level 特徵融合
        self.code_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, metrics, logs=None, traces=None):
        # 1. 處理指標數據
        metric_hidden, _ = self.metric_encoder(metrics)
        attn_out, _ = self.multihead_attention(metric_hidden, metric_hidden, metric_hidden)
        
        # 2. 融合日誌特徵 (如果有的話)
        if logs is not None and self.log_encoder:
            log_hidden = self.log_encoder(logs)
            if len(log_hidden.shape) == 2:
                log_hidden = log_hidden.unsqueeze(1).expand(-1, attn_out.size(1), -1)
            attn_out = attn_out + log_hidden
            
        # 3. 融合追蹤特徵 (如果有的話)
        if traces is not None and self.trace_encoder:
            trace_hidden = self.trace_encoder(traces)
            if len(trace_hidden.shape) == 2:
                trace_hidden = trace_hidden.unsqueeze(1).expand(-1, attn_out.size(1), -1)
            attn_out = attn_out + trace_hidden
        
        # 4. 如果有多模態數據，使用特殊的 code-level 注意力
        if logs is not None and traces is not None:
            # 將三種模態組合進行 code-level 分析
            combined_features = torch.cat([
                attn_out.mean(dim=1),  # metric features
                log_hidden.mean(dim=1) if logs is not None else torch.zeros_like(attn_out.mean(dim=1)),
                trace_hidden.mean(dim=1) if traces is not None else torch.zeros_like(attn_out.mean(dim=1))
            ], dim=-1)
            
            code_level_features = self.code_fusion(combined_features)
            return self.dropout(code_level_features).unsqueeze(1)
        
        # 5. 標準融合
        fused = self.dropout(self.fusion(attn_out))
        return fused

def extract_statistical_features(data):
    """提取統計特徵"""
    features = pd.DataFrame()
    
    # 基本統計量
    features['mean'] = data.mean()
    features['std'] = data.std()
    features['skew'] = data.skew()
    features['kurt'] = data.kurtosis()
    features['max'] = data.max()
    features['min'] = data.min()
    
    # 時間序列特徵
    features['diff_mean'] = data.diff().mean()
    features['diff_std'] = data.diff().std()
    
    return features

def calculate_change_point_score(normal_data, anomaly_data):
    """計算變化點分數"""
    scores = {}
    
    for col in normal_data.columns:
        # 計算均值變化
        mean_change = abs(normal_data[col].mean() - anomaly_data[col].mean())
        
        # 計算方差變化
        std_change = abs(normal_data[col].std() - anomaly_data[col].std())
        
        # 計算分布變化（使用KS統計量）
        try:
            from scipy import stats
            _, p_value = stats.ks_2samp(normal_data[col], anomaly_data[col])
            dist_change = 1 - p_value
        except:
            dist_change = 0
            
        # 綜合分數
        scores[col] = (mean_change + std_change + dist_change) / 3
        
    return scores

def labrca(data, inject_time=None, dataset=None, sli=None, anomalies=None, logs=None, traces=None, **kwargs):
    """增強的根因分析方法，支援 code-level 分析"""
    
    # 1. 數據預處理
    time_col = data["time"]
    data = preprocess(
        data=data,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    data["time"] = time_col
    
    # 2. 分割數據
    if inject_time is not None:
        normal_data = data[data["time"] < inject_time]
        anomaly_data = data[data["time"] >= inject_time]
    else:
        split_point = len(data) // 2
        normal_data = data.head(split_point)
        anomaly_data = data.tail(len(data) - split_point)
    
    # 3. 特徵標準化
    scaler = RobustScaler()
    normal_scaled = pd.DataFrame(
        scaler.fit_transform(normal_data.drop(columns=["time"])),
        columns=normal_data.drop(columns=["time"]).columns
    )
    anomaly_scaled = pd.DataFrame(
        scaler.transform(anomaly_data.drop(columns=["time"])),
        columns=anomaly_data.drop(columns=["time"]).columns
    )
    
    # 4. Code-level 分析：處理日誌和 stack traces
    stack_analyzer = StackTraceAnalyzer()
    log_features = {}
    stack_features = {}
    
    if logs is not None:
        log_features, stack_features = stack_analyzer.analyze_log_patterns(logs, inject_time)
        
        # 顯示發現的stack traces
        has_stack_traces = any(v > 0 for k, v in stack_features.items() if 'stack_count' in k)
        if has_stack_traces:
            print("[LabRCA] Code-level analysis enabled")
    
    # 5. 提取統計特徵
    normal_stats = extract_statistical_features(normal_scaled)
    anomaly_stats = extract_statistical_features(anomaly_scaled)
    
    # 6. 計算變化點分數
    change_point_scores = calculate_change_point_score(normal_scaled, anomaly_scaled)
    
    # 7. 使用IsolationForest檢測異常
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(normal_scaled)
    anomaly_scores_iso = -iso_forest.score_samples(anomaly_scaled)
    
    # 8. 多模態深度學習特徵提取
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 準備多模態數據
    log_dim = len(log_features) if log_features else None
    trace_dim = len(stack_features) if stack_features else None
    
    # 創建多模態編碼器
    multimodal_encoder = MultiModalEncoder(
        metric_dim=normal_scaled.shape[1],
        log_dim=log_dim,
        trace_dim=trace_dim,
        hidden_dim=64
    ).to(device)
    
    feature_extractor = FeatureExtractor(normal_scaled.shape[1], hidden_dim=64).to(device)
    
    # 準備時間窗口數據
    window_size = 10
    normal_windows = []
    anomaly_windows = []
    
    for i in range(len(normal_scaled) - window_size + 1):
        normal_windows.append(normal_scaled.iloc[i:i+window_size].values)
    for i in range(len(anomaly_scaled) - window_size + 1):
        anomaly_windows.append(anomaly_scaled.iloc[i:i+window_size].values)
    
    if len(normal_windows) == 0 or len(anomaly_windows) == 0:
        # 如果數據不夠，直接返回基於變化點的結果
        ranks = sorted(change_point_scores.items(), key=lambda x: x[1], reverse=True)
        return {
            "ranks": [x[0] for x in ranks],
            "node_names": list(normal_data.columns.drop("time")),
            "scores": change_point_scores
        }
    
    normal_tensor = torch.FloatTensor(normal_windows).to(device)
    anomaly_tensor = torch.FloatTensor(anomaly_windows).to(device)
    
    # 準備日誌和追蹤特徵張量
    log_tensor = None
    trace_tensor = None
    
    if log_features:
        log_array = np.array([list(log_features.values())] * len(normal_windows))
        log_tensor = torch.FloatTensor(log_array).to(device)
        
    if stack_features:
        trace_array = np.array([list(stack_features.values())] * len(normal_windows))
        trace_tensor = torch.FloatTensor(trace_array).to(device)
    
    # 訓練特徵提取器
    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters()) + list(multimodal_encoder.parameters()), 
        lr=0.001
    )
    
    n_epochs = 300
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 多模態特徵提取
        multimodal_features = multimodal_encoder(normal_tensor, log_tensor, trace_tensor)
        normal_features = feature_extractor(normal_tensor)
        
        # 結合損失
        reconstruction_loss = F.mse_loss(normal_features, normal_tensor)
        multimodal_loss = F.mse_loss(multimodal_features.squeeze(), normal_tensor.mean(dim=1))
        
        total_loss = reconstruction_loss + 0.3 * multimodal_loss
        total_loss.backward()
        optimizer.step()
    
    # 9. 計算最終分數
    final_scores = {}
    with torch.no_grad():
        # 正常和異常數據的多模態特徵
        normal_multimodal = multimodal_encoder(normal_tensor, log_tensor, trace_tensor)
        anomaly_log_tensor = log_tensor[-len(anomaly_windows):] if log_tensor is not None else None
        anomaly_trace_tensor = trace_tensor[-len(anomaly_windows):] if trace_tensor is not None else None
        anomaly_multimodal = multimodal_encoder(anomaly_tensor, anomaly_log_tensor, anomaly_trace_tensor)
        
        # 傳統特徵
        normal_features = feature_extractor(normal_tensor)
        anomaly_features = feature_extractor(anomaly_tensor)
        
        for i, node in enumerate(normal_scaled.columns):
            # 深度學習特徵分數
            dl_score = torch.mean(torch.abs(anomaly_features[:, :, i] - 
                                          normal_features[:, :, i])).item()
            
            # 多模態特徵分數
            multimodal_score = torch.mean(torch.abs(anomaly_multimodal.squeeze()[:, i] - 
                                                  normal_multimodal.squeeze()[:, i])).item()
            
            # 統計特徵分數
            stats_score = np.mean(np.abs(anomaly_stats.iloc[i] - normal_stats.iloc[i]))
            
            # 變化點分數
            cp_score = change_point_scores[node]
            
            # 孤立森林分數
            iso_score = anomaly_scores_iso[min(i, len(anomaly_scores_iso)-1)]
            
            # Code-level 分數加成
            code_level_bonus = 0
            service_name = node.split('_')[0]
            
            # 檢查是否有相關的 stack trace
            if f"{service_name}_stack_count" in stack_features:
                stack_count = stack_features[f"{service_name}_stack_count"]
                if stack_count > 0:
                    code_level_bonus += 0.5  # stack trace 存在加分
                    
            if f"{service_name}_error_severity" in stack_features:
                error_severity = stack_features[f"{service_name}_error_severity"]
                code_level_bonus += error_severity * 0.3  # 錯誤嚴重性加分
                
            if f"{service_name}_has_code_error" in stack_features:
                has_code_error = stack_features[f"{service_name}_has_code_error"]
                code_level_bonus += has_code_error * 0.4  # 有程式碼錯誤加分
            
            # 計算時序相關性
            temporal_corr = np.corrcoef(normal_scaled[node], anomaly_scaled[node])[0, 1]
            temporal_score = 1 - abs(temporal_corr) if not np.isnan(temporal_corr) else 0.5
            
            # 綜合評分 (增加 code-level 權重)
            final_scores[node] = (
                0.25 * dl_score +           # 深度學習特徵
                0.25 * multimodal_score +   # 多模態特徵  
                0.15 * stats_score +        # 統計特徵
                0.15 * cp_score +           # 變化點分數
                0.1 * iso_score +           # 異常檢測分數
                0.1 * temporal_score +      # 時序相關性
                code_level_bonus            # Code-level 加成
            )
    
    # 10. 排序根因
    ranks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    
    # 11. 如果有 stack trace 信息，將相關服務提前
    if stack_features:
        priority_services = []
        for key, value in stack_features.items():
            if 'has_code_error' in key and value > 0:
                service_name = key.replace('_has_code_error', '')
                priority_services.append(service_name)
        
        if priority_services:
            print(f"[Code-Level] Services with errors: {priority_services}")
            # 重新排列：有 code-level 錯誤的服務優先
            prioritized_ranks = []
            remaining_ranks = []
            
            for rank in ranks:
                service_name = rank.split('_')[0]
                if service_name in priority_services:
                    prioritized_ranks.append(rank)
                else:
                    remaining_ranks.append(rank)
            
            ranks = prioritized_ranks + remaining_ranks
    
    # 輸出 Top 3 根因
    print(f"[Result] Top 3 root causes:")
    for i, rank in enumerate(ranks[:3], 1):
        score = final_scores[rank]
        print(f"  {i}. {rank} (score: {score:.3f})")
    
    return {
        "ranks": ranks,
        "node_names": list(normal_data.columns.drop("time")),
        "scores": final_scores,
        "code_level_analysis": {
            "log_features": log_features,
            "stack_features": stack_features,
            "has_stack_traces": len(stack_features) > 0
        }
    }


if __name__ == "__main__":
    print("LabRCA - Enhanced Root Cause Analysis")
    print("Features: Multi-modal analysis, Code-level error detection") 