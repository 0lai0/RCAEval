import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import het_white
from scipy.spatial.distance import cosine
import networkx as nx
from tqdm import tqdm
import math
import json
from typing import Dict, List, Tuple, Optional

from RCAEval.graph_construction.granger import granger
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    preprocess,
    drop_time,
    select_useful_cols,
)
from RCAEval.e2e import rca


class MultiModalDataProcessor:
    """多模態數據預處理器"""
    
    def __init__(self, time_resolution=15, anomaly_threshold=3.0):
        self.time_resolution = time_resolution
        self.anomaly_threshold = anomaly_threshold
        self.scaler = RobustScaler()
        
    def align_multimodal_data(self, metric_df, logts_df, traces_err_df=None, traces_lat_df=None, inject_time=None):
        """對齊多模態數據到統一時間粒度"""
        aligned_data = {}
        
        # 處理metric數據 - 每15秒採樣一次
        if metric_df is not None:
            metric_resampled = metric_df.iloc[::self.time_resolution, :]
            aligned_data['metric'] = metric_resampled
            
        # 處理logts數據
        if logts_df is not None:
            logts_processed = drop_constant(logts_df)
            aligned_data['logts'] = logts_processed
            
        # 處理traces數據
        if traces_err_df is not None:
            traces_err_filled = traces_err_df.fillna(method='ffill').fillna(0)
            traces_err_processed = drop_constant(traces_err_filled)
            aligned_data['traces_err'] = traces_err_processed
            
        if traces_lat_df is not None:
            traces_lat_filled = traces_lat_df.fillna(method='ffill').fillna(0)
            traces_lat_processed = drop_constant(traces_lat_filled)
            aligned_data['traces_lat'] = traces_lat_processed
            
        return aligned_data
    
    def split_normal_anomal(self, data_dict, inject_time):
        """分割正常和異常數據"""
        normal_data = {}
        anomal_data = {}
        
        for modality, df in data_dict.items():
            if 'time' in df.columns:
                normal_data[modality] = df[df['time'] < inject_time]
                anomal_data[modality] = df[df['time'] >= inject_time]
            else:
                # 對於沒有time列的數據，按比例分割
                split_idx = int(len(df) * 0.7)
                normal_data[modality] = df.iloc[:split_idx]
                anomal_data[modality] = df.iloc[split_idx:]
                
        return normal_data, anomal_data
    
    def preprocess_modality(self, df, dataset=None, dk_select_useful=False):
        """預處理單個模態的數據"""
        if df is None or df.empty:
            return df
            
        # 使用現有的預處理函數
        processed_df = preprocess(
            data=df, 
            dataset=dataset, 
            dk_select_useful=dk_select_useful
        )
        
        return processed_df
    
    def detect_anomalies(self, data):
        """基於Z-score的異常檢測"""
        anomaly_scores = {}
        
        for col in data.columns:
            if col == 'time':
                continue
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            max_z_score = z_scores.max()
            anomaly_scores[col] = max_z_score
            
        return anomaly_scores


class GRUEncoder(nn.Module):
    """GRU編碼器，用於將時間序列映射到統一嵌入空間"""
    
    def __init__(self, input_dim=1, hidden_dim=64, output_embedding_dim=128, num_layers=2, dropout=0.2):
        super(GRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch_size, hidden_dim)
        
        # 取最後一層的隱藏狀態
        last_hidden = h_n[-1, :, :]  # (batch_size, hidden_dim)
        
        # 映射到嵌入空間
        embedding = self.fc(self.dropout(last_hidden))  # (batch_size, output_embedding_dim)
        
        return embedding


class InfoNCELoss(nn.Module):
    """InfoNCE對比學習損失函數"""
    
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        
    def forward(self, query, positive, negatives):
        """
        Args:
            query: 查詢嵌入 (batch_size, embedding_dim)
            positive: 正樣本嵌入 (batch_size, embedding_dim)  
            negatives: 負樣本嵌入 (batch_size, num_negatives, embedding_dim)
        """
        # 計算正樣本相似度
        pos_sim = self.cosine_sim(query, positive) / self.temperature  # (batch_size,)
        
        # 計算負樣本相似度
        query_expanded = query.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        neg_sim = self.cosine_sim(query_expanded, negatives) / self.temperature  # (batch_size, num_negatives)
        
        # 計算InfoNCE損失
        pos_exp = torch.exp(pos_sim)  # (batch_size,)
        neg_exp = torch.exp(neg_sim).sum(dim=1)  # (batch_size,)
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp))
        
        return loss.mean()


class MultiModalTimeSeriesDataset(Dataset):
    """多模態時間序列數據集"""
    
    def __init__(self, data_dict, seq_len=60, positive_pairs=None):
        self.data_dict = data_dict
        self.seq_len = seq_len
        self.positive_pairs = positive_pairs or []
        
        # 準備所有時間序列
        self.time_series = {}
        self.feature_names = []
        
        for modality, df in data_dict.items():
            if df is None or df.empty:
                continue
                
            # 移除時間列
            if 'time' in df.columns:
                df = df.drop(columns=['time'])
                
            for col in df.columns:
                feature_name = f"{modality}_{col}"
                self.time_series[feature_name] = df[col].values
                self.feature_names.append(feature_name)
        
        # 計算可用的序列數量
        min_len = min([len(ts) for ts in self.time_series.values()])
        self.num_sequences = max(0, min_len - seq_len + 1)
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # 隨機選擇一個特徵作為錨點
        anchor_feature = np.random.choice(self.feature_names)
        anchor_seq = self.time_series[anchor_feature][idx:idx+self.seq_len]
        anchor_tensor = torch.FloatTensor(anchor_seq).unsqueeze(-1)  # (seq_len, 1)
        
        # 生成正樣本
        positive_feature = self._get_positive_sample(anchor_feature)
        if positive_feature:
            pos_seq = self.time_series[positive_feature][idx:idx+self.seq_len]
            pos_tensor = torch.FloatTensor(pos_seq).unsqueeze(-1)
        else:
            # 如果沒有正樣本，使用自己
            pos_tensor = anchor_tensor.clone()
            
        # 生成負樣本
        negative_features = self._get_negative_samples(anchor_feature, num_negatives=5)
        neg_tensors = []
        for neg_feature in negative_features:
            neg_seq = self.time_series[neg_feature][idx:idx+self.seq_len]
            neg_tensor = torch.FloatTensor(neg_seq).unsqueeze(-1)
            neg_tensors.append(neg_tensor)
        
        neg_tensors = torch.stack(neg_tensors)  # (num_negatives, seq_len, 1)
        
        return {
            'anchor': anchor_tensor,
            'positive': pos_tensor,
            'negatives': neg_tensors,
            'anchor_name': anchor_feature,
            'positive_name': positive_feature or anchor_feature
        }
    
    def _get_positive_sample(self, anchor_feature):
        """獲取正樣本特徵"""
        # 基於預定義的正樣本對
        for source, target, _ in self.positive_pairs:
            if source == anchor_feature:
                return target
            if target == anchor_feature:
                return source
                
        # 如果沒有預定義的正樣本對，使用啟發式規則
        anchor_modality, anchor_name = anchor_feature.split('_', 1)
        
        # 同服務不同模態
        for feature_name in self.feature_names:
            if feature_name == anchor_feature:
                continue
            modality, name = feature_name.split('_', 1)
            
            # 檢查是否是同一個服務
            if self._is_same_service(anchor_name, name) and modality != anchor_modality:
                return feature_name
                
        return None
    
    def _get_negative_samples(self, anchor_feature, num_negatives=5):
        """獲取負樣本特徵"""
        candidates = [f for f in self.feature_names if f != anchor_feature]
        
        # 過濾掉可能的正樣本
        positive_feature = self._get_positive_sample(anchor_feature)
        if positive_feature:
            candidates = [f for f in candidates if f != positive_feature]
            
        # 隨機選擇負樣本
        num_negatives = min(num_negatives, len(candidates))
        return np.random.choice(candidates, num_negatives, replace=False).tolist()
    
    def _is_same_service(self, name1, name2):
        """判斷兩個特徵是否屬於同一個服務"""
        # 簡單的啟發式規則：檢查服務名稱前綴
        service_prefixes = [
            'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
            'emailservice', 'frontend', 'paymentservice', 'productcatalogservice',
            'recommendationservice', 'redis', 'shippingservice'
        ]
        
        for prefix in service_prefixes:
            if name1.startswith(prefix) and name2.startswith(prefix):
                return True
                
        return False


class UnifiedRepresentationLearner:
    """統一表示學習器"""
    
    def __init__(self, embedding_dim=128, hidden_dim=64, seq_len=60):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 為不同模態創建編碼器
        self.encoders = {
            'metric': GRUEncoder(1, hidden_dim, embedding_dim),
            'logts': GRUEncoder(1, hidden_dim, embedding_dim),
            'traces_err': GRUEncoder(1, hidden_dim, embedding_dim),
            'traces_lat': GRUEncoder(1, hidden_dim, embedding_dim)
        }
        
        self.loss_fn = InfoNCELoss(temperature=0.1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移動模型到設備
        for encoder in self.encoders.values():
            encoder.to(self.device)
            
    def train(self, data_dict, num_epochs=50, batch_size=32, lr=1e-3):
        """訓練統一表示學習器"""
        
        # 創建數據集和數據加載器
        dataset = MultiModalTimeSeriesDataset(data_dict, self.seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if len(dataset) == 0:
            print("Warning: Dataset is empty, skipping training")
            return
        
        # 設置優化器
        all_params = []
        for encoder in self.encoders.values():
            all_params.extend(encoder.parameters())
        optimizer = optim.Adam(all_params, lr=lr)
        
        # 訓練循環
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                anchor = batch['anchor'].to(self.device)  # (batch_size, seq_len, 1)
                positive = batch['positive'].to(self.device)
                negatives = batch['negatives'].to(self.device)  # (batch_size, num_negatives, seq_len, 1)
                
                # 獲取錨點和正樣本的模態
                anchor_names = batch['anchor_name']
                positive_names = batch['positive_name']
                
                # 編碼錨點和正樣本
                anchor_embeddings = []
                positive_embeddings = []
                
                for i, (anchor_name, pos_name) in enumerate(zip(anchor_names, positive_names)):
                    anchor_modality = anchor_name.split('_')[0]
                    pos_modality = pos_name.split('_')[0]
                    
                    if anchor_modality in self.encoders:
                        anchor_emb = self.encoders[anchor_modality](anchor[i:i+1])
                        anchor_embeddings.append(anchor_emb)
                    
                    if pos_modality in self.encoders:
                        pos_emb = self.encoders[pos_modality](positive[i:i+1])
                        positive_embeddings.append(pos_emb)
                
                if not anchor_embeddings or not positive_embeddings:
                    continue
                    
                anchor_embeddings = torch.cat(anchor_embeddings, dim=0)
                positive_embeddings = torch.cat(positive_embeddings, dim=0)
                
                # 編碼負樣本
                batch_size, num_negatives, seq_len, input_dim = negatives.shape
                negatives_flat = negatives.view(-1, seq_len, input_dim)
                
                # 假設負樣本都是metric模態（簡化處理）
                negative_embeddings = self.encoders['metric'](negatives_flat)
                negative_embeddings = negative_embeddings.view(batch_size, num_negatives, -1)
                
                # 計算損失
                loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                
                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    def encode_time_series(self, data_dict):
        """將時間序列編碼為嵌入向量"""
        embeddings = {}
        
        for modality, df in data_dict.items():
            if df is None or df.empty:
                continue
                
            encoder = self.encoders.get(modality)
            if encoder is None:
                continue
                
            encoder.eval()
            
            # 移除時間列
            if 'time' in df.columns:
                df = df.drop(columns=['time'])
            
            for col in df.columns:
                feature_name = f"{modality}_{col}"
                time_series = df[col].values
                
                if len(time_series) < self.seq_len:
                    continue
                
                # 創建滑動窗口
                embeddings_list = []
                for i in range(len(time_series) - self.seq_len + 1):
                    seq = time_series[i:i+self.seq_len]
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1).to(self.device)
                    
                    with torch.no_grad():
                        embedding = encoder(seq_tensor)
                        embeddings_list.append(embedding.cpu().numpy().flatten())
                
                if embeddings_list:
                    embeddings[feature_name] = np.array(embeddings_list)
        
        return embeddings


class MVGCAnalyzer:
    """多變量格蘭傑因果分析器"""
    
    def __init__(self, max_lag=5, p_threshold=0.05):
        self.max_lag = max_lag
        self.p_threshold = p_threshold
        
    def reduce_embeddings_to_1d(self, embeddings_dict, method='pca'):
        """將高維嵌入降維到1維"""
        reduced_embeddings = {}
        
        for feature_name, embedding_matrix in embeddings_dict.items():
            if embedding_matrix.ndim == 2 and embedding_matrix.shape[1] > 1:
                if method == 'pca':
                    pca = PCA(n_components=1)
                    reduced = pca.fit_transform(embedding_matrix).flatten()
                else:  # 使用均值
                    reduced = embedding_matrix.mean(axis=1)
                    
                reduced_embeddings[feature_name] = reduced
            else:
                reduced_embeddings[feature_name] = embedding_matrix.flatten()
                
        return reduced_embeddings
    
    def perform_mvgc(self, reduced_embeddings):
        """執行多變量格蘭傑因果分析"""
        if not reduced_embeddings:
            return []
            
        # 創建DataFrame
        min_length = min([len(ts) for ts in reduced_embeddings.values()])
        
        data_dict = {}
        for feature_name, ts in reduced_embeddings.items():
            data_dict[feature_name] = ts[:min_length]
            
        df = pd.DataFrame(data_dict)
        
        if df.empty or len(df) < self.max_lag * 2:
            return []
        
        try:
            # 使用現有的granger函數
            adj_matrix = granger(df, maxlag=self.max_lag, p_val_threshold=self.p_threshold)
            
            # 轉換為邊列表
            feature_names = df.columns.tolist()
            edges = []
            
            for i, source in enumerate(feature_names):
                for j, target in enumerate(feature_names):
                    if adj_matrix[j, i] > 0:  # 注意索引順序
                        weight = 1.0 - self.p_threshold  # 簡化的權重計算
                        edges.append((source, target, weight))
                        
            return edges
            
        except Exception as e:
            print(f"MVGC analysis failed: {e}")
            return []


class CausalGraphAnalyzer:
    """因果圖分析器"""
    
    def __init__(self, anomaly_threshold=3.0):
        self.anomaly_threshold = anomaly_threshold
        
    def build_causal_graph(self, causal_edges):
        """構建因果圖"""
        G = nx.DiGraph()
        
        for source, target, weight in causal_edges:
            G.add_edge(source, target, weight=weight)
            
        return G
    
    def rank_root_causes(self, causal_graph, anomaly_scores):
        """根因排序"""
        if causal_graph.number_of_nodes() == 0:
            return list(anomaly_scores.keys())
        
        # 識別異常節點
        anomalous_nodes = [
            node for node, score in anomaly_scores.items() 
            if score > self.anomaly_threshold
        ]
        
        if not anomalous_nodes:
            # 如果沒有明顯異常，返回所有節點
            anomalous_nodes = list(anomaly_scores.keys())
        
        # 計算PageRank
        try:
            pagerank_scores = nx.pagerank(causal_graph, weight='weight')
        except:
            pagerank_scores = {node: 1.0 for node in causal_graph.nodes()}
        
        # 綜合排序
        root_cause_candidates = []
        for node in anomalous_nodes:
            if node in causal_graph.nodes():
                anomaly_score = anomaly_scores.get(node, 0)
                pagerank_score = pagerank_scores.get(node, 0)
                in_degree = causal_graph.in_degree(node)
                
                # 綜合評分：異常程度 * PageRank / (1 + 入度)
                if in_degree == 0:
                    score = anomaly_score * pagerank_score * 2  # 給源頭節點更高權重
                else:
                    score = anomaly_score * pagerank_score / (1 + in_degree * 0.1)
                    
                root_cause_candidates.append((node, score))
            else:
                # 孤立異常節點
                anomaly_score = anomaly_scores.get(node, 0)
                root_cause_candidates.append((node, anomaly_score * 0.5))
        
        # 排序
        root_cause_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in root_cause_candidates]


@rca
def grumvgc(data, inject_time=None, dataset=None, num_loop=None, sli=None, **kwargs):
    """
    GRUMVGC: 多模態統一表示學習與多變量格蘭傑因果分析的根因分析方法
    """
    
    # 參數設置
    embedding_dim = kwargs.get('embedding_dim', 128)
    seq_len = kwargs.get('seq_len', 60)
    num_epochs = kwargs.get('num_epochs', 20)  # 減少訓練輪數以加快實驗
    batch_size = kwargs.get('batch_size', 16)
    
    print("Starting GRUMVGC analysis...")
    
    # 檢查是否為多模態數據
    if not isinstance(data, dict):
        print("Warning: Expected multimodal data (dict), got single modality. Using fallback method.")
        # 回退到簡單的granger方法
        processed_data = preprocess(
            data=data,
            dataset=dataset,
            dk_select_useful=kwargs.get("dk_select_useful", False)
        )
        
        node_names = processed_data.columns.tolist()
        adj = granger(processed_data)
        
        if adj.sum().sum() == 0:
            return {
                "adj": adj,
                "node_names": node_names,
                "ranks": node_names,
            }
        
        ranks = page_rank(adj, node_names=node_names)
        ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
        ranks = [x[0] for x in ranks]
        
        return {
            "adj": adj,
            "node_names": node_names,
            "ranks": ranks,
        }
    
    try:
        # 步驟1: 數據預處理
        print("Step 1: Data preprocessing...")
        processor = MultiModalDataProcessor()
        
        # 對齊多模態數據
        aligned_data = processor.align_multimodal_data(
            data.get("metric"),
            data.get("logts"),
            data.get("tracets_err"),
            data.get("tracets_lat"),
            inject_time
        )
        
        # 分割正常和異常數據
        normal_data, anomal_data = processor.split_normal_anomal(aligned_data, inject_time)
        
        # 預處理各模態數據
        processed_normal = {}
        processed_anomal = {}
        
        for modality in aligned_data.keys():
            if normal_data.get(modality) is not None:
                processed_normal[modality] = processor.preprocess_modality(
                    normal_data[modality], 
                    dataset=dataset,
                    dk_select_useful=kwargs.get("dk_select_useful", False)
                )
            
            if anomal_data.get(modality) is not None:
                processed_anomal[modality] = processor.preprocess_modality(
                    anomal_data[modality],
                    dataset=dataset, 
                    dk_select_useful=kwargs.get("dk_select_useful", False)
                )
        
        # 合併正常和異常數據
        combined_data = {}
        for modality in processed_normal.keys():
            if processed_normal[modality] is not None and processed_anomal[modality] is not None:
                # 確保列名一致
                common_cols = set(processed_normal[modality].columns) & set(processed_anomal[modality].columns)
                if common_cols:
                    normal_subset = processed_normal[modality][list(common_cols)]
                    anomal_subset = processed_anomal[modality][list(common_cols)]
                    combined_data[modality] = pd.concat([normal_subset, anomal_subset], axis=0, ignore_index=True)
        
        if not combined_data:
            print("Warning: No valid combined data after preprocessing")
            return {"ranks": [], "adj": np.array([]), "node_names": []}
        
        print(f"Combined data shapes: {[(k, v.shape) for k, v in combined_data.items()]}")
        
        # 步驟2: 統一表示學習
        print("Step 2: Unified representation learning...")
        learner = UnifiedRepresentationLearner(
            embedding_dim=embedding_dim,
            seq_len=seq_len
        )
        
        # 訓練統一表示學習器
        learner.train(
            combined_data,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        
        # 編碼時間序列
        embeddings = learner.encode_time_series(combined_data)
        
        if not embeddings:
            print("Warning: No embeddings generated")
            return {"ranks": [], "adj": np.array([]), "node_names": []}
        
        print(f"Generated embeddings for {len(embeddings)} features")
        
        # 步驟3: 多變量格蘭傑因果分析
        print("Step 3: MVGC analysis...")
        mvgc_analyzer = MVGCAnalyzer()
        
        # 降維到1維
        reduced_embeddings = mvgc_analyzer.reduce_embeddings_to_1d(embeddings)
        
        # 執行MVGC
        causal_edges = mvgc_analyzer.perform_mvgc(reduced_embeddings)
        
        print(f"Found {len(causal_edges)} causal edges")
        
        # 步驟4: 因果圖分析和根因定位
        print("Step 4: Root cause ranking...")
        graph_analyzer = CausalGraphAnalyzer()
        
        # 構建因果圖
        causal_graph = graph_analyzer.build_causal_graph(causal_edges)
        
        # 計算異常分數
        all_combined_data = pd.concat(combined_data.values(), axis=1)
        anomaly_scores = processor.detect_anomalies(all_combined_data)
        
        # 根因排序
        ranked_causes = graph_analyzer.rank_root_causes(causal_graph, anomaly_scores)
        
        print(f"GRUMVGC completed. Top root causes: {ranked_causes[:5]}")
        
        # 構建鄰接矩陣
        node_names = list(reduced_embeddings.keys())
        adj_matrix = np.zeros((len(node_names), len(node_names)))
        
        for source, target, weight in causal_edges:
            if source in node_names and target in node_names:
                i = node_names.index(source)
                j = node_names.index(target)
                adj_matrix[j, i] = weight
        
        return {
            "ranks": ranked_causes,
            "adj": adj_matrix,
            "node_names": node_names,
            "causal_edges": causal_edges,
            "embeddings_info": {
                "num_features": len(embeddings),
                "embedding_dim": embedding_dim
            }
        }
        
    except Exception as e:
        print(f"GRUMVGC failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # 回退到簡單方法
        print("Falling back to simple granger method...")
        
        if 'metric' in data:
            fallback_data = preprocess(
                data=data['metric'],
                dataset=dataset,
                dk_select_useful=kwargs.get("dk_select_useful", False)
            )
        else:
            # 如果沒有metric數據，嘗試使用第一個可用的數據
            first_key = next(iter(data.keys()))
            fallback_data = preprocess(
                data=data[first_key],
                dataset=dataset,
                dk_select_useful=kwargs.get("dk_select_useful", False)
            )
        
        node_names = fallback_data.columns.tolist()
        adj = granger(fallback_data)
        
        if adj.sum().sum() == 0:
            return {
                "adj": adj,
                "node_names": node_names,
                "ranks": node_names,
            }
        
        ranks = page_rank(adj, node_names=node_names)
        ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
        ranks = [x[0] for x in ranks]
        
        return {
            "adj": adj,
            "node_names": node_names,
            "ranks": ranks,
        } 