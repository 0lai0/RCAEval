from __future__ import annotations
from typing import Dict, Tuple, List
from collections import deque
import math
import networkx as nx

from .config import PCMCIShapleyConfig
from .utils import min_max_normalize


def compute_reachability(focus_node: str, edge_weights: Dict[Tuple[str, str], float], local_nodes: List[str]) -> Dict[str, float]:
    """
    計算從每個節點到 focus_node 的可達性分數
    使用最短路徑算法（Dijkstra）替代枚舉所有路徑，大幅提升性能
    
    可達性 = max_{path s->focus} ∏_{edges in path} w_{edge}
    等價於：log(reachability) = max_{path s->focus} Σ_{edges in path} log(w_{edge})
    使用負 log 權重 Dijkstra 來找最大值路徑（理論上等價）
    
    理論等價性證明：
    1. 原始問題：找 max ∏ w_i = max Σ log(w_i)
    2. 轉換：min -Σ log(w_i) = min Σ (-log(w_i))
    3. Dijkstra 找 min Σ weight_i，其中 weight_i = -log(w_i)
    4. 因此結果完全等價
    
    注意：權重 w ∈ [0, 1]（歸一化後），所以 -log(w) ≥ 0，Dijkstra 可正常工作
    數值穩定性：使用 epsilon 避免 w=0 時 log(0) 的問題
    """
    # 建立圖，使用負 log 權重（Dijkstra 找最小路徑，我們要找最大乘積路徑）
    # 使用小的 epsilon 避免數值問題（w 接近 0 時 log 會很大）
    EPSILON = 1e-10
    G = nx.DiGraph()
    for (i, j), w in edge_weights.items():
        if w > EPSILON:
            # 使用負 log，這樣 Dijkstra 最小路徑 = 原始最大乘積路徑
            # 限制 log 的輸入值，避免數值溢出
            w_clamped = max(EPSILON, min(w, 1.0))
            G.add_edge(i, j, weight=-math.log(w_clamped))
    
    r: Dict[str, float] = {}
    
    # 從 focus_node 開始反向搜索（更高效，只需要一次 Dijkstra）
    # 建立反向圖
    G_rev = G.reverse(copy=True)
    
    # 使用單源最短路徑算法（Dijkstra）從 focus_node 反向搜索
    try:
        # 使用 Dijkstra 計算從 focus_node 到所有節點的最短距離（在反向圖中）
        # 這樣只需要一次算法調用，而不是對每個節點都調用
        cutoff_value = len(local_nodes) * 2  # 合理的 cutoff
        distances = nx.single_source_dijkstra_path_length(
            G_rev, focus_node, weight='weight', cutoff=cutoff_value
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        distances = {}
    
    # 計算可達性分數
    for s in local_nodes:
        if s == focus_node:
            r[s] = 1.0
            continue
        if s not in distances:
            r[s] = 0.0
            continue
        # 距離是負 log 權重，所以 exp(-distance) 就是原始權重乘積
        r[s] = math.exp(-distances[s])
    
    return r


def compute_temporal_penalty(focus_node: str, anomaly_time: Dict[str, int], edge_strengths: Dict[Tuple[str, str], float], graph: nx.DiGraph, lambda_penalty: float, local_nodes: List[str]) -> Dict[str, float]:
    """
    計算時間一致性懲罰（優化版本）
    使用 BFS 限制深度，避免枚舉所有路徑造成的性能問題
    """
    p: Dict[str, float] = {s: 1.0 for s in local_nodes}
    
    # 限制搜索深度，避免指數爆炸
    max_depth = min(5, len(local_nodes))
    
    for s in local_nodes:
        if s == focus_node:
            continue
        if s not in graph:
            continue
        
        penalty_sum = 0.0
        visited_edges = set()  # 避免重複計算同一條邊
        
        # 使用 BFS 限制深度搜索
        queue = deque([(s, [s], 0)])  # (current_node, path, depth)
        
        while queue:
            node, path, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            # 檢查路徑上的時間違反
            for u, v in zip(path[:-1], path[1:]):
                tu = anomaly_time.get(u, None)
                tv = anomaly_time.get(v, None)
                if tu is not None and tv is not None and tv < tu:
                    edge_key = (u, v)
                    if edge_key not in visited_edges:
                        penalty_sum += float(edge_strengths.get(edge_key, 0.0))
                        visited_edges.add(edge_key)
            
            # 如果到達目標，不需要繼續
            if node == focus_node:
                continue
            
            # 繼續搜索
            for neighbor in graph.successors(node):
                if neighbor not in path:  # 避免循環
                    queue.append((neighbor, path + [neighbor], depth + 1))
        
        # 如果沒有找到違反，penalty_sum = 0，p[s] = 1.0
        p[s] = float(math.exp(-lambda_penalty * penalty_sum)) if penalty_sum > 0 else 1.0
    
    return p


def compute_comprehensive_score(shapley_norm: Dict[str, float], reach_norm: Dict[str, float], anomaly_norm: Dict[str, float], cfg: PCMCIShapleyConfig) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for s in shapley_norm.keys():
        scores[s] = cfg.score_alpha1 * shapley_norm.get(s, 0.0) + cfg.score_alpha2 * reach_norm.get(s, 0.0) + cfg.score_alpha3 * anomaly_norm.get(s, 0.0)
    return scores


def compute_final_ranking(scores: Dict[str, float], penalties: Dict[str, float]) -> List[str]:
    adjusted = {s: scores.get(s, 0.0) * penalties.get(s, 1.0) for s in scores}
    return [k for k, _ in sorted(adjusted.items(), key=lambda x: x[1], reverse=True)]
