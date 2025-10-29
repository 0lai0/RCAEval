"""
節點剪枝模組 - 減少需要分析的節點數量
"""
from __future__ import annotations
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import networkx as nx


def trace_based_prefiltering(
    all_nodes: List[str],
    trace_graph: nx.DiGraph | None,
    focus_node: str,
    max_hops: int = 2
) -> List[str]:
    """
    基於 trace 圖的預過濾：只保留與 focus_node 在 max_hops 跳內的節點
    
    Args:
        all_nodes: 所有候選節點列表
        trace_graph: 服務依賴圖 (trace)
        focus_node: 焦點節點 (SLI 服務)
        max_hops: 最大跳數 (預設 2)
    
    Returns:
        過濾後的節點列表
    """
    if trace_graph is None or focus_node not in all_nodes:
        return all_nodes
    
    if focus_node not in trace_graph:
        # focus_node 不在 trace 圖中，保留所有節點
        return all_nodes
    
    reachable: Set[str] = {focus_node}
    frontier: Set[str] = {focus_node}
    
    for hop in range(max_hops):
        new_frontier: Set[str] = set()
        for node in frontier:
            if node in trace_graph:
                # 前驅節點 (上游依賴)
                new_frontier.update(trace_graph.predecessors(node))
                # 後繼節點 (下游依賴)
                new_frontier.update(trace_graph.successors(node))
        reachable.update(new_frontier)
        frontier = new_frontier
    
    # 保留在 all_nodes 中且可達的節點
    filtered = [n for n in all_nodes if n in reachable]
    return filtered


def early_anomaly_pruning(
    node_anomaly: Dict[str, float] | pd.Series,
    threshold_percentile: float = 0.3,
    min_nodes: int = 10,
    max_nodes: int = 50
) -> List[str]:
    """
    早期異常分數剪枝：只保留異常分數較高的節點
    
    Args:
        node_anomaly: 各節點的異常分數 (字典或 Series)
        threshold_percentile: 保留的百分位數 (0.3 表示保留 top 70%)
        min_nodes: 最少保留的節點數
        max_nodes: 最多保留的節點數
    
    Returns:
        過濾後的節點列表
    """
    if isinstance(node_anomaly, pd.Series):
        node_anomaly = node_anomaly.to_dict()
    
    if not node_anomaly:
        return []
    
    # 移除 NaN 和負值
    valid_scores = {k: float(v) for k, v in node_anomaly.items() 
                    if np.isfinite(v) and v >= 0}
    
    if len(valid_scores) <= min_nodes:
        return list(valid_scores.keys())
    
    # 計算閾值
    scores = list(valid_scores.values())
    threshold = np.percentile(scores, threshold_percentile * 100)
    
    # 過濾
    filtered = [n for n, score in valid_scores.items() if score >= threshold]
    
    # 確保至少保留 min_nodes 個節點
    if len(filtered) < min_nodes:
        sorted_nodes = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
        filtered = [n for n, _ in sorted_nodes[:min_nodes]]
    
    # 確保不超過 max_nodes 個節點
    if len(filtered) > max_nodes:
        # 按異常分數排序，保留前 max_nodes 個
        filtered_scores = {n: valid_scores[n] for n in filtered}
        sorted_filtered = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        filtered = [n for n, _ in sorted_filtered[:max_nodes]]
    
    return filtered


def combined_pruning(
    all_nodes: List[str],
    node_anomaly: Dict[str, float] | pd.Series,
    trace_graph: nx.DiGraph | None,
    focus_node: str,
    max_hops: int = 2,
    anomaly_percentile: float = 0.3,
    min_nodes: int = 10,
    max_nodes: int = 50
) -> List[str]:
    """
    組合剪枝策略：先 trace 過濾，再異常分數過濾
    
    Returns:
        最終過濾後的節點列表
    """
    # Step 1: Trace-based filtering
    step1 = trace_based_prefiltering(all_nodes, trace_graph, focus_node, max_hops)
    
    # Step 2: Anomaly-based filtering
    if isinstance(node_anomaly, pd.Series):
        node_anomaly_dict = node_anomaly.to_dict()
    else:
        node_anomaly_dict = dict(node_anomaly)
    
    # 只考慮 step1 中的節點
    step1_anomaly = {k: v for k, v in node_anomaly_dict.items() if k in step1}
    step2 = early_anomaly_pruning(step1_anomaly, anomaly_percentile, min_nodes, max_nodes)
    
    # 確保 focus_node 一定保留
    if focus_node not in step2 and focus_node in all_nodes:
        step2 = [focus_node] + step2
    
    return step2
