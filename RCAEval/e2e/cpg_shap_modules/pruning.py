"""
Node pruning module - reduce the number of nodes that need to be analyzed.
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
    Trace-graph-based pre-filtering: keep only nodes within max_hops of focus_node.
    
    Args:
        all_nodes: List of all candidate nodes.
        trace_graph: Service dependency graph (trace).
        focus_node: Focus node (SLI service).
        max_hops: Maximum hop distance (default 2).
    
        Returns:
        Filtered list of nodes.
    """
    if trace_graph is None or focus_node not in all_nodes:
        return all_nodes
    
    if focus_node not in trace_graph:
        # focus_node is not in trace graph; keep all nodes
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
    
    # Keep reachable nodes that are in all_nodes
    filtered = [n for n in all_nodes if n in reachable]
    return filtered


def early_anomaly_pruning(
    node_anomaly: Dict[str, float] | pd.Series,
    threshold_percentile: float = 0.3,
    min_nodes: int = 10,
    max_nodes: int = 50
) -> List[str]:
    """
    Early anomaly-score pruning: keep only nodes with higher anomaly scores.
    
    Args:
        node_anomaly: Per-node anomaly scores (dict or Series).
        threshold_percentile: Percentile cutoff to keep (0.3 means keep top 70%).
        min_nodes: Minimum number of nodes to keep.
        max_nodes: Maximum number of nodes to keep.
    
    Returns:
        過濾後的節點列表
    """
    if isinstance(node_anomaly, pd.Series):
        node_anomaly = node_anomaly.to_dict()
    
    if not node_anomaly:
        return []
    
    # Remove NaN and negative values
    valid_scores = {k: float(v) for k, v in node_anomaly.items() 
                    if np.isfinite(v) and v >= 0}
    
    if len(valid_scores) <= min_nodes:
        return list(valid_scores.keys())
    
    # Compute threshold
    scores = list(valid_scores.values())
    threshold = np.percentile(scores, threshold_percentile * 100)
    
    # Filter by threshold
    filtered = [n for n, score in valid_scores.items() if score >= threshold]
    
    # Ensure at least min_nodes are kept
    if len(filtered) < min_nodes:
        sorted_nodes = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
        filtered = [n for n, _ in sorted_nodes[:min_nodes]]
    
    # Ensure we do not exceed max_nodes
    if len(filtered) > max_nodes:
        # Sort by anomaly score and keep top max_nodes
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
    Combined pruning strategy: first trace-based filtering, then anomaly-score filtering.
    
    Returns:
        Final list of filtered nodes.
    """
    # Step 1: Trace-based filtering
    step1 = trace_based_prefiltering(all_nodes, trace_graph, focus_node, max_hops)
    
    # Step 2: Anomaly-based filtering
    if isinstance(node_anomaly, pd.Series):
        node_anomaly_dict = node_anomaly.to_dict()
    else:
        node_anomaly_dict = dict(node_anomaly)
    
    # Only consider nodes from step1
    step1_anomaly = {k: v for k, v in node_anomaly_dict.items() if k in step1}
    step2 = early_anomaly_pruning(step1_anomaly, anomaly_percentile, min_nodes, max_nodes)
    
    # Ensure focus_node is always kept
    if focus_node not in step2 and focus_node in all_nodes:
        step2 = [focus_node] + step2
    
    return step2
