from __future__ import annotations
from typing import Dict, Tuple, List
from collections import deque
import math
import networkx as nx

from .config import PCMCIShapleyConfig
from .utils import min_max_normalize


def compute_reachability(focus_node: str, edge_weights: Dict[Tuple[str, str], float], local_nodes: List[str]) -> Dict[str, float]:
    """
    Compute reachability scores from each node to the focus_node.
    Uses Dijkstra's shortest-path algorithm instead of enumerating all paths
    to greatly improve performance.
    
    Reachability = max_{path s->focus} ∏_{edges in path} w_{edge}
    Equivalent to: log(reachability) = max_{path s->focus} Σ_{edges in path} log(w_{edge})
    We use Dijkstra with negative log weights to find the maximum-product path (theoretically equivalent).
    
    Sketch of theoretical equivalence:
    1. Original problem: find max ∏ w_i = max Σ log(w_i)
    2. Transform: min -Σ log(w_i) = min Σ (-log(w_i))
    3. Dijkstra finds min Σ weight_i, with weight_i = -log(w_i)
    4. Therefore the result is equivalent to maximizing the product.
    
    Note: weights w ∈ [0, 1] (after normalization), so -log(w) ≥ 0 and Dijkstra works as usual.
    Numerical stability: use epsilon to avoid log(0) when w = 0.
    """
    # Build a directed graph with negative log weights
    # Use a small epsilon to avoid numerical issues when w is close to 0
    EPSILON = 1e-10
    G = nx.DiGraph()
    for (i, j), w in edge_weights.items():
        if w > EPSILON:
            # Use negative log so that Dijkstra's shortest path corresponds to max-product path
            # Clamp input to log to avoid overflow
            w_clamped = max(EPSILON, min(w, 1.0))
            G.add_edge(i, j, weight=-math.log(w_clamped))
    
    r: Dict[str, float] = {}
    
    # Run a single Dijkstra search from focus_node on the reversed graph
    G_rev = G.reverse(copy=True)
    
    # Use single-source shortest path (Dijkstra) from focus_node on the reversed graph
    try:
        # Dijkstra from focus_node to all nodes in the reversed graph
        # Only one call is needed instead of per-node calls
        cutoff_value = len(local_nodes) * 2  # Reasonable cutoff
        distances = nx.single_source_dijkstra_path_length(
            G_rev, focus_node, weight='weight', cutoff=cutoff_value
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        distances = {}
    
    # Compute reachability scores
    for s in local_nodes:
        if s == focus_node:
            r[s] = 1.0
            continue
        if s not in distances:
            r[s] = 0.0
            continue
        # Distance is negative log weight, so exp(-distance) recovers the original product of weights
        r[s] = math.exp(-distances[s])
    
    return r


def compute_temporal_penalty(focus_node: str, anomaly_time: Dict[str, int], edge_strengths: Dict[Tuple[str, str], float], graph: nx.DiGraph, lambda_penalty: float, local_nodes: List[str]) -> Dict[str, float]:
    """
    Compute temporal consistency penalties (optimized version).
    Uses BFS with limited depth instead of enumerating all paths to avoid exponential blow-up.
    """
    p: Dict[str, float] = {s: 1.0 for s in local_nodes}
    
    # Limit search depth to avoid exponential explosion
    max_depth = min(5, len(local_nodes))
    
    for s in local_nodes:
        if s == focus_node:
            continue
        if s not in graph:
            continue
        
        penalty_sum = 0.0
        visited_edges = set()  # Avoid double-counting the same edge
        
        # Use BFS with bounded depth
        queue = deque([(s, [s], 0)])  # (current_node, path, depth)
        
        while queue:
            node, path, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            # Check for temporal violations along the path
            for u, v in zip(path[:-1], path[1:]):
                tu = anomaly_time.get(u, None)
                tv = anomaly_time.get(v, None)
                if tu is not None and tv is not None and tv < tu:
                    edge_key = (u, v)
                    if edge_key not in visited_edges:
                        penalty_sum += float(edge_strengths.get(edge_key, 0.0))
                        visited_edges.add(edge_key)
            
            # If we reached focus_node, we don't need to expand further
            if node == focus_node:
                continue
            
            # Continue exploring neighbors
            for neighbor in graph.successors(node):
                if neighbor not in path:  # Avoid cycles
                    queue.append((neighbor, path + [neighbor], depth + 1))
        
        # If no violation is found, penalty_sum = 0 and p[s] = 1.0
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
