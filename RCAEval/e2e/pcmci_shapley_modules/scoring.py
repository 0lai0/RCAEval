from __future__ import annotations
from typing import Dict, Tuple, List
import math
import networkx as nx

from .config import PCMCIShapleyConfig
from .utils import min_max_normalize


def compute_reachability(focus_node: str, edge_weights: Dict[Tuple[str, str], float], local_nodes: List[str]) -> Dict[str, float]:
    # Convert product of weights to sum of log-weights and find max path to focus
    G = nx.DiGraph()
    for (i, j), w in edge_weights.items():
        if w > 0:
            G.add_edge(i, j, weight=math.log(w))
    r: Dict[str, float] = {}
    for s in local_nodes:
        if s == focus_node:
            r[s] = 1.0
            continue
        if s not in G.nodes:
            r[s] = 0.0
            continue
        best = -math.inf
        # Use BFS-limited simple paths with cutoff to avoid cycles explosion
        try:
            for path in nx.all_simple_paths(G, source=s, target=focus_node, cutoff=len(local_nodes)):
                wsum = 0.0
                for u, v in zip(path[:-1], path[1:]):
                    wsum += G[u][v]["weight"]
                best = max(best, wsum)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            best = -math.inf
        r[s] = 0.0 if best == -math.inf else math.exp(best)
    return r


def compute_temporal_penalty(focus_node: str, anomaly_time: Dict[str, int], edge_strengths: Dict[Tuple[str, str], float], graph: nx.DiGraph, lambda_penalty: float, local_nodes: List[str]) -> Dict[str, float]:
    p: Dict[str, float] = {s: 1.0 for s in local_nodes}
    for s in local_nodes:
        if s == focus_node:
            continue
        if s not in graph:
            continue
        # Any path s -> ... -> focus that violates time order
        penalty_sum = 0.0
        try:
            for path in nx.all_simple_paths(graph, source=s, target=focus_node, cutoff=len(local_nodes)):
                for u, v in zip(path[:-1], path[1:]):
                    tu = anomaly_time.get(u, None)
                    tv = anomaly_time.get(v, None)
                    if tu is None or tv is None:
                        continue
                    if tv < tu:  # violation
                        penalty_sum += float(edge_strengths.get((u, v), 0.0))
        except nx.NetworkXNoPath:
            pass
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
