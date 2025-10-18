from __future__ import annotations
from typing import Dict, List, Tuple
import networkx as nx

from .config import PCMCIShapleyConfig


def extract_trace_weights(trace_graph: nx.DiGraph | None, local_nodes: List[str]) -> Dict[Tuple[str, str], float]:
    weights: Dict[Tuple[str, str], float] = {}
    if trace_graph is None:
        return weights
    Uset = set(local_nodes)
    for u, v, data in trace_graph.edges(data=True):
        if u in Uset and v in Uset:
            w = float(data.get("weight", 1.0))
            weights[(u, v)] = w
    return weights


def fuse_edge_weights(trace_w: Dict[Tuple[str, str], float], pcmci_strengths: Dict[Tuple[str, str], float], isolation_scores: Dict[str, float], cfg: PCMCIShapleyConfig) -> Dict[Tuple[str, str], float]:
    fused: Dict[Tuple[str, str], float] = {}
    nodes = set([i for i, _ in trace_w.keys()] + [j for _, j in trace_w.keys()])
    nodes |= set([i for i, _ in pcmci_strengths.keys()] + [j for _, j in pcmci_strengths.keys()])
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            tw = trace_w.get((i, j), 0.0)
            ps = pcmci_strengths.get((i, j), 0.0)
            Ii = isolation_scores.get(i, 0.0)
            fused[(i, j)] = cfg.theta1 * tw + cfg.theta2 * ps + cfg.theta3 * Ii
    return fused


def apply_conflict_penalty(weights: Dict[Tuple[str, str], float], pcmci_edges: List[Tuple[str, str]], gamma: float) -> Dict[Tuple[str, str], float]:
    penalized = dict(weights)
    pcmci_set = set(pcmci_edges)
    for (i, j) in list(weights.keys()):
        if (j, i) in pcmci_set and (i, j) in penalized:
            penalized[(i, j)] = penalized[(i, j)] * (1.0 - gamma)
    return penalized


def normalize_incoming_weights(weights: Dict[Tuple[str, str], float], local_nodes: List[str]) -> Dict[Tuple[str, str], float]:
    normed: Dict[Tuple[str, str], float] = {}
    incoming_sum: Dict[str, float] = {n: 0.0 for n in local_nodes}
    for (i, j), w in weights.items():
        if j in incoming_sum:
            incoming_sum[j] += max(0.0, w)
    for (i, j), w in weights.items():
        if j not in incoming_sum or incoming_sum[j] == 0:
            continue
        normed[(i, j)] = max(0.0, w) / incoming_sum[j]
    return normed
