from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse as sp


def compute_initial_anomaly(node_anomaly: Dict[str, float], timestamp: int | None = None) -> Dict[str, float]:
    # Use last-time anomaly value (already aggregated) and log1p
    out: Dict[str, float] = {}
    for s, val in node_anomaly.items():
        v = float(val) if np.isfinite(val) else 0.0
        out[s] = float(np.log1p(max(0.0, v)))
    return out


def propagate_one_step(current_h: Dict[str, float], edge_weights: Dict[Tuple[str, str], float], alpha_prop: float) -> Dict[str, float]:
    next_h: Dict[str, float] = {}
    # accumulate incoming weighted influence
    for (i, j), w in edge_weights.items():
        if w <= 0:
            continue
        hi = current_h.get(i, 0.0)
        if hi == 0.0:
            continue
        next_h[j] = next_h.get(j, 0.0) + alpha_prop * w * hi
    return next_h


def propagate_k_steps(initial_delta: Dict[str, float], edge_weights: Dict[Tuple[str, str], float], K: int, alpha_prop: float) -> Dict[str, object]:
    """
    Main entry point: automatically choose between fast or original implementation.
    """
    # Use the fast version (it will decide internally whether it is applicable)
    return fast_propagate_k_steps(initial_delta, edge_weights, K, alpha_prop, use_sparse=True)


def fast_propagate_k_steps(
    initial_delta: Dict[str, float], 
    edge_weights: Dict[Tuple[str, str], float], 
    K: int, 
    alpha_prop: float,
    use_sparse: bool = True
) -> Dict[str, object]:
    """
    Propagation implementation accelerated by sparse matrices.
    
    Args:
        use_sparse: Whether to use sparse matrices (recommended when node count > 20).
    """
    # For small graphs or few edges, fall back to the original implementation
    all_nodes = set(initial_delta.keys())
    all_nodes.update([i for i, j in edge_weights.keys()])
    all_nodes.update([j for i, j in edge_weights.keys()])
    
    if len(all_nodes) <= 10 or len(edge_weights) <= 5 or not use_sparse:
        # 回退到原始實作
        return _propagate_k_steps_original(initial_delta, edge_weights, K, alpha_prop)
    
    # Build node index
    nodes = sorted(all_nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    
    # Build sparse adjacency matrix (transposed for matrix multiplication)
    # A[j, i] = alpha * w(i -> j)
    row, col, data = [], [], []
    for (i, j), w in edge_weights.items():
        if w <= 0:
            continue
        row.append(node_idx[j])
        col.append(node_idx[i])
        data.append(alpha_prop * float(w))
    
    if not data:
        # No valid edges, return initial values
        return {"h_final": dict(initial_delta), "h_by_step": [dict(initial_delta)]}
    
    A = sp.csr_matrix((data, (row, col)), shape=(N, N))
    
    # Initialize vectors
    h0 = np.zeros(N, dtype=float)
    for node, val in initial_delta.items():
        if node in node_idx:
            h0[node_idx[node]] = float(val)
    
    # Iterative propagation: h_{k+1} = A @ h_k
    h_accum = h0.copy()
    h_curr = h0.copy()
    h_by_step = [_vec_to_dict(h0, nodes)]
    
    for k in range(K):
        h_curr = A @ h_curr  # 稀疏矩陣乘法 (O(E) 而非 O(N²))
        h_accum += h_curr
        h_by_step.append(_vec_to_dict(h_curr, nodes))
    
    h_final = _vec_to_dict(h_accum, nodes)
    
    return {"h_final": h_final, "h_by_step": h_by_step}


def _vec_to_dict(vec: np.ndarray, nodes: List[str]) -> Dict[str, float]:
    """Convert vector back to a dict keyed by node."""
    return {nodes[i]: float(vec[i]) for i in range(len(nodes)) if vec[i] != 0}


def _propagate_k_steps_original(
    initial_delta: Dict[str, float], 
    edge_weights: Dict[Tuple[str, str], float], 
    K: int, 
    alpha_prop: float
) -> Dict[str, object]:
    """Original propagation implementation (kept as a backup)."""
    h_accum: Dict[str, float] = {k: float(v) for k, v in initial_delta.items()}
    h_curr: Dict[str, float] = dict(initial_delta)
    h_by_step: List[Dict[str, float]] = [dict(h_curr)]
    for _ in range(K):
        h_next = propagate_one_step(h_curr, edge_weights, alpha_prop)
        for k, v in h_next.items():
            h_accum[k] = h_accum.get(k, 0.0) + float(v)
        h_by_step.append(dict(h_next))
        h_curr = h_next
    return {"h_final": h_accum, "h_by_step": h_by_step}
