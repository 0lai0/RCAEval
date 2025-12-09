"""
PC-based causal discovery module.
Simplified to use only the PC algorithm for instantaneous causal relationships.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import os
import numpy as np
import pandas as pd
import networkx as nx
from tigramite import data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from .config import PCShapleyConfig
from .utils import smart_fillna_matrix


def prepare_data_matrix(
    data: pd.DataFrame, 
    local_nodes: List[str],
    use_pca: bool = False,
    pca_components: int = 10
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare time series matrix from DataFrame.
    
    Args:
        data: Input DataFrame with time series
        local_nodes: List of node/service names to include
        use_pca: Whether to apply PCA dimensionality reduction
        pca_components: Number of PCA components to keep
        
    Returns:
        Tuple of (data_matrix, column_names)
        data_matrix shape: (n_variables, n_timesteps)
    """
    cols: List[str] = []
    for s in local_nodes:
        matched = [c for c in data.columns if c != "time" and (c == s or c.startswith(f"{s}_"))]
        cols.extend(matched)
    cols = list(dict.fromkeys(cols))
    
    if not cols:
        return np.array([]).reshape(0, 0), []
    
    X = data[cols].to_numpy(dtype=float)
    X = smart_fillna_matrix(X, method='forward_backward')
    
    if use_pca and X.shape[1] > pca_components:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(pca_components, X.shape[1]))
        X = pca.fit_transform(X)
        cols = [f"PC{i}" for i in range(X.shape[1])]
    
    X = X.T
    return X, cols


def clean_data_matrix(X: np.ndarray, min_std: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean data matrix by removing constant variables and adding noise if needed.
    
    Args:
        X: Input matrix (variables, time)
        min_std: Minimum standard deviation threshold
        
    Returns:
        Tuple of (cleaned_matrix, mask_of_kept_variables)
    """
    std_vals = np.std(X, axis=1)
    mask = std_vals > min_std
    X_clean = X[mask, :]
    
    for i in range(X_clean.shape[0]):
        if np.std(X_clean[i, :]) < min_std * 10:
            noise = np.random.normal(0, min_std, X_clean.shape[1])
            X_clean[i, :] += noise
    
    return X_clean, mask


def run_pc(
    X: np.ndarray,
    alpha: float = 0.05,
    max_conds_dim: int | None = 3
) -> Dict[str, Any]:
    """
    Run PC algorithm for causal discovery (instantaneous relationships only).
    
    Args:
        X: Input matrix (variables, time)
        alpha: Significance level
        max_conds_dim: Maximum conditioning set size
        
    Returns:
        Dictionary with adjacency matrix
    """
    if X.size == 0 or X.shape[0] < 2:
        raise ValueError(f"Insufficient data: shape={X.shape}")
    
    X_clean, mask = clean_data_matrix(X)

    X_std = X_clean.copy()
    var_std = np.std(X_std, axis=1, keepdims=True)
    var_std[var_std == 0] = 1.0
    X_std = (X_std - np.mean(X_std, axis=1, keepdims=True)) / var_std
    
    if X_clean.shape[0] < 2:
        raise ValueError(f"Not enough variables after cleaning")
    
    dataframe = data_processing.DataFrame(X_std)
    # Note: We use tigramite's PCMCI class but only call run_pc_stable (PC algorithm)
    causal_engine = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    
    try:
        alpha_env = os.environ.get("PC_ALPHA")
        if alpha_env is not None:
            alpha = float(alpha_env)
    except Exception:
        pass
    try:
        mcd_env = os.environ.get("PC_MAXCONDS")
        if mcd_env is not None:
            max_conds_dim = int(mcd_env)
    except Exception:
        pass

    try:
        results = causal_engine.run_pc_stable(
            pc_alpha=alpha,
            tau_max=0,
            max_conds_dim=max_conds_dim
        )
        
        graph = results['graph'][:, :, 0]
        
    except Exception:
        n_vars = X_clean.shape[0]
        graph = np.zeros((n_vars, n_vars))
    
    if not np.any(graph):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "PC algorithm returned empty graph. Falling back to correlation-based edge detection. "
            "This may produce less accurate causal relationships. "
            "Consider adjusting PC_ALPHA or PC_MAXCONDS environment variables."
        )
        try:
            thr = float(os.environ.get("CORR_THRESHOLD", "0.2"))
            topk = int(os.environ.get("CORR_TOPK", "0"))
        except Exception:
            thr, topk = 0.2, 0
        corr = np.corrcoef(X_std)
        np.fill_diagonal(corr, 0.0)
        edges_idx = np.argwhere(np.abs(corr) >= thr)
        if topk > 0 and edges_idx.size > 0:
            graph_fallback = np.zeros_like(corr)
            n_vars = corr.shape[0]
            for j in range(n_vars):
                incoming = [(i, j, abs(corr[i, j])) for i in range(n_vars) if i != j]
                incoming.sort(key=lambda x: x[2], reverse=True)
                for i, j2, _s in incoming[:topk]:
                    graph_fallback[i, j2] = corr[i, j2]
            graph = graph_fallback
        else:
            graph = (np.abs(corr) >= thr).astype(float) * corr

    return {
        "graph": graph,
        "p_matrix": None,
        "val_matrix": None
    }


def extract_edges_and_strengths(
    report: Dict[str, Any],
    alpha: float = 0.05
) -> Tuple[List[Tuple[int, int, int]], Dict[Tuple[int, int], float]]:
    """
    Extract significant edges and their strengths from PC results.
    
    Args:
        report: Results from PC algorithm
        alpha: Significance threshold (unused for PC but kept for compatibility)
        
    Returns:
        Tuple of (edges, edge_strengths)
        edges: List of (cause_idx, effect_idx, lag=0)
        edge_strengths: Dict mapping (cause, effect) to strength [0,1]
    """
    edges: List[Tuple[int, int, int]] = []
    strengths: Dict[Tuple[int, int], float] = {}
    
    graph = report.get("graph")
    
    if graph is None:
        return edges, strengths
    
    n_vars = graph.shape[0]
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and graph[i, j] != 0:
                edges.append((i, j, 0))
                strengths[(i, j)] = float(min(1.0, abs(graph[i, j])))
    
    return edges, strengths


def discover_causal_graph(
    data: pd.DataFrame,
    local_nodes: List[str],
    config: PCShapleyConfig,
    method: str = "pc"
) -> Dict[str, Any]:
    """
    PC-based causal discovery interface.
    
    Args:
        data: Input time series DataFrame
        local_nodes: List of nodes to include
        config: Configuration object
        method: Causal discovery method (only "pc" is supported)
        
    Returns:
        Dictionary containing:
        - edges: List of causal edges
        - edge_strengths: Dict of edge strengths
        - columns: Column names used
    """
    if method != "pc":
        raise ValueError(f"Only 'pc' method is supported, got '{method}'")
    
    X, cols = prepare_data_matrix(data, local_nodes, config.use_pca, config.pca_components)
    
    if X.size == 0:
        return {
            "edges": [],
            "edge_strengths": {},
            "columns": []
        }
    
    try:
        report = run_pc(X, config.pc_alpha, config.pc_max_conds_dim)
        edges, strengths = extract_edges_and_strengths(report, config.pc_alpha)
    except Exception:
        edges = []
        strengths = {}
    
    G = nx.DiGraph()
    G.add_nodes_from(range(len(cols)))
    for (i, j, _tau) in edges:
        G.add_edge(i, j, weight=strengths.get((i, j), 0.0))
    
    return {
        "edges": edges,
        "edge_strengths": strengths,
        "columns": cols
    }
