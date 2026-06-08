"""
Unified causal discovery module supporting multiple algorithms.
Simplifies the causal graph construction process.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Literal
import os
import numpy as np
import pandas as pd
import networkx as nx
from tigramite import data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from .config import CPGShapConfig
from .utils import smart_fillna_matrix


CausalMethod = Literal["pcmci", "pc", "ges", "fci", "lingam"]


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
    # Match columns by service prefix
    cols: List[str] = []
    for s in local_nodes:
        matched = [c for c in data.columns if c != "time" and (c == s or c.startswith(f"{s}_"))]
        cols.extend(matched)
    cols = list(dict.fromkeys(cols))
    
    if not cols:
        return np.array([]).reshape(0, 0), []
    
    # Extract matrix and handle missing values
    X = data[cols].to_numpy(dtype=float)
    X = smart_fillna_matrix(X, method='forward_backward')
    
    # PCA dimensionality reduction (optional)
    if use_pca and X.shape[1] > pca_components:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(pca_components, X.shape[1]))
        X = pca.fit_transform(X)
        cols = [f"PC{i}" for i in range(X.shape[1])]
    
    # Transpose to (variables, time)
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
    
    # Add minimal noise to near-constant variables
    for i in range(X_clean.shape[0]):
        if np.std(X_clean[i, :]) < min_std * 10:
            noise = np.random.normal(0, min_std, X_clean.shape[1])
            X_clean[i, :] += noise
    
    return X_clean, mask


def run_pcmci(
    X: np.ndarray,
    tau_max: int = 5,
    alpha: float = 0.05,
    max_conds_dim: int | None = 3
) -> Dict[str, Any]:
    """
    Run PCMCI+ algorithm for time-series causal discovery.
    
    Args:
        X: Input matrix (variables, time)
        tau_max: Maximum time lag to consider
        alpha: Significance level for independence tests
        max_conds_dim: Maximum conditioning set size
        
    Returns:
        Dictionary with p_matrix, val_matrix, and graph
    """
    if X.size == 0 or X.shape[0] < 2:
        raise ValueError(f"Insufficient data: shape={X.shape}")
    
    if X.shape[1] < tau_max + 2:
        raise ValueError(f"Time series too short: {X.shape[1]} < {tau_max + 2}")
    
    # Clean data
    X_clean, mask = clean_data_matrix(X)
    
    if X_clean.shape[0] < 2:
        raise ValueError(f"Not enough variables after cleaning: {X_clean.shape[0]}")
    
    # Adjust max_conds_dim
    if max_conds_dim is not None:
        max_conds_dim = min(max_conds_dim, X_clean.shape[0] - 2)
        max_conds_dim = max(1, max_conds_dim)
    
    # Run PCMCI
    dataframe = data_processing.DataFrame(X_clean)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    
    try:
        report = pcmci.run_pcmci(
            tau_max=tau_max,
            pc_alpha=alpha,
            max_conds_dim=max_conds_dim
        )
    except Exception as e:
        # Return empty results on failure
        n_vars = X_clean.shape[0]
        report = {
            "p_matrix": np.ones((n_vars, n_vars, tau_max + 1)),
            "val_matrix": np.zeros((n_vars, n_vars, tau_max + 1)),
            "graph": np.zeros((n_vars, n_vars))
        }
    
    return report


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

    # Z-score standardization improves PC stability on heterogeneous scales
    X_std = X_clean.copy()
    var_std = np.std(X_std, axis=1, keepdims=True)
    var_std[var_std == 0] = 1.0
    X_std = (X_std - np.mean(X_std, axis=1, keepdims=True)) / var_std
    
    if X_clean.shape[0] < 2:
        raise ValueError(f"Not enough variables after cleaning")
    
    # Use tigramite's PC algorithm on lag-0 only
    dataframe = data_processing.DataFrame(X_std)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    
    # Allow environment overrides for sensitivity
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
        # Run PC algorithm (tau_max=0 means contemporaneous only)
        results = pcmci.run_pc_stable(
            pc_alpha=alpha,
            tau_max=0,
            max_conds_dim=max_conds_dim
        )
        
        # Extract graph at lag 0
        graph = results['graph'][:, :, 0]
        
    except Exception as e:
        # Return empty graph on failure
        n_vars = X_clean.shape[0]
        graph = np.zeros((n_vars, n_vars))
    
    # Fallback: if PC returns empty graph, use correlation thresholding
    if not np.any(graph):
        try:
            thr = float(os.environ.get("CORR_THRESHOLD", "0.2"))
            topk = int(os.environ.get("CORR_TOPK", "0"))  # 0 means no top-k limit
        except Exception:
            thr, topk = 0.2, 0
        corr = np.corrcoef(X_std)
        np.fill_diagonal(corr, 0.0)
        edges_idx = np.argwhere(np.abs(corr) >= thr)
        # Optionally restrict to top-k absolute correlations per target
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
    method: CausalMethod,
    alpha: float = 0.05,
    tau_max: int = 5
) -> Tuple[List[Tuple[int, int, int]], Dict[Tuple[int, int], float]]:
    """
    Extract significant edges and their strengths from causal discovery results.
    
    Args:
        report: Results from causal discovery algorithm
        method: Which algorithm was used
        alpha: Significance threshold
        tau_max: Maximum time lag (for PCMCI)
        
    Returns:
        Tuple of (edges, edge_strengths)
        edges: List of (cause_idx, effect_idx, lag)
        edge_strengths: Dict mapping (cause, effect) to strength [0,1]
    """
    edges: List[Tuple[int, int, int]] = []
    strengths: Dict[Tuple[int, int], float] = {}
    
    if method == "pcmci":
        # Extract from PCMCI p_matrix
        pmat = report.get("p_matrix")
        val = report.get("val_matrix")
        
        if pmat is None:
            return edges, strengths
        
        C, E, T = pmat.shape
        
        # Find significant edges (lag > 0)
        for i in range(C):
            for j in range(E):
                if i == j:
                    continue
                for tau in range(1, T):
                    if pmat[i, j, tau] <= alpha:
                        edges.append((i, j, tau))
        
        # Compute edge strengths
        if val is not None:
            for (i, j, _) in edges:
                s = np.nanmax(np.abs(val[i, j, 1:]))
                strengths[(i, j)] = float(min(1.0, max(0.0, s if np.isfinite(s) else 0.0)))
        else:
            # Use p-values as proxy
            for (i, j, _) in edges:
                p = np.nanmin(pmat[i, j, 1:])
                strengths[(i, j)] = float(max(0.0, min(1.0, 1.0 - p)) if np.isfinite(p) else 0.0)
    
    elif method in ["pc", "ges", "fci"]:
        # Extract from adjacency matrix (no time lags)
        graph = report.get("graph")
        
        if graph is None:
            return edges, strengths
        
        n_vars = graph.shape[0]
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and graph[i, j] != 0:
                    edges.append((i, j, 0))  # lag=0 for instantaneous
                    # Use absolute value as strength
                    strengths[(i, j)] = float(min(1.0, abs(graph[i, j])))
    
    return edges, strengths


def discover_causal_graph(
    data: pd.DataFrame,
    local_nodes: List[str],
    config: CPGShapConfig,
    method: CausalMethod = "pcmci"
) -> Dict[str, Any]:
    """
    Unified interface for causal discovery.
    
    Args:
        data: Input time series DataFrame
        local_nodes: List of nodes to include
        config: Configuration object
        method: Causal discovery method to use
        
    Returns:
        Dictionary containing:
        - edges: List of causal edges
        - edge_strengths: Dict of edge strengths
        - pcmci_graph: NetworkX graph
        - columns: Column names used
    """
    # Prepare data
    X, cols = prepare_data_matrix(data, local_nodes, config.use_pca, config.pca_components)
    
    if X.size == 0:
        return {
            "edges": [],
            "edge_strengths": {},
            "pcmci_graph": nx.DiGraph(),
            "columns": []
        }
    
    # Run causal discovery
    try:
        if method == "pcmci":
            report = run_pcmci(X, config.tau_max, config.pcmci_alpha, config.pcmci_max_conds_dim)
        elif method == "pc":
            report = run_pc(X, config.pcmci_alpha, config.pcmci_max_conds_dim)
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")
        
        # Extract edges and strengths
        edges, strengths = extract_edges_and_strengths(
            report, method, config.pcmci_alpha, config.tau_max
        )
        
    except Exception as e:
        # Fallback to empty graph
        edges = []
        strengths = {}
    
    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(len(cols)))
    for (i, j, _tau) in edges:
        G.add_edge(i, j, weight=strengths.get((i, j), 0.0))
    
    return {
        "edges": edges,
        "edge_strengths": strengths,
        "pcmci_graph": G,
        "columns": cols
    }

