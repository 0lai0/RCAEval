from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from tigramite import data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from .config import PCMCIShapleyConfig
from .utils import smart_fillna_matrix


def build_timeseries_matrix(data: pd.DataFrame, local_nodes: List[str], use_pca: bool = False, pca_components: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Build time series matrix with intelligent missing value handling.
    
    Args:
        data: Input DataFrame
        local_nodes: List of local node names
        use_pca: Whether to use PCA (not implemented yet)
        pca_components: Number of PCA components
        
    Returns:
        Tuple of (matrix, column_names)
    """
    # Use columns matching local_nodes prefixes (service or exact names)
    cols: List[str] = []
    for s in local_nodes:
        matched = [c for c in data.columns if c != "time" and (c == s or c.startswith(f"{s}_"))]
        cols.extend(matched)
    cols = list(dict.fromkeys(cols))
    
    if not cols:
        # 如果沒有匹配的列，返回空矩陣
        return np.array([]).reshape(0, 0), []
    
    X = data[cols].to_numpy(dtype=float)
    
    # 使用智能插值處理缺失值
    X = smart_fillna_matrix(X, method='forward_backward')
    
    # Arrange to (variables, time)
    X = X.T
    return X, cols


def run_pcmci_plus(X: np.ndarray, tau_max: int, alpha: float, 
                   max_conds_dim: int | None = None) -> Dict[str, Any]:
    """
    Run PCMCI+ algorithm with improved missing value handling.
    
    Args:
        X: Input matrix (variables, time)
        tau_max: Maximum time lag
        alpha: Significance level
        max_conds_dim: Maximum condition set dimension (None means no limit)
    """
    # 檢查輸入
    if X.size == 0:
        raise ValueError("Empty input matrix")
    
    # 數據預處理：移除常數列
    X_clean = X.copy()
    
    # 移除常數列（標準差為0），但使用更寬鬆的閾值
    std_mask = np.std(X_clean, axis=1) > 1e-8  # 更寬鬆的閾值
    X_clean = X_clean[std_mask, :]
    
    # 檢查是否還有足夠的變量
    if X_clean.shape[0] < 2:
        raise ValueError(f"Not enough variables after cleaning: {X_clean.shape[0]} variables remaining")
    
    # 檢查時間序列長度
    if X_clean.shape[1] < tau_max + 2:
        raise ValueError(f"Time series too short: {X_clean.shape[1]} < {tau_max + 2}")
    
    # 創建 DataFrame 並運行 PCMCI
    dataframe = data_processing.DataFrame(X_clean)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    
    # 動態調整 max_conds_dim
    if max_conds_dim is None:
        max_conds_dim_actual = None
    else:
        # 確保不超過變量數
        max_conds_dim_actual = min(max_conds_dim, X_clean.shape[0] - 2)
        # 確保至少為 1
        max_conds_dim_actual = max(1, max_conds_dim_actual)
    
    try:
        report = pcmci.run_pcmci(
            tau_max=tau_max, 
            pc_alpha=alpha, 
            max_conds_dim=max_conds_dim_actual,
            max_conds_py=None,  # 可選：限制 Y 的條件集
            max_conds_px=None   # 可選：限制 X 的條件集
        )
    except Exception as e:
        # 如果 PCMCI 失敗，返回空結果
        print(f"PCMCI failed: {e}")
        return {
            "p_matrix": np.ones((X_clean.shape[0], X_clean.shape[0], tau_max + 1)),
            "val_matrix": np.zeros((X_clean.shape[0], X_clean.shape[0], tau_max + 1)),
            "graph": np.zeros((X_clean.shape[0], X_clean.shape[0])),
        }
    
    return report


def extract_significant_edges(report: Dict[str, Any], alpha: float) -> List[Tuple[int, int, int]]:
    pmat = report["p_matrix"]  # shape: (cause, effect, tau)
    C, E, T = pmat.shape
    edges: List[Tuple[int, int, int]] = []
    for i in range(C):
        for j in range(E):
            if i == j:
                continue
            for tau in range(1, T):
                if pmat[i, j, tau] <= alpha:
                    edges.append((i, j, tau))
    return edges


def compute_edge_strength(report: Dict[str, Any], edges: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], float]:
    val = report.get("val_matrix", None)
    strengths: Dict[Tuple[int, int], float] = {}
    if val is None:
        # fallback: use inverse p-value proxy
        pmat = report["p_matrix"]
        for (i, j, _) in edges:
            p = np.nanmin(pmat[i, j, 1:])
            strengths[(i, j)] = float(max(0.0, min(1.0, 1.0 - p))) if np.isfinite(p) else 0.0
        return strengths
    C, E, T = val.shape
    for (i, j, _) in edges:
        s = np.nanmax(np.abs(val[i, j, 1:]))
        strengths[(i, j)] = float(0.0 if not np.isfinite(s) else min(1.0, max(0.0, s)))
    return strengths


def local_pcmci_causal_test(data: pd.DataFrame, local_nodes: List[str], config: PCMCIShapleyConfig) -> Dict[str, Any]:
    X, cols = build_timeseries_matrix(data, local_nodes, use_pca=config.use_pca, pca_components=config.pca_components)
    report = run_pcmci_plus(
        X, 
        tau_max=config.tau_max, 
        alpha=config.pcmci_alpha,
        max_conds_dim=config.pcmci_max_conds_dim  # 傳遞參數
    )
    edges = extract_significant_edges(report, alpha=config.pcmci_alpha)
    strengths = compute_edge_strength(report, edges)
    # Build graph in variable-space (cols), later map back to services if needed
    G = nx.DiGraph()
    G.add_nodes_from(range(len(cols)))
    for (i, j, _tau) in edges:
        G.add_edge(i, j, weight=strengths.get((i, j), 0.0))
    return {
        "edges": edges,
        "edge_strengths": strengths,
        "pcmci_graph": G,
        "columns": cols,
    }
