"""
Build a PyG HeteroData graph from preprocessed DataFrames.

Node types
----------
- ``metric``  : one per original column (e.g. ``frontend_cpu``)
- ``service`` : one per unique service (e.g. ``frontend``)

Edge types
----------
- ``("service", "calls", "service")`` : causal / correlation edges
- ``("service", "owns", "metric")``   : ownership
- ``("metric", "belongs_to", "service")`` : reverse of *owns*
"""

import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .feature_engineer import compute_deviation_features
from RCAEval.graph_construction.hetRCA_granger import granger

GRAPH_BUILD_CONFIG = {
    "apply_fdr": os.environ.get("GRANGER_FDR", "1") == "1",
    "preprocess_stationarity": os.environ.get("GRANGER_ADF", "1") == "1",
    "fallback": os.environ.get("GRANGER_FALLBACK", "pearson"),
    "version": os.environ.get("GRAPH_BUILD_VERSION", "v2_bh_fdr"),
    "pearson_top_percentile": float(os.environ.get("PEARSON_TOP_PERCENTILE", "90")),
}


# ---------------------------------------------------------------------------
# Service name extraction
# ---------------------------------------------------------------------------

# Known multi-token metric suffixes, ordered longest-first for greedy match.
_KNOWN_SUFFIXES = [
    "lat_50", "lat_90", "lat_99", "lat_mean",
    "cpu", "mem", "latency", "load", "error",
    "diskio", "socket", "requests",
]


def _extract_service_name(col_name: str) -> str:
    """Extract the service name from a ``{service}_{metric_type}`` column.

    Tries known multi-token suffixes first (e.g. ``lat_90``), then falls
    back to splitting on the last ``_``.

    Examples
    --------
    >>> _extract_service_name("frontend_cpu")
    'frontend'
    >>> _extract_service_name("ts-ui-dashboard_latency")
    'ts-ui-dashboard'
    >>> _extract_service_name("front-end_lat_90")
    'front-end'
    """
    for suffix in _KNOWN_SUFFIXES:
        tag = "_" + suffix
        if col_name.endswith(tag):
            return col_name[: -len(tag)]
    # Fallback: last underscore-separated token
    parts = col_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return col_name


def _extract_metric_type(col_name: str) -> str:
    """Return the metric-type suffix of a column name."""
    for suffix in _KNOWN_SUFFIXES:
        tag = "_" + suffix
        if col_name.endswith(tag):
            return suffix
    parts = col_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1]
    return "unknown"


# ---------------------------------------------------------------------------
# Correlation-based service graph
# ---------------------------------------------------------------------------

def _build_causal_edges(
    anomal_df: pd.DataFrame,
    service_names: list,
    service_cols: dict,
    method: str = "granger",
    threshold: float = 0.3,
    cache_dir: str = "cache/graphs",
):
    """Build service-service edges via causal discovery or correlation, with caching."""
    n_srv = len(service_names)
    
    # Pre-compute service-level aggregated series
    srv_series = {}
    for svc in service_names:
        cols = service_cols[svc]
        valid = [c for c in cols if c in anomal_df.columns]
        if valid:
            srv_series[svc] = anomal_df[valid].mean(axis=1).to_numpy()
        else:
            srv_series[svc] = np.zeros(len(anomal_df))
    
    service_level_df = pd.DataFrame(srv_series)
    
    # Include a hash of the actual time-series values so that different
    # failure cases sharing the same service set / window length are NOT
    # silently served from the same cached graph (reviewer #2 bug report).
    data_hash = hashlib.sha256(anomal_df.to_numpy().tobytes()).hexdigest()[:8]
    key_str = (
        "|".join(sorted(service_names))
        + f"|{method}|{len(anomal_df)}|{data_hash}"
        + f"|{GRAPH_BUILD_CONFIG['version']}"
        + f"|fdr={int(GRAPH_BUILD_CONFIG['apply_fdr'])}"
        + f"|adf={int(GRAPH_BUILD_CONFIG['preprocess_stationarity'])}"
        + f"|fb={GRAPH_BUILD_CONFIG['fallback']}"
    )
    cache_key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"{cache_key}.npz")
    
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        build_stats = {}
        if "build_stats" in cached.files:
            build_stats = cached["build_stats"].item()
        return cached["edge_index"], cached["edge_weight"], build_stats
        
    src_list, dst_list, weight_list = [], [], []
    build_stats = {
        "n_granger": 0,
        "n_pearson_fallback": 0,
        "fallback_triggered": False,
        "granger_detail": {},
        "fallback_mode": GRAPH_BUILD_CONFIG["fallback"],
    }

    def _pearson_top_edges():
        corr_pairs = []
        for i in range(n_srv):
            for j in range(n_srv):
                if i == j:
                    continue
                a = srv_series[service_names[i]]
                b = srv_series[service_names[j]]
                std_a, std_b = np.std(a), np.std(b)
                if std_a < 1e-12 or std_b < 1e-12:
                    continue
                corr = float(np.abs(np.corrcoef(a, b)[0, 1]))
                corr_pairs.append((i, j, corr))

        if not corr_pairs:
            return [], [], []

        all_corrs = [c for _, _, c in corr_pairs]
        cutoff = np.percentile(all_corrs, GRAPH_BUILD_CONFIG["pearson_top_percentile"])
        p_src, p_dst, p_w = [], [], []
        for i, j, corr in corr_pairs:
            if corr >= cutoff:
                p_src.append(i)
                p_dst.append(j)
                p_w.append(corr)
        return p_src, p_dst, p_w

    if method == "granger":
        try:
            adj_matrix, granger_stats = granger(
                service_level_df,
                apply_fdr=GRAPH_BUILD_CONFIG["apply_fdr"],
                preprocess_stationarity=GRAPH_BUILD_CONFIG["preprocess_stationarity"],
                return_stats=True,
            )
            build_stats["granger_detail"] = granger_stats
            for i in range(n_srv):
                for j in range(n_srv):
                    if i != j and adj_matrix[i, j] > 0:
                        # adj_matrix[i,j] means j causes i. So src=j, dst=i
                        src_list.append(j)
                        dst_list.append(i)
                        weight_list.append(1.0)
            build_stats["n_granger"] = len(src_list)
        except Exception as exc:
            build_stats["granger_detail"] = {"error": str(exc)}
    elif method == "corr":
        corr_matrix = []
        for i in range(n_srv):
            for j in range(n_srv):
                if i == j:
                    continue
                a = srv_series[service_names[i]]
                b = srv_series[service_names[j]]
                std_a, std_b = np.std(a), np.std(b)
                if std_a < 1e-12 or std_b < 1e-12:
                    continue
                corr = float(np.abs(np.corrcoef(a, b)[0, 1]))
                corr_matrix.append((i, j, corr))
                
        if corr_matrix:
            # Dynamic threshold to ensure graph sparsity (Top 10% edges, or at least base threshold)
            all_corrs = [c for _, _, c in corr_matrix]
            dynamic_threshold = max(threshold, np.percentile(all_corrs, 90))
            for i, j, corr in corr_matrix:
                if corr >= dynamic_threshold:
                    src_list.append(i)
                    dst_list.append(j)
                    weight_list.append(corr)

    min_granger_edges = max(n_srv - 1, 1)
    if method == "granger" and len(src_list) < min_granger_edges:
        build_stats["fallback_triggered"] = True
        if GRAPH_BUILD_CONFIG["fallback"] == "pearson":
            p_src, p_dst, p_w = _pearson_top_edges()
            src_list.extend(p_src)
            dst_list.extend(p_dst)
            weight_list.extend(p_w)
            build_stats["n_pearson_fallback"] = len(p_src)
        elif GRAPH_BUILD_CONFIG["fallback"] == "full":
            for i in range(n_srv):
                for j in range(n_srv):
                    if i != j:
                        src_list.append(i)
                        dst_list.append(j)
                        weight_list.append(1.0 / max(n_srv - 1, 1))
            build_stats["n_pearson_fallback"] = 0

    if not src_list:
        # Final safe fallback: star-like spanning structure
        for i in range(1, n_srv):
            src_list.append(0)
            dst_list.append(i)
            weight_list.append(1.0 / max(n_srv, 1))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(weight_list, dtype=np.float32)
    
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        cache_path,
        edge_index=edge_index,
        edge_weight=edge_weight,
        build_stats=np.array(build_stats, dtype=object),
    )
    
    return edge_index, edge_weight, build_stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_hetero_graph(
    normal_df: pd.DataFrame,
    anomal_df: pd.DataFrame,
    metric_cols: list,
    dataset: str = None,
    sli: str = None,
    corr_threshold: float = 0.3,
    causal_method: str = "granger",
    return_stats: bool = False,
) -> HeteroData:
    """Convert preprocessed DataFrames into a PyG ``HeteroData`` object.

    Parameters
    ----------
    normal_df, anomal_df : pd.DataFrame
        Normal / anomalous period data.  Must share *metric_cols*.
    metric_cols : list[str]
        Metric column names (``time`` already excluded).
    dataset : str, optional
        Dataset identifier (not used in prototype; reserved for future
        dataset-specific graph-building strategies).
    sli : str, optional
        SLI column name.
    corr_threshold : float
        Threshold for Pearson-correlation-based service edges.

    Returns
    -------
    data : HeteroData
    """

    # ---- metric features ------------------------------------------------
    metric_features = compute_deviation_features(normal_df, anomal_df, metric_cols, sli=sli)
    feat_dim = metric_features.shape[1]

    # ---- parse service / metric structure --------------------------------
    col_to_service = {}
    col_to_metric_type = {}
    service_set = set()

    for c in metric_cols:
        svc = _extract_service_name(c)
        mt = _extract_metric_type(c)
        col_to_service[c] = svc
        col_to_metric_type[c] = mt
        service_set.add(svc)

    service_names = sorted(service_set)
    svc_to_idx = {s: i for i, s in enumerate(service_names)}

    # ---- service features (aggregate of owned metrics) -------------------
    service_cols_map = {s: [] for s in service_names}
    for c in metric_cols:
        service_cols_map[col_to_service[c]].append(c)

    service_features = np.zeros((len(service_names), feat_dim), dtype=np.float32)
    for idx, svc in enumerate(service_names):
        owned = service_cols_map[svc]
        metric_idxs = [metric_cols.index(c) for c in owned]
        if metric_idxs:
            service_features[idx] = metric_features[metric_idxs].max(axis=0)

    # ---- build edges -----------------------------------------------------
    # 1. service -> service (correlation / causal)
    ss_edge_index, ss_edge_weight, build_stats = _build_causal_edges(
        anomal_df, service_names, service_cols_map, 
        method=causal_method, threshold=corr_threshold,
    )

    # 2. service -> metric (owns)
    sm_src, sm_dst = [], []
    for mi, c in enumerate(metric_cols):
        si = svc_to_idx[col_to_service[c]]
        sm_src.append(si)
        sm_dst.append(mi)
    sm_edge_index = np.array([sm_src, sm_dst], dtype=np.int64)

    # 3. metric -> service (belongs_to, reverse of owns)
    ms_edge_index = np.array([sm_dst, sm_src], dtype=np.int64)

    # ---- assemble HeteroData --------------------------------------------
    data = HeteroData()

    data["metric"].x = torch.tensor(metric_features, dtype=torch.float)
    data["metric"].col_names = metric_cols
    data["metric"].col_to_service = col_to_service
    data["metric"].col_to_metric_type = col_to_metric_type

    data["service"].x = torch.tensor(service_features, dtype=torch.float)
    data["service"].names = service_names

    data[("service", "calls", "service")].edge_index = torch.tensor(
        ss_edge_index, dtype=torch.long,
    )
    data[("service", "calls", "service")].edge_weight = torch.tensor(
        ss_edge_weight, dtype=torch.float,
    )

    data[("service", "owns", "metric")].edge_index = torch.tensor(
        sm_edge_index, dtype=torch.long,
    )

    data[("metric", "belongs_to", "service")].edge_index = torch.tensor(
        ms_edge_index, dtype=torch.long,
    )

    # SLI metadata
    data.sli_col = sli
    data.sli_idx = metric_cols.index(sli) if (sli and sli in metric_cols) else -1

    if return_stats:
        return data, build_stats
    return data
