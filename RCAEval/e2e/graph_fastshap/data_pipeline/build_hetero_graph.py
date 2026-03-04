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

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .feature_engineer import compute_deviation_features


# ---------------------------------------------------------------------------
# Service name extraction
# ---------------------------------------------------------------------------

def _extract_service_name(col_name: str) -> str:
    """Extract the service name from a ``{service}_{metric_type}`` column.

    The convention used across RCAEval datasets is that the *last* ``_``-
    separated token is the metric type (cpu, mem, latency, load, error, ...).
    Everything before that token is the service name.

    Examples
    --------
    >>> _extract_service_name("frontend_cpu")
    'frontend'
    >>> _extract_service_name("ts-ui-dashboard_latency")
    'ts-ui-dashboard'
    >>> _extract_service_name("front-end_lat_90")
    'front-end_lat'
    """
    parts = col_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return col_name


def _extract_metric_type(col_name: str) -> str:
    """Return the metric-type suffix of a column name."""
    parts = col_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1]
    return "unknown"


# ---------------------------------------------------------------------------
# Correlation-based service graph
# ---------------------------------------------------------------------------

def _build_service_corr_edges(
    anomal_df: pd.DataFrame,
    service_names: list,
    service_cols: dict,
    threshold: float = 0.3,
):
    """Build service-service edges via Pearson correlation.

    For each pair of services we compute the maximum absolute correlation
    among their metric columns and add an edge if it exceeds *threshold*.

    Returns
    -------
    edge_index : np.ndarray, shape ``(2, E)``
    edge_weight : np.ndarray, shape ``(E,)``
    """
    n_srv = len(service_names)
    src_list, dst_list, weight_list = [], [], []

    # Pre-compute service-level aggregated series (mean of their metrics)
    srv_series = {}
    for svc in service_names:
        cols = service_cols[svc]
        valid = [c for c in cols if c in anomal_df.columns]
        if valid:
            srv_series[svc] = anomal_df[valid].mean(axis=1).to_numpy()
        else:
            srv_series[svc] = np.zeros(len(anomal_df))

    for i in range(n_srv):
        for j in range(n_srv):
            if i == j:
                continue
            a = srv_series[service_names[i]]
            b = srv_series[service_names[j]]
            std_a = np.std(a)
            std_b = np.std(b)
            if std_a < 1e-12 or std_b < 1e-12:
                continue
            corr = float(np.abs(np.corrcoef(a, b)[0, 1]))
            if corr > threshold:
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(corr)

    if not src_list:
        # Fallback: fully-connected with uniform weight to avoid empty graph
        for i in range(n_srv):
            for j in range(n_srv):
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)
                    weight_list.append(1.0 / max(n_srv - 1, 1))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(weight_list, dtype=np.float32)
    return edge_index, edge_weight


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
    metric_features = compute_deviation_features(normal_df, anomal_df, metric_cols)
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
    ss_edge_index, ss_edge_weight = _build_service_corr_edges(
        anomal_df, service_names, service_cols_map, threshold=corr_threshold,
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

    return data
