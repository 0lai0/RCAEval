from __future__ import annotations
from typing import Dict, Any
import pandas as pd

from .config import CPGShapConfig
from .utils import robust_standardize, robust_standardize_with_interpolation


def robust_normalize(df: pd.DataFrame, eps: float = 1e-9, use_interpolation: bool = True) -> pd.DataFrame:
    """
    Column-wise robust normalization using MAD with optional interpolation for missing values.
    
    Args:
        df: Input DataFrame
        eps: Small constant to prevent division by zero
        use_interpolation: Whether to use interpolation for missing values
    """
    out = {}
    for c in df.columns:
        if c == "time":
            continue
        
        if use_interpolation:
            # 使用帶插值的魯棒標準化
            out[c] = robust_standardize_with_interpolation(df[c], eps=eps)
        else:
            # 使用原始方法
            out[c] = robust_standardize(df[c], eps=eps)
    
    res = pd.DataFrame(out)
    if "time" in df.columns:
        res.insert(0, "time", df["time"].values)
    return res


def detect_anomaly_zscore(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Compute anomaly scores with robust z-score; values below threshold set to 0."""
    out = {}
    for c in df.columns:
        if c == "time":
            continue
        z = df[c].abs()
        out[c] = (z.where(z > threshold, other=0.0))
    res = pd.DataFrame(out)
    if "time" in df.columns:
        res.insert(0, "time", df["time"].values)
    return res


def aggregate_node_anomaly(anomaly_df: pd.DataFrame, metric_map: Dict[str, list]) -> pd.Series:
    """Aggregate metric anomalies into per-service anomaly score by simple mean."""
    scores = {}
    for service, metrics in metric_map.items():
        cols = [m for m in metrics if m in anomaly_df.columns]
        if not cols:
            scores[service] = 0.0
            continue
        scores[service] = anomaly_df[cols].mean(axis=1).iloc[-1]
    return pd.Series(scores)


def preprocess_data(data: pd.DataFrame, config: CPGShapConfig) -> Dict[str, Any]:
    """
    End-to-end preprocessing pipeline producing normalized data and anomaly scores.
    Now includes intelligent missing value handling.
    """
    # 使用插值進行魯棒標準化
    norm = robust_normalize(data, use_interpolation=True)
    
    if config.anomaly_method == "zscore":
        anomalies = detect_anomaly_zscore(norm, threshold=config.anomaly_threshold)
    else:
        # TODO: SPOT method placeholder; fallback to zscore for now
        anomalies = detect_anomaly_zscore(norm, threshold=config.anomaly_threshold)
    # Derive metric_map by prefix splitting: service_metric
    metric_map: Dict[str, list] = {}
    for c in anomalies.columns:
        if c == "time":
            continue
        if "_" in c:
            service = c.split("_")[0]
        else:
            service = c
        metric_map.setdefault(service, []).append(c)
    node_anomaly = aggregate_node_anomaly(anomalies, metric_map)
    # Build per-service anomaly time series a_s(t) by row-wise mean
    node_anomaly_ts = {}
    for service, metrics in metric_map.items():
        cols = [m for m in metrics if m in anomalies.columns]
        if cols:
            node_anomaly_ts[service] = anomalies[cols].mean(axis=1)
    return {
        "normalized_df": norm,
        "anomaly_scores": anomalies,
        "node_anomaly": node_anomaly,
        "node_anomaly_ts": node_anomaly_ts,
        "metric_mapping": metric_map,
        }
