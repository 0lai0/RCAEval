"""
Feature engineering: compute deviation features from normal/anomalous time series.

For each metric column, we compute a deviation vector relative to the
normal-period baseline (ToD/EWMA).  The anomalous window is then summarised
into a fixed-length feature vector per metric.
"""

import numpy as np
import pandas as pd


def compute_deviation_features(
    normal_df: pd.DataFrame,
    anomal_df: pd.DataFrame,
    metric_cols: list,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute per-metric deviation features.

    Parameters
    ----------
    normal_df : pd.DataFrame
        Normal-period data (columns must include *metric_cols*).
    anomal_df : pd.DataFrame
        Anomalous-period data.
    metric_cols : list[str]
        Metric column names (excluding ``time``).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    features : np.ndarray, shape ``(len(metric_cols), 4)``
        Each row = [mean_dev, max_abs_dev, slope, mean_diff].
    """
    n_metrics = len(metric_cols)
    features = np.zeros((n_metrics, 4), dtype=np.float32)

    for i, col in enumerate(metric_cols):
        normal_vals = normal_df[col].to_numpy(dtype=np.float64)
        anomal_vals = anomal_df[col].to_numpy(dtype=np.float64)

        # --- ToD baseline ---
        mu = np.mean(normal_vals)
        sigma = np.std(normal_vals)

        # --- deviation in anomal window ---
        x_dev = (anomal_vals - mu) / (sigma + eps)

        # 1. mean deviation
        mean_dev = float(np.mean(x_dev))

        # 2. max absolute deviation
        max_abs_dev = float(np.max(np.abs(x_dev))) if len(x_dev) > 0 else 0.0

        # 3. slope via simple linear regression
        if len(x_dev) >= 2:
            t = np.arange(len(x_dev), dtype=np.float64)
            t_mean = t.mean()
            x_mean = x_dev.mean()
            denom = np.sum((t - t_mean) ** 2)
            slope = float(np.sum((t - t_mean) * (x_dev - x_mean)) / (denom + eps))
        else:
            slope = 0.0

        # 4. mean of first-order differences
        if len(x_dev) >= 2:
            mean_diff = float(np.mean(np.diff(x_dev)))
        else:
            mean_diff = 0.0

        features[i] = [mean_dev, max_abs_dev, slope, mean_diff]

    return features
