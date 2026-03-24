import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

METRIC_TYPES = ["cpu", "mem", "latency", "load", "error", "diskio", "socket", "other"]

def compute_deviation_features(
    normal_df: pd.DataFrame,
    anomal_df: pd.DataFrame,
    metric_cols: list,
    sli: str = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute per-metric deviation features (19 dimensional).

    Parameters
    ----------
    normal_df : pd.DataFrame
        Normal-period data (columns must include *metric_cols*).
    anomal_df : pd.DataFrame
        Anomalous-period data.
    metric_cols : list[str]
        Metric column names (excluding ``time``).
    sli : str
        SLI column name for calculating sli_correlation.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    features : np.ndarray, shape ``(len(metric_cols), 19)``
    """
    n_metrics = len(metric_cols)
    if n_metrics == 0:
        return np.zeros((0, 11 + len(METRIC_TYPES)), dtype=np.float32)

    normal_vals = normal_df[metric_cols].to_numpy(dtype=np.float64)
    anomal_vals = anomal_df[metric_cols].to_numpy(dtype=np.float64)

    # --- ToD baseline ---
    mu = np.mean(normal_vals, axis=0)
    sigma = np.std(normal_vals, axis=0)

    # --- deviation in anomal window ---
    x_dev = (anomal_vals - mu) / (sigma + eps) # shape: (T_a, N_met)

    # 1. mean deviation
    mean_dev = np.mean(x_dev, axis=0)

    # 2. max absolute deviation
    max_abs_dev = np.max(np.abs(x_dev), axis=0) if len(x_dev) > 0 else np.zeros(n_metrics)

    # 3. slope via vectorized linear regression
    T = len(x_dev)
    if T >= 2:
        t = np.arange(T, dtype=np.float64)
        t_mean = t.mean()
        t_var = np.sum((t - t_mean) ** 2)
        slopes = np.sum((x_dev - mean_dev[None, :]) * (t - t_mean)[:, None], axis=0) / (t_var + eps)
    else:
        slopes = np.zeros(n_metrics)

    # 4. mean of first-order differences
    if T >= 2:
        mean_diff = np.mean(np.diff(x_dev, axis=0), axis=0)
    else:
        mean_diff = np.zeros(n_metrics)

    # 5. onset_time (normalized 0 to 1) & 6. rise_time (normalized)
    # Define onset as first time deviation exceeds 2.0
    is_above = x_dev > 2.0
    has_onset = np.any(is_above, axis=0)
    onset_idx = np.argmax(is_above, axis=0)  # first True index
    onset_time = np.where(has_onset, onset_idx / T, 1.0)
    
    peak_idx = np.argmax(np.abs(x_dev), axis=0)
    rise_time = np.where(has_onset, np.clip((peak_idx - onset_idx) / T, 0.0, 1.0), 1.0)

    # 7. std_dev
    std_dev = np.std(x_dev, axis=0)

    # 8. skewness & 9. kurtosis
    if T >= 3:
        skew_vals = skew(x_dev, axis=0, nan_policy='omit')
        kurt_vals = kurtosis(x_dev, axis=0, nan_policy='omit')
        skew_vals = np.nan_to_num(skew_vals, nan=0.0)
        kurt_vals = np.nan_to_num(kurt_vals, nan=0.0)
    else:
        skew_vals = np.zeros(n_metrics)
        kurt_vals = np.zeros(n_metrics)

    # 10. rank_percentile (based on |mean_dev|)
    abs_mean_dev = np.abs(mean_dev)
    if n_metrics > 1:
        # np.argsort twice gives the ranks
        ranks = np.argsort(np.argsort(abs_mean_dev))
        rank_percentile = ranks / (n_metrics - 1)
    else:
        rank_percentile = np.zeros(n_metrics)

    # 11. is_sli_correlated
    is_sli_correlated = np.zeros(n_metrics)
    if sli is not None and sli in metric_cols:
        sli_idx = metric_cols.index(sli)
        sli_vals = x_dev[:, sli_idx]
        if np.std(sli_vals) > eps:
            for i in range(n_metrics):
                if np.std(x_dev[:, i]) > eps:
                    is_sli_correlated[i] = np.corrcoef(x_dev[:, i], sli_vals)[0, 1]
    is_sli_correlated = np.nan_to_num(is_sli_correlated, nan=0.0)

    # 12. fault_type_hint (one-hot vectors)
    one_hots = np.zeros((n_metrics, len(METRIC_TYPES)))
    for i, col in enumerate(metric_cols):
        parts = col.rsplit('_', 1)
        suffix = parts[-1].lower() if len(parts) > 1 else "unknown"
        
        found_idx = len(METRIC_TYPES) - 1 # default to 'other'
        for j, mt in enumerate(METRIC_TYPES[:-1]):
            if mt in suffix:
                found_idx = j
                break
        one_hots[i, found_idx] = 1.0

    # Assemble final (N_met, 19) matrix
    features = np.column_stack([
        mean_dev, max_abs_dev, slopes, mean_diff,
        onset_time, rise_time, std_dev, skew_vals, kurt_vals,
        rank_percentile, is_sli_correlated, one_hots
    ]).astype(np.float32)

    return features
