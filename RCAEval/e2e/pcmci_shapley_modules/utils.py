from __future__ import annotations
from typing import Dict, Iterable
import numpy as np
import pandas as pd


def min_max_normalize(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin == 0:
        return {k: 0.0 for k in values}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in values.items()}


def robust_standardize(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    denom = mad if abs(mad) > eps else eps
    return (series - med) / denom


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    if denom == 0 or not np.isfinite(denom):
        return default
    return float(numer / denom)


def forward_backward_interpolate(series: pd.Series) -> pd.Series:
    """
    Handle missing values using a combination of forward-fill and back-fill.
    
    Args:
        series: Input time series that may contain missing values.
        
    Returns:
        Interpolated time series.
    """
    if series.isna().sum() == 0:
        return series
    
    # First apply forward fill
    series_ffill = series.fillna(method='ffill')
    
    # Then apply backward fill
    series_bfill = series_ffill.fillna(method='bfill')
    
    # If there are still NaNs at the beginning or end, use linear interpolation
    if series_bfill.isna().sum() > 0:
        series_bfill = series_bfill.interpolate(method='linear')
    
    # Final fallback: if there are still NaNs, fill with the mean
    if series_bfill.isna().sum() > 0:
        mean_val = series_bfill.dropna().mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        series_bfill = series_bfill.fillna(mean_val)
    
    return series_bfill


def smart_fillna_matrix(X: np.ndarray, method: str = 'forward_backward') -> np.ndarray:
    """
    Perform intelligent missing-value imputation on a matrix.
    
    Args:
        X: Input matrix (samples, features).
        method: Imputation method ('forward_backward', 'linear', 'mean').
        
    Returns:
        Imputed matrix.
    """
    if not np.isnan(X).any():
        return X
    
    X_df = pd.DataFrame(X)
    
    if method == 'forward_backward':
        # Apply forward-backward filling to each column
        for col in X_df.columns:
            X_df[col] = forward_backward_interpolate(X_df[col])
    elif method == 'linear':
        # Linear interpolation
        X_df = X_df.interpolate(method='linear', axis=0)
        # If there are still NaNs, apply forward-backward fill
        X_df = X_df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'mean':
        # Use column mean to fill remaining NaNs
        X_df = X_df.fillna(X_df.mean())
    
    # Final check: if there are still NaNs, fill with 0
    X_df = X_df.fillna(0.0)
    
    return X_df.values


def robust_standardize_with_interpolation(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    """
    Robust standardization with interpolation for missing values.
    
    Args:
        series: Input time series.
        eps: Small constant to avoid division-by-zero.
        
    Returns:
        Standardized time series.
    """
    # First interpolate missing values
    series_interpolated = forward_backward_interpolate(series)
    
    # Check that there is enough data after interpolation
    if len(series_interpolated.dropna()) < 2:
        return pd.Series([0.0] * len(series), index=series.index)
    
    # Use robust statistics for standardization
    med = series_interpolated.median()
    mad = (series_interpolated - med).abs().median()
    
    # Critical fix: ensure MAD is not too small
    if mad < eps:
        # If MAD is too small, fall back to standard deviation
        std_val = series_interpolated.std()
        if std_val < eps:
            # If standard deviation is also too small, add small random noise
            noise = np.random.normal(0, eps, len(series_interpolated))
            series_interpolated = series_interpolated + noise
            mad = (series_interpolated - series_interpolated.median()).abs().median()
        else:
            mad = std_val
    
    denom = mad if abs(mad) > eps else eps
    
    return (series_interpolated - med) / denom
