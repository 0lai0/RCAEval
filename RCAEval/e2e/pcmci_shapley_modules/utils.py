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
    使用前向填充和後向填充的組合來處理缺失值
    
    Args:
        series: 包含可能缺失值的時間序列
        
    Returns:
        插值後的時間序列
    """
    if series.isna().sum() == 0:
        return series
    
    # 先進行前向填充
    series_ffill = series.fillna(method='ffill')
    
    # 再進行後向填充
    series_bfill = series_ffill.fillna(method='bfill')
    
    # 如果開頭或結尾仍有 NaN，使用線性插值
    if series_bfill.isna().sum() > 0:
        series_bfill = series_bfill.interpolate(method='linear')
    
    # 最後的兜底：如果還有 NaN，使用均值填充
    if series_bfill.isna().sum() > 0:
        mean_val = series_bfill.dropna().mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        series_bfill = series_bfill.fillna(mean_val)
    
    return series_bfill


def smart_fillna_matrix(X: np.ndarray, method: str = 'forward_backward') -> np.ndarray:
    """
    對矩陣進行智能缺失值填充
    
    Args:
        X: 輸入矩陣 (samples, features)
        method: 填充方法 ('forward_backward', 'linear', 'mean')
        
    Returns:
        填充後的矩陣
    """
    if not np.isnan(X).any():
        return X
    
    X_df = pd.DataFrame(X)
    
    if method == 'forward_backward':
        # 對每一列進行前向-後向填充
        for col in X_df.columns:
            X_df[col] = forward_backward_interpolate(X_df[col])
    elif method == 'linear':
        # 線性插值
        X_df = X_df.interpolate(method='linear', axis=0)
        # 如果還有 NaN，使用前向-後向填充
        X_df = X_df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'mean':
        # 使用列均值填充
        X_df = X_df.fillna(X_df.mean())
    
    # 最後檢查：如果還有 NaN，填充為 0
    X_df = X_df.fillna(0.0)
    
    return X_df.values


def robust_standardize_with_interpolation(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    """
    使用插值處理缺失值的魯棒標準化
    
    Args:
        series: 輸入時間序列
        eps: 防止除零的小常數
        
    Returns:
        標準化後的時間序列
    """
    # 先進行插值處理
    series_interpolated = forward_backward_interpolate(series)
    
    # 檢查插值後是否還有足夠的數據
    if len(series_interpolated.dropna()) < 2:
        return pd.Series([0.0] * len(series), index=series.index)
    
    # 使用魯棒統計量進行標準化
    med = series_interpolated.median()
    mad = (series_interpolated - med).abs().median()
    
    # 關鍵修復：確保 MAD 不會太小
    if mad < eps:
        # 如果 MAD 太小，使用標準差作為替代
        std_val = series_interpolated.std()
        if std_val < eps:
            # 如果標準差也太小，添加微小的隨機噪聲
            noise = np.random.normal(0, eps, len(series_interpolated))
            series_interpolated = series_interpolated + noise
            mad = (series_interpolated - series_interpolated.median()).abs().median()
        else:
            mad = std_val
    
    denom = mad if abs(mad) > eps else eps
    
    return (series_interpolated - med) / denom
