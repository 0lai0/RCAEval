from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging

from .config import PCMCIShapleyConfig
from .utils import robust_standardize, robust_standardize_with_interpolation

logger = logging.getLogger(__name__)


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
        # 应用阈值过滤
        anomaly_scores = z.where(z > threshold, other=0.0)
        
        # 关键修复：对异常分数进行范围限制，避免极值
        # 使用tanh函数将异常分数限制在合理范围内
        max_score = 10.0  # 设置最大异常分数为10
        anomaly_scores = np.tanh(anomaly_scores / max_score) * max_score
        
        # 进一步限制：如果分数仍然过大，进行对数缩放
        if anomaly_scores.max() > 5.0:
            logger.warning(f"Column {c} has extreme anomaly scores (max={anomaly_scores.max():.2f}), applying log scaling")
            # 使用log1p进行缩放，避免log(0)
            anomaly_scores = np.log1p(anomaly_scores)
        
        out[c] = anomaly_scores
    
    res = pd.DataFrame(out)
    if "time" in df.columns:
        res.insert(0, "time", df["time"].values)
    
    # 记录异常分数的统计信息
    logger.info(f"Anomaly scores range: min={res.min().min():.4f}, max={res.max().max():.4f}")
    
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


def preprocess_data(data: pd.DataFrame, config: PCMCIShapleyConfig) -> Dict[str, Any]:
    """
    End-to-end preprocessing pipeline producing normalized data and anomaly scores.
    Now includes SPOT extreme value theory analysis.
    """
    # 使用插值進行魯棒標準化
    norm = robust_normalize(data, use_interpolation=True)
    
    # Z-score 異常檢測
    if config.anomaly_method == "zscore":
        anomalies = detect_anomaly_zscore(norm, threshold=config.anomaly_threshold)
    else:
        # 默認使用 zscore
        anomalies = detect_anomaly_zscore(norm, threshold=config.anomaly_threshold)
    
    # SPOT 極值理論分析（新增）
    if getattr(config, 'enable_spot', False):
        try:
            from .spot_detector import spot_anomaly_detection, aggregate_spot_scores
            
            logger.info("Running SPOT extreme value theory analysis")
            spot_scores_df = spot_anomaly_detection(norm, config)
            
            # 構建 metric_map 用於 SPOT 聚合
            metric_map = {}
            for c in norm.columns:
                if c == "time":
                    continue
                if "_" in c:
                    service = c.split("_")[0]
                else:
                    service = c
                metric_map.setdefault(service, []).append(c)
            
            node_spot = aggregate_spot_scores(spot_scores_df, metric_map)
            logger.info(f"SPOT analysis completed: {len(node_spot)} services analyzed")
            
        except Exception as e:
            logger.warning(f"SPOT analysis failed: {e}, using zero scores")
            spot_scores_df = None
            node_spot = pd.Series({})
    else:
        spot_scores_df = None
        node_spot = pd.Series({})
    
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
    
    logger.info(f"Preprocessing completed successfully")
    logger.info(f"Output shapes: normalized_df={norm.shape}, anomaly_scores={anomalies.shape}")
    logger.info(f"Node counts: anomaly={len(node_anomaly)}, spot={len(node_spot)}")
    logger.info(f"Metric mapping: {len(metric_map)} services")
    
    return {
        "normalized_df": norm,
        "anomaly_scores": anomalies,
        "node_anomaly": node_anomaly,
        "node_anomaly_ts": node_anomaly_ts,
        "metric_mapping": metric_map,
        "spot_scores": spot_scores_df,  # 新增
        "node_spot": node_spot,  # 新增
    }
