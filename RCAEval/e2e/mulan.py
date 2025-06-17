"""
MULAN: Multi-Modal Causal Structure Learning for Effective Root Cause Analysis
Simplified implementation for RCAEval framework supporting RE1, RE2, RE3 datasets.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

from RCAEval.io.time_series import (
    preprocess, drop_constant, convert_mem_mb, drop_time, select_useful_cols
)


def extract_log_representations(logs_data, logts_data, n_entities, target_dim=1):
    """
    Extract log representations from log data
    
    Args:
        logs_data: Raw log DataFrame or None
        logts_data: Log time series DataFrame or None
        n_entities: Number of entities
        target_dim: Target dimension after PCA
    
    Returns:
        Log representations [n_entities, target_dim] or None
    """
    if logts_data is not None:
        # Use preprocessed log time series
        log_features = logts_data.drop(columns=['time'], errors='ignore')
        log_features = drop_constant(log_features)
        
        if log_features.shape[1] == 0:
            return None
        
        # Apply PCA for dimensionality reduction
        if log_features.shape[1] > target_dim:
            try:
                pca = PCA(n_components=target_dim)
                log_repr = pca.fit_transform(log_features.T)  # [n_entities, target_dim]
            except:
                log_repr = log_features.T.values[:, :target_dim]
        else:
            log_repr = log_features.T.values  # [n_entities, n_features]
        
        # Ensure we have n_entities rows
        if log_repr.shape[0] < n_entities:
            padding = np.zeros((n_entities - log_repr.shape[0], log_repr.shape[1]))
            log_repr = np.vstack([log_repr, padding])
        elif log_repr.shape[0] > n_entities:
            log_repr = log_repr[:n_entities]
        
        return log_repr
    
    return None


def extract_trace_representations(traces_data, tracets_lat_data, tracets_err_data, n_entities):
    """
    Extract trace representations from trace data
    
    Args:
        traces_data: Raw trace DataFrame or None
        tracets_lat_data: Trace latency time series DataFrame or None
        tracets_err_data: Trace error time series DataFrame or None
        n_entities: Number of entities
    
    Returns:
        Trace representations [n_entities, trace_dim] or None
    """
    trace_features = []
    
    if tracets_lat_data is not None:
        lat_features = tracets_lat_data.drop(columns=['time'], errors='ignore')
        lat_features = drop_constant(lat_features)
        if lat_features.shape[1] > 0:
            trace_features.append(lat_features)
    
    if tracets_err_data is not None:
        err_features = tracets_err_data.drop(columns=['time'], errors='ignore')
        err_features = drop_constant(err_features)
        if err_features.shape[1] > 0:
            trace_features.append(err_features)
    
    if not trace_features:
        return None
    
    # Combine trace features
    combined_trace = pd.concat(trace_features, axis=1)
    trace_repr = combined_trace.T.values  # [n_entities, trace_dim]
    
    # Ensure we have n_entities rows
    if trace_repr.shape[0] < n_entities:
        padding = np.zeros((n_entities - trace_repr.shape[0], trace_repr.shape[1]))
        trace_repr = np.vstack([trace_repr, padding])
    elif trace_repr.shape[0] > n_entities:
        trace_repr = trace_repr[:n_entities]
    
    return trace_repr


def compute_kpi_correlation(features, kpi, tau=3):
    """
    Compute correlation between features and KPI with time lag
    
    Args:
        features: Feature matrix [n_features, n_timesteps]
        kpi: KPI time series [n_timesteps]
        tau: Maximum time lag
    
    Returns:
        Correlation scores [n_features]
    """
    n_features, n_timesteps = features.shape
    correlations = np.zeros(n_features)
    
    # Clean data
    kpi = np.nan_to_num(kpi, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    for i in range(n_features):
        max_corr = 0
        feature_i = features[i]
        
        # Skip if feature is constant
        if np.std(feature_i) == 0 or np.std(kpi) == 0:
            correlations[i] = 0
            continue
            
        for p in range(min(tau + 1, n_timesteps)):
            try:
                if p == 0:
                    corr, _ = pearsonr(feature_i, kpi)
                else:
                    # Time-lagged correlation
                    if len(feature_i[:-p]) > 1 and len(kpi[p:]) > 1:
                        corr, _ = pearsonr(feature_i[:-p], kpi[p:])
                    else:
                        corr = 0
                
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))
            except:
                continue
                
        correlations[i] = max_corr
    
    return correlations


def build_correlation_graph(metric_data, log_data=None, trace_data=None, threshold=0.3):
    """
    Build correlation-based adjacency matrix
    
    Args:
        metric_data: Metric features [n_entities, n_timesteps]
        log_data: Log features [n_entities, log_dim] or None
        trace_data: Trace features [n_entities, trace_dim] or None
        threshold: Correlation threshold
    
    Returns:
        Adjacency matrix [n_entities, n_entities]
    """
    n_entities = metric_data.shape[0]
    adj_matrix = np.zeros((n_entities, n_entities))
    
    # Clean data
    metric_data = np.nan_to_num(metric_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute pairwise correlations for metrics
    for i in range(n_entities):
        for j in range(n_entities):
            if i == j:
                continue
                
            try:
                # Use time series correlation
                corr, _ = pearsonr(metric_data[i], metric_data[j])
                if not np.isnan(corr) and abs(corr) > threshold:
                    adj_matrix[i, j] = abs(corr)
            except:
                continue
    
    return adj_matrix


def pagerank_ranking(adj_matrix, kpi_index, alpha=0.85, max_iter=100, tol=1e-6):
    """
    PageRank-based ranking for root cause analysis
    
    Args:
        adj_matrix: Adjacency matrix [n_nodes, n_nodes]
        kpi_index: Index of KPI node
        alpha: Damping factor
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Ranking scores [n_nodes]
    """
    n_nodes = adj_matrix.shape[0]
    
    # Clean adjacency matrix
    adj_matrix = np.nan_to_num(adj_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize adjacency matrix
    row_sums = np.sum(adj_matrix, axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    P = adj_matrix / row_sums[:, np.newaxis]
    
    # Initialize PageRank vector
    pr = np.ones(n_nodes) / n_nodes
    
    # PersonalizedPageRank: bias towards KPI
    personalization = np.zeros(n_nodes)
    personalization[kpi_index] = 1.0
    
    # PageRank iterations
    for _ in range(max_iter):
        pr_new = alpha * np.dot(P.T, pr) + (1 - alpha) * personalization
        
        if np.linalg.norm(pr_new - pr) < tol:
            break
        
        pr = pr_new
    
    return pr


def mulan(data, inject_time=None, dataset=None, sli=None, anomalies=None, **kwargs):
    """
    MULAN: Multi-Modal Causal Structure Learning for Effective Root Cause Analysis
    Simplified implementation using correlation analysis and PageRank
    
    Args:
        data: Input data (DataFrame for RE1, dict for RE2/RE3)
        inject_time: Fault injection timestamp
        dataset: Dataset name (re1, re2, re3)
        sli: Service Level Indicator (KPI)
        anomalies: Anomaly indices (optional)
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with ranked root causes
    """
    
    # Check if multi-modal data (RE2/RE3) or single-modal (RE1)
    is_multimodal = isinstance(data, dict)
    
    if is_multimodal:
        # RE2/RE3: Multi-modal data
        metric_data = data.get("metric")
        logs_data = data.get("logs")
        logts_data = data.get("logts")
        traces_data = data.get("traces")
        tracets_lat_data = data.get("tracets_lat")
        tracets_err_data = data.get("tracets_err")
        
        if metric_data is None:
            raise ValueError("Metric data is required for MULAN")
        
        # Preprocess metric data
        if inject_time is not None:
            normal_metric = metric_data[metric_data["time"] < inject_time]
            anomal_metric = metric_data[metric_data["time"] >= inject_time]
        else:
            # Use anomalies parameter if available
            if anomalies is not None:
                normal_metric = metric_data.head(anomalies[0])
                anomal_metric = metric_data.tail(len(metric_data) - anomalies[0])
            else:
                # Split data in half
                split_point = len(metric_data) // 2
                normal_metric = metric_data.head(split_point)
                anomal_metric = metric_data.tail(len(metric_data) - split_point)
        
        # Combine normal and anomal data for full time series
        full_metric = pd.concat([normal_metric, anomal_metric], axis=0, ignore_index=True)
        full_metric = preprocess(data=full_metric, dataset=dataset, 
                                dk_select_useful=kwargs.get("dk_select_useful", False))
        
        # Extract log representations (RE2/RE3)
        log_repr = None
        if logts_data is not None or logs_data is not None:
            log_repr = extract_log_representations(logs_data, logts_data, 
                                                 full_metric.shape[1], target_dim=1)
        
        # Extract trace representations (optional for RE2/RE3)
        trace_repr = None
        use_traces = kwargs.get("use_traces", False)
        if use_traces and (tracets_lat_data is not None or tracets_err_data is not None):
            trace_repr = extract_trace_representations(traces_data, tracets_lat_data, 
                                                     tracets_err_data, full_metric.shape[1])
        
    else:
        # RE1: Single-modal data (metrics only)
        if inject_time is not None:
            normal_data = data[data["time"] < inject_time]
            anomal_data = data[data["time"] >= inject_time]
        else:
            if anomalies is not None:
                normal_data = data.head(anomalies[0])
                anomal_data = data.tail(len(data) - anomalies[0])
            else:
                split_point = len(data) // 2
                normal_data = data.head(split_point)
                anomal_data = data.tail(len(data) - split_point)
        
        full_metric = pd.concat([normal_data, anomal_data], axis=0, ignore_index=True)
        full_metric = preprocess(data=full_metric, dataset=dataset,
                                dk_select_useful=kwargs.get("dk_select_useful", False))
        
        log_repr = None
        trace_repr = None
    
    # Prepare metric features
    metric_features = full_metric.values.T  # [n_entities, n_timesteps]
    n_entities, n_timesteps = metric_features.shape
    
    # Clean metric features
    metric_features = np.nan_to_num(metric_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Identify KPI
    kpi_index = 0  # Default to first metric
    if sli is not None and sli in full_metric.columns:
        kpi_index = list(full_metric.columns).index(sli)
    elif 'frontend_latency' in full_metric.columns:
        kpi_index = list(full_metric.columns).index('frontend_latency')
    elif any('latency' in col for col in full_metric.columns):
        latency_cols = [col for col in full_metric.columns if 'latency' in col]
        kpi_index = list(full_metric.columns).index(latency_cols[0])
    
    kpi_series = metric_features[kpi_index]
    
    # Build correlation graph
    threshold = kwargs.get("correlation_threshold", 0.3)
    adj_matrix = build_correlation_graph(metric_features, log_repr, trace_repr, threshold)
    
    # Compute KPI correlations
    kpi_correlations = compute_kpi_correlation(metric_features, kpi_series)
    
    # Multi-modal fusion (if available)
    if log_repr is not None:
        log_adj = build_correlation_graph(log_repr.T)
        log_correlations = compute_kpi_correlation(log_repr.T, kpi_series)
        
        # Simple weighted fusion
        log_weight = np.mean(log_correlations) / (np.mean(kpi_correlations) + 1e-8)
        log_weight = min(log_weight, 1.0)  # Cap at 1.0
        
        if log_adj.shape == adj_matrix.shape:
            adj_matrix = 0.7 * adj_matrix + 0.3 * log_weight * log_adj
    
    if trace_repr is not None:
        trace_adj = build_correlation_graph(trace_repr.T)
        trace_correlations = compute_kpi_correlation(trace_repr.T, kpi_series)
        
        # Simple weighted fusion
        trace_weight = np.mean(trace_correlations) / (np.mean(kpi_correlations) + 1e-8)
        trace_weight = min(trace_weight, 1.0)  # Cap at 1.0
        
        if trace_adj.shape == adj_matrix.shape:
            adj_matrix = 0.8 * adj_matrix + 0.2 * trace_weight * trace_adj
    
    # PageRank-based ranking
    alpha = kwargs.get("alpha", 0.85)
    ranking_scores = pagerank_ranking(adj_matrix, kpi_index, alpha=alpha)
    
    # Create ranked list
    entity_names = list(full_metric.columns)
    ranked_entities = [(entity_names[i], ranking_scores[i]) for i in range(len(entity_names))]
    ranked_entities.sort(key=lambda x: x[1], reverse=True)
    
    # Extract ranked root causes
    ranks = [entity for entity, score in ranked_entities]
    
    return {
        "ranks": ranks,
        "adj_matrix": adj_matrix,
        "ranking_scores": ranking_scores,
        "node_names": entity_names
    }


if __name__ == "__main__":
    # Test with sample data
    print("MULAN simplified implementation for RCAEval") 