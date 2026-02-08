"""
CausalSHAP: Shapley Value-based Root Cause Analysis with Causal Propagation

This module implements the CausalSHAP method for microservice root cause analysis.
It combines:
1. Multi-layer causal graph construction (PCMCI/Granger + Trace + Correlation)
2. Non-additive value function based on anomaly propagation
3. Monte Carlo Shapley value estimation

Key advantages over linear scoring:
- Captures redundancy effects (multiple upstream sources)
- Distinguishes root causes from propagated anomalies
- Provides interpretable contribution scores

Reference Implementation for RCAEval Framework
Author: RCAEval Team
Date: 2026-02-07
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Optional
from datetime import datetime
from sklearn.preprocessing import RobustScaler

from RCAEval.io.time_series import preprocess
from RCAEval.e2e.pc_shapley_modules.propagation_shapley import (
    sampling_shapley_with_propagation,
    build_causal_graph_for_shapley,
    extract_service_name,
)


def _log_with_timestamp(message: str, verbose: bool = True, always: bool = False):
    """
    Helper function to print log message with timestamp
    
    Parameters
    ----------
    message : str
        Log message
    verbose : bool
        Whether verbose logging is enabled
    always : bool
        If True, always output regardless of verbose flag (for critical stages)
    """
    if verbose or always:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")


def _log_phase_start(phase_name: str, verbose: bool = True, always: bool = True):
    """Log the start of a phase (always output by default)"""
    _log_with_timestamp(f"[CausalSHAP] ===== Phase: {phase_name} START =====", verbose, always)


def _log_phase_end(phase_name: str, elapsed_time: float, verbose: bool = True, always: bool = True):
    """Log the end of a phase with elapsed time (always output by default)"""
    _log_with_timestamp(
        f"[CausalSHAP] ===== Phase: {phase_name} END (elapsed: {elapsed_time:.3f}s) =====",
        verbose,
        always
    )


def causalshap(
    data,
    inject_time=None,
    dataset=None,
    n_permutations=1000,
    decay=0.7,
    propagation_mode="max",
    target_avg_degree=3.0,
    causal_method="pcmci",
    **kwargs,
):
    """
    CausalSHAP: Root Cause Analysis using Shapley values with causal propagation.
    
    Pipeline:
    1. Anomaly Detection: Compute anomaly scores using RobustScaler
    2. Causal Graph Construction: Multi-layer fusion (PCMCI/Granger + Trace + Corr)
    3. Shapley Computation: Monte Carlo sampling with propagation value function
    
    Parameters
    ----------
    data : pd.DataFrame or Dict[str, pd.DataFrame]
        Time series data. If dict, must contain "metric" key.
    inject_time : int, optional
        Anomaly injection timestamp. If None, splits data in half.
    dataset : str, optional
        Dataset name for format-specific parsing (e.g., "sock-shop", "train-ticket")
    n_permutations : int, default=1000
        Number of permutations for Shapley sampling.
        Higher = more accurate but slower. 1000 is a good balance for N~14 services.
    decay : float, default=0.7
        Decay factor for anomaly propagation along edges.
        0.7 means anomaly reduces to 70% per hop.
    propagation_mode : str, default="max"
        Propagation mode for value function:
        - "max": node anomaly = max(self, upstream) [recommended]
        - "attenuated_sum": upstream contributions have diminishing returns
    target_avg_degree : float, default=3.0
        Target average degree for causal graph.
        Controls graph density: 2.0 = sparse, 5.0 = dense.
    causal_method : str, default="pcmci"
        Causal discovery method:
        - "pcmci": PCMCI (more rigorous, slower)
        - "granger": Granger causality (faster, less rigorous)
        - "both": Use both (may double-count linear causality)
    **kwargs : dict
        Additional arguments:
        - dk_select_useful: bool, whether to select useful columns
        - verbose: bool, whether to print debug info
    
    Returns
    -------
    dict
        {
            "ranks": List[str],  # Ranked list of service names
            "shapley_values": Dict[str, float],  # Shapley value per service
        }
    
    Examples
    --------
    >>> result = causalshap(
    ...     data=metric_df,
    ...     inject_time=1234567890,
    ...     dataset="sock-shop",
    ...     n_permutations=1000,
    ...     causal_method="pcmci"
    ... )
    >>> print(result["ranks"][:5])  # Top 5 root causes
    ['checkoutservice', 'cartservice', 'frontend', ...]
    
    Notes
    -----
    - Time complexity: O(n_permutations × N × E) where N = services, E = edges
    - For N=14, n_permutations=1000: ~100-200ms
    - Requires at least 30 timestamps for reliable causal discovery
    - Falls back to anomaly score ranking if graph construction fails
    """
    import time
    
    # Track overall execution time
    start_time = time.time()
    verbose = kwargs.get("verbose", False)
    
    # Pipeline start/end always logged
    _log_with_timestamp("[CausalSHAP] ===== CausalSHAP Pipeline START =====", verbose, always=True)
    
    # ========== Input Processing ==========
    _log_with_timestamp("[CausalSHAP] Input processing...", verbose)
    
    if isinstance(data, dict):
        metric = data.get("metric", pd.DataFrame())
    else:
        metric = data

    if metric.empty:
        _log_with_timestamp("[CausalSHAP] Warning: Empty input data", verbose)
        return {"ranks": [], "shapley_values": {}}
    
    _log_with_timestamp(
        f"[CausalSHAP] Input data: {len(metric)} rows, {len(metric.columns)} columns",
        verbose
    )

    # ========== Phase 0: Unified Preprocessing ==========
    phase_start = time.time()
    _log_phase_start("0: Unified Preprocessing", verbose)
    
    # Preprocess first, then split to avoid column inconsistency
    all_data = preprocess(
        data=metric,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False),
    )
    
    _log_with_timestamp(
        f"[CausalSHAP] After preprocessing: {len(all_data)} rows, {len(all_data.columns)} columns",
        verbose
    )

    # Check if time column exists after preprocessing
    has_time_col = "time" in all_data.columns
    
    if inject_time is not None and has_time_col:
        normal_df = all_data[all_data["time"] < inject_time].copy()
        anomal_df = all_data[all_data["time"] >= inject_time].copy()
        _log_with_timestamp(
            f"[CausalSHAP] Split by inject_time={inject_time}: normal={len(normal_df)}, anomal={len(anomal_df)}",
            verbose
        )
    elif inject_time is not None and not has_time_col:
        # If preprocess removed time, use original metric's time for splitting
        if "time" in metric.columns:
            time_col = metric["time"].values
            split_mask = time_col < inject_time
            split_idx = int(split_mask.sum())
            normal_df = all_data.iloc[:split_idx].copy()
            anomal_df = all_data.iloc[split_idx:].copy()
            _log_with_timestamp(
                f"[CausalSHAP] Split by index (inject_time={inject_time}): normal={len(normal_df)}, anomal={len(anomal_df)}",
                verbose
            )
        else:
            # No time info, split in half
            mid = len(all_data) // 2
            normal_df = all_data.iloc[:mid].copy()
            anomal_df = all_data.iloc[mid:].copy()
            _log_with_timestamp(
                f"[CausalSHAP] Split in half (no time column): normal={len(normal_df)}, anomal={len(anomal_df)}",
                verbose
            )
    else:
        # No inject_time, split in half
        mid = len(all_data) // 2
        normal_df = all_data.iloc[:mid].copy()
        anomal_df = all_data.iloc[mid:].copy()
        _log_with_timestamp(
            f"[CausalSHAP] Split in half (no inject_time): normal={len(normal_df)}, anomal={len(anomal_df)}",
            verbose
        )

    if normal_df.empty or anomal_df.empty:
        _log_with_timestamp("[CausalSHAP] Warning: Empty normal or anomalous data", verbose)
        return {"ranks": [], "shapley_values": {}}
    
    phase_elapsed = time.time() - phase_start
    _log_phase_end("0: Unified Preprocessing", phase_elapsed, verbose)

    # ========== Phase 1: Anomaly Detection ==========
    phase_start = time.time()
    _log_phase_start("1: Anomaly Detection", verbose)
    
    anomaly_scores_metric = {}
    cols = [c for c in normal_df.columns if c != "time"]
    _log_with_timestamp(f"[CausalSHAP] Computing anomaly scores for {len(cols)} metrics...", verbose)

    for col in cols:
        try:
            n_vals = normal_df[[col]].dropna()
            a_vals = anomal_df[[col]].dropna()
            if len(n_vals) < 3 or len(a_vals) < 3:
                anomaly_scores_metric[col] = 0.0
                continue
            
            # Use RobustScaler (robust to outliers)
            scaler = RobustScaler().fit(n_vals)
            zscores = scaler.transform(a_vals)[:, 0]
            anomaly_scores_metric[col] = float(np.max(np.abs(zscores)))
        except Exception:
            anomaly_scores_metric[col] = 0.0

    # Aggregate to service level (take max across metrics)
    anomaly_scores = {}
    for metric_name, score in anomaly_scores_metric.items():
        svc = extract_service_name(metric_name, dataset)
        anomaly_scores[svc] = max(anomaly_scores.get(svc, 0.0), score)

    if not anomaly_scores:
        _log_with_timestamp("[CausalSHAP] Warning: No anomaly scores computed", verbose)
        return {"ranks": [], "shapley_values": {}}

    # Focus node = service with highest anomaly
    focus_node = max(anomaly_scores.items(), key=lambda x: x[1])[0]
    
    _log_with_timestamp(
        f"[CausalSHAP] Anomaly scores computed for {len(anomaly_scores)} services",
        verbose
    )
    _log_with_timestamp(
        f"[CausalSHAP] Focus node: {focus_node} (anomaly={anomaly_scores[focus_node]:.2f})",
        verbose
    )
    
    if verbose:
        top5 = sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        _log_with_timestamp(f"[CausalSHAP] Top 5 anomalies: {top5}", verbose)
    
    phase_elapsed = time.time() - phase_start
    _log_phase_end("1: Anomaly Detection", phase_elapsed, verbose)

    # ========== Phase 2: Causal Graph Construction ==========
    phase_start = time.time()
    _log_phase_start("2: Causal Graph Construction", verbose)
    
    # Prefer anomalous period data (captures anomaly propagation)
    # Fall back to all data if anomalous period is too short
    if len(anomal_df) >= 30:
        graph_data = anomal_df
        _log_with_timestamp(
            f"[CausalSHAP] Using anomalous period for graph construction (n={len(anomal_df)})",
            verbose
        )
    else:
        graph_data = all_data
        _log_with_timestamp(
            f"[CausalSHAP] Anomalous period too short, using all data (n={len(all_data)})",
            verbose
        )
    
    _log_with_timestamp(
        f"[CausalSHAP] Causal discovery method: {causal_method}, target_avg_degree: {target_avg_degree}",
        verbose
    )

    # Optional: Trace graph (if available in data dict)
    trace_graph = None
    if isinstance(data, dict) and "traces" in data:
        # TODO: Implement trace graph construction if needed
        _log_with_timestamp("[CausalSHAP] Trace graph available but not implemented yet", verbose)
        pass

    try:
        _log_with_timestamp("[CausalSHAP] Building causal graph...", verbose)
        causal_graph = build_causal_graph_for_shapley(
            data=graph_data,
            anomaly_scores=anomaly_scores,
            focus_node=focus_node,
            dataset=dataset,
            trace_graph=trace_graph,
            target_avg_degree=target_avg_degree,
            causal_method=causal_method,
        )
        
        n_nodes = causal_graph.number_of_nodes()
        n_edges = causal_graph.number_of_edges()
        avg_deg = n_edges / n_nodes if n_nodes > 0 else 0
        _log_with_timestamp(
            f"[CausalSHAP] Causal graph built: {n_nodes} nodes, {n_edges} edges, avg_degree={avg_deg:.2f}",
            verbose
        )
        
        phase_elapsed = time.time() - phase_start
        _log_phase_end("2: Causal Graph Construction", phase_elapsed, verbose)
    except Exception as e:
        _log_with_timestamp(f"[CausalSHAP] Warning: Causal graph construction failed: {e}", verbose, always=True)
        _log_with_timestamp(
            "[CausalSHAP] Fallback mode: Using anomaly score ranking (NOT using CausalSHAP method)",
            verbose,
            always=True
        )
        ranked = sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results for evaluation (same as main return)
        ranks = [s for s, _ in ranked]
        shapley_values = dict(ranked)
        
        if dataset in ["train-ticket", "mm-tt", "fse-tt", "re1-tt", "re2-tt", "re3-tt"]:
            ranks = [f"ts-{s}" if not s.startswith("ts-") else s for s in ranks]
            shapley_values = {
                f"ts-{s}" if not s.startswith("ts-") else s: v 
                for s, v in shapley_values.items()
            }
        
        total_elapsed = time.time() - start_time
        _log_with_timestamp(
            f"[CausalSHAP] ===== CausalSHAP Pipeline END (FALLBACK MODE, total elapsed: {total_elapsed:.3f}s) =====",
            verbose,
            always=True
        )
        return {
            "ranks": ranks,
            "shapley_values": shapley_values,
        }

    # ========== Phase 3: Shapley Value Computation ==========
    phase_start = time.time()
    _log_phase_start("3: Shapley Value Computation", verbose)
    
    _log_with_timestamp(
        f"[CausalSHAP] Computing Shapley values: n_permutations={n_permutations}, "
        f"decay={decay}, propagation_mode={propagation_mode}",
        verbose
    )
    
    try:
        ranked = sampling_shapley_with_propagation(
            causal_graph=causal_graph,
            anomaly_scores=anomaly_scores,
            focus_node=focus_node,
            n_permutations=n_permutations,
            decay=decay,
            propagation_mode=propagation_mode,
        )
        
        _log_with_timestamp(
            f"[CausalSHAP] Shapley values computed for {len(ranked)} services",
            verbose
        )
        
        # Mark that we successfully used CausalSHAP method
        _log_with_timestamp(
            "[CausalSHAP] Using CausalSHAP method (Shapley values with causal propagation)",
            verbose,
            always=True
        )
        
        if verbose:
            _log_with_timestamp("[CausalSHAP] Top 5 Shapley values:", verbose)
            for svc, val in ranked[:5]:
                _log_with_timestamp(f"[CausalSHAP]   {svc}: {val:.4f}", verbose)
        
        phase_elapsed = time.time() - phase_start
        _log_phase_end("3: Shapley Value Computation", phase_elapsed, verbose)
    except Exception as e:
        _log_with_timestamp(f"[CausalSHAP] Warning: Shapley computation failed: {e}", verbose, always=True)
        _log_with_timestamp(
            "[CausalSHAP] Fallback mode: Using anomaly score ranking (NOT using CausalSHAP method)",
            verbose,
            always=True
        )
        ranked = sorted(
            anomaly_scores.items(), key=lambda x: x[1], reverse=True
        )
        phase_elapsed = time.time() - phase_start
        _log_phase_end("3: Shapley Value Computation (fallback)", phase_elapsed, verbose)
        
        # Format results for evaluation (same as main return)
        ranks = [s for s, _ in ranked]
        shapley_values = dict(ranked)
        
        if dataset in ["train-ticket", "mm-tt", "fse-tt", "re1-tt", "re2-tt", "re3-tt"]:
            ranks = [f"ts-{s}" if not s.startswith("ts-") else s for s in ranks]
            shapley_values = {
                f"ts-{s}" if not s.startswith("ts-") else s: v 
                for s, v in shapley_values.items()
            }
        
        total_elapsed = time.time() - start_time
        _log_with_timestamp(
            f"[CausalSHAP] ===== CausalSHAP Pipeline END (FALLBACK MODE, total elapsed: {total_elapsed:.3f}s) =====",
            verbose,
            always=True
        )
        return {
            "ranks": ranks,
            "shapley_values": shapley_values,
        }

    # ========== Format Results for Evaluation ==========
    # For train-ticket dataset, ground truth uses "ts-" prefix in service names
    # We need to add it back to match the evaluation format
    ranks = [s for s, _ in ranked]
    shapley_values = {s: v for s, v in ranked}
    
    if dataset in ["train-ticket", "mm-tt", "fse-tt", "re1-tt", "re2-tt", "re3-tt"]:
        # Add "ts-" prefix to match ground truth format
        ranks = [f"ts-{s}" if not s.startswith("ts-") else s for s in ranks]
        shapley_values = {
            f"ts-{s}" if not s.startswith("ts-") else s: v 
            for s, v in shapley_values.items()
        }
        if verbose:
            _log_with_timestamp(
                f"[CausalSHAP] Added 'ts-' prefix for train-ticket format compatibility",
                verbose
            )

    total_elapsed = time.time() - start_time
    _log_with_timestamp(
        f"[CausalSHAP] ===== CausalSHAP Pipeline END (total elapsed: {total_elapsed:.3f}s) =====",
        verbose,
        always=True
    )

    return {
        "ranks": ranks,
        "shapley_values": shapley_values,
    }
