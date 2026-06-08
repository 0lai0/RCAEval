from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed


def compute_lagged_correlation(focus: pd.Series, other: pd.Series, tau_max: int) -> np.ndarray:
    arr = np.zeros(tau_max, dtype=float)
    for tau in range(1, tau_max + 1):
        if len(focus) <= tau or len(other) <= tau:
            arr[tau - 1] = 0.0
            continue
        a = focus.values[tau:]
        b = other.values[:-tau]
        if np.std(a) == 0 or np.std(b) == 0:
            arr[tau - 1] = 0.0
        else:
            arr[tau - 1] = float(np.corrcoef(a, b)[0, 1])
    return arr


def statistical_neighborhood(focus_node: str, node_anomaly_ts: Dict[str, pd.Series], tau_max: int, top_m1: int, n_jobs: int = -1) -> List[str]:
    """
    Parallelized computation of the statistical neighborhood.
    
    Args:
        n_jobs: Number of parallel jobs (-1 means all cores).
    """
    focus = node_anomaly_ts.get(focus_node)
    if focus is None:
        return []
    
    # 準備平行化任務
    def compute_score(node_series_pair):
        node, series = node_series_pair
        if node == focus_node:
            return None
        corr = compute_lagged_correlation(focus, series, tau_max)
        score = float(np.nanmax(np.abs(corr))) if corr.size else 0.0
        return (node, score)
    
    # 決定是否使用平行化
    nodes_to_check = [(node, series) for node, series in node_anomaly_ts.items()]
    
    if len(nodes_to_check) <= 10 or n_jobs == 1:
        # 小規模或禁用平行化時使用原始方法
        scores: List[Tuple[str, float]] = []
        for node, series in nodes_to_check:
            if node == focus_node:
                continue
            corr = compute_lagged_correlation(focus, series, tau_max)
            score = float(np.nanmax(np.abs(corr))) if corr.size else 0.0
            scores.append((node, score))
    else:
        # 平行化計算
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_score)(pair) for pair in nodes_to_check
        )
        scores = [r for r in results if r is not None]
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scores[:top_m1]]


def statistical_neighborhood_original(focus_node: str, node_anomaly_ts: Dict[str, pd.Series], tau_max: int, top_m1: int) -> List[str]:
    """原始版本 (保留作為備份)"""
    focus = node_anomaly_ts.get(focus_node)
    if focus is None:
        return []
    scores: List[Tuple[str, float]] = []
    for node, series in node_anomaly_ts.items():
        if node == focus_node:
            continue
        corr = compute_lagged_correlation(focus, series, tau_max)
        score = float(np.nanmax(np.abs(corr))) if corr.size else 0.0
        scores.append((node, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scores[:top_m1]]


def build_isolation_features(candidates: List[str], focus_node: str, node_anomaly_ts: Dict[str, pd.Series], tau_max: int) -> pd.DataFrame:
    rows = {}
    focus = node_anomaly_ts.get(focus_node)
    for node in candidates:
        series = node_anomaly_ts.get(node)
        if series is None or focus is None:
            continue
        rows[node] = compute_lagged_correlation(focus, series, tau_max)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index")


def isolation_forest_selection(features: pd.DataFrame, top_m2: int, random_state: int = 0) -> Tuple[List[str], Dict[str, float]]:
    
    if len(features) <= top_m2:
        return list(features.index), {k: 1.0 for k in features.index}
    
    if features.empty:
        return [], {}
    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state)
    clf.fit(features.values)
    # score_samples: higher is less anomalous; convert to [0,1]
    raw = clf.score_samples(features.values)
    raw_min, raw_max = raw.min(), raw.max()
    if raw_max - raw_min == 0:
        norm = np.zeros_like(raw)
    else:
        norm = (raw - raw_min) / (raw_max - raw_min)
    scores = {node: float(v) for node, v in zip(features.index.tolist(), norm.tolist())}
    selected = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_m2]
    return [n for n, _ in selected], scores


def trace_augmentation(selected_nodes: List[str], trace_graph: nx.DiGraph | None, focus_node: str, max_size: int) -> List[str]:
    U = list(dict.fromkeys([focus_node] + selected_nodes))
    if trace_graph is None:
        return U[:max_size]
    onehop = []
    for n in selected_nodes:
        if n in trace_graph:
            onehop.extend(list(trace_graph.predecessors(n)))
            onehop.extend(list(trace_graph.successors(n)))
    U = list(dict.fromkeys(U + onehop))
    return U[:max_size]
