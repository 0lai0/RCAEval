from __future__ import annotations
from typing import Dict, List, Set, Tuple, Callable
import numpy as np
from joblib import Parallel, delayed

from .config import PCMCIShapleyConfig
from .propagation import propagate_k_steps


def compute_system_anomaly(coalition: Set[str], local_nodes: List[str], edge_weights: Dict[Tuple[str, str], float], K: int, alpha_prop: float, node_init: Dict[str, float] | None = None) -> float:
    # Only keep initial anomalies for nodes in coalition; others set to 0
    init = {s: (node_init.get(s, 0.0) if node_init is not None else 0.0) for s in local_nodes}
    for s in list(init.keys()):
        if s not in coalition:
            init[s] = 0.0
    res = propagate_k_steps(init, edge_weights, K=K, alpha_prop=alpha_prop)
    h_final = res["h_final"]
    return float(sum(h_final.values()))


def compute_shapley_sampling(local_nodes: List[str], value_func: Callable[[Set[str]], float], R: int, rng: np.random.Generator | None = None, n_jobs: int = -1) -> Dict[str, float]:
    """
    平行化版本的 Shapley 採樣
    
    Args:
        n_jobs: 平行化進程數 (-1 表示使用所有核心)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    nodes = list(local_nodes)
    
    # 決定是否使用平行化
    if R <= 50 or n_jobs == 1 or len(nodes) <= 5:
        # 小規模時使用原始方法
        return _compute_shapley_sampling_original(local_nodes, value_func, R, rng)
    
    # 平行化版本：將 R 個採樣分配到多個進程
    def single_sample_shapley(seed):
        """單次採樣實作"""
        local_rng = np.random.default_rng(seed)
        phi_local = {s: 0.0 for s in nodes}
        perm = nodes.copy()
        local_rng.shuffle(perm)
        prefix = set()
        for s in perm:
            marginal = value_func(prefix | {s}) - value_func(prefix)
            phi_local[s] += marginal
            prefix.add(s)
        return phi_local
    
    # 平行執行 R 個樣本
    seeds = [rng.integers(0, 2**32) for _ in range(R)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_sample_shapley)(seed) for seed in seeds
    )
    
    # 聚合結果
    phi = {s: 0.0 for s in nodes}
    for phi_local in results:
        for s, v in phi_local.items():
            phi[s] += v
    return {s: v / R for s, v in phi.items()}


def _compute_shapley_sampling_original(local_nodes: List[str], value_func: Callable[[Set[str]], float], R: int, rng: np.random.Generator | None = None) -> Dict[str, float]:
    """原始版本 (保留作為備份)"""
    if rng is None:
        rng = np.random.default_rng(0)
    nodes = list(local_nodes)
    phi = {s: 0.0 for s in nodes}
    for _ in range(R):
        perm = nodes.copy()
        rng.shuffle(perm)
        prefix: Set[str] = set()
        for s in perm:
            marginal = value_func(prefix | {s}) - value_func(prefix)
            phi[s] += float(marginal)
            prefix.add(s)
    return {s: v / float(R) for s, v in phi.items()}


def normalize_shapley(shapley_values: Dict[str, float]) -> Dict[str, float]:
    if not shapley_values:
        return {}
    vals = np.array(list(shapley_values.values()), dtype=float)
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmax - vmin == 0:
        return {k: 0.0 for k in shapley_values}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in shapley_values.items()}


def compute_shapley_values(local_nodes: List[str], edge_weights: Dict[Tuple[str, str], float], node_init: Dict[str, float], cfg: PCMCIShapleyConfig) -> Dict[str, float]:
    def v(C: Set[str]) -> float:
        return compute_system_anomaly(C, local_nodes, edge_weights, K=cfg.K, alpha_prop=cfg.alpha_prop, node_init=node_init)

    if cfg.shapley_method == "exact" and len(local_nodes) <= 15:
        # Fallback to sampling with high R if exact not implemented yet
        R = max(cfg.sampling_rounds, 2000)
        return compute_shapley_sampling(local_nodes, v, R=R, n_jobs=cfg.shapley_n_jobs)
    elif cfg.shapley_method == "adaptive":
        # Adaptive sampling
        return compute_adaptive_shapley_sampling(local_nodes, v, cfg)
    # default sampling
    return compute_shapley_sampling(local_nodes, v, R=cfg.sampling_rounds, n_jobs=cfg.shapley_n_jobs)


def compute_adaptive_shapley_sampling(
    local_nodes: List[str], 
    value_func: Callable[[Set[str]], float], 
    cfg: PCMCIShapleyConfig
) -> Dict[str, float]:
    """
    自適應 Shapley 採樣：根據方差自動調整採樣輪數
    
    Args:
        cfg: 配置物件，包含 adaptive sampling 參數
    """
    nodes = list(local_nodes)
    max_R = cfg.adaptive_max_rounds
    min_R = cfg.adaptive_min_rounds
    confidence = cfg.adaptive_confidence
    tolerance = cfg.adaptive_tolerance
    check_interval = cfg.adaptive_check_interval
    
    phi_sum = {s: 0.0 for s in nodes}
    phi_sq_sum = {s: 0.0 for s in nodes}
    
    # 初始採樣
    initial_samples = min(min_R, max_R)
    for r in range(initial_samples):
        phi_r = _single_sample_shapley(nodes, value_func)
        for s in nodes:
            phi_sum[s] += phi_r[s]
            phi_sq_sum[s] += phi_r[s] ** 2
    
    # 自適應採樣
    for r in range(initial_samples, max_R + 1):
        phi_r = _single_sample_shapley(nodes, value_func)
        for s in nodes:
            phi_sum[s] += phi_r[s]
            phi_sq_sum[s] += phi_r[s] ** 2
        
        # 每 check_interval 輪檢查一次收斂
        if r % check_interval == 0 and r >= min_R:
            converged = True
            for s in nodes:
                mean = phi_sum[s] / r
                variance = (phi_sq_sum[s] / r) - mean ** 2
                std_err = np.sqrt(variance / r) if variance > 0 else 0.0
                
                # 計算置信區間寬度
                if confidence == 0.95:
                    ci_width = 1.96 * std_err
                elif confidence == 0.99:
                    ci_width = 2.58 * std_err
                else:
                    ci_width = 1.96 * std_err  # 預設 95%
                
                # 檢查是否收斂
                if abs(mean) > 1e-10 and ci_width > tolerance * abs(mean):
                    converged = False
                    break
            
            if converged:
                print(f"Adaptive Shapley converged at round {r}/{max_R}")
                break
    
    return {s: phi_sum[s] / r for s in nodes}


def _single_sample_shapley(nodes: List[str], value_func: Callable[[Set[str]], float]) -> Dict[str, float]:
    """單次 Shapley 採樣"""
    rng = np.random.default_rng()
    phi_local = {s: 0.0 for s in nodes}
    perm = nodes.copy()
    rng.shuffle(perm)
    prefix = set()
    for s in perm:
        marginal = value_func(prefix | {s}) - value_func(prefix)
        phi_local[s] += marginal
        prefix.add(s)
    return phi_local
