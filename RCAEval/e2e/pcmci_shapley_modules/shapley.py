from __future__ import annotations
from typing import Dict, List, Set, Tuple, Callable
import numpy as np
from joblib import Parallel, delayed
import logging

from .config import PCMCIShapleyConfig
from .propagation import propagate_k_steps

logger = logging.getLogger(__name__)


def compute_system_anomaly(coalition: Set[str], local_nodes: List[str], edge_weights: Dict[Tuple[str, str], float], K: int, alpha_prop: float, node_init: Dict[str, float] | None = None) -> float:
    """計算 coalition 的系統異常值（每次調用都會執行傳播計算，無緩存）"""
    # Only keep initial anomalies for nodes in coalition; others set to 0
    init = {s: (node_init.get(s, 0.0) if node_init is not None else 0.0) for s in local_nodes}
    for s in list(init.keys()):
        if s not in coalition:
            init[s] = 0.0
    res = propagate_k_steps(init, edge_weights, K=K, alpha_prop=alpha_prop)
    h_final = res["h_final"]
    return float(sum(h_final.values()))


def _make_cached_value_func(base_value_func: Callable[[Set[str]], float], cache_size: int = 2048) -> Tuple[Callable[[Set[str]], float], Dict]:
    """
    創建帶緩存的 value function wrapper
    
    Args:
        base_value_func: 原始的 coalition value function
        cache_size: 緩存大小（LRU cache 的最大條目數）
    
    Returns:
        (cached_value_func, cache_stats_dict) - 帶緩存的函數和緩存統計信息
    """
    cache: Dict[Tuple[str, ...], float] = {}
    cache_hits = 0
    cache_misses = 0
    
    def cached_func(coalition: Set[str]) -> float:
        nonlocal cache_hits, cache_misses
        
        # 使用排序的 tuple 作為 cache key（Set 無法直接作為 dict key）
        key = tuple(sorted(coalition))
        
        if key in cache:
            cache_hits += 1
            return cache[key]
        
        cache_misses += 1
        value = base_value_func(coalition)
        
        # 簡單的 LRU：如果緩存太大，刪除最舊的條目（簡單策略：隨機刪除）
        if len(cache) >= cache_size:
            # 刪除最早的幾個條目（簡單的 LRU 近似）
            keys_to_remove = list(cache.keys())[:cache_size // 4]
            for k in keys_to_remove:
                del cache[k]
        
        cache[key] = value
        return value
    
    stats = {"hits": 0, "misses": 0, "size": 0}
    
    def get_stats():
        stats["hits"] = cache_hits
        stats["misses"] = cache_misses
        stats["size"] = len(cache)
        return stats
    
    # 附加統計函數
    cached_func.get_stats = get_stats  # type: ignore
    
    return cached_func, stats


def compute_shapley_sampling(local_nodes: List[str], value_func: Callable[[Set[str]], float], R: int, rng: np.random.Generator | None = None, n_jobs: int = -1, enable_cache: bool = True, cache_size: int = 2048) -> Dict[str, float]:
    """
    平行化版本的 Shapley 採樣（帶緩存優化）
    
    Args:
        local_nodes: 本地節點列表
        value_func: Coalition value 函數
        R: 採樣輪數
        rng: 隨機數生成器
        n_jobs: 平行化進程數 (-1 表示使用所有核心)
        enable_cache: 是否啟用緩存
        cache_size: 每個進程的緩存大小
    """
    if rng is None:
        rng = np.random.default_rng(0)
    nodes = list(local_nodes)
    
    # 決定是否使用平行化
    if R <= 50 or n_jobs == 1 or len(nodes) <= 5:
        # 小規模時使用原始方法（帶緩存）
        if enable_cache:
            cached_func, _ = _make_cached_value_func(value_func, cache_size=cache_size)
            return _compute_shapley_sampling_original(local_nodes, cached_func, R, rng)
        else:
            return _compute_shapley_sampling_original(local_nodes, value_func, R, rng)
    
    # 平行化版本：將 R 個採樣分配到多個進程
    # 每個進程會有自己的緩存實例
    def single_sample_shapley_with_cache(seed, base_func):
        """
        單次採樣實作（進程內緩存）
        每個進程都有獨立的緩存，避免重複計算相同 coalition
        """
        # 為每個進程創建獨立的緩存
        if enable_cache:
            cached_func, _ = _make_cached_value_func(base_func, cache_size=cache_size)
        else:
            cached_func = base_func
        
        local_rng = np.random.default_rng(seed)
        phi_local = {s: 0.0 for s in nodes}
        perm = nodes.copy()
        local_rng.shuffle(perm)
        prefix = set()
        
        # 在單個 permutation 中，prefix 會逐步增長，可以重用緩存
        for s in perm:
            prefix_with_s = prefix | {s}
            marginal = cached_func(prefix_with_s) - cached_func(prefix)
            phi_local[s] += marginal
            prefix.add(s)
        
        return phi_local
    
    # 平行執行 R 個樣本
    seeds = [rng.integers(0, 2**32) for _ in range(R)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_sample_shapley_with_cache)(seed, value_func) for seed in seeds
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
    """
    計算 Shapley 值（帶緩存優化）
    """
    def v(C: Set[str]) -> float:
        return compute_system_anomaly(C, local_nodes, edge_weights, K=cfg.K, alpha_prop=cfg.alpha_prop, node_init=node_init)

    # 根據配置決定是否啟用緩存
    enable_cache = cfg.enable_shapley_cache
    
    if cfg.shapley_method == "exact" and len(local_nodes) <= 15:
        # Fallback to sampling with high R if exact not implemented yet
        R = max(cfg.sampling_rounds, 2000)
        result = compute_shapley_sampling(
            local_nodes, v, R=R, n_jobs=cfg.shapley_n_jobs,
            enable_cache=enable_cache, cache_size=cfg.shapley_cache_size
        )
    elif cfg.shapley_method == "adaptive":
        # Adaptive sampling（內部已使用緩存）
        result = compute_adaptive_shapley_sampling(local_nodes, v, cfg)
    else:
        # default sampling
        result = compute_shapley_sampling(
            local_nodes, v, R=cfg.sampling_rounds, n_jobs=cfg.shapley_n_jobs,
            enable_cache=enable_cache, cache_size=cfg.shapley_cache_size
        )
    
    return result


def compute_adaptive_shapley_sampling(
    local_nodes: List[str], 
    value_func: Callable[[Set[str]], float], 
    cfg: PCMCIShapleyConfig
) -> Dict[str, float]:
    """
    自適應 Shapley 採樣：根據方差自動調整採樣輪數（帶緩存優化）
    
    Args:
        local_nodes: 本地節點列表
        value_func: Coalition value 函數
        cfg: 配置物件，包含 adaptive sampling 參數
    """
    nodes = list(local_nodes)
    max_R = cfg.adaptive_max_rounds
    min_R = cfg.adaptive_min_rounds
    confidence = cfg.adaptive_confidence
    tolerance = cfg.adaptive_tolerance
    check_interval = cfg.adaptive_check_interval
    
    # 創建帶緩存的 value function
    cached_func, cache_stats = _make_cached_value_func(
        value_func, 
        cache_size=cfg.shapley_cache_size
    )
    
    phi_sum = {s: 0.0 for s in nodes}
    phi_sq_sum = {s: 0.0 for s in nodes}
    
    # 初始採樣
    initial_samples = min(min_R, max_R)
    for r in range(initial_samples):
        phi_r = _single_sample_shapley(nodes, cached_func)
        for s in nodes:
            phi_sum[s] += phi_r[s]
            phi_sq_sum[s] += phi_r[s] ** 2
    
    # 自適應採樣
    final_r = initial_samples
    for r in range(initial_samples, max_R + 1):
        phi_r = _single_sample_shapley(nodes, cached_func)
        for s in nodes:
            phi_sum[s] += phi_r[s]
            phi_sq_sum[s] += phi_r[s] ** 2
        
        final_r = r + 1
        
        # 每 check_interval 輪檢查一次收斂
        if r % check_interval == 0 and r >= min_R:
            converged = True
            for s in nodes:
                mean = phi_sum[s] / (r + 1)
                variance = (phi_sq_sum[s] / (r + 1)) - mean ** 2
                std_err = np.sqrt(variance / (r + 1)) if variance > 0 else 0.0
                
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
                stats = cache_stats()
                logger.info(f"Adaptive Shapley converged at round {final_r}/{max_R}, cache hits={stats['hits']}, misses={stats['misses']}, hit_rate={stats['hits']/(stats['hits']+stats['misses']) if (stats['hits']+stats['misses']) > 0 else 0:.2%}")
                break
    
    # 輸出緩存統計
    stats = cache_stats()
    if stats['hits'] + stats['misses'] > 0:
        hit_rate = stats['hits'] / (stats['hits'] + stats['misses'])
        logger.debug(f"Shapley cache stats: hits={stats['hits']}, misses={stats['misses']}, hit_rate={hit_rate:.2%}, size={stats['size']}")
    
    return {s: phi_sum[s] / final_r for s in nodes}


def _single_sample_shapley(nodes: List[str], value_func: Callable[[Set[str]], float]) -> Dict[str, float]:
    """
    單次 Shapley 採樣
    注意：如果 value_func 已經帶緩存，這裡會自動受益
    """
    rng = np.random.default_rng()
    phi_local = {s: 0.0 for s in nodes}
    perm = nodes.copy()
    rng.shuffle(perm)
    prefix = set()
    for s in perm:
        prefix_with_s = prefix | {s}
        marginal = value_func(prefix_with_s) - value_func(prefix)
        phi_local[s] += marginal
        prefix.add(s)
    return phi_local
