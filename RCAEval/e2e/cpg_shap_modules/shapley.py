from __future__ import annotations
from typing import Dict, List, Set, Tuple, Callable
import numpy as np
from joblib import Parallel, delayed
import logging

from .config import CPGShapConfig
from .propagation import propagate_k_steps

logger = logging.getLogger(__name__)


def compute_system_anomaly(coalition: Set[str], local_nodes: List[str], edge_weights: Dict[Tuple[str, str], float], K: int, alpha_prop: float, node_init: Dict[str, float] | None = None) -> float:
    """Compute the system anomaly value of a coalition (always runs propagation, no caching)."""
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
    Create a cached wrapper around the value function.
    
    Args:
        base_value_func: Original coalition value function.
        cache_size: Cache size (maximum number of LRU entries).
    
    Returns:
        (cached_value_func, cache_stats_dict) - cached function and cache statistics.
    """
    cache: Dict[Tuple[str, ...], float] = {}
    cache_hits = 0
    cache_misses = 0
    
    def cached_func(coalition: Set[str]) -> float:
        nonlocal cache_hits, cache_misses
        
        # Use a sorted tuple as the cache key (set cannot be used as a dict key)
        key = tuple(sorted(coalition))
        
        if key in cache:
            cache_hits += 1
            return cache[key]
        
        cache_misses += 1
        value = base_value_func(coalition)
        
        # Simple LRU: if cache is too large, drop the oldest entries (simple strategy: drop first few keys)
        if len(cache) >= cache_size:
            # Remove the first few entries (simple LRU approximation)
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
    
    # Attach statistics function
    cached_func.get_stats = get_stats  # type: ignore
    
    return cached_func, stats


def compute_shapley_sampling(local_nodes: List[str], value_func: Callable[[Set[str]], float], R: int, rng: np.random.Generator | None = None, n_jobs: int = -1, enable_cache: bool = True, cache_size: int = 2048) -> Dict[str, float]:
    """
    Parallel Shapley sampling with caching optimizations.
    
    Args:
        local_nodes: List of local nodes.
        value_func: Coalition value function.
        R: Number of sampling rounds.
        rng: Random number generator.
        n_jobs: Number of parallel jobs (-1 means all cores).
        enable_cache: Whether to enable caching.
        cache_size: Cache size per worker process.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    nodes = list(local_nodes)
    
    # Decide whether to use parallelism
    if R <= 50 or n_jobs == 1 or len(nodes) <= 5:
        # For small-scale cases, use the original (single-process) method with optional cache
        if enable_cache:
            cached_func, _ = _make_cached_value_func(value_func, cache_size=cache_size)
            return _compute_shapley_sampling_original(local_nodes, cached_func, R, rng)
        else:
            return _compute_shapley_sampling_original(local_nodes, value_func, R, rng)
    
    # Parallel version: distribute R samples across multiple processes,
    # each with its own cache instance.
    def single_sample_shapley_with_cache(seed, base_func):
        """
        Single sampling implementation (with per-process cache).
        Each process has its own cache to avoid recomputing the same coalitions.
        """
        # Create an independent cache for each process
        if enable_cache:
            cached_func, _ = _make_cached_value_func(base_func, cache_size=cache_size)
        else:
            cached_func = base_func
        
        local_rng = np.random.default_rng(seed)
        phi_local = {s: 0.0 for s in nodes}
        perm = nodes.copy()
        local_rng.shuffle(perm)
        prefix = set()
        
        # Within a single permutation, prefix grows step by step, so the cache can be reused
        for s in perm:
            prefix_with_s = prefix | {s}
            marginal = cached_func(prefix_with_s) - cached_func(prefix)
            phi_local[s] += marginal
            prefix.add(s)
        
        return phi_local
    
    # Run R samples in parallel
    seeds = [rng.integers(0, 2**32) for _ in range(R)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_sample_shapley_with_cache)(seed, value_func) for seed in seeds
    )
    
    # Aggregate results
    phi = {s: 0.0 for s in nodes}
    for phi_local in results:
        for s, v in phi_local.items():
            phi[s] += v
    return {s: v / R for s, v in phi.items()}


def _compute_shapley_sampling_original(local_nodes: List[str], value_func: Callable[[Set[str]], float], R: int, rng: np.random.Generator | None = None) -> Dict[str, float]:
    """Original single-process version (kept as a backup)."""
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


def compute_shapley_values(local_nodes: List[str], edge_weights: Dict[Tuple[str, str], float], node_init: Dict[str, float], cfg: CPGShapConfig) -> Dict[str, float]:
    """
    Compute Shapley values with caching optimizations.
    """
    def v(C: Set[str]) -> float:
        return compute_system_anomaly(C, local_nodes, edge_weights, K=cfg.K, alpha_prop=cfg.alpha_prop, node_init=node_init)

    # Decide whether to enable caching based on config
    enable_cache = cfg.enable_shapley_cache
    
    if cfg.shapley_method == "exact" and len(local_nodes) <= 15:
        # Fallback to sampling with high R if exact is not implemented
        R = max(cfg.sampling_rounds, 2000)
        result = compute_shapley_sampling(
            local_nodes, v, R=R, n_jobs=cfg.shapley_n_jobs,
            enable_cache=enable_cache, cache_size=cfg.shapley_cache_size
        )
    elif cfg.shapley_method == "adaptive":
        # Adaptive sampling (internally uses caching)
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
    cfg: CPGShapConfig
) -> Dict[str, float]:
    """
    Adaptive Shapley sampling: automatically adjust the number of samples based on variance,
    with caching optimizations.
    
    Args:
        local_nodes: List of local nodes.
        value_func: Coalition value function.
        cfg: Configuration object containing adaptive sampling parameters.
    """
    nodes = list(local_nodes)
    max_R = cfg.adaptive_max_rounds
    min_R = cfg.adaptive_min_rounds
    confidence = cfg.adaptive_confidence
    tolerance = cfg.adaptive_tolerance
    check_interval = cfg.adaptive_check_interval
    
    # Create cached value function
    cached_func, cache_stats = _make_cached_value_func(
        value_func, 
        cache_size=cfg.shapley_cache_size
    )
    
    phi_sum = {s: 0.0 for s in nodes}
    phi_sq_sum = {s: 0.0 for s in nodes}
    
    # Initial sampling
    initial_samples = min(min_R, max_R)
    for r in range(initial_samples):
        phi_r = _single_sample_shapley(nodes, cached_func)
        for s in nodes:
            phi_sum[s] += phi_r[s]
            phi_sq_sum[s] += phi_r[s] ** 2
    
    # Adaptive sampling loop
    final_r = initial_samples
    for r in range(initial_samples, max_R + 1):
        phi_r = _single_sample_shapley(nodes, cached_func)
        for s in nodes:
            phi_sum[s] += phi_r[s]
            phi_sq_sum[s] += phi_r[s] ** 2
        
        final_r = r + 1
        
        # Check convergence every check_interval rounds
        if r % check_interval == 0 and r >= min_R:
            converged = True
            for s in nodes:
                mean = phi_sum[s] / (r + 1)
                variance = (phi_sq_sum[s] / (r + 1)) - mean ** 2
                std_err = np.sqrt(variance / (r + 1)) if variance > 0 else 0.0
                
                # Compute confidence interval width
                if confidence == 0.95:
                    ci_width = 1.96 * std_err
                elif confidence == 0.99:
                    ci_width = 2.58 * std_err
                else:
                    ci_width = 1.96 * std_err  # Default to 95%
                
                # Check convergence condition
                if abs(mean) > 1e-10 and ci_width > tolerance * abs(mean):
                    converged = False
                    break
            
            if converged:
                stats = cache_stats()
                logger.info(f"Adaptive Shapley converged at round {final_r}/{max_R}, cache hits={stats['hits']}, misses={stats['misses']}, hit_rate={stats['hits']/(stats['hits']+stats['misses']) if (stats['hits']+stats['misses']) > 0 else 0:.2%}")
                break
    
    # Output cache statistics
    stats = cache_stats()
    if stats['hits'] + stats['misses'] > 0:
        hit_rate = stats['hits'] / (stats['hits'] + stats['misses'])
        logger.debug(f"Shapley cache stats: hits={stats['hits']}, misses={stats['misses']}, hit_rate={hit_rate:.2%}, size={stats['size']}")
    
    return {s: phi_sum[s] / final_r for s in nodes}


def _single_sample_shapley(nodes: List[str], value_func: Callable[[Set[str]], float]) -> Dict[str, float]:
    """
    Single Shapley sampling round.
    Note: if value_func is already cached, this automatically benefits from it.
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
