import pytest
import pandas as pd
import numpy as np
from RCAEval.e2e.pcmci_shapley import pcmci_shapley
from RCAEval.e2e.pcmci_shapley_modules import PCMCIShapleyConfig


def test_parallel_node_isolation():
    """Test Node Isolation parallelization."""
    np.random.seed(42)
    T = 100
    data = pd.DataFrame({
        'time': range(T),
        'A_latency': np.random.randn(T) + 1,
        'B_latency': np.random.randn(T) + 0.5,
        'C_latency': np.random.randn(T),
        'D_latency': np.random.randn(T) + 0.3,
        'E_latency': np.random.randn(T) + 0.8,
    })
    
    # Parallel version
    config_parallel = PCMCIShapleyConfig(
        enable_parallel=True,
        node_isolation_n_jobs=2,
        sampling_rounds=50
    )
    
    result_parallel = pcmci_shapley(data, config=config_parallel, dataset="test")
    
    # Non-parallel version
    config_serial = PCMCIShapleyConfig(
        enable_parallel=False,
        node_isolation_n_jobs=1,
        sampling_rounds=50
    )
    
    result_serial = pcmci_shapley(data, config=config_serial, dataset="test")
    
    # Results should match
    assert len(result_parallel['ranks']) == len(result_serial['ranks'])
    assert 'adj' in result_parallel
    assert 'adj' in result_serial


def test_parallel_shapley_sampling():
    """Test Shapley sampling parallelization."""
    np.random.seed(42)
    T = 200
    data = pd.DataFrame({
        'time': range(T),
        **{f'S{i}_latency': np.random.randn(T) for i in range(15)}
    })
    
    # Parallel version
    config_parallel = PCMCIShapleyConfig(
        enable_parallel=True,
        shapley_n_jobs=2,
        sampling_rounds=200
    )
    
    result_parallel = pcmci_shapley(data, config=config_parallel, dataset="test")
    
    # Non-parallel version
    config_serial = PCMCIShapleyConfig(
        enable_parallel=False,
        shapley_n_jobs=1,
        sampling_rounds=200
    )
    
    result_serial = pcmci_shapley(data, config=config_serial, dataset="test")
    
    # Results should match
    assert len(result_parallel['ranks']) == len(result_serial['ranks'])


def test_adaptive_sampling():
    """Test Adaptive Sampling."""
    np.random.seed(42)
    T = 150
    data = pd.DataFrame({
        'time': range(T),
        **{f'S{i}_latency': np.random.randn(T) for i in range(10)}
    })
    
    # Adaptive sampling
    config_adaptive = PCMCIShapleyConfig(
        shapley_method="adaptive",
        adaptive_max_rounds=500,
        adaptive_min_rounds=50,
        adaptive_tolerance=0.05,  # 5% error
        adaptive_check_interval=25
    )
    
    result = pcmci_shapley(data, config=config_adaptive, dataset="test")
    
    assert 'ranks' in result
    assert len(result['ranks']) > 0


def test_phase2_performance():
    """Test overall performance improvement in Phase 2."""
    import time
    
    np.random.seed(42)
    T = 300
    data = pd.DataFrame({
        'time': range(T),
        **{f'S{i}_latency': np.random.randn(T) for i in range(25)}
    })
    
    # Phase 1 config
    config_phase1 = PCMCIShapleyConfig(
        enable_parallel=False,
        shapley_method="sampling",
        sampling_rounds=200
    )
    
    start = time.time()
    result1 = pcmci_shapley(data, config=config_phase1, dataset="test")
    time_phase1 = time.time() - start
    
    # Phase 2 config
    config_phase2 = PCMCIShapleyConfig(
        enable_parallel=True,
        node_isolation_n_jobs=2,
        shapley_n_jobs=2,
        shapley_method="adaptive",
        adaptive_max_rounds=200,
        adaptive_min_rounds=50,
        adaptive_tolerance=0.05
    )
    
    start = time.time()
    result2 = pcmci_shapley(data, config=config_phase2, dataset="test")
    time_phase2 = time.time() - start
    
    # Phase 2 should be faster
    assert time_phase2 < time_phase1
    speedup = time_phase1 / time_phase2
    print(f"Phase 2 speedup: {speedup:.2f}x")
    
    # Sanity check: results should be reasonable
    assert len(result1['ranks']) == len(result2['ranks'])
