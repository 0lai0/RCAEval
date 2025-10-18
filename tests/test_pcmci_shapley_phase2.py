import pytest
import pandas as pd
import numpy as np
from RCAEval.e2e.pcmci_shapley import pcmci_shapley
from RCAEval.e2e.pcmci_shapley_modules import PCMCIShapleyConfig


def test_parallel_node_isolation():
    """測試 Node Isolation 平行化"""
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
    
    # 測試平行化版本
    config_parallel = PCMCIShapleyConfig(
        enable_parallel=True,
        node_isolation_n_jobs=2,
        sampling_rounds=50
    )
    
    result_parallel = pcmci_shapley(data, config=config_parallel, dataset="test")
    
    # 測試非平行化版本
    config_serial = PCMCIShapleyConfig(
        enable_parallel=False,
        node_isolation_n_jobs=1,
        sampling_rounds=50
    )
    
    result_serial = pcmci_shapley(data, config=config_serial, dataset="test")
    
    # 結果應該相同
    assert len(result_parallel['ranks']) == len(result_serial['ranks'])
    assert 'adj' in result_parallel
    assert 'adj' in result_serial


def test_parallel_shapley_sampling():
    """測試 Shapley 採樣平行化"""
    np.random.seed(42)
    T = 200
    data = pd.DataFrame({
        'time': range(T),
        **{f'S{i}_latency': np.random.randn(T) for i in range(15)}
    })
    
    # 測試平行化版本
    config_parallel = PCMCIShapleyConfig(
        enable_parallel=True,
        shapley_n_jobs=2,
        sampling_rounds=200
    )
    
    result_parallel = pcmci_shapley(data, config=config_parallel, dataset="test")
    
    # 測試非平行化版本
    config_serial = PCMCIShapleyConfig(
        enable_parallel=False,
        shapley_n_jobs=1,
        sampling_rounds=200
    )
    
    result_serial = pcmci_shapley(data, config=config_serial, dataset="test")
    
    # 結果應該相同
    assert len(result_parallel['ranks']) == len(result_serial['ranks'])


def test_adaptive_sampling():
    """測試 Adaptive Sampling"""
    np.random.seed(42)
    T = 150
    data = pd.DataFrame({
        'time': range(T),
        **{f'S{i}_latency': np.random.randn(T) for i in range(10)}
    })
    
    # 測試 adaptive sampling
    config_adaptive = PCMCIShapleyConfig(
        shapley_method="adaptive",
        adaptive_max_rounds=500,
        adaptive_min_rounds=50,
        adaptive_tolerance=0.05,  # 5% 誤差
        adaptive_check_interval=25
    )
    
    result = pcmci_shapley(data, config=config_adaptive, dataset="test")
    
    assert 'ranks' in result
    assert len(result['ranks']) > 0


def test_phase2_performance():
    """測試 Phase 2 整體效能提升"""
    import time
    
    np.random.seed(42)
    T = 300
    data = pd.DataFrame({
        'time': range(T),
        **{f'S{i}_latency': np.random.randn(T) for i in range(25)}
    })
    
    # Phase 1 配置
    config_phase1 = PCMCIShapleyConfig(
        enable_parallel=False,
        shapley_method="sampling",
        sampling_rounds=200
    )
    
    start = time.time()
    result1 = pcmci_shapley(data, config=config_phase1, dataset="test")
    time_phase1 = time.time() - start
    
    # Phase 2 配置
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
    
    # Phase 2 應該更快
    assert time_phase2 < time_phase1
    speedup = time_phase1 / time_phase2
    print(f"Phase 2 speedup: {speedup:.2f}x")
    
    # 結果應該合理
    assert len(result1['ranks']) == len(result2['ranks'])
