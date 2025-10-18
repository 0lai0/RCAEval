import pytest
import pandas as pd
import numpy as np
from RCAEval.e2e.pcmci_shapley import pcmci_shapley
from RCAEval.e2e.pcmci_shapley_modules import PCMCIShapleyConfig


def test_pcmci_shapley_with_pruning():
    """測試啟用剪枝後的完整流程"""
    # 建立簡單測試數據
    np.random.seed(42)
    T = 100
    data = pd.DataFrame({
        'time': range(T),
        'A_latency': np.random.randn(T) + 1,
        'B_latency': np.random.randn(T) + 0.5,
        'C_latency': np.random.randn(T),
    })
    
    # 配置: 啟用剪枝
    config = PCMCIShapleyConfig(
        enable_pruning=True,
        pruning_max_hops=1,
        pruning_min_nodes=2,
        sampling_rounds=100  # 減少採樣以加快測試
    )
    
    result = pcmci_shapley(data, config=config, dataset="test")
    
    assert 'ranks' in result
    assert 'adj' in result
    assert len(result['ranks']) > 0


def test_pruning_reduces_computation():
    """測試剪枝確實減少計算量"""
    import time
    
    np.random.seed(42)
    T = 200
    # 建立較大的數據集
    data = pd.DataFrame({
        'time': range(T),
        **{f'S{i}_latency': np.random.randn(T) for i in range(20)}
    })
    
    # 不啟用剪枝
    config_no_prune = PCMCIShapleyConfig(enable_pruning=False, sampling_rounds=50)
    start = time.time()
    result1 = pcmci_shapley(data, config=config_no_prune, dataset="test")
    time_no_prune = time.time() - start
    
    # 啟用剪枝
    config_prune = PCMCIShapleyConfig(enable_pruning=True, sampling_rounds=50)
    start = time.time()
    result2 = pcmci_shapley(data, config=config_prune, dataset="test")
    time_prune = time.time() - start
    
    # 應該要更快
    assert time_prune < time_no_prune
    print(f"Speedup: {time_no_prune / time_prune:.2f}x")
