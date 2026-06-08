import pytest
import pandas as pd
import numpy as np
from RCAEval.e2e.cpg_shap import cpg_shap
from RCAEval.e2e.cpg_shap_modules import CPGShapConfig


def test_cpg_shap_with_pruning():
    """Test the full pipeline with pruning enabled."""
    # Build simple test data
    np.random.seed(42)
    T = 100
    data = pd.DataFrame({
        'time': range(T),
        'A_latency': np.random.randn(T) + 1,
        'B_latency': np.random.randn(T) + 0.5,
        'C_latency': np.random.randn(T),
    })
    
    # Config: enable pruning
    config = CPGShapConfig(
        enable_pruning=True,
        pruning_max_hops=1,
        pruning_min_nodes=2,
        sampling_rounds=100  # Reduce sampling to speed up the test
    )
    
    result = cpg_shap(data, config=config, dataset="test")
    
    assert 'ranks' in result
    assert 'adj' in result
    assert len(result['ranks']) > 0


def test_pruning_reduces_computation():
    """Test that pruning reduces computation."""
    import time
    
    np.random.seed(42)
    T = 200
    # Build a larger dataset
    data = pd.DataFrame({
        'time': range(T),
        **{f'S{i}_latency': np.random.randn(T) for i in range(20)}
    })
    
    # Without pruning
    config_no_prune = CPGShapConfig(enable_pruning=False, sampling_rounds=50)
    start = time.time()
    result1 = cpg_shap(data, config=config_no_prune, dataset="test")
    time_no_prune = time.time() - start
    
    # With pruning
    config_prune = CPGShapConfig(enable_pruning=True, sampling_rounds=50)
    start = time.time()
    result2 = cpg_shap(data, config=config_prune, dataset="test")
    time_prune = time.time() - start
    
    # Should be faster
    assert time_prune < time_no_prune
    print(f"Speedup: {time_no_prune / time_prune:.2f}x")
