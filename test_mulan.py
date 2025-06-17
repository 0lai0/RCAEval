"""
Test script for MULAN implementation in RCAEval
Testing on RE1 (single-modal), RE2/RE3 (multi-modal) datasets
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add RCAEval to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from RCAEval.e2e.mulan import mulan
from RCAEval.io.time_series import preprocess


def create_sample_re1_data(n_entities=10, n_timesteps=100):
    """Create sample RE1 data (single-modal metrics only)"""
    np.random.seed(42)
    
    # Generate time series data
    time = pd.date_range('2023-01-01', periods=n_timesteps, freq='1min')
    data = {
        'time': time,
    }
    
    # Add metric columns
    for i in range(n_entities):
        if i == 0:
            # First metric as KPI (with anomaly pattern)
            normal_values = np.random.normal(100, 10, n_timesteps // 2)
            anomal_values = np.random.normal(150, 15, n_timesteps - n_timesteps // 2)
            data[f'frontend_latency'] = np.concatenate([normal_values, anomal_values])
        else:
            # Other metrics with some correlation to KPI
            base_values = np.random.normal(50, 5, n_timesteps)
            if i <= 3:  # Some correlated metrics
                correlation = 0.3 * data['frontend_latency'] / 100
                data[f'metric_{i}'] = base_values + correlation
            else:
                data[f'metric_{i}'] = base_values
    
    return pd.DataFrame(data)


def create_sample_re2_data(n_entities=8, n_timesteps=100):
    """Create sample RE2/RE3 data (multi-modal: metrics + logs + traces)"""
    np.random.seed(42)
    
    # Generate time series data
    time = pd.date_range('2023-01-01', periods=n_timesteps, freq='1min')
    
    # Metric data
    metric_data = {
        'time': time,
        'frontend_latency': np.concatenate([
            np.random.normal(100, 10, n_timesteps // 2),  # normal
            np.random.normal(150, 15, n_timesteps - n_timesteps // 2)  # anomal
        ])
    }
    
    for i in range(1, n_entities):
        base_values = np.random.normal(50, 5, n_timesteps)
        if i <= 3:
            correlation = 0.3 * metric_data['frontend_latency'] / 100
            metric_data[f'service_{i}_cpu'] = base_values + correlation
        else:
            metric_data[f'service_{i}_memory'] = base_values
    
    metric_df = pd.DataFrame(metric_data)
    
    # Log time series data (logts)
    log_data = {
        'time': time,
    }
    
    for i in range(n_entities):
        # Log event counts
        base_log_count = np.random.poisson(10, n_timesteps)
        if i <= 2:  # Some services have anomalous log patterns
            anomal_pattern = np.concatenate([
                np.zeros(n_timesteps // 2),
                np.random.poisson(30, n_timesteps - n_timesteps // 2)
            ])
            log_data[f'service_{i}_error_count'] = base_log_count + anomal_pattern
        else:
            log_data[f'service_{i}_info_count'] = base_log_count
    
    logts_df = pd.DataFrame(log_data)
    
    # Trace data (simplified)
    trace_lat_data = {
        'time': time,
    }
    
    trace_err_data = {
        'time': time,
    }
    
    for i in range(n_entities):
        # Trace latency
        base_latency = np.random.gamma(2, 50, n_timesteps)
        if i <= 2:
            anomal_latency = np.concatenate([
                np.zeros(n_timesteps // 2),
                np.random.gamma(3, 30, n_timesteps - n_timesteps // 2)
            ])
            trace_lat_data[f'service_{i}_trace_latency'] = base_latency + anomal_latency
        else:
            trace_lat_data[f'service_{i}_trace_latency'] = base_latency
        
        # Trace error rate
        base_error_rate = np.random.beta(1, 100, n_timesteps)
        if i <= 1:
            anomal_error = np.concatenate([
                np.zeros(n_timesteps // 2),
                np.random.beta(2, 20, n_timesteps - n_timesteps // 2)
            ])
            trace_err_data[f'service_{i}_error_rate'] = base_error_rate + anomal_error
        else:
            trace_err_data[f'service_{i}_error_rate'] = base_error_rate
    
    tracets_lat_df = pd.DataFrame(trace_lat_data)
    tracets_err_df = pd.DataFrame(trace_err_data)
    
    return {
        "metric": metric_df,
        "logts": logts_df,
        "tracets_lat": tracets_lat_df,
        "tracets_err": tracets_err_df
    }


def test_mulan_re1():
    """Test MULAN on RE1 dataset (single-modal)"""
    print("🧪 Testing MULAN on RE1 (single-modal)...")
    
    # Create sample data
    data = create_sample_re1_data()
    inject_time = data['time'].iloc[len(data) // 2]
    
    try:
        # Run MULAN
        results = mulan(
            data=data,
            inject_time=inject_time,
            dataset="re1",
            sli="frontend_latency",
            num_epochs=20,  # Reduced for testing
            learning_rate=0.01
        )
        
        print(f"✅ RE1 Test Passed!")
        print(f"   - Number of entities: {len(results['node_names'])}")
        print(f"   - Top 5 root causes: {results['ranks'][:5]}")
        print(f"   - Adjacency matrix shape: {results['adj_matrix'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ RE1 Test Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mulan_re2():
    """Test MULAN on RE2/RE3 dataset (multi-modal)"""
    print("\n🧪 Testing MULAN on RE2/RE3 (multi-modal)...")
    
    # Create sample data
    data = create_sample_re2_data()
    inject_time = data["metric"]['time'].iloc[len(data["metric"]) // 2]
    
    try:
        # Run MULAN without traces
        results = mulan(
            data=data,
            inject_time=inject_time,
            dataset="re2",
            sli="frontend_latency",
            use_traces=False,  # Test without traces first
            num_epochs=20,
            learning_rate=0.01
        )
        
        print(f"✅ RE2 Test (without traces) Passed!")
        print(f"   - Number of entities: {len(results['node_names'])}")
        print(f"   - Top 5 root causes: {results['ranks'][:5]}")
        print(f"   - Adjacency matrix shape: {results['adj_matrix'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ RE2 Test Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mulan_re2_with_traces():
    """Test MULAN on RE2/RE3 dataset with traces (full multi-modal)"""
    print("\n🧪 Testing MULAN on RE2/RE3 (with traces)...")
    
    # Create sample data
    data = create_sample_re2_data()
    inject_time = data["metric"]['time'].iloc[len(data["metric"]) // 2]
    
    try:
        # Run MULAN with traces
        results = mulan(
            data=data,
            inject_time=inject_time,
            dataset="re2",
            sli="frontend_latency",
            use_traces=True,  # Test with traces
            num_epochs=20,
            learning_rate=0.01
        )
        
        print(f"✅ RE2 Test (with traces) Passed!")
        print(f"   - Number of entities: {len(results['node_names'])}")
        print(f"   - Top 5 root causes: {results['ranks'][:5]}")
        print(f"   - Adjacency matrix shape: {results['adj_matrix'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ RE2 Test (with traces) Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mulan_performance():
    """Test MULAN performance and evaluation metrics"""
    print("\n📊 Testing MULAN evaluation metrics...")
    
    # Create sample data with known ground truth
    data = create_sample_re1_data()
    inject_time = data['time'].iloc[len(data) // 2]
    
    # Assume ground truth: first 3 metrics are true root causes
    ground_truth = ['frontend_latency', 'metric_1', 'metric_2']
    
    try:
        results = mulan(
            data=data,
            inject_time=inject_time,
            dataset="re1",
            sli="frontend_latency",
            num_epochs=20
        )
        
        # Calculate Avg@5
        predicted_top5 = results['ranks'][:5]
        hits = sum(1 for gt in ground_truth if gt in predicted_top5)
        avg_at_5 = hits / len(ground_truth)
        
        print(f"✅ Performance Test Passed!")
        print(f"   - Ground truth: {ground_truth}")
        print(f"   - Predicted top 5: {predicted_top5}")
        print(f"   - Avg@5: {avg_at_5:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance Test Failed: {str(e)}")
        return False


def main():
    """Main test function"""
    print("🚀 Starting MULAN Integration Tests for RCAEval\n")
    
    # Check PyTorch availability
    if torch.cuda.is_available():
        print(f"🖥️  Using CUDA: {torch.cuda.get_device_name()}")
    else:
        print("🖥️  Using CPU")
    
    # Run tests
    tests = [
        test_mulan_re1,
        test_mulan_re2,
        test_mulan_re2_with_traces,
        test_mulan_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   - Passed: {passed}/{total}")
    print(f"   - Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! MULAN is ready for integration with RCAEval.")
        print("\n📝 Next steps:")
        print("   1. Run: python main.py --method mulan --dataset re1-ob")
        print("   2. Run: python main.py --method mulan --dataset re2-tt")
        print("   3. Run: python main.py --method mulan --dataset re3-tk")
        print("   4. Compare results with baseline methods (Baro, CausalRCA)")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 