"""
Tests for the simplified causal discovery module.
"""
import numpy as np
import pandas as pd
import pytest

from RCAEval.e2e.cpg_shap_modules import causal_discovery, CPGShapConfig


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    np.random.seed(42)
    n_timesteps = 200
    
    # Create causal relationships: A -> B -> C
    A = np.random.randn(n_timesteps)
    B = 0.5 * np.roll(A, 1) + np.random.randn(n_timesteps) * 0.3
    C = 0.6 * np.roll(B, 1) + np.random.randn(n_timesteps) * 0.3
    
    df = pd.DataFrame({
        "service_a": A,
        "service_b": B,
        "service_c": C
    })
    
    return df


@pytest.fixture
def config():
    """Create default configuration."""
    return CPGShapConfig(
        causal_method="pcmci",
        pcmci_alpha=0.05,
        tau_max=5,
        pcmci_max_conds_dim=2
    )


def test_prepare_data_matrix(sample_data):
    """Test data matrix preparation."""
    X, cols = causal_discovery.prepare_data_matrix(
        sample_data,
        local_nodes=["service_a", "service_b"]
    )
    
    assert X.shape[0] == 2  # 2 variables
    assert X.shape[1] == 200  # 200 timesteps
    assert len(cols) == 2
    assert "service_a" in cols
    assert "service_b" in cols


def test_clean_data_matrix():
    """Test data matrix cleaning."""
    # Create matrix with one constant variable
    X = np.array([
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],  # constant
        [5, 4, 3, 2, 1]
    ], dtype=float)
    
    X_clean, mask = causal_discovery.clean_data_matrix(X)
    
    assert X_clean.shape[0] == 2  # constant variable removed
    assert mask.sum() == 2
    assert not mask[1]  # constant variable marked as False


def test_discover_causal_graph_pcmci(sample_data, config):
    """Test PCMCI causal discovery."""
    config.causal_method = "pcmci"
    
    result = causal_discovery.discover_causal_graph(
        sample_data,
        local_nodes=["service_a", "service_b", "service_c"],
        config=config,
        method="pcmci"
    )
    
    assert "edges" in result
    assert "edge_strengths" in result
    assert "pcmci_graph" in result
    assert "columns" in result
    
    assert len(result["columns"]) == 3


def test_discover_causal_graph_pc(sample_data, config):
    """Test PC causal discovery."""
    config.causal_method = "pc"
    
    result = causal_discovery.discover_causal_graph(
        sample_data,
        local_nodes=["service_a", "service_b", "service_c"],
        config=config,
        method="pc"
    )
    
    assert "edges" in result
    assert "edge_strengths" in result
    assert "pcmci_graph" in result
    assert "columns" in result


def test_discover_causal_graph_empty_data(config):
    """Test handling of empty data."""
    empty_df = pd.DataFrame()
    
    result = causal_discovery.discover_causal_graph(
        empty_df,
        local_nodes=[],
        config=config,
        method="pcmci"
    )
    
    assert len(result["edges"]) == 0
    assert len(result["edge_strengths"]) == 0
    assert len(result["columns"]) == 0


def test_discover_causal_graph_insufficient_variables(config):
    """Test handling of insufficient variables."""
    df = pd.DataFrame({
        "service_a": [1, 2, 3, 4, 5]
    })
    
    result = causal_discovery.discover_causal_graph(
        df,
        local_nodes=["service_a"],
        config=config,
        method="pcmci"
    )
    
    # Should return empty results gracefully
    assert len(result["edges"]) == 0


def test_run_pcmci_basic():
    """Test basic PCMCI run."""
    np.random.seed(42)
    n_vars = 3
    n_time = 200
    
    X = np.random.randn(n_vars, n_time)
    
    report = causal_discovery.run_pcmci(
        X,
        tau_max=3,
        alpha=0.05,
        max_conds_dim=2
    )
    
    assert "p_matrix" in report
    assert "val_matrix" in report
    assert report["p_matrix"].shape[0] == n_vars
    assert report["p_matrix"].shape[2] == 4  # tau_max + 1


def test_run_pc_basic():
    """Test basic PC run."""
    np.random.seed(42)
    n_vars = 3
    n_time = 200
    
    X = np.random.randn(n_vars, n_time)
    
    report = causal_discovery.run_pc(
        X,
        alpha=0.05,
        max_conds_dim=2
    )
    
    assert "graph" in report
    assert report["graph"].shape == (n_vars, n_vars)


def test_extract_edges_pcmci():
    """Test edge extraction from PCMCI results."""
    n_vars = 3
    tau_max = 2
    
    # Create mock report with one significant edge
    p_matrix = np.ones((n_vars, n_vars, tau_max + 1))
    p_matrix[0, 1, 1] = 0.01  # significant edge from 0->1 at lag 1
    
    val_matrix = np.random.randn(n_vars, n_vars, tau_max + 1)
    val_matrix[0, 1, 1] = 0.8
    
    report = {
        "p_matrix": p_matrix,
        "val_matrix": val_matrix
    }
    
    edges, strengths = causal_discovery.extract_edges_and_strengths(
        report, method="pcmci", alpha=0.05, tau_max=tau_max
    )
    
    assert len(edges) == 1
    assert edges[0] == (0, 1, 1)
    assert (0, 1) in strengths
    assert 0.0 <= strengths[(0, 1)] <= 1.0


def test_extract_edges_pc():
    """Test edge extraction from PC results."""
    n_vars = 3
    
    # Create mock graph with edges
    graph = np.zeros((n_vars, n_vars))
    graph[0, 1] = 1.0
    graph[1, 2] = 0.8
    
    report = {"graph": graph}
    
    edges, strengths = causal_discovery.extract_edges_and_strengths(
        report, method="pc", alpha=0.05
    )
    
    assert len(edges) == 2
    assert (0, 1, 0) in edges  # lag=0 for PC
    assert (1, 2, 0) in edges
    assert strengths[(0, 1)] == 1.0
    assert strengths[(1, 2)] == 0.8


def test_config_validation():
    """Test configuration validation."""
    # Valid config
    config = CPGShapConfig(causal_method="pcmci")
    assert config.validate()
    
    # Invalid method
    with pytest.raises(AssertionError):
        config = CPGShapConfig(causal_method="invalid_method")
        config.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

