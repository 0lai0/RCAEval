import pytest
import numpy as np
import pandas as pd
import networkx as nx
from RCAEval.e2e.cpg_shap_modules import pruning


def test_trace_based_prefiltering():
    """Test trace-based filtering."""
    # Build a simple trace graph: A -> B -> C, A -> D
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'D')])
    
    all_nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # 1-hop from A: should keep A, B, D
    result = pruning.trace_based_prefiltering(all_nodes, G, 'A', max_hops=1)
    assert set(result) == {'A', 'B', 'D'}
    
    # 2-hop from A: should keep A, B, C, D
    result = pruning.trace_based_prefiltering(all_nodes, G, 'A', max_hops=2)
    assert set(result) == {'A', 'B', 'C', 'D'}
    
    # No trace graph: should keep all
    result = pruning.trace_based_prefiltering(all_nodes, None, 'A', max_hops=1)
    assert result == all_nodes


def test_early_anomaly_pruning():
    """Test anomaly-score-based pruning."""
    node_anomaly = {
        'A': 10.0,
        'B': 8.0,
        'C': 6.0,
        'D': 4.0,
        'E': 2.0,
        'F': 0.5
    }
    
    # Keep top 50% (percentile=0.5)
    result = pruning.early_anomaly_pruning(node_anomaly, threshold_percentile=0.5, min_nodes=2)
    assert len(result) >= 3  # Keep at least top 50%
    assert 'A' in result  # The highest score must be included
    assert 'B' in result
    
    # Test min_nodes guarantee
    result = pruning.early_anomaly_pruning({'A': 1.0, 'B': 0.5}, threshold_percentile=0.8, min_nodes=2)
    assert len(result) == 2


def test_combined_pruning():
    """Test combined pruning."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C')])
    
    all_nodes = ['A', 'B', 'C', 'D']
    node_anomaly = {'A': 10, 'B': 8, 'C': 2, 'D': 1}
    
    result = pruning.combined_pruning(
        all_nodes, node_anomaly, G, 'A', 
        max_hops=1, anomaly_percentile=0.5, min_nodes=2
    )
    
    # Should keep only nodes within 1-hop in the trace (A, B) and with higher anomaly scores
    assert 'A' in result
    assert 'D' not in result  # D is outside the trace neighborhood
