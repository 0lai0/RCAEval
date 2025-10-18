import pytest
import numpy as np
import pandas as pd
import networkx as nx
from RCAEval.e2e.pcmci_shapley_modules import pruning


def test_trace_based_prefiltering():
    """測試基於 trace 的過濾"""
    # 建立簡單的 trace 圖: A -> B -> C, A -> D
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
    """測試基於異常分數的過濾"""
    node_anomaly = {
        'A': 10.0,
        'B': 8.0,
        'C': 6.0,
        'D': 4.0,
        'E': 2.0,
        'F': 0.5
    }
    
    # 保留 top 50% (percentile=0.5)
    result = pruning.early_anomaly_pruning(node_anomaly, threshold_percentile=0.5, min_nodes=2)
    assert len(result) >= 3  # 至少保留 top 50%
    assert 'A' in result  # 最高分一定在
    assert 'B' in result
    
    # 測試 min_nodes 保證
    result = pruning.early_anomaly_pruning({'A': 1.0, 'B': 0.5}, threshold_percentile=0.8, min_nodes=2)
    assert len(result) == 2


def test_combined_pruning():
    """測試組合剪枝"""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C')])
    
    all_nodes = ['A', 'B', 'C', 'D']
    node_anomaly = {'A': 10, 'B': 8, 'C': 2, 'D': 1}
    
    result = pruning.combined_pruning(
        all_nodes, node_anomaly, G, 'A', 
        max_hops=1, anomaly_percentile=0.5, min_nodes=2
    )
    
    # 應該只保留 trace 1-hop 內 (A, B) 且異常分數較高的節點
    assert 'A' in result
    assert 'D' not in result  # D 不在 trace 範圍內
