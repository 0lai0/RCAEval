from typing import List, Set, Dict, Tuple
import networkx as nx
import pandas as pd
import numpy as np
from .config import PCMCIShapleyConfig
import logging

logger = logging.getLogger(__name__)


def identify_focus_neighbors(
    df: pd.DataFrame,
    focus_node: str,
    radius: int = 2,
    trace_graph: nx.DiGraph = None,
    anomaly_scores: pd.Series = None,
    max_neighbors: int = 20
) -> List[str]:
    """
    识别焦点节点的邻居（类似RCD的局部化）
    
    Args:
        df: 数据框
        focus_node: 焦点节点
        radius: 搜索半径
        trace_graph: trace依赖图
        anomaly_scores: 异常分数
        max_neighbors: 最大邻居数
    
    Returns:
        邻居节点列表
    """
    neighbors = set()
    
    # 策略1: 基于trace graph的邻居
    if trace_graph is not None and focus_node in trace_graph:
        # 向前和向后radius跳
        for r in range(1, radius + 1):
            # 前驱（可能的原因）
            for pred in trace_graph.predecessors(focus_node):
                neighbors.add(pred)
                # 递归前驱
                if r > 1:
                    for pred2 in nx.ancestors(trace_graph, pred):
                        if len(neighbors) < max_neighbors:
                            neighbors.add(pred2)
            
            # 后继（影响的节点）
            for succ in trace_graph.successors(focus_node):
                if len(neighbors) < max_neighbors:
                    neighbors.add(succ)
    
    # 策略2: 基于时间序列相关性的邻居
    if len(neighbors) < max_neighbors:
        all_nodes = [col for col in df.columns if col != 'time' and col != focus_node]
        
        if focus_node in df.columns:
            focus_series = df[focus_node]
            correlations = {}
            
            for node in all_nodes:
                if node not in neighbors:
                    try:
                        corr = abs(focus_series.corr(df[node]))
                        if not pd.isna(corr):
                            correlations[node] = corr
                    except:
                        continue
            
            # 取相关性最高的节点
            sorted_nodes = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            for node, _ in sorted_nodes[:max_neighbors - len(neighbors)]:
                neighbors.add(node)
    
    # 策略3: 基于异常分数的邻居
    if anomaly_scores is not None and len(neighbors) < max_neighbors:
        high_anomaly_nodes = anomaly_scores.nlargest(max_neighbors).index.tolist()
        for node in high_anomaly_nodes:
            if node != focus_node and len(neighbors) < max_neighbors:
                neighbors.add(node)
    
    return list(neighbors)


def build_localized_graph(
    focus_node: str,
    neighbors: List[str],
    trace_graph: nx.DiGraph = None
) -> nx.DiGraph:
    """
    构建局部图（只包含焦点节点和邻居）
    
    Args:
        focus_node: 焦点节点
        neighbors: 邻居列表
        trace_graph: 完整trace图
    
    Returns:
        局部图
    """
    local_graph = nx.DiGraph()
    all_nodes = [focus_node] + neighbors
    local_graph.add_nodes_from(all_nodes)
    
    if trace_graph is not None:
        # 只保留局部节点之间的边
        for u in all_nodes:
            for v in all_nodes:
                if trace_graph.has_edge(u, v):
                    local_graph.add_edge(u, v, **trace_graph[u][v])
    
    return local_graph


def compute_localization_score(
    focus_node: str,
    candidate_nodes: List[str],
    df: pd.DataFrame,
    trace_graph: nx.DiGraph = None,
    anomaly_scores: pd.Series = None
) -> Dict[str, float]:
    """
    计算每个候选节点的局部化分数
    
    Args:
        focus_node: 焦点节点
        candidate_nodes: 候选节点列表
        df: 数据框
        trace_graph: trace图
        anomaly_scores: 异常分数
    
    Returns:
        节点到分数的映射
    """
    scores = {}
    
    for node in candidate_nodes:
        if node == focus_node:
            scores[node] = 1.0
            continue
        
        score = 0.0
        
        # 基于trace graph的距离分数
        if trace_graph is not None and focus_node in trace_graph and node in trace_graph:
            try:
                # 计算最短路径距离
                if nx.has_path(trace_graph, focus_node, node):
                    distance = nx.shortest_path_length(trace_graph, focus_node, node)
                    score += 1.0 / (1.0 + distance)  # 距离越近分数越高
                elif nx.has_path(trace_graph, node, focus_node):
                    distance = nx.shortest_path_length(trace_graph, node, focus_node)
                    score += 0.8 / (1.0 + distance)  # 反向路径分数稍低
            except:
                pass
        
        # 基于时间序列相关性的分数
        if focus_node in df.columns and node in df.columns:
            try:
                corr = abs(df[focus_node].corr(df[node]))
                if not pd.isna(corr):
                    score += corr * 0.5
            except:
                pass
        
        # 基于异常分数的分数
        if anomaly_scores is not None and node in anomaly_scores.index:
            score += anomaly_scores[node] * 0.3
        
        scores[node] = score
    
    return scores


def select_top_localized_nodes(
    focus_node: str,
    all_nodes: List[str],
    df: pd.DataFrame,
    config: PCMCIShapleyConfig,
    trace_graph: nx.DiGraph = None,
    anomaly_scores: pd.Series = None
) -> List[str]:
    """
    选择局部化分数最高的节点
    
    Args:
        focus_node: 焦点节点
        all_nodes: 所有节点
        df: 数据框
        config: 配置
        trace_graph: trace图
        anomaly_scores: 异常分数
    
    Returns:
        选中的节点列表
    """
    # 计算局部化分数
    scores = compute_localization_score(
        focus_node, all_nodes, df, trace_graph, anomaly_scores
    )
    
    # 按分数排序
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 选择top-k节点
    max_nodes = min(config.u_max, len(all_nodes))
    selected_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
    
    logger.info(f"Selected {len(selected_nodes)} localized nodes for focus_node={focus_node}")
    logger.info(f"Top-5 localized scores: {dict(sorted_nodes[:5])}")
    
    return selected_nodes


def validate_localization_inputs(
    focus_node: str,
    df: pd.DataFrame,
    trace_graph: nx.DiGraph = None
) -> Tuple[bool, str]:
    """
    验证局部化输入的合法性
    
    Args:
        focus_node: 焦点节点
        df: 数据框
        trace_graph: trace图
    
    Returns:
        (是否有效, 错误信息)
    """
    if not focus_node:
        return False, "focus_node is empty"
    
    if focus_node not in df.columns:
        return False, f"focus_node '{focus_node}' not found in dataframe columns"
    
    if trace_graph is not None and focus_node not in trace_graph:
        logger.warning(f"focus_node '{focus_node}' not found in trace_graph")
    
    return True, ""


def get_localization_statistics(
    focus_node: str,
    neighbors: List[str],
    trace_graph: nx.DiGraph = None
) -> Dict[str, int]:
    """
    获取局部化统计信息
    
    Args:
        focus_node: 焦点节点
        neighbors: 邻居列表
        trace_graph: trace图
    
    Returns:
        统计信息
    """
    stats = {
        'focus_node': focus_node,
        'n_neighbors': len(neighbors),
        'n_trace_connections': 0,
        'avg_trace_distance': 0.0
    }
    
    if trace_graph is not None and focus_node in trace_graph:
        distances = []
        for neighbor in neighbors:
            if neighbor in trace_graph:
                try:
                    if nx.has_path(trace_graph, focus_node, neighbor):
                        dist = nx.shortest_path_length(trace_graph, focus_node, neighbor)
                        distances.append(dist)
                        stats['n_trace_connections'] += 1
                except:
                    pass
        
        if distances:
            stats['avg_trace_distance'] = np.mean(distances)
    
    return stats
