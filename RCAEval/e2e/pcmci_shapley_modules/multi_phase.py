from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from .config import PCMCIShapleyConfig
import logging

logger = logging.getLogger(__name__)


def select_important_nodes(
    pcmci_result: Dict,
    all_nodes: List[str],
    focus_node: str = None,
    top_k: int = 15
) -> List[str]:
    """
    从PCMCI结果中选择重要节点（类似RCD的节点筛选）
    
    Args:
        pcmci_result: PCMCI结果
        all_nodes: 所有节点列表
        focus_node: 焦点节点（必须保留）
        top_k: 保留节点数
    
    Returns:
        重要节点列表
    """
    # 计算每个节点的重要性分数
    node_importance = {node: 0.0 for node in all_nodes}
    
    # 基于边的强度计算重要性
    edges = pcmci_result.get('edges', [])
    edge_strengths = pcmci_result.get('edge_strengths', {})
    columns = pcmci_result.get('columns', all_nodes)
    
    for (i, j, tau) in edges:
        if i < len(columns) and j < len(columns):
            node_i = columns[i]
            node_j = columns[j]
            
            strength = edge_strengths.get((i, j), 1.0)
            
            # 出边节点（原因）重要性更高
            node_importance[node_i] += strength * 2.0
            # 入边节点也有一定重要性
            node_importance[node_j] += strength * 1.0
    
    # 焦点节点必须保留
    if focus_node and focus_node in node_importance:
        node_importance[focus_node] = float('inf')
    
    # 选择top-k节点
    sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
    important_nodes = [node for node, _ in sorted_nodes[:top_k]]
    
    return important_nodes


def run_phase1_iteration(
    df: pd.DataFrame,
    candidate_nodes: List[str],
    config: PCMCIShapleyConfig,
    pcmci_runner,
    focus_node: str = None
) -> Tuple[List[str], Dict]:
    """
    执行Phase-1的一次迭代
    
    Args:
        df: 数据框
        candidate_nodes: 候选节点
        config: 配置
        pcmci_runner: PCMCI执行函数
        focus_node: 焦点节点
    
    Returns:
        (筛选后的节点, PCMCI结果)
    """
    logger.info(f"Phase-1 iteration: {len(candidate_nodes)} candidate nodes")
    
    # 在候选节点上运行PCMCI（使用Phase-1的大块）
    from . import chunking
    
    if config.enable_chunking and len(candidate_nodes) > config.phase1_chunk_size:
        # 分块运行
        chunks = chunking.create_pcmci_chunks(
            candidate_nodes, 
            config.phase1_chunk_size,
            overlap=1
        )
        
        chunk_results = []
        for chunk in chunks:
            try:
                result = pcmci_runner(df, chunk, config)
                chunk_results.append(result)
            except Exception as e:
                logger.warning(f"Phase-1 chunk failed: {e}")
                continue
        
        # 合并结果
        pcmci_result = chunking.merge_chunk_results(chunk_results, candidate_nodes)
    else:
        # 直接运行
        pcmci_result = pcmci_runner(df, candidate_nodes, config)
    
    # 选择重要节点
    important_nodes = select_important_nodes(
        pcmci_result, 
        candidate_nodes, 
        focus_node,
        top_k=config.phase2_node_limit
    )
    
    logger.info(f"Phase-1 selected: {len(important_nodes)} important nodes")
    
    return important_nodes, pcmci_result


def run_multi_phase_iteration(
    df: pd.DataFrame,
    candidate_nodes: List[str],
    config: PCMCIShapleyConfig,
    pcmci_runner,
    focus_node: str = None,
    iteration: int = 0
) -> Tuple[List[str], Dict, Dict]:
    """
    执行多阶段迭代的一次完整迭代
    
    Args:
        df: 数据框
        candidate_nodes: 候选节点
        config: 配置
        pcmci_runner: PCMCI执行函数
        focus_node: 焦点节点
        iteration: 当前迭代次数
    
    Returns:
        (筛选后的节点, PCMCI结果, 统计信息)
    """
    logger.info(f"Multi-phase iteration {iteration+1}: {len(candidate_nodes)} candidates")
    
    # 运行Phase-1迭代
    important_nodes, pcmci_result = run_phase1_iteration(
        df, candidate_nodes, config, pcmci_runner, focus_node
    )
    
    # 计算统计信息
    stats = {
        'iteration': iteration + 1,
        'input_nodes': len(candidate_nodes),
        'output_nodes': len(important_nodes),
        'reduction_ratio': len(important_nodes) / len(candidate_nodes) if candidate_nodes else 0,
        'edges_found': len(pcmci_result.get('edges', [])),
        'strategy_used': pcmci_result.get('strategy_used', 'unknown')
    }
    
    # 检查收敛条件
    converged = (
        len(important_nodes) == len(candidate_nodes) or  # 没有减少
        len(important_nodes) <= config.phase2_node_limit or  # 达到目标大小
        stats['reduction_ratio'] > 0.8  # 减少幅度太小
    )
    
    stats['converged'] = converged
    
    if converged:
        logger.info(f"Multi-phase converged at iteration {iteration+1}")
    
    return important_nodes, pcmci_result, stats


def compute_node_importance_scores(
    pcmci_result: Dict,
    all_nodes: List[str],
    focus_node: str = None
) -> Dict[str, float]:
    """
    计算节点重要性分数（用于排序和筛选）
    
    Args:
        pcmci_result: PCMCI结果
        all_nodes: 所有节点
        focus_node: 焦点节点
    
    Returns:
        节点到分数的映射
    """
    scores = {node: 0.0 for node in all_nodes}
    
    edges = pcmci_result.get('edges', [])
    edge_strengths = pcmci_result.get('edge_strengths', {})
    columns = pcmci_result.get('columns', all_nodes)
    
    # 基于边的出度和入度计算分数
    for (i, j, tau) in edges:
        if i < len(columns) and j < len(columns):
            node_i = columns[i]
            node_j = columns[j]
            
            strength = edge_strengths.get((i, j), 1.0)
            
            # 出边节点（原因）得分更高
            scores[node_i] += strength * 2.0
            # 入边节点也有一定得分
            scores[node_j] += strength * 1.0
    
    # 焦点节点额外加分
    if focus_node and focus_node in scores:
        scores[focus_node] += 10.0
    
    return scores


def validate_multi_phase_inputs(
    df: pd.DataFrame,
    candidate_nodes: List[str],
    config: PCMCIShapleyConfig
) -> Tuple[bool, str]:
    """
    验证多阶段迭代输入的合法性
    
    Args:
        df: 数据框
        candidate_nodes: 候选节点
        config: 配置
    
    Returns:
        (是否有效, 错误信息)
    """
    if len(candidate_nodes) < 2:
        return False, f"Insufficient candidate nodes: {len(candidate_nodes)}"
    
    if len(candidate_nodes) <= config.phase2_node_limit:
        return False, f"Candidate nodes ({len(candidate_nodes)}) already <= phase2_node_limit ({config.phase2_node_limit})"
    
    # 检查节点是否都在数据框中
    missing_nodes = [node for node in candidate_nodes if node not in df.columns]
    if missing_nodes:
        return False, f"Missing nodes in dataframe: {missing_nodes}"
    
    return True, ""


def get_multi_phase_statistics(
    phase_results: List[Dict],
    final_nodes: List[str],
    initial_nodes: List[str]
) -> Dict[str, any]:
    """
    获取多阶段迭代的统计信息
    
    Args:
        phase_results: 各阶段结果
        final_nodes: 最终节点
        initial_nodes: 初始节点
    
    Returns:
        统计信息
    """
    if not phase_results:
        return {'n_iterations': 0, 'total_reduction': 0, 'avg_reduction': 0}
    
    total_edges = sum(result.get('edges_found', 0) for result in phase_results)
    total_reduction = len(initial_nodes) - len(final_nodes)
    avg_reduction = total_reduction / len(phase_results) if phase_results else 0
    
    strategies_used = [result.get('strategy_used', 'unknown') for result in phase_results]
    strategy_counts = {strategy: strategies_used.count(strategy) for strategy in set(strategies_used)}
    
    return {
        'n_iterations': len(phase_results),
        'initial_nodes': len(initial_nodes),
        'final_nodes': len(final_nodes),
        'total_reduction': total_reduction,
        'reduction_ratio': total_reduction / len(initial_nodes) if initial_nodes else 0,
        'avg_reduction_per_iteration': avg_reduction,
        'total_edges_found': total_edges,
        'strategy_counts': strategy_counts,
        'converged': phase_results[-1].get('converged', False) if phase_results else False
    }
