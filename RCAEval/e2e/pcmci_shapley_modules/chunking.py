from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from .config import PCMCIShapleyConfig
import logging

logger = logging.getLogger(__name__)


def create_pcmci_chunks(
    columns: List[str], 
    chunk_size: int, 
    overlap: int = 1,
    shuffle: bool = True
) -> List[List[str]]:
    """
    创建分块，类似RCD的create_chunks
    
    Args:
        columns: 列名列表
        chunk_size: 每块大小
        overlap: 重叠节点数
        shuffle: 是否随机打乱（减少相邻节点偏差）
    
    Returns:
        分块列表
    """
    cols = columns.copy()
    if shuffle:
        np.random.shuffle(cols)
    
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(cols), step):
        chunk = cols[i:i + chunk_size]
        if len(chunk) > 0:
            chunks.append(chunk)
    
    return chunks


def create_smart_chunks(
    df: pd.DataFrame,
    columns: List[str],
    chunk_size: int,
    anomaly_scores: pd.Series = None,
    correlation_threshold: float = 0.5
) -> List[List[str]]:
    """
    智能分块：将高度相关或异常分数相近的节点分到同一块
    
    Args:
        df: 数据框
        columns: 列名
        chunk_size: 块大小
        anomaly_scores: 节点异常分数（用于优先级排序）
        correlation_threshold: 相关性阈值
    
    Returns:
        智能分块列表
    """
    # 计算相关性矩阵
    corr_matrix = df[columns].corr().abs()
    
    # 贪心分块算法
    chunks = []
    remaining = set(columns)
    
    while remaining:
        # 选择异常分数最高的节点作为种子
        if anomaly_scores is not None:
            seed = max(remaining, key=lambda x: anomaly_scores.get(x, 0))
        else:
            seed = list(remaining)[0]
        
        # 找到与种子高度相关的节点
        chunk = [seed]
        remaining.remove(seed)
        
        for node in sorted(remaining, key=lambda x: corr_matrix.loc[seed, x], reverse=True):
            if len(chunk) >= chunk_size:
                break
            if corr_matrix.loc[seed, node] > correlation_threshold:
                chunk.append(node)
                remaining.remove(node)
        
        # 填充到chunk_size
        while len(chunk) < chunk_size and remaining:
            chunk.append(remaining.pop())
        
        chunks.append(chunk)
    
    return chunks


def merge_chunk_results(
    chunk_results: List[Dict],
    columns: List[str]
) -> Dict:
    """
    合并多个分块的PCMCI结果
    
    Args:
        chunk_results: 每个分块的结果
        columns: 完整列名列表
    
    Returns:
        合并后的结果
    """
    # 创建列名到索引的映射
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    
    all_edges = []
    all_strengths = {}
    
    for result in chunk_results:
        chunk_cols = result['columns']
        chunk_col_to_idx = {col: idx for idx, col in enumerate(chunk_cols)}
        
        # 转换边索引到全局索引
        for (i, j, tau) in result['edges']:
            col_i = chunk_cols[i]
            col_j = chunk_cols[j]
            
            global_i = col_to_idx[col_i]
            global_j = col_to_idx[col_j]
            
            all_edges.append((global_i, global_j, tau))
            
            # 合并边强度（取最大值）
            key = (global_i, global_j)
            chunk_key = (i, j)
            if chunk_key in result['edge_strengths']:
                strength = result['edge_strengths'][chunk_key]
                all_strengths[key] = max(all_strengths.get(key, 0), strength)
    
    return {
        'edges': all_edges,
        'edge_strengths': all_strengths,
        'columns': columns,
        'strategy_used': 'chunked_merged'
    }


def validate_chunks(chunks: List[List[str]], min_chunk_size: int = 2) -> bool:
    """
    验证分块的有效性
    
    Args:
        chunks: 分块列表
        min_chunk_size: 最小块大小
    
    Returns:
        是否有效
    """
    if not chunks:
        return False
    
    for chunk in chunks:
        if len(chunk) < min_chunk_size:
            logger.warning(f"Chunk too small: {len(chunk)} < {min_chunk_size}")
            return False
    
    return True


def get_chunk_statistics(chunks: List[List[str]]) -> Dict[str, int]:
    """
    获取分块统计信息
    
    Args:
        chunks: 分块列表
    
    Returns:
        统计信息字典
    """
    if not chunks:
        return {'n_chunks': 0, 'avg_size': 0, 'min_size': 0, 'max_size': 0}
    
    sizes = [len(chunk) for chunk in chunks]
    
    return {
        'n_chunks': len(chunks),
        'avg_size': np.mean(sizes),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'total_nodes': sum(sizes)
    }
