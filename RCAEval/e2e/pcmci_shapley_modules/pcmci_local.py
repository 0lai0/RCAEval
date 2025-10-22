from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from tigramite import data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from .config import PCMCIShapleyConfig
from .utils import smart_fillna_matrix
import logging

logger = logging.getLogger(__name__)


def validate_and_clean_for_pcmci(df: pd.DataFrame, columns: List[str], config: PCMCIShapleyConfig) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    在 PCMCI 執行前進行嚴格的數據驗證和清理
    
    Args:
        df: 輸入 DataFrame
        columns: 要檢查的列名列表
        config: 配置對象
        
    Returns:
        Tuple of (cleaned_df, valid_columns, diagnostics)
    """
    diagnostics = {
        'removed_constant': [],
        'removed_low_variance': [],
        'removed_collinear': [],
        'removed_high_nan': [],  # 新增
        'nan_handled': 0,
        'input_shape': df.shape,
        'output_shape': None
    }
    
    # 確保只檢查指定的列
    if columns:
        df_subset = df[columns].copy()
    else:
        df_subset = df.copy()
    
    logger.info(f"Data validation input: shape={df_subset.shape}, columns={len(columns)}")
    
    # 1. 檢測並移除常數列（方差 < 1e-9）
    variances = df_subset.var()
    constant_cols = variances[variances < 1e-9].index.tolist()
    diagnostics['removed_constant'] = constant_cols
    df_clean = df_subset.drop(columns=constant_cols)
    
    if constant_cols:
        logger.warning(f"Removed constant columns: {constant_cols}")
    
    # 2. 檢測並移除低變異列（變異係數 < 0.01）
    if len(df_clean.columns) > 0:
        cv = df_clean.std() / (df_clean.mean().abs() + 1e-10)
        low_var_cols = cv[cv < 0.01].index.tolist()
        diagnostics['removed_low_variance'] = low_var_cols
        df_clean = df_clean.drop(columns=low_var_cols)
        
        if low_var_cols:
            logger.warning(f"Removed low variance columns: {low_var_cols}")
    
    # 3. 處理 NaN 值（前向填充 + 後向填充 + 零填充）
    nan_count = df_clean.isna().sum().sum()
    diagnostics['nan_handled'] = nan_count
    
    if nan_count > 0:
        logger.info(f"Handling {nan_count} NaN values")
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 3.5. 額外檢查：確保沒有 NaN 或 Inf
    if df_clean.isna().any().any():
        logger.warning("Still have NaN values after cleaning, filling with 0")
        df_clean = df_clean.fillna(0)
    
    if np.isinf(df_clean).any().any():
        logger.warning("Found infinite values, replacing with finite values")
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 4. 檢測共線性（相關係數 > 0.99）
    if len(df_clean.columns) > 1:
        corr_matrix = df_clean.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        collinear_cols = [
            col for col in upper_tri.columns 
            if any(upper_tri[col] > 0.99)
        ]
        # 只移除一半的共線性列，避免過度清理
        diagnostics['removed_collinear'] = collinear_cols[:len(collinear_cols)//2]
        df_clean = df_clean.drop(columns=diagnostics['removed_collinear'])
        
        if diagnostics['removed_collinear']:
            logger.warning(f"Removed collinear columns: {diagnostics['removed_collinear']}")
    
    # 新增：移除NaN过多的列（>30%）
    nan_threshold = 0.3
    nan_ratio = df_clean.isna().sum() / len(df_clean)
    high_nan_cols = nan_ratio[nan_ratio > nan_threshold].index.tolist()
    diagnostics['removed_high_nan'] = high_nan_cols
    df_clean = df_clean.drop(columns=high_nan_cols)
    
    if high_nan_cols:
        logger.warning(f"Removed high-NaN columns: {high_nan_cols}")
    
    # 新增：更积极的共线性移除
    if len(df_clean.columns) > 1:
        corr_matrix = df_clean.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        # 大幅降低阈值到0.85，移除近似共线
        collinear_cols = [
            col for col in upper_tri.columns 
            if any(upper_tri[col] > 0.85)
        ]
        diagnostics['removed_collinear'] = collinear_cols
        df_clean = df_clean.drop(columns=collinear_cols)
        
        if collinear_cols:
            logger.warning(f"Removed collinear columns (>0.85): {collinear_cols}")
    
    # 5. 最終檢查：確保數據適合 PCMCI
    if len(df_clean.columns) > 0:
        # 檢查每列的變異性
        final_variances = df_clean.var()
        still_constant = final_variances[final_variances < 1e-10].index.tolist()
        if still_constant:
            logger.warning(f"Still have constant columns after cleaning: {still_constant}")
            df_clean = df_clean.drop(columns=still_constant)
            diagnostics['removed_constant'].extend(still_constant)
        
        # 檢查數據範圍和數值穩定性
        problematic_cols = []
        for col in df_clean.columns:
            col_data = df_clean[col]
            
            # 檢查標準差
            if col_data.std() == 0:
                logger.warning(f"Column {col} has zero standard deviation")
                problematic_cols.append(col)
            elif np.isnan(col_data.std()) or np.isinf(col_data.std()):
                logger.warning(f"Column {col} has invalid standard deviation")
                problematic_cols.append(col)
            
            # 檢查數據範圍（避免極大值）
            col_range = col_data.max() - col_data.min()
            if col_range > 1e6:  # 降低閾值到100萬
                logger.warning(f"Column {col} has extreme range: {col_range:.2e}")
                
                # 使用更激進的縮放策略
                if col_range > 1e8:  # 極大值：使用對數縮放
                    logger.warning(f"Applying log scaling to column {col}")
                    # 先平移數據使其為正
                    min_val = col_data.min()
                    if min_val <= 0:
                        df_clean[col] = col_data - min_val + 1e-6
                    else:
                        df_clean[col] = col_data
                    # 對數縮放
                    df_clean[col] = np.log1p(df_clean[col])
                else:  # 中等極值：使用標準化縮放
                    df_clean[col] = df_clean[col] / (col_data.std() + 1e-10)
                
                logger.info(f"Scaled column {col} to reduce range")
                
                # 檢查縮放後的效果
                new_range = df_clean[col].max() - df_clean[col].min()
                logger.info(f"Column {col} range after scaling: {new_range:.2e}")
            
            # 檢查是否有異常的數值
            if np.any(np.isinf(col_data)) or np.any(np.isnan(col_data)):
                logger.warning(f"Column {col} contains inf or nan values")
                problematic_cols.append(col)
        
        # 移除有問題的列
        if problematic_cols:
            df_clean = df_clean.drop(columns=problematic_cols)
            diagnostics['removed_constant'].extend(problematic_cols)
            logger.warning(f"Removed problematic columns: {problematic_cols}")
        
        # 最終數據質量檢查
        if len(df_clean.columns) > 0:
            # 確保沒有inf或nan
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
            df_clean = df_clean.fillna(0)
            
            # 檢查協方差矩陣的條件
            try:
                cov_matrix = df_clean.cov()
                if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                    logger.warning("Covariance matrix contains invalid values")
                    # 添加小的正則化項
                    cov_matrix = cov_matrix + np.eye(len(df_clean.columns)) * 1e-6
                    logger.info("Added regularization to covariance matrix")
            except Exception as e:
                logger.warning(f"Covariance matrix computation failed: {e}")
    
    diagnostics['output_shape'] = df_clean.shape
    valid_columns = df_clean.columns.tolist()
    
    logger.info(f"Data validation output: shape={df_clean.shape}, valid_columns={len(valid_columns)}")
    
    return df_clean, valid_columns, diagnostics


def run_chunked_pcmci(
    df: pd.DataFrame, 
    columns: List[str], 
    config: PCMCIShapleyConfig
) -> Dict[str, Any]:
    """
    分块执行PCMCI，类似RCD的分块策略
    
    复杂度: O(n_chunks × γ² × τ_max × T) ≈ O(γ × |U| × τ_max × T)
    其中 n_chunks = |U| / γ, γ = chunk_size
    """
    logger.info(f"Starting chunked PCMCI with {len(columns)} columns, chunk_size={config.chunk_size}")
    
    # 1. 数据清理
    df_clean, valid_columns, diagnostics = validate_and_clean_for_pcmci(df, columns, config)
    
    if len(valid_columns) < 2:
        return {
            'edges': [],
            'edge_strengths': {},
            'columns': valid_columns,
            'strategy_used': 'empty',
            'diagnostics': diagnostics
        }
    
    # 2. 创建分块
    if config.enable_smart_chunking:
        from . import chunking
        # 获取异常分数（如果有）
        anomaly_scores = None
        chunks = chunking.create_smart_chunks(
            df_clean, 
            valid_columns, 
            config.chunk_size,
            anomaly_scores
        )
        logger.info(f"Created {len(chunks)} smart chunks")
    else:
        from . import chunking
        chunks = chunking.create_pcmci_chunks(
            valid_columns, 
            config.chunk_size,
            overlap=config.chunk_overlap
        )
        logger.info(f"Created {len(chunks)} random chunks")
    
    # 3. 并行执行每个分块
    chunk_results = []
    for idx, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {idx+1}/{len(chunks)}: {len(chunk)} nodes")
        
        try:
            # 在分块上运行标准PCMCI
            chunk_result = run_pcmci_with_progressive_fallback(
                df_clean, 
                chunk, 
                config
            )
            chunk_results.append(chunk_result)
            
            logger.info(f"Chunk {idx+1} completed: {len(chunk_result['edges'])} edges found")
            
        except Exception as e:
            logger.warning(f"Chunk {idx+1} failed: {e}, continuing...")
            continue
    
    # 4. 合并分块结果
    if not chunk_results:
        logger.warning("All chunks failed, returning empty result")
        return {
            'edges': [],
            'edge_strengths': {},
            'columns': valid_columns,
            'strategy_used': 'chunked_failed',
            'diagnostics': diagnostics
        }
    
    from . import chunking
    merged_result = chunking.merge_chunk_results(chunk_results, valid_columns)
    merged_result['diagnostics'] = diagnostics
    merged_result['strategy_used'] = 'chunked_pcmci'
    merged_result['n_chunks'] = len(chunks)
    merged_result['successful_chunks'] = len(chunk_results)
    
    logger.info(f"Chunked PCMCI completed: {len(merged_result['edges'])} total edges from {len(chunk_results)} chunks")
    
    return merged_result


def run_localized_pcmci(
    df: pd.DataFrame,
    columns: List[str],
    focus_node: str,
    config: PCMCIShapleyConfig,
    trace_graph: nx.DiGraph = None,
    anomaly_scores: pd.Series = None
) -> Dict[str, Any]:
    """
    局部化PCMCI：只关注焦点节点的局部邻域
    
    类似RCD的local_skeleton_discovery
    """
    logger.info(f"Starting localized PCMCI with focus_node={focus_node}")
    
    # 1. 识别焦点节点的邻居
    from . import localization
    
    neighbors = localization.identify_focus_neighbors(
        df,
        focus_node,
        radius=config.localization_radius,
        trace_graph=trace_graph if config.use_trace_for_localization else None,
        anomaly_scores=anomaly_scores,
        max_neighbors=config.u_max - 1  # 减去focus_node本身
    )
    
    logger.info(f"Identified {len(neighbors)} neighbors for focus_node")
    
    # 2. 构建局部变量集
    local_vars = [focus_node] + neighbors
    local_vars = [v for v in local_vars if v in columns]
    
    if len(local_vars) < 2:
        logger.warning(f"Insufficient local variables: {len(local_vars)}")
        return {
            'edges': [],
            'edge_strengths': {},
            'columns': local_vars,
            'strategy_used': 'localized_insufficient'
        }
    
    logger.info(f"Local variable set size: {len(local_vars)}")
    
    # 3. 在局部集上运行PCMCI
    if config.enable_chunking and len(local_vars) > config.chunk_size:
        # 局部集仍然较大，继续分块
        result = run_chunked_pcmci(df, local_vars, config)
        result['strategy_used'] = 'localized_chunked'
    else:
        # 直接运行
        result = run_pcmci_with_progressive_fallback(df, local_vars, config)
        result['strategy_used'] = 'localized'
    
    result['focus_node'] = focus_node
    result['n_neighbors'] = len(neighbors)
    
    logger.info(f"Localized PCMCI completed: {len(result['edges'])} edges")
    
    return result


def run_multi_phase_pcmci(
    df: pd.DataFrame,
    columns: List[str],
    config: PCMCIShapleyConfig,
    focus_node: str = None
) -> Dict[str, Any]:
    """
    多阶段PCMCI：Phase-1快速筛选 + Phase-2精细分析
    
    类似RCD的run_multi_phase
    """
    logger.info(f"Starting multi-phase PCMCI with {len(columns)} columns")
    
    from . import multi_phase
    
    candidate_nodes = columns.copy()
    prev_size = len(candidate_nodes)
    phase1_results = []
    
    # Phase-1: 迭代收缩候选集
    for iteration in range(config.phase1_max_iterations):
        logger.info(f"Phase-1 iteration {iteration+1}: {len(candidate_nodes)} candidates")
        
        # 运行一次Phase-1迭代
        important_nodes, iter_result = multi_phase.run_phase1_iteration(
            df,
            candidate_nodes,
            config,
            run_pcmci_with_progressive_fallback,
            focus_node
        )
        
        phase1_results.append(iter_result)
        
        # 检查收敛
        if len(important_nodes) == prev_size or len(important_nodes) <= config.phase2_node_limit:
            logger.info(f"Phase-1 converged at iteration {iteration+1}")
            candidate_nodes = important_nodes
            break
        
        candidate_nodes = important_nodes
        prev_size = len(candidate_nodes)
    
    logger.info(f"Phase-1 completed: {len(candidate_nodes)} nodes selected for Phase-2")
    
    # Phase-2: 在最终候选集上运行精细PCMCI
    logger.info("Starting Phase-2: fine-grained PCMCI")
    
    # Phase-2使用更严格的参数
    phase2_config = PCMCIShapleyConfig(**config.__dict__)
    phase2_config.tau_max = min(config.tau_max, 5)  # 允许更大的tau
    phase2_config.pcmci_alpha = min(config.pcmci_alpha, 0.05)  # 更严格的alpha
    phase2_config.pcmci_max_conds_dim = min(config.pcmci_max_conds_dim or 3, 3)
    phase2_config.enable_chunking = False  # Phase-2不再分块
    
    phase2_result = run_pcmci_with_progressive_fallback(
        df,
        candidate_nodes,
        phase2_config
    )
    
    phase2_result['strategy_used'] = 'multi_phase'
    phase2_result['phase1_iterations'] = len(phase1_results)
    phase2_result['phase1_results'] = phase1_results
    phase2_result['initial_nodes'] = len(columns)
    phase2_result['final_nodes'] = len(candidate_nodes)
    
    logger.info(f"Multi-phase PCMCI completed: {len(phase2_result['edges'])} edges")
    
    return phase2_result


def run_adaptive_pcmci(
    df: pd.DataFrame,
    columns: List[str],
    config: PCMCIShapleyConfig,
    focus_node: str = None,
    trace_graph: nx.DiGraph = None,
    anomaly_scores: pd.Series = None
) -> Dict[str, Any]:
    """
    自适应PCMCI：根据数据规模自动选择最优策略
    
    决策树:
    - |U| <= 10: 标准PCMCI
    - 10 < |U| <= 15 且有focus_node: 局部化PCMCI
    - 15 < |U| <= 30 且enable_chunking: 分块PCMCI
    - |U| > 30: 多阶段PCMCI
    """
    n_nodes = len(columns)
    logger.info(f"Adaptive PCMCI: {n_nodes} nodes, selecting strategy...")
    
    # 决策逻辑
    if n_nodes <= 10:
        # 小规模：直接运行标准PCMCI
        logger.info("Strategy: STANDARD (small scale)")
        return run_pcmci_with_progressive_fallback(df, columns, config)
    
    elif n_nodes <= 15 and config.enable_localization and focus_node:
        # 中小规模 + 有焦点：局部化PCMCI
        logger.info("Strategy: LOCALIZED (medium scale with focus)")
        return run_localized_pcmci(df, columns, focus_node, config, trace_graph, anomaly_scores)
    
    elif n_nodes <= 30 and config.enable_chunking:
        # 中等规模：分块PCMCI
        logger.info("Strategy: CHUNKED (medium-large scale)")
        return run_chunked_pcmci(df, columns, config)
    
    else:
        # 大规模：多阶段PCMCI
        logger.info("Strategy: MULTI-PHASE (large scale)")
        return run_multi_phase_pcmci(df, columns, config, focus_node)


def run_pcmci_with_progressive_fallback(df: pd.DataFrame, columns: List[str], config: PCMCIShapleyConfig) -> Dict[str, Any]:
    """
    實現漸進式降級策略的 PCMCI 執行器
    
    策略優先序:
    1. 標準 PCMCI+ (alpha=0.05, tau_max=5, max_conds_dim=3)
    2. 寬鬆 PCMCI (alpha=0.1, tau_max=3, max_conds_dim=2)
    3. 極寬鬆 PCMCI (alpha=0.2, tau_max=2, max_conds_dim=1)
    4. 簡單滯後相關 (Pearson + 顯著性檢定)
    5. 空圖（返回空邊集合但保留節點）
    
    Args:
        df: 輸入 DataFrame
        columns: 列名列表
        config: 配置對象
        
    Returns:
        PCMCI 結果字典，包含 strategy_used 字段
    """
    # 首先進行數據驗證和清理
    df_clean, valid_columns, diagnostics = validate_and_clean_for_pcmci(df, columns, config)
    
    if len(valid_columns) < 2:
        logger.warning(f"Insufficient valid columns ({len(valid_columns)}), returning empty result")
        return {
            'edges': [],
            'edge_strengths': {},
            'columns': valid_columns,
            'strategy_used': 'empty',
            'diagnostics': diagnostics
        }
    
    # 構建時間序列矩陣
    X, cols = build_timeseries_matrix(df_clean, valid_columns, use_pca=config.use_pca, pca_components=config.pca_components)
    
    # 新增：對數據進行標準化以提升PCMCI穩定性
    if X.size > 0:
        logger.info(f"Applying standardization to improve PCMCI stability")
        # 對每個變量進行標準化
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True)
        # 避免除零
        X_std = np.where(X_std < 1e-10, 1.0, X_std)
        X = (X - X_mean) / X_std
        
        logger.info(f"Standardized data range: min={X.min():.4f}, max={X.max():.4f}")
        logger.info(f"Standardized data std: min={X.std(axis=1).min():.4f}, max={X.std(axis=1).max():.4f}")
    
    if X.size == 0:
        logger.warning("Empty time series matrix, returning empty result")
        return {
            'edges': [],
            'edge_strengths': {},
            'columns': valid_columns,
            'strategy_used': 'empty',
            'diagnostics': diagnostics
        }
    
    # 定義策略序列
    strategies = [
        {'name': 'standard', 'alpha': 0.05, 'tau_max': 5, 'max_conds_dim': 3},
        {'name': 'relaxed', 'alpha': 0.1, 'tau_max': 3, 'max_conds_dim': 2},
        {'name': 'very_relaxed', 'alpha': 0.2, 'tau_max': 2, 'max_conds_dim': 1},
    ]
    
    # 嘗試每個策略
    for strategy in strategies:
        try:
            logger.info(f"Trying PCMCI strategy: {strategy['name']}")
            report = run_pcmci_plus(
                X, 
                tau_max=strategy['tau_max'],
                alpha=strategy['alpha'],
                max_conds_dim=strategy['max_conds_dim']
            )
            
            # 提取邊和強度
            edges = extract_significant_edges(report, alpha=strategy['alpha'])
            strengths = compute_edge_strength(report, edges)
            
            logger.info(f"PCMCI succeeded with strategy: {strategy['name']}, found {len(edges)} edges")
            logger.info(f"Edge strength range: min={min(strengths.values()) if strengths else 0:.4f}, max={max(strengths.values()) if strengths else 0:.4f}")
            logger.info(f"Significant edges: {edges[:5] if edges else 'None'}")
            
            return {
                'edges': edges,
                'edge_strengths': strengths,
                'columns': cols,
                'strategy_used': strategy['name'],
                'diagnostics': diagnostics,
                'pcmci_report': report
            }
            
        except Exception as e:
            logger.warning(f"PCMCI strategy {strategy['name']} failed: {e}")
            
            # 提供更詳細的錯誤診斷
            if "negative dimensions" in str(e):
                logger.error("Negative dimensions error - likely data quality issue")
                logger.error(f"Data shape: {X.shape}, columns: {len(cols)}")
                logger.error(f"Data range: min={X.min():.2e}, max={X.max():.2e}")
                logger.error(f"Data std: min={X.std(axis=0).min():.2e}, max={X.std(axis=0).max():.2e}")
                
                # 嘗試修復數據問題
                try:
                    # 檢查是否有零方差列
                    zero_var_cols = np.where(X.std(axis=0) < 1e-10)[0]
                    if len(zero_var_cols) > 0:
                        logger.warning(f"Found {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
                        # 移除零方差列
                        valid_indices = np.where(X.std(axis=0) >= 1e-10)[0]
                        if len(valid_indices) >= 2:
                            X = X[:, valid_indices]
                            cols = [cols[i] for i in valid_indices]
                            logger.info(f"Removed zero-variance columns, new shape: {X.shape}")
                            continue  # 重試當前策略
                    
                    # 檢查是否有極大值
                    if X.max() > 1e6:
                        logger.warning("Data contains extreme values, applying scaling")
                        X = X / (X.std(axis=0) + 1e-10)
                        logger.info("Applied scaling to reduce extreme values")
                        continue  # 重試當前策略
                        
                except Exception as fix_e:
                    logger.error(f"Data repair failed: {fix_e}")
            
            continue
    
    # 所有 PCMCI 策略都失敗，降級到簡單相關性
    logger.warning("All PCMCI strategies failed, using lagged correlation")
    return fallback_to_lagged_correlation(df_clean, valid_columns, config)


def fallback_to_lagged_correlation(df: pd.DataFrame, columns: List[str], config: PCMCIShapleyConfig) -> Dict[str, Any]:
    """
    使用滯後相關性作為因果關係的代理
    保持與 PCMCI 相同的輸出格式
    
    Args:
        df: 輸入 DataFrame
        columns: 列名列表
        config: 配置對象
        
    Returns:
        與 PCMCI 格式一致的結果字典
    """
    edges = []
    edge_strengths = {}
    
    logger.info(f"Using lagged correlation fallback for {len(columns)} columns")
    
    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            if i == j:
                continue
            
            # 計算滯後相關性
            max_corr = 0.0
            best_tau = 0
            
            for tau in range(1, min(config.tau_max + 1, len(df))):
                if len(df) <= tau:
                    continue
                
                x = df[col_i].values[:-tau]
                y = df[col_j].values[tau:]
                
                if len(x) > 0 and len(y) > 0 and np.std(x) > 0 and np.std(y) > 0:
                    try:
                        corr = abs(np.corrcoef(x, y)[0, 1])
                        if corr > max_corr:
                            max_corr = corr
                            best_tau = tau
                    except:
                        continue
            
            # 使用閾值篩選（相當於 alpha 檢定）
            threshold = 0.3  # 可調整
            if max_corr > threshold:
                edges.append((i, j, best_tau))
                edge_strengths[(i, j)] = max_corr
    
    logger.info(f"Lagged correlation found {len(edges)} edges")
    
    return {
        'edges': edges,
        'edge_strengths': edge_strengths,
        'columns': columns,
        'strategy_used': 'lagged_correlation',
        'diagnostics': {'method': 'lagged_correlation', 'threshold': 0.3}
    }


def build_timeseries_matrix(data: pd.DataFrame, local_nodes: List[str], use_pca: bool = False, pca_components: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Build time series matrix with intelligent missing value handling.
    
    Args:
        data: Input DataFrame
        local_nodes: List of local node names
        use_pca: Whether to use PCA (not implemented yet)
        pca_components: Number of PCA components
        
    Returns:
        Tuple of (matrix, column_names)
    """
    # Use columns matching local_nodes prefixes (service or exact names)
    cols: List[str] = []
    for s in local_nodes:
        matched = [c for c in data.columns if c != "time" and (c == s or c.startswith(f"{s}_"))]
        cols.extend(matched)
    cols = list(dict.fromkeys(cols))
    
    if not cols:
        # 如果沒有匹配的列，返回空矩陣
        return np.array([]).reshape(0, 0), []
    
    X = data[cols].to_numpy(dtype=float)
    
    # 使用智能插值處理缺失值
    X = smart_fillna_matrix(X, method='forward_backward')
    
    # Arrange to (variables, time)
    X = X.T
    return X, cols


def run_pcmci_plus(X: np.ndarray, tau_max: int, alpha: float, 
                   max_conds_dim: int | None = None) -> Dict[str, Any]:
    """
    Run PCMCI+ algorithm with improved missing value handling.
    
    Args:
        X: Input matrix (variables, time)
        tau_max: Maximum time lag
        alpha: Significance level
        max_conds_dim: Maximum condition set dimension (None means no limit)
    """
    # 檢查輸入
    if X.size == 0:
        raise ValueError("Empty input matrix")
    
    # 數據預處理：移除常數列
    X_clean = X.copy()
    
    # 移除常數列（標準差為0），但使用更寬鬆的閾值
    std_mask = np.std(X_clean, axis=1) > 1e-8  # 更寬鬆的閾值
    X_clean = X_clean[std_mask, :]
    
    # 檢查是否還有足夠的變量
    if X_clean.shape[0] < 2:
        raise ValueError(f"Not enough variables after cleaning: {X_clean.shape[0]} variables remaining")
    
    # 檢查時間序列長度
    if X_clean.shape[1] < tau_max + 2:
        raise ValueError(f"Time series too short: {X_clean.shape[1]} < {tau_max + 2}")
    
    # 創建 DataFrame 並運行 PCMCI
    dataframe = data_processing.DataFrame(X_clean)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    
    # 動態調整 max_conds_dim
    if max_conds_dim is None:
        max_conds_dim_actual = None
    else:
        # 確保不超過變量數
        max_conds_dim_actual = min(max_conds_dim, X_clean.shape[0] - 2)
        # 確保至少為 1
        max_conds_dim_actual = max(1, max_conds_dim_actual)
    
    try:
        report = pcmci.run_pcmci(
            tau_max=tau_max, 
            pc_alpha=alpha, 
            max_conds_dim=max_conds_dim_actual,
            max_conds_py=None,  # 可選：限制 Y 的條件集
            max_conds_px=None   # 可選：限制 X 的條件集
        )
    except Exception as e:
        # 如果 PCMCI 失敗，返回空結果
        print(f"PCMCI failed: {e}")
        return {
            "p_matrix": np.ones((X_clean.shape[0], X_clean.shape[0], tau_max + 1)),
            "val_matrix": np.zeros((X_clean.shape[0], X_clean.shape[0], tau_max + 1)),
            "graph": np.zeros((X_clean.shape[0], X_clean.shape[0])),
        }
    
    return report


def extract_significant_edges(report: Dict[str, Any], alpha: float) -> List[Tuple[int, int, int]]:
    pmat = report["p_matrix"]  # shape: (cause, effect, tau)
    C, E, T = pmat.shape
    edges: List[Tuple[int, int, int]] = []
    for i in range(C):
        for j in range(E):
            if i == j:
                continue
            for tau in range(1, T):
                if pmat[i, j, tau] <= alpha:
                    edges.append((i, j, tau))
    return edges


def compute_edge_strength(report: Dict[str, Any], edges: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], float]:
    val = report.get("val_matrix", None)
    strengths: Dict[Tuple[int, int], float] = {}
    if val is None:
        # fallback: use inverse p-value proxy
        pmat = report["p_matrix"]
        for (i, j, _) in edges:
            p = np.nanmin(pmat[i, j, 1:])
            strengths[(i, j)] = float(max(0.0, min(1.0, 1.0 - p))) if np.isfinite(p) else 0.0
        return strengths
    C, E, T = val.shape
    for (i, j, _) in edges:
        s = np.nanmax(np.abs(val[i, j, 1:]))
        strengths[(i, j)] = float(0.0 if not np.isfinite(s) else min(1.0, max(0.0, s)))
    return strengths


def local_pcmci_causal_test(data: pd.DataFrame, local_nodes: List[str], config: PCMCIShapleyConfig) -> Dict[str, Any]:
    X, cols = build_timeseries_matrix(data, local_nodes, use_pca=config.use_pca, pca_components=config.pca_components)
    report = run_pcmci_plus(
        X, 
        tau_max=config.tau_max, 
        alpha=config.pcmci_alpha,
        max_conds_dim=config.pcmci_max_conds_dim  # 傳遞參數
    )
    edges = extract_significant_edges(report, alpha=config.pcmci_alpha)
    strengths = compute_edge_strength(report, edges)
    # Build graph in variable-space (cols), later map back to services if needed
    G = nx.DiGraph()
    G.add_nodes_from(range(len(cols)))
    for (i, j, _tau) in edges:
        G.add_edge(i, j, weight=strengths.get((i, j), 0.0))
    return {
        "edges": edges,
        "edge_strengths": strengths,
        "pcmci_graph": G,
        "columns": cols,
    }
