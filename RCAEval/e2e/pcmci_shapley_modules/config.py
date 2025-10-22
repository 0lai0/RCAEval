from dataclasses import dataclass
from typing import Tuple, Union, Optional


@dataclass
class PCMCIShapleyConfig:
    # Node Isolation
    tau_max: int = 5
    top_m1: int = 60
    top_m2: int = 30
    u_max: int = 40

    # PCMCI
    pcmci_alpha: float = 0.05
    use_pca: bool = False
    pca_components: int = 10

    # Edge Fusion
    theta1: float = 0.6
    theta2: float = 0.3
    theta3: float = 0.1
    gamma: float = 0.5

    # Propagation
    K: int = 5
    alpha_prop: float = 0.85

    # Shapley
    shapley_method: str = "auto"  # auto, exact, sampling
    sampling_rounds: int = 500

    # Scoring
    score_alpha1: float = 0.5
    score_alpha2: float = 0.3
    score_alpha3: float = 0.2
    lambda_penalty: float = 1.0

    # Preprocessing
    anomaly_threshold: float = 3.0
    anomaly_method: str = "zscore"  # zscore or spot
    
    # SPOT 極值理論參數（新增）
    enable_spot: bool = True
    spot_risk_param: float = 0.01  # q 參數：0.001 表示捕捉 0.1% 的極值
    spot_init_window: int = 200  # 初始訓練窗口大小
    spot_depth: int = 10  # 極值池深度
    
    # 聯合篩選器參數（新增）
    joint_screener_enabled: bool = True
    joint_weight_fallback: float = 0.5  # Fallback 異常分數權重
    joint_weight_spot: float = 0.5  # SPOT 罕見度權重
    joint_top_n: int = 20  # 篩選後保留的節點數
    
    # Fallback 優化參數（新增）
    fallback_on_sparse_graph: bool = True  # 圖稀疏時是否觸發 Fallback
    fallback_sparse_threshold: float = 0.5  # 邊數/節點數 < 此值視為稀疏
    
    # 混合模式參數（新增）
    enable_hybrid_mode: bool = True
    hybrid_confidence_threshold: float = 0.7  # > 此值用 PCMCI，< 1-此值用 Fallback
    
    # 增強異常分數參數（新增）
    enhanced_anomaly_weights: tuple = (0.4, 0.3, 0.2, 0.1)  # (即時, 持續, 嚴重, 變化)

    # Pruning (新增)
    enable_pruning: bool = True
    pruning_max_hops: int = 2
    pruning_anomaly_percentile: float = 0.3
    pruning_min_nodes: int = 10

    # PCMCI 優化 (新增)
    pcmci_max_conds_dim: Optional[int] = 3  # None 表示不限制
    pcmci_max_conds_py: Optional[int] = None
    pcmci_max_conds_px: Optional[int] = None

    # Phase 2: 平行化 (新增)
    enable_parallel: bool = True
    node_isolation_n_jobs: int = -1  # -1 表示使用所有核心
    shapley_n_jobs: int = -1  # -1 表示使用所有核心

    # Phase 2: Adaptive Sampling (新增)
    adaptive_max_rounds: int = 2000
    adaptive_min_rounds: int = 100
    adaptive_confidence: float = 0.95  # 95% 或 99%
    adaptive_tolerance: float = 0.01  # 1% 相對誤差
    adaptive_check_interval: int = 50  # 每 50 輪檢查一次

    # ===== RCD风格优化参数 =====
    # 分块处理
    enable_chunking: bool = True
    chunk_size: int = 5  # 每块节点数（类似RCD的gamma）
    chunk_overlap: int = 1  # 块之间的重叠节点数，避免遗漏跨块边
    
    # 局部化搜索
    enable_localization: bool = True
    localization_radius: int = 2  # 焦点节点的邻居半径
    use_trace_for_localization: bool = True  # 使用trace graph辅助局部化
    
    # 多阶段迭代
    enable_multi_phase: bool = True
    phase1_max_iterations: int = 3  # Phase-1最大迭代次数
    phase1_chunk_size: int = 10  # Phase-1分块大小（更大，更快）
    phase2_node_limit: int = 15  # Phase-2最终节点数上限
    
    # 智能筛选
    enable_smart_chunking: bool = True  # 基于异常分数智能分块
    chunk_by_correlation: bool = True  # 基于相关性分块
    
    # 时间窗口优化
    enable_window_truncation: bool = True
    window_length: Optional[int] = 600  # 自动截断时间窗口（None=不截断）

    def validate(self) -> bool:
        assert 0 < self.tau_max <= 10
        assert 0 < self.pcmci_alpha < 1
        assert abs(self.theta1 + self.theta2 + self.theta3 - 1.0) < 1e-9
        assert 0 < self.alpha_prop < 1
        assert self.K >= 1
        assert 0 <= self.lambda_penalty
        assert self.score_alpha1 >= 0 and self.score_alpha2 >= 0 and self.score_alpha3 >= 0
        assert abs(self.score_alpha1 + self.score_alpha2 + self.score_alpha3 - 1.0) < 1e-9
        
        # Phase 2 驗證
        assert self.node_isolation_n_jobs >= -1
        assert self.shapley_n_jobs >= -1
        assert self.adaptive_max_rounds >= self.adaptive_min_rounds
        assert self.adaptive_min_rounds >= 10
        assert self.adaptive_confidence in [0.95, 0.99]
        assert 0 < self.adaptive_tolerance < 1
        assert self.adaptive_check_interval >= 10
        
        # SPOT 參數驗證
        assert 0 < self.spot_risk_param < 1
        assert self.spot_init_window >= 50
        assert self.spot_depth >= 5
        
        # 聯合篩選器參數驗證
        assert 0 <= self.joint_weight_fallback <= 1
        assert 0 <= self.joint_weight_spot <= 1
        assert abs(self.joint_weight_fallback + self.joint_weight_spot - 1.0) < 1e-9
        assert self.joint_top_n >= 5
        
        # Fallback 優化參數驗證
        assert 0 <= self.fallback_sparse_threshold <= 1
        assert 0 <= self.hybrid_confidence_threshold <= 1
        assert len(self.enhanced_anomaly_weights) == 4
        assert abs(sum(self.enhanced_anomaly_weights) - 1.0) < 1e-9
        assert all(w >= 0 for w in self.enhanced_anomaly_weights)
        
        # RCD风格优化参数验证
        assert self.chunk_size >= 2
        assert self.chunk_overlap >= 0
        assert self.localization_radius >= 1
        assert self.phase1_max_iterations >= 1
        assert self.phase1_chunk_size >= 2
        assert self.phase2_node_limit >= 5
        if self.window_length is not None:
            assert self.window_length >= 100
        
        return True
    
    @classmethod
    def fast_preset(cls) -> 'PCMCIShapleyConfig':
        """快速模式：牺牲少量精度换取5-10倍速度"""
        return cls(
            tau_max=2,
            pcmci_alpha=0.1,
            pcmci_max_conds_dim=1,
            enable_chunking=True,
            chunk_size=5,
            chunk_overlap=0,
            enable_localization=True,
            localization_radius=1,
            enable_multi_phase=True,
            phase1_chunk_size=10,
            phase2_node_limit=10,
            enable_window_truncation=True,
            window_length=400,
            shapley_method="sampling",
            adaptive_max_rounds=1000,
            sampling_rounds=300
        )
    
    @classmethod
    def balanced_preset(cls) -> 'PCMCIShapleyConfig':
        """平衡模式：精度与速度兼顾"""
        return cls(
            tau_max=3,
            pcmci_alpha=0.05,
            pcmci_max_conds_dim=2,
            enable_chunking=True,
            chunk_size=7,
            chunk_overlap=1,
            enable_localization=True,
            localization_radius=2,
            enable_multi_phase=True,
            phase1_chunk_size=12,
            phase2_node_limit=15,
            enable_window_truncation=True,
            window_length=600,
            shapley_method="sampling",
            adaptive_max_rounds=2000,
            sampling_rounds=500
        )
    
    @classmethod
    def accurate_preset(cls) -> 'PCMCIShapleyConfig':
        """精确模式：追求最高精度"""
        return cls(
            tau_max=5,
            pcmci_alpha=0.01,
            pcmci_max_conds_dim=3,
            enable_chunking=False,  # 不分块
            enable_localization=False,  # 不局部化
            enable_multi_phase=True,  # 保留多阶段
            phase2_node_limit=20,
            enable_window_truncation=False,  # 不截断窗口
            shapley_method="adaptive",  # 使用自适应采样
            adaptive_max_rounds=5000,
            sampling_rounds=3000  # 提升采样轮数
        )
    
    @classmethod
    def high_precision_preset(cls) -> 'PCMCIShapleyConfig':
        """高精度模式：专门为提升精度到0.8+设计"""
        return cls(
            # PCMCI参数：更保守但更稳定
            tau_max=2,  # 降低滞后期
            pcmci_alpha=0.1,  # 放宽显著性
            pcmci_max_conds_dim=1,  # 限制条件集
            
            # 优化策略：启用所有优化
            enable_chunking=True,
            chunk_size=3,  # 更小的分块
            chunk_overlap=1,
            enable_localization=True,
            localization_radius=1,
            enable_multi_phase=True,
            phase1_chunk_size=8,
            phase2_node_limit=12,
            
            # 时间窗口：适度截断
            enable_window_truncation=True,
            window_length=400,
            
            # Shapley：高精度采样
            shapley_method="adaptive",
            adaptive_max_rounds=8000,  # 大幅提升
            sampling_rounds=4000,  # 大幅提升
            
            # 异常检测：更敏感
            anomaly_threshold=1.5,  # 降低阈值
            
            # 联合筛选：更严格
            joint_top_n=15,  # 减少节点数
            joint_weight_fallback=0.7,  # 更重视异常分数
            joint_weight_spot=0.3,
            
            # 增强异常分数权重
            enhanced_anomaly_weights=[0.4, 0.2, 0.3, 0.1]  # 即时性40%，严重性30%
        )
