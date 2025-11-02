from dataclasses import dataclass


@dataclass
class PCMCIShapleyConfig:
    # Node Isolation
    tau_max: int = 5
    top_m1: int = 60
    top_m2: int = 30
    u_max: int = 40

    # Causal Discovery
    causal_method: str = "pc"  # pcmci, pc, ges, fci, lingam
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

    # Pruning (新增)
    enable_pruning: bool = True
    pruning_max_hops: int = 2
    pruning_anomaly_percentile: float = 0.3
    pruning_min_nodes: int = 10
    pruning_max_nodes: int = 20  # 剪枝後最多保留的節點數

    # PCMCI 優化 (新增)
    pcmci_max_conds_dim: int | None = 3  # None 表示不限制
    pcmci_max_conds_py: int | None = None
    pcmci_max_conds_px: int | None = None

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

    # Phase 2: Shapley 緩存優化 (新增)
    enable_shapley_cache: bool = True  # 是否啟用 coalition value 緩存
    shapley_cache_size: int = 2048  # 每個進程的緩存大小（LRU cache 最大條目數）

    def validate(self) -> bool:
        assert self.causal_method in ["pcmci", "pc", "ges", "fci", "lingam"]
        assert 0 < self.tau_max <= 10
        assert 0 < self.pcmci_alpha < 1
        assert abs(self.theta1 + self.theta2 + self.theta3 - 1.0) < 1e-9
        assert 0 < self.alpha_prop < 1
        assert self.K >= 1
        assert 0 <= self.lambda_penalty
        assert self.score_alpha1 >= 0 and self.score_alpha2 >= 0 and self.score_alpha3 >= 0
        assert abs(self.score_alpha1 + self.score_alpha2 + self.score_alpha3 - 1.0) < 1e-9
        assert self.pruning_max_nodes >= self.pruning_min_nodes
        
        # Phase 2 驗證
        assert self.node_isolation_n_jobs >= -1
        assert self.shapley_n_jobs >= -1
        assert self.adaptive_max_rounds >= self.adaptive_min_rounds
        assert self.adaptive_min_rounds >= 10
        assert self.adaptive_confidence in [0.95, 0.99]
        assert 0 < self.adaptive_tolerance < 1
        assert self.adaptive_check_interval >= 10
        assert self.shapley_cache_size >= 64  # 緩存大小至少為 64
        
        return True
