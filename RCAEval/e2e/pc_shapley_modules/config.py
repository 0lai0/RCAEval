from dataclasses import dataclass


@dataclass
class PCShapleyConfig:
    # Node Isolation
    tau_max: int = 5
    top_m1: int = 60
    top_m2: int = 30
    u_max: int = 40

    # Causal Discovery (PC algorithm only)
    pc_alpha: float = 0.05
    pc_max_conds_dim: int | None = 3
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
    anomaly_method: str = "zscore"

    # Pruning
    enable_pruning: bool = True
    pruning_max_hops: int = 2
    pruning_anomaly_percentile: float = 0.3
    pruning_min_nodes: int = 10
    pruning_max_nodes: int = 20

    # Parallelization
    enable_parallel: bool = True
    node_isolation_n_jobs: int = -1
    shapley_n_jobs: int = -1

    # Adaptive Sampling
    adaptive_max_rounds: int = 2000
    adaptive_min_rounds: int = 100
    adaptive_confidence: float = 0.95
    adaptive_tolerance: float = 0.01
    adaptive_check_interval: int = 50

    # Shapley Cache Optimization
    enable_shapley_cache: bool = True
    shapley_cache_size: int = 2048

    def validate(self) -> bool:
        assert 0 < self.tau_max <= 10
        assert 0 < self.pc_alpha < 1
        assert abs(self.theta1 + self.theta2 + self.theta3 - 1.0) < 1e-9
        assert 0 < self.alpha_prop < 1
        assert self.K >= 1
        assert 0 <= self.lambda_penalty
        assert self.score_alpha1 >= 0 and self.score_alpha2 >= 0 and self.score_alpha3 >= 0
        assert abs(self.score_alpha1 + self.score_alpha2 + self.score_alpha3 - 1.0) < 1e-9
        assert self.pruning_max_nodes >= self.pruning_min_nodes
        assert self.node_isolation_n_jobs >= -1
        assert self.shapley_n_jobs >= -1
        assert self.adaptive_max_rounds >= self.adaptive_min_rounds
        assert self.adaptive_min_rounds >= 10
        assert self.adaptive_confidence in [0.95, 0.99]
        assert 0 < self.adaptive_tolerance < 1
        assert self.adaptive_check_interval >= 10
        assert self.shapley_cache_size >= 64
        
        return True
