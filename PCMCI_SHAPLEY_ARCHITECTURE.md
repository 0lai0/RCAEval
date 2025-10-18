# PCMCI-Shapley 方法架構設計文件

**方法名稱**: Local PCMCI-lag Causal + Shapley Propagation Ranking  
**設計日期**: 2025-10-15  
**版本**: v1.0

---

## 一、整體架構概覽

### 1.1 系統層次結構

```
┌─────────────────────────────────────────────────────────────┐
│                   RCAEval Evaluation Framework               │
│                         (main.py)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              RCAEval/e2e/pcmci_shapley.py                    │
│         (Main Entry Point with @rca decorator)               │
│                                                              │
│  def pcmci_shapley(data, inject_time, dataset, **kwargs)    │
│      → returns: {adj, node_names, ranks, ...}               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          RCAEval/e2e/pcmci_shapley_modules/                  │
│                   (Modular Components)                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │preprocessing │  │node_isolation│  │pcmci_local   │      │
│  │    .py       │  │    .py       │  │    .py       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │edge_fusion   │  │propagation   │  │shapley       │      │
│  │    .py       │  │    .py       │  │    .py       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │scoring       │  │utils         │                        │
│  │    .py       │  │    .py       │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                External Dependencies                         │
│                                                              │
│  • tigramite (PCMCI algorithm)                              │
│  • sklearn (IsolationForest, PCA)                           │
│  • networkx (Graph operations)                              │
│  • numpy, pandas (Data processing)                          │
│  • scipy (Statistical functions)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、資料流程設計

### 2.1 主流程資料流

```
Input: Raw Metrics DataFrame
  ├─ columns: [time, service1_metric1, service1_metric2, ...]
  └─ shape: (T_samples, N_services × M_metrics)
  
  ↓ [Step 0: Preprocessing]
  
Preprocessed Data
  ├─ normalized_df: Robust normalized metrics
  ├─ anomaly_scores: a_{s,m}(t) per metric
  └─ node_anomaly: a_s(t) per service
  
  ↓ [Step 1: Local Node Construction]
  
Local Subgraph
  ├─ local_nodes: U (≤40 nodes)
  └─ isolation_scores: I_s per node
  
  ↓ [Step 2: PCMCI+ Causal Testing]
  
Causal Graph
  ├─ edges: E_PCMCI with time lags
  ├─ edge_strengths: s_{i→j}
  └─ pcmci_graph: nx.DiGraph
  
  ↓ [Step 3: Edge Fusion]
  
Weighted Graph
  ├─ fused_weights: w_{i→j}
  └─ normalized_weights: ŵ_{i→j}
  
  ↓ [Step 4: Anomaly Propagation]
  
Propagation Result
  ├─ h_final: Final influence scores
  └─ h_by_step: Propagation trajectory
  
  ↓ [Step 5: Shapley Calculation]
  
Contribution Scores
  └─ shapley_values: φ_s per node
  
  ↓ [Step 6-7: Scoring & Ranking]
  
Final Output
  ├─ adj: Adjacency matrix
  ├─ node_names: All node names
  ├─ ranks: Sorted root cause candidates
  └─ scores: Detailed scoring breakdown
```

### 2.2 關鍵資料結構

#### 2.2.1 預處理階段輸出
```python
PreprocessResult = {
    'normalized_df': pd.DataFrame,        # Shape: (T, N×M)
    'anomaly_scores': pd.DataFrame,       # Shape: (T, N×M)
    'node_anomaly': pd.Series,            # Shape: (N,)
    'metric_mapping': Dict[str, List[str]] # service → metrics
}
```

#### 2.2.2 局部節點建構輸出
```python
LocalNodeResult = {
    'local_nodes': List[str],             # Length: ≤40
    'isolation_scores': Dict[str, float], # I_s scores
    'correlation_matrix': np.ndarray,     # ρ_{F,s}(τ)
    'trace_graph': nx.Graph               # Local trace subgraph
}
```

#### 2.2.3 PCMCI 檢定輸出
```python
PCMCIResult = {
    'edges': List[Tuple[str, str, int]],  # (source, target, tau)
    'edge_strengths': Dict[Tuple, float], # (i, j) → s_{i→j}
    'p_values': Dict[Tuple, float],       # Statistical significance
    'pcmci_graph': nx.DiGraph,            # Causal graph
    'val_matrix': np.ndarray              # Full correlation matrix
}
```

#### 2.2.4 邊權融合輸出
```python
EdgeFusionResult = {
    'raw_weights': Dict[Tuple, float],    # w_{i→j} before normalization
    'normalized_weights': Dict[Tuple, float], # ŵ_{i→j}
    'conflict_flags': Dict[Tuple, bool],  # Direction conflicts
    'weighted_graph': nx.DiGraph          # Final weighted graph
}
```

#### 2.2.5 傳播輸出
```python
PropagationResult = {
    'h_final': Dict[str, float],          # Final influence
    'h_by_step': List[Dict[str, float]],  # Trajectory
    'initial_delta': Dict[str, float],    # δ_s
    'convergence_step': int               # If converged
}
```

#### 2.2.6 Shapley 輸出
```python
ShapleyResult = {
    'shapley_values': Dict[str, float],   # φ_s
    'normalized_shapley': Dict[str, float], # Normalized to [0,1]
    'coalition_values': List[float],      # v(C) samples
    'confidence_intervals': Dict[str, Tuple] # Optional CI
}
```

#### 2.2.7 最終輸出
```python
FinalResult = {
    'adj': np.ndarray,                    # Adjacency matrix
    'node_names': List[str],              # All node names
    'ranks': List[str],                   # Ranked root causes
    'scores': Dict[str, float],           # Comprehensive scores
    'shapley_values': Dict[str, float],   # Shapley contributions
    'reachability': Dict[str, float],     # r_s scores
    'temporal_penalties': Dict[str, float], # p_s penalties
    'local_graph': nx.DiGraph,            # Local causal graph
    'metadata': Dict                      # Hyperparameters, timing
}
```

---

## 三、模組設計詳解

### 3.1 preprocessing.py

**職責**: 資料標準化、異常偵測、聚合

**核心函數**:
```python
def robust_normalize(df: pd.DataFrame) -> pd.DataFrame
    """Robust standardization using MAD"""
    
def detect_anomaly_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame
    """Anomaly detection using robust z-score"""
    
def detect_anomaly_spot(df: pd.DataFrame) -> pd.DataFrame
    """SPOT-based anomaly detection (optional)"""
    
def aggregate_node_anomaly(anomaly_df: pd.DataFrame, metric_map: Dict) -> pd.Series
    """Aggregate metrics to service-level anomaly scores"""
    
def preprocess_data(data: pd.DataFrame, config: Config) -> PreprocessResult
    """Main preprocessing pipeline"""
```

**設計考量**:
- 使用 MAD (Median Absolute Deviation) 而非標準差，更 robust
- 支援多種異常偵測方法（Z-score, SPOT）
- 聚合時考慮 metric 權重（可選）

---

### 3.2 node_isolation.py

**職責**: 局部節點篩選，降低 PCMCI 計算複雜度

**核心函數**:
```python
def compute_lagged_correlation(focus_series: pd.Series, 
                               other_series: pd.Series, 
                               tau_max: int) -> np.ndarray
    """Compute ρ_{F,s}(τ) for τ ∈ [1, τ_max]"""
    
def statistical_neighborhood(focus_node: str, 
                             node_anomaly: Dict, 
                             tau_max: int, 
                             top_m1: int) -> List[str]
    """Select top-M1 nodes by lagged correlation"""
    
def build_isolation_features(candidates: List[str], 
                             focus_node: str, 
                             node_anomaly: Dict, 
                             tau_max: int) -> pd.DataFrame
    """Build feature vectors z_s for IsolationForest"""
    
def isolation_forest_selection(features: pd.DataFrame, 
                                top_m2: int) -> Tuple[List[str], Dict]
    """IsolationForest-based node selection"""
    
def trace_augmentation(selected_nodes: List[str], 
                       trace_graph: nx.Graph, 
                       focus_node: str, 
                       max_size: int = 40) -> List[str]
    """Augment with 1-hop trace neighbors"""
    
def build_local_nodes(data: PreprocessResult, 
                      focus_node: str, 
                      trace_graph: nx.Graph, 
                      config: Config) -> LocalNodeResult
    """Main local node construction pipeline"""
```

**設計考量**:
- 兩階段篩選：統計相關性 → Isolation Forest
- IsolationForest 可捕捉非線性交互模式
- Trace 補強保證結構完整性
- 嚴格控制 |U| ≤ 40

---

### 3.3 pcmci_local.py

**職責**: 在局部節點集合上執行 PCMCI+ 時間因果檢定

**核心函數**:
```python
def build_timeseries_matrix(data: pd.DataFrame, 
                            local_nodes: List[str], 
                            use_pca: bool = False) -> np.ndarray
    """Construct time series matrix for PCMCI"""
    
def run_pcmci_plus(data_matrix: np.ndarray, 
                   tau_max: int, 
                   alpha: float) -> Dict
    """Wrapper for tigramite PCMCI"""
    
def extract_significant_edges(pcmci_result: Dict, 
                               alpha: float) -> List[Tuple]
    """Extract significant causal edges E_PCMCI"""
    
def compute_edge_strength(pcmci_result: Dict, 
                          edges: List[Tuple]) -> Dict
    """Compute s_{j→i} = max_τ |r_{j→i|Z}(τ)|"""
    
def local_pcmci_causal_test(data: pd.DataFrame, 
                             local_nodes: List[str], 
                             config: Config) -> PCMCIResult
    """Main PCMCI testing pipeline"""
```

**設計考量**:
- 重用現有 `RCAEval/graph_construction/pcmci.py` 的部分邏輯
- 增強版：保留時滯資訊 τ
- 使用 ParCorr 條件獨立檢定
- 可選 PCA 降維（當 metrics 過多時）

---

### 3.4 edge_fusion.py

**職責**: 融合 Trace、PCMCI、Isolation 三種邊權來源

**核心函數**:
```python
def extract_trace_weights(trace_graph: nx.Graph, 
                          local_nodes: List[str]) -> Dict
    """Extract w^{trace}_{i→j} from trace graph"""
    
def fuse_edge_weights(trace_weights: Dict, 
                      pcmci_strengths: Dict, 
                      isolation_scores: Dict, 
                      config: Config) -> Dict
    """Fuse three sources: w = θ1·trace + θ2·pcmci + θ3·isolation"""
    
def apply_conflict_penalty(weights: Dict, 
                           pcmci_edges: List, 
                           trace_edges: List, 
                           gamma: float) -> Dict
    """Penalize direction conflicts"""
    
def normalize_incoming_weights(weights: Dict, 
                                local_nodes: List[str]) -> Dict
    """Normalize incoming edges: ŵ_{i→j} = w_{i→j} / Σ_k w_{k→j}"""
    
def fuse_and_normalize(trace_graph: nx.Graph, 
                       pcmci_result: PCMCIResult, 
                       isolation_scores: Dict, 
                       config: Config) -> EdgeFusionResult
    """Main edge fusion pipeline"""
```

**設計考量**:
- θ1, θ2, θ3 可調整（預設 0.6:0.3:0.1）
- 方向衝突懲罰避免 trace 與因果相悖
- 入邊歸一化確保傳播穩定性

---

### 3.5 propagation.py

**職責**: K 步異常傳播模擬

**核心函數**:
```python
def compute_initial_anomaly(node_anomaly: pd.Series, 
                            timestamp: int) -> Dict
    """Compute δ_s = log(1 + a_s(T))"""
    
def propagate_one_step(current_h: Dict, 
                       edge_weights: Dict, 
                       alpha_prop: float) -> Dict
    """One-step propagation: h^{(k+1)}_j = α_prop Σ_i ŵ_{i→j} h^{(k)}_i"""
    
def propagate_k_steps(initial_delta: Dict, 
                      edge_weights: Dict, 
                      K: int, 
                      alpha_prop: float = 0.85) -> PropagationResult
    """K-step propagation with accumulation"""
    
def visualize_propagation(h_by_step: List[Dict], 
                          graph: nx.DiGraph, 
                          save_path: str = None) -> None
    """Optional: Visualize propagation dynamics"""
```

**設計考量**:
- 線性衰減模型（類似 PageRank）
- α_prop 控制衰減速率（預設 0.85）
- K 通常取 5-10 步
- 累積所有步驟的影響

---

### 3.6 shapley.py

**職責**: 基於聯盟博弈理論計算節點貢獻

**核心函數**:
```python
def compute_system_anomaly(coalition: Set[str], 
                           edge_weights: Dict, 
                           local_nodes: List[str], 
                           alpha_prop: float, 
                           K: int) -> float
    """Compute v(C): system anomaly with coalition C"""
    
def compute_shapley_sampling(local_nodes: List[str], 
                             edge_weights: Dict, 
                             config: Config) -> ShapleyResult
    """Monte Carlo sampling approximation of Shapley values"""
    
def compute_shapley_exact(local_nodes: List[str], 
                          edge_weights: Dict, 
                          config: Config) -> ShapleyResult
    """Exact Shapley computation for |U| ≤ 15"""
    
def normalize_shapley(shapley_values: Dict) -> Dict
    """Normalize Shapley values to [0, 1]"""
    
def compute_shapley_values(local_nodes: List[str], 
                           edge_weights: Dict, 
                           config: Config) -> ShapleyResult
    """Main Shapley computation pipeline (auto-select method)"""
```

**設計考量**:
- 小規模 (|U| ≤ 15)：精確計算
- 大規模 (|U| > 15)：蒙特卡洛抽樣 (R=500-1000)
- 可選：計算信賴區間
- v(C) 定義：保留 C 中節點的異常源，執行傳播

---

### 3.7 scoring.py

**職責**: 綜合評分與排名

**核心函數**:
```python
def compute_reachability(node: str, 
                         focus_node: str, 
                         edge_weights: Dict, 
                         graph: nx.DiGraph) -> float
    """Compute r_s = max_path Π ŵ_{i→j}"""
    
def compute_temporal_penalty(node: str, 
                             focus_node: str, 
                             anomaly_timestamps: Dict, 
                             graph: nx.DiGraph, 
                             lambda_penalty: float) -> float
    """Compute p_s = exp(-λ Σ penalty)"""
    
def compute_comprehensive_score(shapley_values: Dict, 
                                reachability: Dict, 
                                node_anomaly: Dict, 
                                config: Config) -> Dict
    """Compute Score_s = α1·φ̂ + α2·r̂ + α3·â"""
    
def compute_final_ranking(scores: Dict, 
                          penalties: Dict) -> List[str]
    """Final ranking: argsort(Score_s · p_s)"""
    
def score_and_rank(shapley_result: ShapleyResult, 
                   edge_weights: Dict, 
                   node_anomaly: Dict, 
                   graph: nx.DiGraph, 
                   focus_node: str, 
                   config: Config) -> FinalResult
    """Main scoring and ranking pipeline"""
```

**設計考量**:
- α1, α2, α3 預設 0.5:0.3:0.2
- 可達性使用最大權重路徑
- 時間懲罰檢測因果時序違反
- 所有分數先正規化再加權

---

### 3.8 utils.py

**職責**: 通用工具函數

**核心函數**:
```python
def min_max_normalize(values: Dict) -> Dict
    """Min-max normalization to [0, 1]"""
    
def robust_standardize(series: pd.Series) -> pd.Series
    """Robust standardization using MAD"""
    
def ensure_2d(arr: np.ndarray) -> np.ndarray
    """Ensure array is 2D"""
    
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float
    """Safe division avoiding divide-by-zero"""
    
def build_adjacency_matrix(graph: nx.DiGraph, node_names: List[str]) -> np.ndarray
    """Convert graph to adjacency matrix"""
    
def validate_config(config: Config) -> bool
    """Validate configuration parameters"""
    
def log_step(step_name: str, result: Dict) -> None
    """Logging utility for debugging"""
```

---

## 四、配置管理

### 4.1 PCMCIShapleyConfig 類別

```python
@dataclass
class PCMCIShapleyConfig:
    # === Node Isolation ===
    tau_max: int = 5                    # Max time lag
    top_m1: int = 60                    # Statistical neighborhood size
    top_m2: int = 30                    # IsolationForest selection size
    u_max: int = 40                     # Max local nodes
    
    # === PCMCI ===
    pcmci_alpha: float = 0.05           # Significance level
    use_pca: bool = False               # Use PCA for dimension reduction
    pca_components: int = 10            # PCA components if enabled
    
    # === Edge Fusion ===
    theta1: float = 0.6                 # Trace weight
    theta2: float = 0.3                 # PCMCI weight
    theta3: float = 0.1                 # Isolation weight
    gamma: float = 0.5                  # Conflict penalty
    
    # === Propagation ===
    K: int = 5                          # Propagation steps
    alpha_prop: float = 0.85            # Propagation decay
    
    # === Shapley ===
    shapley_method: str = 'auto'        # 'auto', 'exact', 'sampling'
    sampling_rounds: int = 500          # Monte Carlo rounds
    
    # === Scoring ===
    score_alpha1: float = 0.5           # Shapley weight
    score_alpha2: float = 0.3           # Reachability weight
    score_alpha3: float = 0.2           # Anomaly weight
    lambda_penalty: float = 1.0         # Temporal penalty strength
    
    # === Preprocessing ===
    anomaly_threshold: float = 3.0      # Z-score threshold
    anomaly_method: str = 'zscore'      # 'zscore' or 'spot'
    
    def validate(self) -> bool:
        """Validate configuration"""
        assert 0 < self.tau_max <= 10
        assert 0 < self.pcmci_alpha < 1
        assert self.theta1 + self.theta2 + self.theta3 == 1.0
        # ... more validations
        return True
```

---

## 五、介面與契約

### 5.1 主函數介面

```python
@rca
def pcmci_shapley(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    dk_select_useful: bool = False,
    focus_node: Optional[str] = None,
    trace_graph: Optional[nx.Graph] = None,
    config: Optional[PCMCIShapleyConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    PCMCI-Shapley Root Cause Analysis
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with columns [time, metric1, metric2, ...]
    inject_time : int, optional
        Fault injection timestamp
    dataset : str, optional
        Dataset name for preprocessing
    dk_select_useful : bool
        Whether to select useful columns
    focus_node : str, optional
        Focal service (if None, auto-detect)
    trace_graph : nx.Graph, optional
        Service call graph
    config : PCMCIShapleyConfig, optional
        Configuration object
    
    Returns
    -------
    result : dict
        {
            'adj': np.ndarray,              # Adjacency matrix
            'node_names': List[str],        # Node names
            'ranks': List[str],             # Ranked root causes
            'scores': Dict[str, float],     # Scores
            'shapley_values': Dict[str, float],
            'local_graph': nx.DiGraph,
            'metadata': Dict                # Timing, parameters
        }
    """
```

### 5.2 與 RCAEval 框架整合

```python
# In RCAEval/e2e/__init__.py
if is_py310():
    # ... existing imports
    from .pcmci_shapley import pcmci_shapley

# In main.py
METHODS = {
    # ... existing methods
    'pcmci_shapley': pcmci_shapley,
}
```

---

## 六、錯誤處理與容錯

### 6.1 錯誤處理策略

1. **@rca 裝飾器**: 最外層錯誤捕捉，返回 dummy 結果
2. **模組級 try-except**: 每個模組的主函數包裝
3. **降級策略**:
   - 無 trace graph → 純統計方法
   - PCMCI 失敗 → 使用 Granger causality
   - Shapley 超時 → 使用簡化評分

### 6.2 輸入驗證

```python
def validate_input(data: pd.DataFrame, config: Config) -> bool:
    # Check data shape
    assert data.shape[0] > config.tau_max
    # Check for NaN
    assert not data.isnull().all().any()
    # Check focus node exists
    # ... more checks
```

### 6.3 Logging 策略

```python
import logging

logger = logging.getLogger('pcmci_shapley')
logger.setLevel(logging.INFO)

# In each module
logger.info(f"Step 1: Preprocessing completed, {len(df)} samples")
logger.debug(f"Anomaly scores: {anomaly_scores.describe()}")
logger.warning(f"Large local graph: |U|={len(U)} > 40")
logger.error(f"PCMCI failed: {e}")
```

---

## 七、性能優化策略

### 7.1 計算優化

1. **PCMCI 優化**:
   - 限制 |U| ≤ 40
   - 平行化條件獨立檢定（可選）
   - 早停機制

2. **Shapley 優化**:
   - 小規模精確計算
   - 大規模抽樣近似
   - 可選：TreeSHAP 演算法

3. **傳播優化**:
   - 稀疏矩陣運算
   - 收斂檢測（早停）

### 7.2 記憶體優化

1. **資料結構選擇**:
   - 使用 scipy.sparse 存儲稀疏圖
   - numpy array 避免重複複製

2. **批次處理**:
   - Shapley 抽樣分批計算

### 7.3 性能監控

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.time()
    yield
    end = time.time()
    logger.info(f"{name} took {end - start:.2f}s")

# Usage
with timer("PCMCI Testing"):
    pcmci_result = run_pcmci_plus(...)
```

---

## 八、測試架構

### 8.1 測試層次

```
tests/
├── unit/
│   ├── test_preprocessing.py
│   ├── test_node_isolation.py
│   ├── test_pcmci_local.py
│   ├── test_edge_fusion.py
│   ├── test_propagation.py
│   ├── test_shapley.py
│   └── test_scoring.py
├── integration/
│   ├── test_end_to_end.py
│   └── test_pipeline.py
└── performance/
    ├── test_scalability.py
    └── test_timing.py
```

### 8.2 測試資料

```python
# Fixture for test data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'time': range(100),
        'service1_cpu': np.random.randn(100),
        'service1_mem': np.random.randn(100),
        'service2_cpu': np.random.randn(100),
        # ...
    })

@pytest.fixture
def sample_trace_graph():
    G = nx.DiGraph()
    G.add_edges_from([
        ('service1', 'service2'),
        ('service2', 'service3'),
    ])
    return G
```

---

## 九、文件與範例

### 9.1 使用範例

```python
# Example 1: Basic usage
from RCAEval.e2e import pcmci_shapley
import pandas as pd

data = pd.read_csv('metrics.csv')
result = pcmci_shapley(
    data=data,
    inject_time=1000,
    dataset='online-boutique'
)

print("Root causes:", result['ranks'][:5])
print("Shapley values:", result['shapley_values'])
```

```python
# Example 2: Custom configuration
from RCAEval.e2e.pcmci_shapley_modules import PCMCIShapleyConfig

config = PCMCIShapleyConfig(
    tau_max=10,
    K=8,
    shapley_method='sampling',
    sampling_rounds=1000
)

result = pcmci_shapley(
    data=data,
    config=config
)
```

### 9.2 README 結構

```markdown
# PCMCI-Shapley Root Cause Analysis

## Overview
Brief description of the method

## Installation
Dependencies and setup

## Quick Start
Minimal example

## Configuration
Parameter descriptions

## Advanced Usage
Custom trace graphs, hyperparameter tuning

## Algorithm Details
Link to plan.md

## Performance
Complexity analysis, benchmarks

## Citation
If applicable

## License
```

---

## 十、未來擴展方向

### 10.1 短期擴展
1. 支援多故障點同時分析
2. 即時串流資料處理
3. 互動式視覺化介面

### 10.2 中期擴展
1. 深度學習增強的異常偵測
2. 自適應超參數調整
3. 增量學習支援

### 10.3 長期擴展
1. 多模態資料融合（logs, traces, metrics）
2. 因果圖的線上學習與更新
3. 分散式計算支援

---

**文件版本**: v1.0  
**最後更新**: 2025-10-15  
**維護者**: RCAEval Team


