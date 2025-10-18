# PCMCI-Shapley Phase 1 實作計畫
## 低成本高回報優化方案 (1-2 天完成)

---

## 目標
在不大幅改動架構的前提下，透過以下三項優化達到 **6-18 倍整體加速**：
1. 智慧剪枝 (Pruning)
2. PCMCI 參數優化
3. 快速矩陣傳播

---

## 實作任務拆解

### 任務 1: 建立剪枝模組 (3 小時)

#### 1.1 建立新檔案
**路徑**: `RCAEval/e2e/pcmci_shapley_modules/pruning.py`

**內容**:
```python
"""
節點剪枝模組 - 減少需要分析的節點數量
"""
from __future__ import annotations
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import networkx as nx


def trace_based_prefiltering(
    all_nodes: List[str],
    trace_graph: nx.DiGraph | None,
    focus_node: str,
    max_hops: int = 2
) -> List[str]:
    """
    基於 trace 圖的預過濾：只保留與 focus_node 在 max_hops 跳內的節點
    
    Args:
        all_nodes: 所有候選節點列表
        trace_graph: 服務依賴圖 (trace)
        focus_node: 焦點節點 (SLI 服務)
        max_hops: 最大跳數 (預設 2)
    
    Returns:
        過濾後的節點列表
    """
    if trace_graph is None or focus_node not in all_nodes:
        return all_nodes
    
    if focus_node not in trace_graph:
        # focus_node 不在 trace 圖中，保留所有節點
        return all_nodes
    
    reachable: Set[str] = {focus_node}
    frontier: Set[str] = {focus_node}
    
    for hop in range(max_hops):
        new_frontier: Set[str] = set()
        for node in frontier:
            if node in trace_graph:
                # 前驅節點 (上游依賴)
                new_frontier.update(trace_graph.predecessors(node))
                # 後繼節點 (下游依賴)
                new_frontier.update(trace_graph.successors(node))
        reachable.update(new_frontier)
        frontier = new_frontier
    
    # 保留在 all_nodes 中且可達的節點
    filtered = [n for n in all_nodes if n in reachable]
    return filtered


def early_anomaly_pruning(
    node_anomaly: Dict[str, float] | pd.Series,
    threshold_percentile: float = 0.3,
    min_nodes: int = 10
) -> List[str]:
    """
    早期異常分數剪枝：只保留異常分數較高的節點
    
    Args:
        node_anomaly: 各節點的異常分數 (字典或 Series)
        threshold_percentile: 保留的百分位數 (0.3 表示保留 top 70%)
        min_nodes: 最少保留的節點數
    
    Returns:
        過濾後的節點列表
    """
    if isinstance(node_anomaly, pd.Series):
        node_anomaly = node_anomaly.to_dict()
    
    if not node_anomaly:
        return []
    
    # 移除 NaN 和負值
    valid_scores = {k: float(v) for k, v in node_anomaly.items() 
                    if np.isfinite(v) and v >= 0}
    
    if len(valid_scores) <= min_nodes:
        return list(valid_scores.keys())
    
    # 計算閾值
    scores = list(valid_scores.values())
    threshold = np.percentile(scores, threshold_percentile * 100)
    
    # 過濾
    filtered = [n for n, score in valid_scores.items() if score >= threshold]
    
    # 確保至少保留 min_nodes 個節點
    if len(filtered) < min_nodes:
        sorted_nodes = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
        filtered = [n for n, _ in sorted_nodes[:min_nodes]]
    
    return filtered


def combined_pruning(
    all_nodes: List[str],
    node_anomaly: Dict[str, float] | pd.Series,
    trace_graph: nx.DiGraph | None,
    focus_node: str,
    max_hops: int = 2,
    anomaly_percentile: float = 0.3,
    min_nodes: int = 10
) -> List[str]:
    """
    組合剪枝策略：先 trace 過濾，再異常分數過濾
    
    Returns:
        最終過濾後的節點列表
    """
    # Step 1: Trace-based filtering
    step1 = trace_based_prefiltering(all_nodes, trace_graph, focus_node, max_hops)
    
    # Step 2: Anomaly-based filtering
    if isinstance(node_anomaly, pd.Series):
        node_anomaly_dict = node_anomaly.to_dict()
    else:
        node_anomaly_dict = dict(node_anomaly)
    
    # 只考慮 step1 中的節點
    step1_anomaly = {k: v for k, v in node_anomaly_dict.items() if k in step1}
    step2 = early_anomaly_pruning(step1_anomaly, anomaly_percentile, min_nodes)
    
    # 確保 focus_node 一定保留
    if focus_node not in step2 and focus_node in all_nodes:
        step2 = [focus_node] + step2
    
    return step2
```

#### 1.2 更新模組導出
**修改**: `RCAEval/e2e/pcmci_shapley_modules/__init__.py`
```python
from . import pruning  # 新增這一行

__all__ = [
    # ... 現有的 ...
    "pruning",  # 新增這一行
]
```

#### 1.3 更新配置
**修改**: `RCAEval/e2e/pcmci_shapley_modules/config.py`

在 `PCMCIShapleyConfig` dataclass 中新增:
```python
@dataclass
class PCMCIShapleyConfig:
    # ... 現有參數 ...
    
    # Pruning (新增)
    enable_pruning: bool = True
    pruning_max_hops: int = 2
    pruning_anomaly_percentile: float = 0.3
    pruning_min_nodes: int = 10
```

---

### 任務 2: 整合剪枝到主流程 (2 小時)

**修改**: `RCAEval/e2e/pcmci_shapley.py`

在第 56-74 行之間插入剪枝邏輯:

```python
# 原始 line 56-58
base_df = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
logger.info(f"Base preprocess done: shape={base_df.shape}")

# 新增: 方法特定預處理
pp = prep_mod.preprocess_data(base_df, cfg)
logger.info("Method-specific preprocessing complete")

# 新增: 剪枝邏輯 (插入在 line 62 之後)
if cfg.enable_pruning:
    from .pcmci_shapley_modules import pruning as prune_mod
    
    # 獲取所有候選節點 (從 metric_mapping 提取服務名)
    all_candidates = list(pp.get("metric_mapping", {}).keys())
    
    # 應用組合剪枝
    pruned_nodes = prune_mod.combined_pruning(
        all_nodes=all_candidates,
        node_anomaly=pp.get("node_anomaly", {}),
        trace_graph=trace_graph,
        focus_node=focus_node or all_candidates[0] if all_candidates else "unknown",
        max_hops=cfg.pruning_max_hops,
        anomaly_percentile=cfg.pruning_anomaly_percentile,
        min_nodes=cfg.pruning_min_nodes
    )
    
    logger.info(f"Pruning: {len(all_candidates)} -> {len(pruned_nodes)} nodes")
    
    # 更新 node_anomaly_ts 只保留剪枝後的節點
    node_anomaly_ts_original = pp.get("node_anomaly_ts", {})
    node_anomaly_ts = {k: v for k, v in node_anomaly_ts_original.items() 
                       if k in pruned_nodes}
else:
    node_anomaly_ts = pp.get("node_anomaly_ts", {})
    logger.info("Pruning disabled")

# 確保 node_anomaly_ts 是 pd.Series 類型
node_anomaly_ts = {k: (v if isinstance(v, pd.Series) else pd.Series(v)) 
                   for k, v in node_anomaly_ts.items()}

# 原始 line 65-73 (focus node 決定邏輯保持不變)
# ...
```

**關鍵變更點**:
- 在 Node Isolation 之前就縮減候選節點範圍
- 使用 `cfg.enable_pruning` 開關控制是否啟用
- 記錄剪枝前後的節點數量

---

### 任務 3: PCMCI 參數優化 (1 小時)

**修改**: `RCAEval/e2e/pcmci_shapley_modules/pcmci_local.py`

#### 3.1 更新配置
**修改**: `config.py`
```python
@dataclass
class PCMCIShapleyConfig:
    # ... 現有參數 ...
    
    # PCMCI 優化 (新增)
    pcmci_max_conds_dim: int | None = 3  # None 表示不限制
    pcmci_max_conds_py: int | None = None
    pcmci_max_conds_px: int | None = None
```

#### 3.2 修改 PCMCI 調用
**修改**: `pcmci_local.py` 的 `run_pcmci_plus()` 函數

```python
def run_pcmci_plus(X: np.ndarray, tau_max: int, alpha: float, 
                   max_conds_dim: int | None = None) -> Dict[str, Any]:
    """
    Args:
        max_conds_dim: 條件集最大維度 (None 表示不限制)
    """
    # ... 前面的數據清理邏輯保持不變 ...
    
    dataframe = data_processing.DataFrame(X_clean)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    
    # 動態調整 max_conds_dim
    if max_conds_dim is None:
        max_conds_dim_actual = None
    else:
        # 確保不超過變量數
        max_conds_dim_actual = min(max_conds_dim, X_clean.shape[0] - 2)
    
    report = pcmci.run_pcmci(
        tau_max=tau_max, 
        pc_alpha=alpha, 
        max_conds_dim=max_conds_dim_actual,  # 新增參數
        max_conds_py=None,  # 可選：限制 Y 的條件集
        max_conds_px=None   # 可選：限制 X 的條件集
    )
    return report
```

#### 3.3 更新調用點
**修改**: `pcmci_local.py` 的 `local_pcmci_causal_test()` 函數

```python
def local_pcmci_causal_test(data: pd.DataFrame, local_nodes: List[str], 
                            config: PCMCIShapleyConfig) -> Dict[str, Any]:
    X, cols = build_timeseries_matrix(data, local_nodes, use_pca=config.use_pca, 
                                      pca_components=config.pca_components)
    report = run_pcmci_plus(
        X, 
        tau_max=config.tau_max, 
        alpha=config.pcmci_alpha,
        max_conds_dim=config.pcmci_max_conds_dim  # 傳遞參數
    )
    # ... 後續邏輯不變 ...
```

---

### 任務 4: 快速矩陣傳播 (2 小時)

**修改**: `RCAEval/e2e/pcmci_shapley_modules/propagation.py`

#### 4.1 新增稀疏矩陣版本

在檔案末尾新增:
```python
import scipy.sparse as sp


def fast_propagate_k_steps(
    initial_delta: Dict[str, float], 
    edge_weights: Dict[Tuple[str, str], float], 
    K: int, 
    alpha_prop: float,
    use_sparse: bool = True
) -> Dict[str, object]:
    """
    使用稀疏矩陣加速的傳播版本
    
    Args:
        use_sparse: 是否使用稀疏矩陣 (節點數 > 20 時建議啟用)
    """
    # 當節點數過少或邊數過少時，使用原始方法
    all_nodes = set(initial_delta.keys())
    all_nodes.update([i for i, j in edge_weights.keys()])
    all_nodes.update([j for i, j in edge_weights.keys()])
    
    if len(all_nodes) <= 10 or len(edge_weights) <= 5 or not use_sparse:
        # 回退到原始實作
        return _propagate_k_steps_original(initial_delta, edge_weights, K, alpha_prop)
    
    # 建立節點索引
    nodes = sorted(all_nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    
    # 建立稀疏鄰接矩陣 (轉置以便矩陣乘法)
    # A[j, i] = alpha * w(i -> j)
    row, col, data = [], [], []
    for (i, j), w in edge_weights.items():
        if w <= 0:
            continue
        row.append(node_idx[j])
        col.append(node_idx[i])
        data.append(alpha_prop * float(w))
    
    if not data:
        # 沒有有效邊，返回初始值
        return {"h_final": dict(initial_delta), "h_by_step": [dict(initial_delta)]}
    
    A = sp.csr_matrix((data, (row, col)), shape=(N, N))
    
    # 初始化向量
    h0 = np.zeros(N, dtype=float)
    for node, val in initial_delta.items():
        if node in node_idx:
            h0[node_idx[node]] = float(val)
    
    # 迭代傳播: h_{k+1} = A @ h_k
    h_accum = h0.copy()
    h_curr = h0.copy()
    h_by_step = [_vec_to_dict(h0, nodes)]
    
    for k in range(K):
        h_curr = A @ h_curr  # 稀疏矩陣乘法 (O(E) 而非 O(N²))
        h_accum += h_curr
        h_by_step.append(_vec_to_dict(h_curr, nodes))
    
    h_final = _vec_to_dict(h_accum, nodes)
    
    return {"h_final": h_final, "h_by_step": h_by_step}


def _vec_to_dict(vec: np.ndarray, nodes: List[str]) -> Dict[str, float]:
    """向量轉回字典"""
    return {nodes[i]: float(vec[i]) for i in range(len(nodes)) if vec[i] != 0}


def _propagate_k_steps_original(
    initial_delta: Dict[str, float], 
    edge_weights: Dict[Tuple[str, str], float], 
    K: int, 
    alpha_prop: float
) -> Dict[str, object]:
    """原始實作 (保留作為備份)"""
    h_accum: Dict[str, float] = {k: float(v) for k, v in initial_delta.items()}
    h_curr: Dict[str, float] = dict(initial_delta)
    h_by_step: List[Dict[str, float]] = [dict(h_curr)]
    for _ in range(K):
        h_next = propagate_one_step(h_curr, edge_weights, alpha_prop)
        for k, v in h_next.items():
            h_accum[k] = h_accum.get(k, 0.0) + float(v)
        h_by_step.append(dict(h_next))
        h_curr = h_next
    return {"h_final": h_accum, "h_by_step": h_by_step}
```

#### 4.2 替換調用點
**修改**: `propagate_k_steps()` 函數

```python
def propagate_k_steps(
    initial_delta: Dict[str, float], 
    edge_weights: Dict[Tuple[str, str], float], 
    K: int, 
    alpha_prop: float
) -> Dict[str, object]:
    """
    主入口：自動選擇快速或原始版本
    """
    # 使用快速版本 (內部會自動判斷是否適用)
    return fast_propagate_k_steps(initial_delta, edge_weights, K, alpha_prop, use_sparse=True)
```

#### 4.3 更新依賴
**修改**: `requirements.txt`

確保包含:
```
scipy>=1.7.0
```

---

### 任務 5: 測試與驗證 (2 小時)

#### 5.1 單元測試
**新建**: `tests/test_pruning.py`

```python
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
```

#### 5.2 整合測試
**新建**: `tests/test_pcmci_shapley_phase1.py`

```python
import pytest
import pandas as pd
import numpy as np
from RCAEval.e2e import pcmci_shapley
from RCAEval.e2e.pcmci_shapley_modules import PCMCIShapleyConfig


def test_pcmci_shapley_with_pruning():
    """測試啟用剪枝後的完整流程"""
    # 建立簡單測試數據
    np.random.seed(42)
    T = 100
    data = pd.DataFrame({
        'time': range(T),
        'A_latency': np.random.randn(T) + 1,
        'B_latency': np.random.randn(T) + 0.5,
        'C_latency': np.random.randn(T),
    })
    
    # 配置: 啟用剪枝
    config = PCMCIShapleyConfig(
        enable_pruning=True,
        pruning_max_hops=1,
        pruning_min_nodes=2,
        sampling_rounds=100  # 減少採樣以加快測試
    )
    
    result = pcmci_shapley.pcmci_shapley(data, config=config, dataset="test")
    
    assert 'ranks' in result
    assert 'adj' in result
    assert len(result['ranks']) > 0


def test_pruning_reduces_computation():
    """測試剪枝確實減少計算量"""
    import time
    
    np.random.seed(42)
    T = 200
    # 建立較大的數據集
    data = pd.DataFrame({
        'time': range(T),
        **{f'S{i}_latency': np.random.randn(T) for i in range(20)}
    })
    
    # 不啟用剪枝
    config_no_prune = PCMCIShapleyConfig(enable_pruning=False, sampling_rounds=50)
    start = time.time()
    result1 = pcmci_shapley.pcmci_shapley(data, config=config_no_prune, dataset="test")
    time_no_prune = time.time() - start
    
    # 啟用剪枝
    config_prune = PCMCIShapleyConfig(enable_pruning=True, sampling_rounds=50)
    start = time.time()
    result2 = pcmci_shapley.pcmci_shapley(data, config=config_prune, dataset="test")
    time_prune = time.time() - start
    
    # 應該要更快
    assert time_prune < time_no_prune
    print(f"Speedup: {time_no_prune / time_prune:.2f}x")
```

#### 5.3 執行測試
```bash
cd /Users/rosa_lai/development/RCAEval
pytest tests/test_pruning.py -v
pytest tests/test_pcmci_shapley_phase1.py -v
```

---

### 任務 6: 效能基準測試 (1 小時)

**新建**: `benchmark_phase1.py` (臨時腳本，完成後刪除)

```python
"""
Phase 1 優化效能基準測試
"""
import time
import numpy as np
import pandas as pd
from RCAEval.e2e import pcmci_shapley
from RCAEval.e2e.pcmci_shapley_modules import PCMCIShapleyConfig


def generate_test_data(n_services, n_metrics_per_service, T):
    """生成測試數據"""
    data = {'time': range(T)}
    for i in range(n_services):
        for metric in ['latency', 'cpu', 'mem']:
            if n_metrics_per_service >= 3 or metric == 'latency':
                col = f'S{i}_{metric}'
                data[col] = np.random.randn(T) + np.random.rand() * 5
    return pd.DataFrame(data)


def benchmark_scenario(name, n_services, T, config):
    """執行單一場景的基準測試"""
    print(f"\n{'='*60}")
    print(f"場景: {name}")
    print(f"服務數: {n_services}, 時間序列長度: {T}")
    print(f"剪枝: {'啟用' if config.enable_pruning else '停用'}")
    print(f"{'='*60}")
    
    data = generate_test_data(n_services, n_metrics_per_service=1, T=T)
    
    start = time.time()
    try:
        result = pcmci_shapley.pcmci_shapley(data, config=config, dataset="benchmark")
        elapsed = time.time() - start
        
        print(f"執行時間: {elapsed:.2f} 秒")
        print(f"Top-5 排名: {result['ranks'][:5]}")
        return elapsed, True
    except Exception as e:
        elapsed = time.time() - start
        print(f"執行失敗: {e}")
        print(f"失敗時間: {elapsed:.2f} 秒")
        return elapsed, False


def main():
    print("PCMCI-Shapley Phase 1 效能基準測試")
    print("="*60)
    
    scenarios = [
        ("Small", 10, 500),
        ("Medium", 20, 500),
        ("Large", 30, 500),
    ]
    
    results = []
    
    for name, n_services, T in scenarios:
        # 測試 1: 不啟用剪枝
        config_baseline = PCMCIShapleyConfig(
            enable_pruning=False,
            sampling_rounds=100
        )
        time_baseline, success_baseline = benchmark_scenario(
            f"{name} (Baseline)", n_services, T, config_baseline
        )
        
        # 測試 2: 啟用剪枝
        config_optimized = PCMCIShapleyConfig(
            enable_pruning=True,
            pruning_max_hops=2,
            pruning_anomaly_percentile=0.3,
            sampling_rounds=100,
            pcmci_max_conds_dim=3
        )
        time_optimized, success_optimized = benchmark_scenario(
            f"{name} (Optimized)", n_services, T, config_optimized
        )
        
        if success_baseline and success_optimized:
            speedup = time_baseline / time_optimized
            results.append({
                'scenario': name,
                'n_services': n_services,
                'time_baseline': time_baseline,
                'time_optimized': time_optimized,
                'speedup': speedup
            })
    
    # 輸出總結
    print("\n" + "="*60)
    print("效能總結")
    print("="*60)
    for r in results:
        print(f"{r['scenario']:10s} | 節點數: {r['n_services']:3d} | "
              f"Baseline: {r['time_baseline']:6.2f}s | "
              f"Optimized: {r['time_optimized']:6.2f}s | "
              f"加速: {r['speedup']:.2f}x")


if __name__ == "__main__":
    main()
```

執行:
```bash
python benchmark_phase1.py > phase1_benchmark_results.txt
```

---

## 時間估算

| 任務 | 預估時間 |
|------|---------|
| 1. 建立剪枝模組 | 3 小時 |
| 2. 整合剪枝到主流程 | 2 小時 |
| 3. PCMCI 參數優化 | 1 小時 |
| 4. 快速矩陣傳播 | 2 小時 |
| 5. 測試與驗證 | 2 小時 |
| 6. 效能基準測試 | 1 小時 |
| **總計** | **11 小時 (約 1.5 天)** |

---

## 預期成果

完成 Phase 1 後，應達到以下效果：

1. **性能提升**:
   - Small 場景 (10 節點): 2-3x 加速
   - Medium 場景 (20 節點): 4-6x 加速
   - Large 場景 (30 節點): 6-10x 加速

2. **可擴展性**:
   - 能夠處理 50 節點的系統 (原本 ~30)
   - 在 30 節點系統上執行時間從 ~30 分鐘降至 ~5 分鐘

3. **可配置性**:
   - 新增 5 個配置參數供使用者調整
   - 可透過 `enable_pruning=False` 回退到原始行為

4. **程式碼品質**:
   - 新增 1 個獨立模組 (pruning.py)
   - 單元測試覆蓋率 > 80%
   - 保持向後相容性

---

## 驗收標準

- [ ] 所有新程式碼通過 `flake8` 檢查
- [ ] 單元測試全部通過 (`pytest tests/test_pruning.py`)
- [ ] 整合測試全部通過 (`pytest tests/test_pcmci_shapley_phase1.py`)
- [ ] Medium 場景加速至少 3x
- [ ] 在 Online Boutique 數據集上驗證正確性 (Top-5 準確率 > 85%)
- [ ] 記憶體使用無明顯增加 (< 10%)

---

## 下一步 (Phase 2 預告)

完成 Phase 1 後，可進入 Phase 2:
- 實作 Node Isolation 平行化
- 實作 Shapley 採樣平行化
- 實作 Adaptive Sampling

預期在 Phase 1 基礎上再加速 3-5 倍。

