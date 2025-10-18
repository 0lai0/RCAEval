# PCMCI-Shapley 公式快速參考

**快速查閱**: 實作時的公式對照表  
**版本**: v1.0  
**日期**: 2025-10-15

---

## 步驟 0: 資料預處理

### 0.1 Robust 標準化
```
x̃_{s,m}(t) = (x_{s,m}(t) - median(x_{s,m})) / MAD(x_{s,m})

MAD(x) = median(|x - median(x)|)
```

**實作**: `preprocessing.py::robust_normalize()`

---

### 0.2 異常偵測 (Z-score)
```
a_{s,m}(t) = {
    |x̃_{s,m}(t)|,  if |x̃_{s,m}(t)| > threshold
    0,              otherwise
}
```

**實作**: `preprocessing.py::detect_anomaly_zscore()`  
**預設閾值**: threshold = 3.0

---

### 0.3 節點異常聚合
```
a_s(t) = (1/M_s) Σ_{m=1}^{M_s} a_{s,m}(t)
```

**實作**: `preprocessing.py::aggregate_node_anomaly()`

---

## 步驟 1: 局部節點建構

### 1.1 滯後互相關
```
ρ_{F,s}(τ) = corr(a_F(t), a_s(t-τ)),  τ ∈ [1, τ_max]
```

**實作**: `node_isolation.py::compute_lagged_correlation()`  
**預設**: τ_max = 5

---

### 1.2 Isolation 特徵向量
```
z_s = [ρ_{F,s}(1), ρ_{F,s}(2), ..., ρ_{F,s}(τ_max)]
```

**實作**: `node_isolation.py::build_isolation_features()`

---

### 1.3 Isolation 分數
```
I_s = score(s) / max_{k∈U_0} score(k)

score(s) = 1 - outlier_rate_{IsolationForest}(z_s)
```

**實作**: `node_isolation.py::isolation_forest_selection()`

---

### 1.4 局部節點集合
```
U = U_1 ∪ trace_1hop(U_1) ∪ {F}

|U| ≤ U_max
```

**實作**: `node_isolation.py::trace_augmentation()`  
**預設**: U_max = 40

---

## 步驟 2: PCMCI+ 因果檢定

### 2.1 條件獨立檢定
```
H_0: X_j(t-τ) ⊥ X_i(t) | Z
H_1: X_j(t-τ) ⫫ X_i(t) | Z
```

**實作**: `pcmci_local.py::run_pcmci_plus()`

---

### 2.2 偏相關檢定統計量
```
stat_{j→i}(τ) = r_{j→i|Z} × √(N - |Z| - 3) / √(1 - r_{j→i|Z}²)
```

**實作**: 由 tigramite.PCMCI 內部實作

---

### 2.3 顯著邊集合
```
E_PCMCI = {(j→i, τ) : p_{j→i}(τ) ≤ α, τ ≥ 1}
```

**實作**: `pcmci_local.py::extract_significant_edges()`  
**預設**: α = 0.05

---

### 2.4 時滯關聯強度
```
s_{j→i} = max_{τ∈[1,τ_max]} |r_{j→i|Z}(τ)|
```

**實作**: `pcmci_local.py::compute_edge_strength()`

---

## 步驟 3: 邊權融合

### 3.1 三源加權融合
```
w_{i→j} = θ_1 × w^{trace}_{i→j} + θ_2 × s_{i→j} + θ_3 × I_i
```

**實作**: `edge_fusion.py::fuse_edge_weights()`  
**預設權重**: θ_1:θ_2:θ_3 = 0.6:0.3:0.1

---

### 3.2 方向衝突懲罰
```
w_{i→j} = w_{i→j} × (1 - γ × 𝟙[PCMCI(j→i) 成立])
```

**實作**: `edge_fusion.py::apply_conflict_penalty()`  
**預設**: γ = 0.5

---

### 3.3 入邊歸一化
```
ŵ_{i→j} = w_{i→j} / Σ_{k∈N_in(j)} w_{k→j}
```

**實作**: `edge_fusion.py::normalize_incoming_weights()`

---

## 步驟 4: 異常傳播

### 4.1 初始異常強度
```
δ_s = log(1 + a_s(T))

h^{(0)}_s = δ_s
```

**實作**: `propagation.py::compute_initial_anomaly()`

---

### 4.2 單步傳播更新
```
h_j^{(k+1)} = α_prop × Σ_{i∈N_in(j)} ŵ_{i→j} × h_i^{(k)}
```

**實作**: `propagation.py::propagate_one_step()`  
**預設**: α_prop = 0.85

---

### 4.3 累積影響
```
h_s = Σ_{k=0}^{K} h_s^{(k)}
```

**實作**: `propagation.py::propagate_k_steps()`  
**預設**: K = 5

---

## 步驟 5: Shapley Value

### 5.1 聯盟值函數
```
v(C) = Σ_{j∈U} v_j × h_j^{(C)}(K)

h_j^{(C)}(K) = K步傳播結果（僅保留 C 中節點的初始異常）
```

**實作**: `shapley.py::compute_system_anomaly()`

---

### 5.2 精確 Shapley 值
```
φ_s = (1/|U|!) × Σ_{π∈Π(U)} [v(P_π(s) ∪ {s}) - v(P_π(s))]

P_π(s) = π 中 s 之前的節點集合
```

**實作**: `shapley.py::compute_shapley_exact()`  
**適用**: |U| ≤ 15

---

### 5.3 抽樣近似 Shapley 值
```
φ̂_s = (1/R) × Σ_{r=1}^{R} [v(C_{-s}^{(r)} ∪ {s}) - v(C_{-s}^{(r)})]

C_{-s}^{(r)} ~ Uniform(2^{U\{s}})
```

**實作**: `shapley.py::compute_shapley_sampling()`  
**預設**: R = 500

---

### 5.4 Shapley 值性質
```
Σ_{s∈U} φ_s = v(U) - v(∅)  (Efficiency)
φ_s = 0 if s is dummy      (Dummy)
```

---

## 步驟 6: 可達性與時間懲罰

### 6.1 可達性分數
```
r_s = max_{path∈Paths(s→F)} Π_{(i→j)∈path} ŵ_{i→j}
```

**實作**: `scoring.py::compute_reachability()`  
**演算法**: Dijkstra 最大權重路徑

---

### 6.2 時間一致性懲罰
```
p_s = exp(-λ × Σ_{violated paths} s_{i→j})

Violated: ∃ path s→...→u, t_u < t_s (異常時間違反因果順序)
```

**實作**: `scoring.py::compute_temporal_penalty()`  
**預設**: λ = 1.0

---

## 步驟 7: 綜合評分與排名

### 7.1 正規化
```
x̂ = (x - min(x)) / (max(x) - min(x))
```

**實作**: `utils.py::min_max_normalize()`

---

### 7.2 綜合分數
```
Score_s = α_1 × φ̂_s + α_2 × r̂_s + α_3 × â_s

φ̂_s: 正規化 Shapley 值
r̂_s: 正規化可達性
â_s: 正規化異常分數
```

**實作**: `scoring.py::compute_comprehensive_score()`  
**預設權重**: α_1:α_2:α_3 = 0.5:0.3:0.2

---

### 7.3 最終排名
```
Rank(s) = argsort(Score_s × p_s, descending)
```

**實作**: `scoring.py::compute_final_ranking()`

---

## 輔助公式

### MAD (Median Absolute Deviation)
```
MAD(x) = median(|x - median(x)|)
```

---

### 皮爾森相關係數
```
corr(X, Y) = Cov(X, Y) / (σ_X × σ_Y)
```

---

### 偏相關係數
```
r_{X,Y|Z} = (r_{X,Y} - r_{X,Z} × r_{Y,Z}) / √((1 - r_{X,Z}²) × (1 - r_{Y,Z}²))
```

---

### 對數變換
```
log(1 + x)  # 避免 log(0)，壓縮大值
```

---

## 複雜度分析

| 模組 | 時間複雜度 | 空間複雜度 |
|------|-----------|-----------|
| 預處理 | O(T × N × M) | O(T × N × M) |
| 節點隔離 | O(N × τ_max) | O(N × τ_max) |
| IsolationForest | O(n_trees × U_0 × τ_max × log(U_0)) | O(U_0 × τ_max) |
| PCMCI | O(\|U\|² × τ_max × T) | O(\|U\|² × τ_max) |
| 邊權融合 | O(\|E_U\|) | O(\|E_U\|) |
| K步傳播 | O(K × \|E_U\|) | O(\|U\|) |
| Shapley 抽樣 | O(R × K × \|E_U\|) | O(\|U\|) |
| Shapley 精確 | O(\|U\|! × K × \|E_U\|) | O(\|U\|) |
| 評分排名 | O(\|U\| × \|E_U\|) | O(\|U\|) |

**整體**: O(R × K × \|E_U\|) 當 |U| ≤ 40, R = 500, K = 5

---

## 超參數快速參考

| 參數 | 符號 | 預設值 | 範圍 | 說明 |
|------|------|--------|------|------|
| 最大時滯 | τ_max | 5 | [3, 10] | PCMCI 檢測的時間滯後 |
| PCMCI 顯著性 | α | 0.05 | (0, 0.1] | 條件獨立檢定閾值 |
| 局部節點數 | U_max | 40 | [20, 60] | 最大局部節點數 |
| Trace 權重 | θ_1 | 0.6 | [0, 1] | 邊權融合：trace |
| PCMCI 權重 | θ_2 | 0.3 | [0, 1] | 邊權融合：因果 |
| Isolation 權重 | θ_3 | 0.1 | [0, 1] | 邊權融合：隔離分數 |
| 衝突懲罰 | γ | 0.5 | [0, 1] | 方向衝突懲罰 |
| 傳播步數 | K | 5 | [3, 10] | 異常傳播迭代次數 |
| 傳播衰減 | α_prop | 0.85 | (0, 1) | 傳播衰減係數 |
| Shapley 抽樣 | R | 500 | [100, 2000] | 蒙特卡洛抽樣次數 |
| Shapley 權重 | α_1 | 0.5 | [0, 1] | 綜合評分：Shapley |
| 可達性權重 | α_2 | 0.3 | [0, 1] | 綜合評分：可達性 |
| 異常權重 | α_3 | 0.2 | [0, 1] | 綜合評分：異常 |
| 時間懲罰 | λ | 1.0 | [0, 5] | 時間一致性懲罰強度 |

**約束**: θ_1 + θ_2 + θ_3 = 1, α_1 + α_2 + α_3 = 1

---

## 程式碼片段示例

### 計算滯後相關
```python
def compute_lagged_correlation(focus, other, tau_max):
    correlations = np.zeros(tau_max)
    for tau in range(1, tau_max + 1):
        correlations[tau - 1] = np.corrcoef(
            focus[tau:], 
            other[:-tau]
        )[0, 1]
    return correlations
```

### 單步傳播
```python
def propagate_one_step(h_current, adj_weights, alpha_prop):
    h_next = {}
    for j in nodes:
        incoming = sum(
            adj_weights[i, j] * h_current[i]
            for i in neighbors_in(j)
        )
        h_next[j] = alpha_prop * incoming
    return h_next
```

### Shapley 抽樣
```python
def compute_shapley_sampling(nodes, value_func, R):
    phi = {s: 0.0 for s in nodes}
    for _ in range(R):
        # 隨機排列
        perm = np.random.permutation(nodes)
        for idx, s in enumerate(perm):
            coalition = set(perm[:idx])
            marginal = value_func(coalition | {s}) - value_func(coalition)
            phi[s] += marginal
    return {s: phi[s] / R for s in nodes}
```

---

## 常見問題 (FAQ)

### Q1: τ_max 應該設多少？
**A**: 取決於資料採樣頻率。若每分鐘採樣，τ_max=5 表示考慮 5 分鐘內的因果關係。建議根據系統響應時間調整。

### Q2: Shapley 計算太慢怎麼辦？
**A**: 
- 減少 R（但不低於 100）
- 使用精確計算（若 |U| ≤ 15）
- 考慮降級為簡單加權分數

### Q3: 無 trace graph 怎麼辦？
**A**: 設置 θ_1 = 0, θ_2 = 0.8, θ_3 = 0.2，降級為純統計 + 因果方法。

### Q4: 如何調整 θ_1, θ_2, θ_3？
**A**: 
- 高品質 trace：θ_1 = 0.7, θ_2 = 0.2, θ_3 = 0.1
- 低品質 trace：θ_1 = 0.3, θ_2 = 0.5, θ_3 = 0.2
- 無 trace：θ_1 = 0, θ_2 = 0.8, θ_3 = 0.2

### Q5: |U| 超過 40 怎麼辦？
**A**: 增強 IsolationForest 篩選，或增加 top_m2 的篩選力度，或提高相關性閾值。

---

## 實作檢查清單

- [ ] 所有公式都有對應的函數實作
- [ ] 每個函數都有單元測試
- [ ] 預設超參數已配置
- [ ] 輸入驗證已實作
- [ ] 異常處理已添加
- [ ] Logging 已添加
- [ ] 文檔字符串已完成
- [ ] 性能優化已考慮
- [ ] 端到端測試通過

---

**文件版本**: v1.0  
**最後更新**: 2025-10-15  
**用途**: 實作時的快速參考


