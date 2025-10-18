

# 方法：PCMCI 局部時滯因果檢定與 Shapley 傳播根因排名  
*(Local PCMCI-lag Causal + Shapley Propagation Ranking)*

---

## 一、方法概述（Overview）

### 目標與動機
在大型多服務系統中（如 RCAEval），要對故障焦點服務 \( F \) 進行根因分析 (Root Cause Analysis, RCA)，  
若直接推斷全網路的因果結構，會導致：
- 維度過高，因果檢定成本呈二次爆炸；
- 大量虛假相關與反向邊；
- 無法保證時間一致性。

因此，本方法提出一種 **局部化的因果推理流程：**

1. **Isolation Node Screening:**  
   篩選疑似受 \( F \) 事件影響的節點集合 \( U \)；
2. **Local PCMCI-lag 因果檢定:**  
   僅在 \( U \) 內對時間滯後進行條件因果測試；
3. **Trace–PCMCI 邊融合與加權傳播:**
   建立融合的局部加權圖 \( G_U = (U, E_U, w_{i→j}) \)；
4. **K步異常傳播與 Shapley 貢獻分解:**  
   模擬異常信號在 \( G_U \) 中的傳遞貢獻，
   最終輸出根因節點排名。

該流程保留方向性(time-aware)、具可解釋性(Shapley-level 解釋)，且運算負荷可控。

---

## 二、資料與記號定義（Data and Notation）

| 符號 | 定義 |
|:------|:------|
| \( S = \{s_1,...,s_N\} \) | 系統中所有服務節點 |
| \( F \) | 焦點服務（故障節點） |
| \([T−W, T]\) | 分析時間窗 |
| \( x_{s,m}(t) \) | 節點 \( s \) 的度量 \( m \) 在時間 \( t \) 的值 |
| \( a_{s,m}(t) \) | 一維異常分數（z-score/SPOT） |
| \( a_s(t) = \text{Agg}_m a_{s,m}(t) \) | 節點異常聚合分數 |
| \( X_s(t) \) | PCMCI 使用的表徵信號 (PCA 或 \( a_s(t) \)) |
| \( τ \in [1, τ_{max}] \) | 滯後階數 |
| \( U \subseteq S \) | 局部節點集合 (\(|U| ≤ 40\)) |
| \( E_U \subseteq U \times U \) | 局部邊集合 |
| \( w_{i→j} \) | 節點 \( i→j \) 邊權重 |
| \( δ_s \) | 節點瞬時異常強度 |
| \( φ_s \) | Shapley 貢獻值 |
| \( r_s \), \( p_s \) | 可達性指標與時間一致性懲罰 |

---

## 三、資料流程總覽（Data Flow）

```text
Raw metrics → Normalization → Anomaly Scoring a_{s,m}(t)
     ↓
[焦點 F, 窗口 [T–W,T]]
     ↓
(F-node Isolation + Trace Hop)
     ↓
局部節點集合 U
     ↓
PCMCI+ 滯後因果檢定
     ↓
融合 trace + PCMCI + Isolation 權重 w_{i→j}
     ↓
K步異常傳播
     ↓
Shapley 貢獻分解
     ↓
r_s 可達性 + p_s 時間懲罰修正
     ↓
Score 統合排序 → 根因排名
```

---

## 四、具體流程與演算法公式

---

### 步驟 0｜資料預處理與異常偵測

1. 指標標準化：
   \[
   \tilde{x}_{s,m}(t) = \frac{x_{s,m}(t) - \text{median}(x_{s,m})}{\text{MAD}(x_{s,m})}
   \]

2. 異常偵測（Robust Z 或 SPOT）：
   \[
   a_{s,m}(t) = 
   \begin{cases}
   |\tilde{x}_{s,m}(t)|, & \text{若超過閾值} \\
   0, & \text{否則}
   \end{cases}
   \]

3. 聚合服務異常：
   \[
   a_s(t) = \frac{1}{M_s} \sum_m a_{s,m}(t)
   \]

---

### 步驟 1｜局部節點建構

#### (1.1) 統計鄰域擴展
使用滯後互相關或偏相關：
\[
\rho_{F,s}(τ) = \text{corr}(a_F(t), a_s(t−τ))
\]
挑選 Top-M1 節點作為初步候選 \( U_0 \)。

#### (1.2) Isolation-based 節點篩選

對每候選節點，構造特徵向量：

\[
\mathbf{z}_s = [\rho_{F,s}(τ_1), \ldots, \rho_{F,s}(τ_{max})]
\]

使用 Isolation Forest \( \mathcal{I} \) 建模異常交互：
\[
\text{score}(s) = 1 - \text{outlier\_rate}_\mathcal{I}(\mathbf{z}_s)
\]

取前 Top-M2 個節點成為 \( U_1 \)。  
定義 isolation 信賴分數：
\[
I_s = \frac{\text{score}(s)}{\max_{k∈U_0}\text{score}(k)}
\]

#### (1.3) Trace 補強：
\[
U = U_1 ∪ \text{trace\_1hop}(U_1) ∪ \{F\}, \quad |U| ≤ U_{max}
\]

---

### 步驟 2｜PCMCI+ 局部滯後因果檢定

在 \( U \) 範圍內：

1. 建立時間序列矩陣 \( X = [X_s(t)]_{s∈U, t∈[T−W,T]} \)  
2. PCMCI 檢定：
   - 對每pair \( (i,j) \)，檢定：
     \[
     H_0: X_j(t−τ)\ ⟂\ X_i(t)\ |\ Z
     \]
     \[
     H_1: X_j(t−τ) \not\!\perp X_i(t)\ | Z
     \]
   - 使用偏相關 (ParCorr) 檢定統計量：
     \[
     \text{stat}_{j→i}(τ) = \frac{r_{j→i|Z} \sqrt{N−|Z|−3}}{\sqrt{1−r_{j→i|Z}^2}}
     \]
3. 保留顯著邊：
   \[
   E_{PCMCI} = \{(j→i, τ): p_{j→i}(τ) \le α, τ≥1\}
   \]
4. 定義時滯關聯強度：
   \[
   s_{j→i} = \max_{τ} |r_{j→i|Z}(τ)|
   \]

---

### 步驟 3｜多源邊權融合

三來源加權公式：

\[
w_{i→j} =
θ_1 · w^{trace}_{i→j}
+ θ_2 · s_{i→j}
+ θ_3 · I_i
\]

常用：\( θ_1:θ_2:θ_3 = 0.6:0.3:0.1 \)

若方向衝突：
\[
w_{i→j} = w_{i→j}(1 - γ \, \mathbf{1}[PCMCI(j→i)\text{成立}])
\]
其中懲罰係數 \( γ∈[0.3, 0.7] \)。

對於 \( j \) 的入邊歸一化：
\[
\hat{w}_{i→j} = \frac{w_{i→j}}{\sum_{k∈N_{in}(j)}w_{k→j}}
\]

---

### 步驟 4｜異常傳播模型（K步線性衰減）

初始化：
\[
h^{(0)}_s = δ_s = \log (1 + a_s(T))
\]

更新規則：
\[
h_j^{(k+1)} = α_{prop} \sum_{i∈N_{in}(j)} \hat{w}_{i→j} h_i^{(k)}
\]

最終整體影響：
\[
h_s = \sum_{k=0}^{K} h_s^{(k)}
\]

---

### 步驟 5｜Shapley Value 貢獻計算

定義局部系統異常指標 \( v(C) \)：  
當僅保留節點集合 \( C \) 的異常源度 \( δ_C \) 時，經傳播後系統異常為：

\[
v(C) = \sum_{j∈U} v_j \cdot h_j^{(C)}(K)
\]

Shapley 值：

\[
φ_s = \frac{1}{|U|!} 
       \sum_{π ∈ Π(U)} 
       [v(P_π(s) ∪ \{s\}) - v(P_π(s))]
\]

實作上採抽樣近似：
\[
\hat{φ}_s = \frac{1}{R} \sum_{r=1}^{R}
[v(C^{(r)}_{-s} ∪ \{s\}) - v(C^{(r)}_{-s})]
\]

---

### 步驟 6｜可達性與時間一致性懲罰

- **可達性 (Reachability)：**
  \[
  r_s = \max_{path(s→F)} \prod_{(i→j)∈path} \hat{w}_{i→j}
  \]

- **時間懲罰 (Temporal Penalty)：**
  若存在 \( s→…→u \) 且異常出現時間 \( t_u < t_s \)：
  \[
  p_s = \exp \left( -λ \sum_{violated\ paths} s_{i→j} \right)
  \]
  \( λ \)：懲罰幅度。

---

### 步驟 7｜綜合排名指標

經 Min–Max 正規化後：

\[
\text{Score}_s = α_1 \, \hat{φ}_s + α_2 \, \hat{r}_s + α_3 \, \hat{a}_s
\]
最終：
\[
\text{Rank}(s) = \text{argsort}(\text{Score}_s \cdot p_s)
\]

建議權重： \( α_1,α_2,α_3 = 0.5, 0.3, 0.2 \)

---

## 五、演算法流程（Algorithmic Form）

```python
Algorithm 1 : Local PCMCI-lag Causal + Shapley Propagation
Input: metrics {x_{s,m}(t)}, trace graph, focal F, window [T−W,T]
Output: ranked list of root-cause candidates

1: Preprocess data → a_{s,m}(t), a_s(t)
2: Compute lagged correlation ρ_{F,s}(τ), get U0
3: IsolationForest on {ρ_{F,s}} → get top nodes U1
4: U ← U1 ∪ TraceNeighbors(U1) ∪ {F}

5: Run PCMCI+(U, τ_max, α) → get E_PCMCI, s_{i→j}
6: Fuse edges and weights:
       w_{i→j} = θ1*w_trace + θ2*s_{i→j} + θ3*I_i
       normalize incoming weights

7: Compute δ_s = log(1+a_s(T))
8: Propagate anomalies over K steps:
       h_j^(k+1) = α_prop * Σ_i w_{i→j} h_i^(k)
       h ← Σ_k h^(k)
9: Estimate Shapley values φ_s by permutation sampling
10: Compute r_s and p_s for reachability & penalty
11: Score_s = α1*φ̂+α2*r̂+α3*â̂
12: Rank ← sort(Score_s * p_s, descending)
13: return Rank
```

---

## 六、整體數據流形式（Data Flow Model）

**資料流變換關係：**

\[
\begin{aligned}
x_{s,m}(t) &\xrightarrow[]{normalize} \tilde{x}_{s,m}(t)
\\
\tilde{x}_{s,m}(t) &\xrightarrow[]{SPOT/z} a_{s,m}(t)
\\
\{a_{s,m}(t)\}_m &\xrightarrow[]{agg} a_s(t)
\\
\{a_s(t)\}_{s∈S} &\xrightarrow[]{isolation,trace} U
\\
X_U(t) &\xrightarrow[]{PCMCI+} E_{PCMCI}, s_{i→j}
\\
TRACE + PCMCI + Isolation  &\xrightarrow[]{fusion} w_{i→j}
\\
δ_s &\xrightarrow[]{propagation} h_s
\\
h_s &\xrightarrow[]{Shapley,v(C)} φ_s
\\
(φ_s, r_s, p_s) &\xrightarrow[]{scoring} Rank(s)
\end{aligned}
\]

---

## 七、複雜度分析

| 模組 | 布朗界複雜度 | 備註 |
|------|----------------|------|
| Isolation-screening | \(O(|U_0|·τ_{max})\) | 輕量級 |
| PCMCI+ Local | \(O(|U|^2·τ_{max})\) | |U|≤40，秒級 |
| K步傳播 | \(O(K·|E_U|)\) | 線性 |
| Shapley Sampling | \(O(R·K·|E_U|)\) | 分鐘內 |
| Overall | 可線上運行或批次 |

---

## 八、方法特性與優勢
- ⚙️ **時間因果一致性**：PCMCI+ 提供嚴格的滯後向性；
- 🧩 **噪音與維度去除**：IsolationNode 預篩減少冗餘節點；
- 🔁 **混合加權因果網路**：融合觀察 trace、統計與因果；
- 💡 **可解釋性**：Shapley 分解展示各節點對整體異常的貢獻；
- 🚀 **可伸縮性**：每個 \(F\) 僅需局部子圖 \(U\)。
