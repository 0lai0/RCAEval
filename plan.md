## Graph-guided FastSHAP for RCAeval -- 完整實作計畫 (v7, 對齊專案)

---

## 零、前言

本計畫將 FastSHAP 的數學機制（代理模型、基準遮蔽、配對抽樣、效率懲罰）
與微服務 RCA 領域知識深度融合，在 RCAeval 基準上實現 GNN-based 的
根因定位方法。所有與 GNN、異質圖、ToD 基準等相關的部分均屬創新延伸設計。

本次改版（v7）基於對專案每一個關鍵檔案的逐行查證，
修正了過去版本在資料流、函式簽名、輸出格式、實作順序上的不一致。

---

## 一、目標

- **核心目標**：單次前向傳播（Amortized inference），直接輸出 metric-level
  Shapley 型根因分數，在毫秒級完成根因定位。
- **對照基線**：現有 `causalshap`（Monte Carlo Shapley，只輸出 service-level
  排序，metric-level 評估無法命中）。我們的方法同時涵蓋 service-level 與
  metric-level 評估。

---

## 二、資料流與對接點（以 `main.py` 為準）

### 2.1 main.py 呼叫方式（行 241-257）

```python
func = globals()[args.method]          # 以 method name 查到函式
out = func(
    data,               # pd.DataFrame，已是 normal+anomal 合併，含 time 欄位
    inject_time,         # int（unix timestamp）
    dataset=args.dataset,
    anomalies=None,
    dk_select_useful=False,
    sli=sli,             # str，如 "frontend_latency"
    verbose=False,
    n_iter=num_node,
    args=run_args,
)
root_causes = out.get("ranks")         # List[str]
```

因此我們的函式簽名必須是：

```python
def graph_fastshap(
    data,                  # pd.DataFrame
    inject_time=None,      # int
    dataset=None,          # str
    sli=None,              # str
    **kwargs,
) -> dict:                 # {"ranks": List[str]}
```

### 2.2 輸入 DataFrame 的欄位格式

實際 `data.csv` 的欄位名稱：

```
time, adservice_cpu, adservice_mem, adservice_latency,
      cartservice_cpu, cartservice_mem, cartservice_latency,
      frontend_cpu, frontend_mem, frontend_latency, ...
```

- 格式：`{service}_{metric_type}`
- metric_type 包括：`cpu`, `mem`, `latency` (P90), `load`, `error`,
  (`diskio`, `socket` 在某些資料集中)
- 經 `main.py` 行 209-215 預處理後，`_latency-90` 已被重命名為 `_latency`
- `main_*`, `PassthroughCluster_*`, `redis_*`, `frontend-external_*`
  等列在 `preprocess` / `drop_extra` 中會被過濾（但 `main.py` 預設
  `dk_select_useful=False`，所以這些列仍然存在）

### 2.3 inject_time 與資料切分

`main.py` 已經完成切分（行 200-203）：

```python
normal_df = data[data["time"] < inject_time].tail(length)
anomal_df = data[data["time"] >= inject_time].head(length)
data = pd.concat([normal_df, anomal_df], ignore_index=True)
```

傳進方法的 `data` 已經是裁剪好的（預設 20 分鐘 = 600 rows），
但方法內部仍需自行依 `inject_time` 再切 normal/anomal。

### 2.4 SLI 目標定義（行 218-238）

| 資料集 | 預設 SLI | 備註 |
|--------|----------|------|
| online-boutique / RE-OB | `frontend_latency` | 若故障 service 的 latency 存在則切換 |
| sock-shop-1 | `front-end_cpu` 或 `{service}_lat_90` | |
| sock-shop-2 / RE-SS | `front-end_cpu` 或 `{service}_latency` | |
| train-ticket / RE-TT | `ts-ui-dashboard_latency` | 若故障 service 的 latency 存在則切換 |

`main.py` 會把選好的 `sli` 字串傳入方法，我們直接使用即可。

### 2.5 輸出格式與評估（行 329-403）

**輸出**：`{"ranks": List[str]}`

- 字串格式：`"{service}_{metric_type}"` -- 即原始欄位名
  例如 `["checkoutservice_cpu", "frontend_latency", "cartservice_mem"]`

**評估**：

- **Service-level**：`main.py` 把每個 rank 字串以 `_` 切分取 `parts[0]`
  作為 service name，比對 `Node(service, "unknown")`。
  重複的 service 會去重。

- **Metric-level**：`main.py` 把每個 rank 字串以 `_` 切分取
  `parts[0]` 為 service、`parts[1]` 為 metric，
  比對 `Node(service, fault_metric)`。

- **Fault → Ground Truth metric 對應**（行 364-403）：

  | 故障類型 | Ground Truth metric |
  |---------|-------------------|
  | cpu | cpu |
  | mem | mem |
  | delay | latency |
  | loss | latency |
  | disk | diskio |
  | socket | socket |

**關鍵發現**：現有 `causalshap` 只回傳 service name（如 `"checkoutservice"`），
以 `_` 切分後 `parts` 長度為 1，metric 被設為 `"unknown"`，
因此 metric-level AC@K 必定為 0。
我們的方法直接回傳欄位名 `"service_metric"`，兩種評估都能正確命中。

### 2.6 方法註冊

在 `RCAEval/e2e/__init__.py` 中加入：

```python
from .graph_fastshap_method import graph_fastshap
```

`main.py` 透過 `globals()[args.method]` 即可找到。

---

## 三、技術棧

| 用途 | 套件 |
|------|------|
| 資料處理 | `pandas`, `numpy`, `networkx`（已在 requirements.txt） |
| GNN 模型 | `torch`（已在 requirements.txt, v1.12.1）, `torch_geometric`（**需新增**） |
| 因果發現 | `RCAEval/graph_construction/` 中的 pcmci / granger / pc（已存在） |
| 統計 | `scikit-learn`（已在 requirements.txt）, `scipy`（已在） |
| 預處理 | `RCAEval/io/time_series.preprocess`（已存在） |
| 評估 | `RCAEval/benchmark/evaluation.Evaluator`（已存在） |

**注意**：`torch_geometric` 未在 `requirements.txt` 中，需安裝並在
README/SETUP 中註明安裝方式（與 torch 1.12.1 / CUDA 版本對應）。

---

## 四、專案目錄結構

所有新增程式碼位於 `RCAEval/e2e/` 下，不干擾現有方法：

```
RCAEval/e2e/
  graph_fastshap_method.py          # 對外入口函式 graph_fastshap()
  graph_fastshap/
    __init__.py
    data_pipeline/
      __init__.py
      feature_engineer.py           # M1：偏差特徵、趨勢特徵計算
      build_hetero_graph.py         # M1：DataFrame → PyG HeteroData
      baseline_store.py             # M1：ToD/EWMA 基準維護
    models/
      __init__.py
      surrogate_gnn.py              # M2：HeteroSAGE 代理模型
      explainer_gnn.py              # M3：HeteroGAT 解釋器
    trainers/
      __init__.py
      train_surrogate.py            # M2：遮蔽抽樣 + 單調性約束訓練
      train_explainer.py            # M3：Paired Sampling + 三合一 Loss
      custom_loss.py                # WLS + Efficiency + Asymmetric Loss
    inference.py                    # M4：phi → ranks 轉換
```

---

## 五、四大模組詳細設計

### M1：Data Pipeline（DataFrame → HeteroData）

#### 5.1.1 輸入

方法收到的 `data` DataFrame 包含 `time` 欄位與 N 個 metric 欄位。
方法同時收到 `inject_time`（int）和 `sli`（str）。

#### 5.1.2 前處理步驟

```python
# 1. 過濾無用欄位（沿用現有 preprocess）
from RCAEval.io.time_series import preprocess
all_data = preprocess(data=data, dataset=dataset, dk_select_useful=False)

# 2. 依 inject_time 切分
normal_df = all_data[all_data["time"] < inject_time] if "time" in all_data else ...
anomal_df = all_data[all_data["time"] >= inject_time] if "time" in all_data else ...

# 3. 取得所有 metric 欄位（排除 time）
metric_cols = [c for c in all_data.columns if c != "time"]
```

#### 5.1.3 偏差特徵計算（feature_engineer.py）

對每個 metric column `c`：

1. **ToD 基準**：用 `normal_df[c]` 計算均值 mu 與標準差 sigma。
2. **偏差率**：`x_dev = (anomal_df[c] - mu) / (sigma + eps)`
3. **統計特徵**（在偏差空間上濃縮整個 anomal 時間窗為一個向量）：
   - 均值 `mean(x_dev)`
   - 最大值 `max(abs(x_dev))`
   - 斜率 slope（線性回歸）-- 區分「正在惡化」vs「已惡化到底」
   - 一階差分均值 `mean(diff(x_dev))`

每個 metric column 最終濃縮成一個 d 維特徵向量（d=4 或更多）。

#### 5.1.4 解析 service/metric 結構

```python
from RCAEval.e2e.pc_shapley_modules.propagation_shapley import extract_service_name

col_to_service = {}
col_to_metric_type = {}
services = set()

for c in metric_cols:
    svc = extract_service_name(c, dataset)
    parts = c.split("_")
    # metric type 是最後一個 _ 之後的部分
    metric_type = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
    col_to_service[c] = svc
    col_to_metric_type[c] = metric_type
    services.add(svc)
```

**重要**：直接沿用現有 `extract_service_name`，確保 service name
解析邏輯與 `causalshap` 一致。

#### 5.1.5 圖建構（build_hetero_graph.py）

**節點**：

- `metric` 節點：每個原始欄位一個（如 `frontend_cpu`, `frontend_latency`）
  特徵 = 偏差特徵向量 [d]
- `service` 節點：每個 unique service 一個（如 `frontend`）
  特徵 = 該 service 下所有 metric 特徵的均值或最大值 [d]

**邊**（三種 edge type）：

1. `("service", "calls", "service")`：
   - 來源：用 `RCAEval/graph_construction/` 的 pcmci 或 granger
     在 service-level 聚合資料上跑因果發現，得到 adjacency matrix，
     轉為 edge_index + edge_weight。
   - 原型階段可先用 Pearson correlation 建立。
   - **注意**：`pcmci()` / `granger()` 接受 DataFrame，回傳 np.ndarray。

2. `("service", "owns", "metric")`：
   - 依 `col_to_service` 對應建立。

3. `("metric", "belongs_to", "service")`：
   - 雙向邊，確保 GNN message passing 可雙向流動。

**輸出**：PyG `HeteroData` 物件。

```python
from torch_geometric.data import HeteroData
import torch

data_pyg = HeteroData()

# metric 節點特徵 [N_met, d]
data_pyg["metric"].x = torch.tensor(metric_features, dtype=torch.float)
data_pyg["metric"].col_names = metric_cols  # 記住原始欄位名，推論時要用

# service 節點特徵 [N_srv, d]
data_pyg["service"].x = torch.tensor(service_features, dtype=torch.float)
data_pyg["service"].names = list(services)

# 邊
data_pyg[("service", "calls", "service")].edge_index = ...  # [2, E_ss]
data_pyg[("service", "calls", "service")].edge_weight = ...  # [E_ss]
data_pyg[("service", "owns", "metric")].edge_index = ...     # [2, E_sm]
data_pyg[("metric", "belongs_to", "service")].edge_index = ...  # [2, E_ms]

# SLI 標籤
data_pyg.sli_col = sli                                # 方法收到的 sli 字串
data_pyg.sli_idx = metric_cols.index(sli) if sli in metric_cols else -1
```

### M2：Surrogate Trainer（代理模型）

#### 5.2.1 角色

代理模型 v(s) 評估「在遮蔽子集 s 下，系統 SLI 的異常程度」。
Surrogate 訓練完後凍結，作為 Explainer 的固定價值函數。

#### 5.2.2 模型架構

```
HeteroConv(SAGEConv) x n_layers → global readout → MLP → sigmoid → v(s)
```

- 使用 `torch_geometric.nn.HeteroConv` 包裝 `SAGEConv`。
- `n_layers`：預設 2-3，可增到 4-5 並加 residual/skip（train-ticket
  有較長因果鏈）。
- global readout：`global_mean_pool` + MLP。
- 最終輸出：scalar in [0,1]（異常機率模式，推薦），Loss 用 BCE。

#### 5.2.3 遮蔽機制

訓練時對每個樣本隨機抽 Bernoulli mask `s` (0/1 per metric node)：

- **特徵遮蔽**：`x_metric[i] = x_metric[i] * s[i]`。
  在偏差空間中 `s[i]=0` 等同「回到 ToD 健康基準」。
- **邊遮蔽**（稀疏，避免 OOM）：

  ```python
  src = edge_index[0]
  eps = 1e-4
  masked_edge_weight = edge_weight * (eps + (1 - eps) * s[src])
  ```

  保留極弱連通性避免圖斷鏈（soft masking）。

- **Target Masking**：SLI 所屬的 metric 節點特徵在訓練時強制設為 0，
  防止 Surrogate 直接看到 SLI 數值而捷徑學習。

#### 5.2.4 損失函數

```
L_surr = L_pred + mu * L_mono
```

- `L_pred`：BCE（對真實 SLI 異常機率的預測誤差）。
  真實標籤：SLI metric 在 anomal 時段的偏差值 > threshold 即為 1。
- `L_mono`（單調性約束）：在 batch 中取 s 與 s'（s' 比 s 多遮蔽
  一個異常節點），加入 hinge loss：
  `L_mono = max(0, v(s') - v(s) + eps)`

#### 5.2.5 訓練資料

- 使用每個故障 case 的 anomal 時段作為異常樣本（label=1）。
- 正常時段作為健康樣本（label=0）。
- 透過 Bernoulli masking 擴增「半異常」樣本。
- 跨資料集混合訓練以增加資料量（schema 需對齊）。

#### 5.2.6 凍結

```python
for param in surrogate.parameters():
    param.requires_grad = False
```

### M3：Explainer Trainer（FastSHAP 解釋器）

#### 5.3.1 模型架構

```
HeteroConv(GATConv) x n_layers → Linear (no ReLU) → phi_hat per metric node
```

- 使用 `HeteroConv(GATConv)` 讓注意力機制自動學習訊息路由。
- **最後一層必須是無激活函數的 Linear**：Shapley 值允許正負，
  ReLU 會破壞數學意義。
- 只對 `metric` 節點輸出 phi_hat，維度 = N_met。

#### 5.3.2 Paired Sampling 訓練迴圈

```python
for batch in dataloader:
    # 1. Explainer forward（完整輸入，不遮蔽）
    phi_hat = explainer(batch)               # [N_met]

    # 2. 抽樣 mask 與其補集
    s = torch.bernoulli(0.5 * torch.ones(N_met))
    s_bar = 1 - s

    # 3. Surrogate forward（凍結，no_grad）
    with torch.no_grad():
        v_0 = surrogate(batch, s=torch.zeros(N_met))   # 全遮蔽
        v_1 = surrogate(batch, s=torch.ones(N_met))     # 全保留
        v_s = surrogate(batch, s=s)
        v_s_bar = surrogate(batch, s=s_bar)

    # 4. 三合一 Loss
    loss = loss_fn(phi_hat, s, s_bar, v_0, v_1, v_s, v_s_bar, prior)
    loss.backward()
    optimizer.step()
```

#### 5.3.3 三合一 Loss（custom_loss.py）

```
Loss = L_wls + gamma * L_eff + lambda_ * L_asym
```

- **WLS（Weighted Least Squares）**：
  `L_wls = mean( (v_s - v_0 - s @ phi_hat)^2 + (v_s_bar - v_0 - s_bar @ phi_hat)^2 )`

- **Efficiency Penalty**：
  `L_eff = (v_1 - v_0 - sum(phi_hat))^2`
  - gamma 初始值設 0.01，不超過 0.1。RCA 場景相對排序比加總精確度重要。
  - 超參搜尋時包含 gamma=0 作為 ablation。

- **Asymmetric RCA Loss**：
  `L_asym = sum( (1 - prior[i]) * phi_hat[i]^2 )`
  - `prior[i]`：來自圖先驗（PageRank 或入度/出度比例）。
    低先驗 = 可能是下游受害者 → 施加更強 L2 懲罰壓抑其 phi。

### M4：Inference（推論 + ranks 輸出）

#### 5.4.1 推論流程

```python
def graph_fastshap(data, inject_time=None, dataset=None, sli=None, **kwargs):
    # 1. 前處理 + 切分
    normal_df, anomal_df, metric_cols = preprocess_and_split(data, inject_time, dataset)

    # 2. 特徵工程 → HeteroData
    hetero_data = build_hetero_graph(normal_df, anomal_df, metric_cols, dataset, sli)

    # 3. Explainer 前向（單次）
    phi_hat = explainer(hetero_data)  # [N_met]

    # 4. 只保留 phi > 0 的 metric，排序
    scores = {
        metric_cols[i]: phi_hat[i].item()
        for i in range(len(metric_cols))
        if phi_hat[i] > 0
    }

    # 5. 排序
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranks = [col_name for col_name, _ in ranked]

    # 6. 如果有 phi<=0 的 metric，附加到尾端（確保完整 coverage）
    remaining = [c for c in metric_cols if c not in ranks]
    ranks.extend(remaining)

    return {"ranks": ranks}
```

#### 5.4.2 輸出格式對齊

回傳的 `ranks` 即為原始欄位名列表，例如：

```python
["checkoutservice_cpu", "frontend_latency", "cartservice_mem", ...]
```

`main.py` 在評估時：
- Service-level：取 `"checkoutservice"` 部分，去重後比對
- Metric-level：取 `("checkoutservice", "cpu")`，比對 `Node(service, fault_metric)`

兩種評估都能正確命中。

#### 5.4.3 Service-level 聚合（Evaluator 已自動處理，但排序可優化）

若要在 service-level 得到更好排序：
- `phi_service[svc] = max(phi_hat[i] for i where col_to_service[metric_cols[i]] == svc)`
- 依 `phi_service` 排序後，展開回 metric-level 並置於前方。

但因為 `main.py` 的 service-level 評估已自動從 metric-level ranks 中
提取 service 並去重（行 329-346），所以只要 metric-level 排序合理，
service-level 也會合理。**無需額外聚合邏輯**。

---

## 六、與現有程式碼的交互關係

### 6.1 可直接沿用的現有工具

| 模組 | 位置 | 用途 |
|------|------|------|
| `preprocess()` | `RCAEval/io/time_series.py` | 過濾常數列、轉換 mem 單位 |
| `extract_service_name()` | `RCAEval/e2e/pc_shapley_modules/propagation_shapley.py` | 從欄位名解析 service name |
| `pcmci()` | `RCAEval/graph_construction/pcmci.py` | 因果發現，回傳 np.ndarray |
| `granger()` | `RCAEval/graph_construction/granger.py` | 因果發現，回傳 np.ndarray |
| `Evaluator` | `RCAEval/benchmark/evaluation.py` | AC@K / Avg@5 計算 |
| `Node` | `RCAEval/classes/graph.py` | 評估比對用 |

### 6.2 不能修改的現有檔案（只讀）

- `main.py`（評估入口）
- `RCAEval/benchmark/evaluation.py`
- `RCAEval/classes/graph.py`
- `RCAEval/graph_construction/*.py`

### 6.3 需新增/修改的檔案

- **新增** `RCAEval/e2e/graph_fastshap_method.py`（方法入口）
- **新增** `RCAEval/e2e/graph_fastshap/`（模型、訓練、推論子目錄）
- **修改** `RCAEval/e2e/__init__.py`（加入 import）
- **修改** `requirements.txt`（加入 torch_geometric 及相關依賴）

---

## 七、盲點總覽與修正策略

### 原始盲點（A-D）

| # | 盲點 | 修正 |
|---|------|------|
| A | 遮蔽的物理意義不明確 | 偏差空間中 s=0 即為 ToD 健康基準 |
| B | 動態拓樸與圖結構隨時間變動 | 從 metrics 用因果發現推圖（pcmci/granger/correlation） |
| C | Explainer 在破碎子集上 OOD 崩潰 | Surrogate 預訓練 + Bernoulli 遮蔽擴增 |
| D | 先驗計算成本與符號解讀問題 | 先驗拼接特徵離線計算；僅正值 Shapley 排序 |

### 進階盲點（E-W）

| # | 盲點 | 修正 |
|---|------|------|
| E | GNN message passing 與特徵遮蔽物理衝突 | Edge masking：`edge_weight * (eps + (1-eps)*s[src])` |
| F | Surrogate 反事實 Ground Truth 缺失 | 單調性約束 L_mono |
| G | 靜態特徵無法捕捉因果時序延遲 | Slope + 一階差分均值作為趨勢特徵 |
| H | Efficiency Penalty 非線性失真 | gamma 極小初始化（0.01），可設 0 |
| I | SLI 尺度與 Loss 衝突 | 機率模式：SLI 映射到 [0,1]，用 BCE |
| J | 邊遮蔽實作不當導致 OOM | 全程稀疏 `edge_index` + `edge_weight * s[src]` |
| K | 網路級故障導致動態圖斷鏈 | 本 repo 無 traces；改用 metric-inferred 圖 + 靜態混合 |
| L | Service-Metric 邊方向性不足 | 雙向邊 `owns` / `belongs_to` |
| M | Surrogate 訓練嚴重樣本不平衡 | 分層抽樣 + oversampling 異常 case |
| N | 特徵工程拖垮推論速度 | 偏差特徵計算輕量（只需 mean/std），slope 用 numpy 即可 |
| O | Efficiency Penalty 與子圖數學衝突 | 全圖 message passing；若用 subgraph 則 gamma=0 |
| P | Metric→Service 聚合不公平 | main.py 已自動處理 service-level 去重 |
| Q | 因果圖建構時間抵銷推論優勢 | 離線/背景建圖 + cache；線上讀 cache |
| R | Surrogate 捷徑學習 | Target Masking（SLI 節點特徵強制為 0） |
| S | 概念漂移導致 ToD 基準失效 | EWMA 更新 baseline；或按 case 重新計算 |
| T | GAT 輸出帶 ReLU 破壞 Shapley 意義 | 最後一層 Linear 無激活函數 |
| U | GNN 感受野與微服務深度不匹配 | n_layers 可增到 4-5 + residual/skip |
| V | Surrogate 資料飢渴與 Overfitting | 跨資料集預訓練 + Gaussian noise augmentation |
| W | Edge Masking 導致圖斷鏈 | Soft masking（eps=1e-4 保留微弱連通性） |

---

## 八、超參數建議

| 超參數 | 含義 | 建議初始值 | 搜尋範圍 |
|--------|------|------------|----------|
| gamma | Efficiency Penalty 權重 | 0.01 | {0, 0.01, 0.05, 0.1} |
| lambda_ | Asymmetric RCA Loss 權重 | 1.0 | {0.1, 0.5, 1.0, 5.0} |
| mu | 單調性約束權重 | 0.1 | {0.01, 0.1, 0.5} |
| n_layers | GNN 層數 | 3 | {2, 3, 4, 5} |
| hidden_dim | GNN 隱藏維度 | 64 | {32, 64, 128} |
| n_samples | 每 batch 遮蔽抽樣數 | 16 | {8, 16, 32} |
| feature_dim | 每 metric 特徵維度 | 4 | {4, 6, 8} |
| causal_method | 圖建構方法 | "pcmci" | {"pcmci", "granger", "correlation"} |

---

## 九、Ablation 實驗建議

1. **Edge masking vs 僅特徵 masking**：驗證盲點 E 修正效果
2. **有/無單調性約束**：驗證 L_mono 對 Surrogate 品質的影響
3. **有/無趨勢特徵**：驗證 slope + diff 在級聯故障上的區分力
4. **gamma=0 vs gamma>0**：驗證 Efficiency Penalty 對 Top-K 的影響
5. **Paired Sampling vs 單一抽樣**：驗證梯度穩定度
6. **本方法 vs causalshap (Monte Carlo)**：推論速度與精度比較
7. **pcmci vs granger vs correlation**：圖建構方法對結果的影響

---

## 十、實作順序（Bottom-Up）

### Phase 1：Data Pipeline + HeteroData 原型

**目標**：把一個 RCAEval case 的 DataFrame 穩定轉成 PyG HeteroData。

**行動**：
1. 寫 `feature_engineer.py`：normal/anomal 切分 → 偏差特徵計算
2. 寫 `build_hetero_graph.py`：
   - 解析 service/metric 結構
   - Pearson correlation 建 service-service 邊（原型，後續可換 pcmci）
   - 建立 owns / belongs_to 邊
   - 輸出 HeteroData
3. 驗證 HeteroData 的 shape 與 edge_index 正確

**驗收**：
- 能對 `data/online-boutique/adservice_cpu/1/data.csv` 產出有效 HeteroData
- 全程稀疏，無 dense adjacency

### Phase 2：Surrogate 訓練

**目標**：在 Phase 1 產出的 HeteroData 上訓練 Surrogate，能正確預測 SLI 異常。

**行動**：
1. 寫 `surrogate_gnn.py`：HeteroConv(SAGEConv) + readout + MLP
2. 寫 `train_surrogate.py`：
   - Bernoulli masking + edge masking
   - Target Masking
   - L_pred (BCE) + L_mono (hinge)
3. 掃過多個 case 訓練

**驗收**：
- Loss 穩定下降
- 手動遮蔽已知根因後，v(s) 顯著降低
- 凍結權重匯出

### Phase 3：Explainer 訓練

**目標**：在凍結 Surrogate 上訓練 Explainer，能輸出合理的 phi_hat。

**行動**：
1. 寫 `explainer_gnn.py`：HeteroConv(GATConv) + Linear 輸出
2. 寫 `custom_loss.py`：WLS + Efficiency + Asymmetric
3. 寫 `train_explainer.py`：Paired Sampling 迴圈

**驗收**：
- phi > 0 集中在少數 metric 節點，不均勻塗抹
- 已知根因的 metric 排名靠前

### Phase 4：推論對接 RCAeval

**目標**：封裝成 `graph_fastshap()` 函式，接入 `main.py` 跑完整評估。

**行動**：
1. 寫 `graph_fastshap_method.py`：串接 M1→M3 推論，輸出 `{"ranks": [...]}`
2. 在 `__init__.py` 註冊
3. 跑 `python main.py --method graph_fastshap --dataset online-boutique`

**驗收**：
- 能跑完整個資料集的所有 case
- 產出 service-level 與 metric-level 的 AC@K / Avg@5
- 與 causalshap 做 side-by-side 比較

---

## 十一、總結

本計畫的核心差異化：

1. **Amortized inference**：相比 causalshap 的 O(N! * n_permutations) 取樣，
   我們只需一次 GNN forward pass。

2. **Metric-level 根因定位**：causalshap 只能回傳 service-level 排序
   （metric-level AC@K 永遠為 0），我們直接輸出 `"service_metric"`
   格式，兩種評估都能命中。

3. **Graph-guided**：透過異質圖建模 service-metric 結構與因果關係，
   比線性排序方法（如 baro, nsigma）能更好處理級聯故障。

4. **可解釋**：Shapley 值具有嚴格的博弈論基礎，每個 metric 的
   貢獻分數有明確的邊際貢獻解釋。

所有新增程式碼位於 `RCAEval/e2e/graph_fastshap/` 下，
不影響現有方法，輸出格式完全對齊 `main.py` 的評估管線。
