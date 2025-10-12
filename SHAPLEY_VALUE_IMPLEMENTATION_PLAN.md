# Shapley Value 實現計畫

## 目標
將 Shapley Value（夏普利值）整合到 CPG 框架中，提升根因分析的精準度和可解釋性。

---

## 階段一：理論研究與背景調查

### 1.1 深入理解 Shapley Value 數學基礎
**目標**：掌握合作賽局理論核心概念

**關鍵概念**：
- **合作賽局**：N 個玩家組成聯盟，共同創造價值
- **特徵函數 v(S)**：聯盟 S 的總價值
- **Shapley Value 公式**：
  ```
  φᵢ = Σ [|S|!(n-|S|-1)! / n!] × [v(S∪{i}) - v(S)]
  ```
  其中：
  - φᵢ：玩家 i 的 Shapley Value
  - S：不包含玩家 i 的所有可能聯盟
  - v(S∪{i}) - v(S)：玩家 i 的邊際貢獻

**優良性質**：
1. **效率性**：所有玩家的 Shapley Value 總和 = 總價值
2. **對稱性**：可互換玩家獲得相同的價值
3. **虛擬性**：對任何聯盟都沒貢獻的玩家，Shapley Value = 0
4. **可加性**：兩個獨立賽局的 Shapley Value = 各自 Shapley Value 之和
5. **唯一性**：滿足上述公理的分配方案唯一

**參考資料**：
- Shapley, L. S. (1953). "A value for n-person games"
- Lundberg & Lee (2017). "A unified approach to interpreting model predictions" (SHAP)

---

### 1.2 文獻調研：Shapley Value 在相關領域的應用

**研究方向**：

1. **XAI 領域（可解釋性 AI）**
   - SHAP (SHapley Additive exPlanations)
   - 解釋深度學習模型的預測
   - TreeSHAP、KernelSHAP 等高效算法

2. **系統診斷領域**
   - 微服務根因分析
   - 網路故障定位
   - 分散式系統異常歸因

3. **相關論文**：
   - "Explainable AI for Root Cause Analysis"
   - "Game-Theoretic Attribution in Distributed Systems"
   - "Shapley Values for Microservice Dependency Analysis"

**調研產出**：
- 整理現有方法的優缺點對比表
- 識別可借鑒的算法優化技巧
- 找出與 CPG 框架的結合點

---

### 1.3 算法研究：精確計算 vs 近似算法

**精確計算**：
- **時間複雜度**：O(2^n × n)
- **空間複雜度**：O(2^n)
- **適用場景**：n ≤ 10（服務數量少）
- **優點**：完全準確
- **缺點**：組合爆炸

**近似算法**：

1. **蒙特卡羅採樣（本專案採用）**
   - 隨機生成服務排列
   - 計算每個排列下的邊際貢獻
   - 平均化得到近似值
   - 時間複雜度：O(M × n)，M 為採樣次數

2. **排列採樣（Permutation Sampling）**
   - 類似蒙特卡羅，但更系統化
   - 確保重要排列被採樣

3. **KernelSHAP**
   - 使用加權線性回歸
   - 適合高維特徵空間

**選擇策略**：
```
if 服務數量 ≤ 10:
    使用精確計算
else:
    使用蒙特卡羅近似（M = 1000 次採樣）
```

---

## 階段二：系統設計

### 2.1 設計系統異常函數 v(S)

**定義**：給定服務子集 S，計算系統整體異常分數

**設計考量**：

1. **異常分數聚合**
   ```python
   def v(service_subset):
       total_anomaly = 0
       for service in service_subset:
           # 獲取服務的最大異常分數（跨所有時間窗口）
           service_anomaly = max(event.anomaly_score 
                                 for event in all_events 
                                 if event.service_name == service)
           
           # 加權：考慮服務優先級
           priority_weight = get_service_priority(service)
           total_anomaly += service_anomaly * priority_weight
       
       # 歸一化
       return total_anomaly / len(service_subset) if service_subset else 0
   ```

2. **拓撲結構考慮**
   - 上游服務異常影響下游，需要傳播效應
   - 關鍵服務（如 frontend）權重更高

3. **時間相關性**
   - 同時發生的異常可能有因果關係
   - 時間窗口內的服務互動頻率

**多種設計方案**：

| 方案 | 異常聚合方式 | 優先級權重 | 拓撲權重 |
|------|-------------|-----------|---------|
| 基礎版 | 平均值 | 無 | 無 |
| 加權版 | 加權平均 | 是 | 無 |
| 拓撲版 | 傳播模型 | 是 | 是 |

**建議**：先實現基礎版和加權版，測試後決定是否需要拓撲版。

---

### 2.2 設計緩存策略

**目標**：避免重複計算相同服務子集的異常分數

**實現方案**：

1. **基礎緩存（字典）**
   ```python
   self.cache = {}  # key: tuple(sorted(subset)), value: anomaly_score
   
   def v_cached(service_subset):
       key = tuple(sorted(service_subset))
       if key in self.cache:
           return self.cache[key]
       
       score = v(service_subset)
       self.cache[key] = score
       return score
   ```

2. **LRU 緩存（限制內存）**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=10000)
   def v_cached(service_subset_tuple):
       return v(list(service_subset_tuple))
   ```

3. **緩存效益分析**
   - 精確計算：2^n 個子集，但只需計算一次
   - 蒙特卡羅：M × n 次計算，但許多子集重複
   - 預期緩存命中率：40-60%

**內存管理**：
- 設置最大緩存大小（如 10,000 條記錄）
- 超過限制時，清除最舊的記錄

---

### 2.3 設計組合爆炸閾值

**問題**：何時切換到近似算法？

**決策因素**：

1. **服務數量閾值**
   - n ≤ 10：精確計算（最多 1,024 組合）
   - n > 10：蒙特卡羅（組合數爆炸）

2. **計算時間限制**
   - 設置最大計算時間（如 30 秒）
   - 超時自動切換到近似算法

3. **內存限制**
   - 估計緩存內存需求
   - 超過系統內存 10% 則切換

**實現**：
```python
def calculate_shapley_values(self, services):
    n = len(services)
    
    # 決策邏輯
    if n <= 10:
        print(f"使用精確計算（{n} 個服務，{2**n} 組合）")
        return self._exact_shapley(services)
    else:
        print(f"使用蒙特卡羅近似（{n} 個服務，避免 {2**n} 組合）")
        return self._approximate_shapley(services, num_samples=1000)
```

---

## 階段三：核心實現

### 3.1 ShapleyValueCalculator 類結構

```python
class ShapleyValueCalculator:
    def __init__(self):
        self.cache = {}  # 緩存子集異常分數
        self.stats = {   # 統計信息
            'exact_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    # 主入口
    def calculate_shapley_values(self, services, system_anomaly_function, 
                                max_combinations=1000):
        """計算所有服務的 Shapley Value"""
        pass
    
    # 精確計算
    def _exact_shapley(self, services, system_anomaly_function):
        """遍歷所有可能的聯盟組合"""
        pass
    
    # 近似計算
    def _approximate_shapley(self, services, system_anomaly_function, 
                            num_samples=1000):
        """蒙特卡羅採樣估計"""
        pass
    
    # 緩存輔助
    def _get_subset_value(self, subset, system_anomaly_function):
        """帶緩存的子集價值計算"""
        pass
    
    # 統計報告
    def get_statistics(self):
        """返回計算統計信息"""
        return self.stats
```

---

### 3.2 精確計算實現細節

**算法流程**：

```python
def _exact_shapley(self, services, system_anomaly_function):
    n = len(services)
    shapley_values = {service: 0.0 for service in services}
    
    # 遍歷所有子集大小
    for subset_size in range(n + 1):
        # 遍歷該大小的所有子集
        for subset in itertools.combinations(services, subset_size):
            subset_list = list(subset)
            
            # 計算子集的異常分數（帶緩存）
            v_S = self._get_subset_value(subset_list, system_anomaly_function)
            
            # 對每個不在子集中的服務
            for service in services:
                if service not in subset:
                    # 計算加入該服務後的異常分數
                    extended_subset = subset_list + [service]
                    v_S_union_i = self._get_subset_value(
                        extended_subset, 
                        system_anomaly_function
                    )
                    
                    # 邊際貢獻
                    marginal_contribution = v_S_union_i - v_S
                    
                    # Shapley Value 公式的權重
                    weight = (math.factorial(subset_size) * 
                             math.factorial(n - subset_size - 1) / 
                             math.factorial(n))
                    
                    # 累加
                    shapley_values[service] += weight * marginal_contribution
    
    return shapley_values
```

**時間複雜度分析**：
- 外層循環：O(n)
- 子集枚舉：O(2^n)
- 每個服務計算：O(n)
- 總計：O(2^n × n)

---

### 3.3 蒙特卡羅近似實現細節

**算法流程**：

```python
def _approximate_shapley(self, services, system_anomaly_function, 
                        num_samples=1000):
    n = len(services)
    shapley_values = {service: 0.0 for service in services}
    
    # 蒙特卡羅採樣
    for sample_idx in range(num_samples):
        # 隨機排列服務順序
        permuted_services = np.random.permutation(services)
        
        # 對每個位置的服務計算邊際貢獻
        for i, service in enumerate(permuted_services):
            # 前綴（已加入的服務）
            prefix = list(permuted_services[:i])
            
            # 包含當前服務
            with_service = list(permuted_services[:i+1])
            
            # 計算邊際貢獻
            v_prefix = self._get_subset_value(prefix, system_anomaly_function)
            v_with = self._get_subset_value(with_service, system_anomaly_function)
            
            marginal_contribution = v_with - v_prefix
            
            # 累加
            shapley_values[service] += marginal_contribution
    
    # 平均化
    for service in shapley_values:
        shapley_values[service] /= num_samples
    
    return shapley_values
```

**時間複雜度**：O(M × n)，M 為採樣次數

**準確性分析**：
- M = 100：快速但較不準確
- M = 1000：平衡點（推薦）
- M = 10000：高準確但較慢

---

### 3.4 緩存機制實現

```python
def _get_subset_value(self, subset, system_anomaly_function):
    """帶緩存的子集價值計算"""
    # 生成緩存鍵（排序後的元組）
    cache_key = tuple(sorted(subset))
    
    # 檢查緩存
    if cache_key in self.cache:
        self.stats['cache_hits'] += 1
        return self.cache[cache_key]
    
    # 緩存未命中，計算
    self.stats['cache_misses'] += 1
    value = system_anomaly_function(subset)
    
    # 存入緩存
    self.cache[cache_key] = value
    
    return value
```

**緩存效益**：
- 精確計算：每個子集只計算一次
- 蒙特卡羅：許多排列共享相同的前綴子集

---

### 3.5 系統異常函數實現

```python
def create_system_anomaly_function(all_events, knowledge_base):
    """工廠函數：創建系統異常評估函數"""
    
    def system_anomaly_function(service_subset):
        if not service_subset:
            return 0.0
        
        total_anomaly = 0.0
        
        for service in service_subset:
            # 1. 獲取服務的異常分數（跨所有窗口的最大值）
            service_anomaly = 0.0
            for event in all_events:
                base_service = extract_base_service_name(event.service_name)
                if base_service == service and hasattr(event, 'anomaly_score'):
                    service_anomaly = max(service_anomaly, event.anomaly_score)
            
            # 2. 獲取服務優先級權重
            priority_weight = get_service_priority(service)
            
            # 3. 加權累加
            total_anomaly += service_anomaly * priority_weight
        
        # 4. 歸一化（平均）
        return total_anomaly / len(service_subset)
    
    return system_anomaly_function
```

**進階版本（考慮拓撲）**：
```python
def system_anomaly_function_with_topology(service_subset):
    # ... 基礎異常分數 ...
    
    # 額外考慮：服務間的依賴關係
    topology_bonus = 0.0
    for service in service_subset:
        # 如果該服務的下游也在子集中，增加異常分數（因為影響範圍大）
        downstream = get_downstream_services(service, knowledge_base)
        affected_downstream = [s for s in downstream if s in service_subset]
        topology_bonus += len(affected_downstream) * 0.1
    
    return (total_anomaly + topology_bonus) / len(service_subset)
```

---

## 階段四：整合到 CPG 框架

### 4.1 修改 CPGFramework 類

**位置**：`quantify_root_causes` 方法

```python
class CPGFramework:
    def __init__(self):
        self.pipeline = AdaptivePipeline()
        self.models = {}
        self.knowledge_base = _load_service_topology()
        self.shapley_calculator = ShapleyValueCalculator()  # 新增
    
    def quantify_root_causes(self, vertices, edges, all_events):
        """
        步驟5: 根因貢獻度量化
        使用 Shapley Value 替代 PageRank
        """
        print("Step 5: Root Cause Contribution Quantification (Shapley Value)")
        
        if not vertices:
            return {}
        
        # 1. 提取基礎服務名稱
        base_services = set()
        for vertex in vertices:
            base_service = extract_base_service_name(vertex)
            base_services.add(base_service)
        
        # 2. 創建系統異常函數
        system_anomaly_func = create_system_anomaly_function(
            all_events, 
            self.knowledge_base
        )
        
        # 3. 計算 Shapley Values
        shapley_values = self.shapley_calculator.calculate_shapley_values(
            list(base_services),
            system_anomaly_func
        )
        
        # 4. 歸一化到 [0, 1]
        if shapley_values:
            max_value = max(shapley_values.values())
            if max_value > 0:
                shapley_values = {
                    k: v / max_value 
                    for k, v in shapley_values.items()
                }
        
        print(f"Computed Shapley Values for {len(shapley_values)} services")
        print(f"Statistics: {self.shapley_calculator.get_statistics()}")
        
        return shapley_values
```

---

### 4.2 設計降級機制

```python
def quantify_root_causes(self, vertices, edges, all_events):
    """帶降級機制的根因量化"""
    
    try:
        # 嘗試使用 Shapley Value
        return self._quantify_with_shapley(vertices, edges, all_events)
    
    except Exception as e:
        print(f"Shapley Value 計算失敗: {e}")
        print("降級到 PageRank 方法")
        
        # 降級到原始 PageRank 方法
        return self._quantify_with_pagerank(vertices, edges, all_events)

def _quantify_with_shapley(self, vertices, edges, all_events):
    """使用 Shapley Value 的量化方法"""
    # ... Shapley 實現 ...
    pass

def _quantify_with_pagerank(self, vertices, edges, all_events):
    """使用 PageRank 的量化方法（原始方法）"""
    # ... 原始 PageRank 實現 ...
    pass
```

**降級觸發條件**：
1. Shapley 計算異常
2. 計算時間超過 60 秒
3. 內存不足

---

## 階段五：測試與驗證

### 5.1 單元測試

**測試案例設計**：

```python
# tests/test_shapley_value.py

def test_shapley_basic():
    """測試基本 Shapley Value 計算"""
    calculator = ShapleyValueCalculator()
    
    # 簡單案例：3 個服務
    services = ['serviceA', 'serviceB', 'serviceC']
    
    # 模擬異常函數：serviceA 最異常
    def v(subset):
        score = 0
        if 'serviceA' in subset:
            score += 0.8
        if 'serviceB' in subset:
            score += 0.3
        if 'serviceC' in subset:
            score += 0.2
        return score / len(subset) if subset else 0
    
    shapley_values = calculator.calculate_shapley_values(services, v)
    
    # 驗證：serviceA 應該有最高的 Shapley Value
    assert shapley_values['serviceA'] > shapley_values['serviceB']
    assert shapley_values['serviceA'] > shapley_values['serviceC']
    
    # 驗證效率性：總和 = 全集價值
    total_shapley = sum(shapley_values.values())
    full_set_value = v(services)
    assert abs(total_shapley - full_set_value) < 0.01

def test_shapley_approximation_accuracy():
    """測試近似算法的準確性"""
    calculator = ShapleyValueCalculator()
    services = ['s1', 's2', 's3', 's4', 's5']
    
    def v(subset):
        return len(subset) * 0.5
    
    # 精確計算
    exact_values = calculator._exact_shapley(services, v)
    
    # 近似計算（多次）
    approx_errors = []
    for _ in range(10):
        approx_values = calculator._approximate_shapley(
            services, v, num_samples=1000
        )
        
        # 計算誤差
        error = sum(abs(exact_values[s] - approx_values[s]) 
                   for s in services)
        approx_errors.append(error)
    
    avg_error = np.mean(approx_errors)
    print(f"平均誤差: {avg_error:.4f}")
    
    # 誤差應該 < 10%
    assert avg_error < 0.25  # 5 個服務，每個最多 0.05 誤差

def test_cache_effectiveness():
    """測試緩存效果"""
    calculator = ShapleyValueCalculator()
    services = ['s1', 's2', 's3', 's4']
    
    call_count = [0]
    def v(subset):
        call_count[0] += 1
        return len(subset)
    
    calculator.calculate_shapley_values(services, v)
    
    total_calls = call_count[0]
    cache_hits = calculator.stats['cache_hits']
    
    print(f"總調用: {total_calls}, 緩存命中: {cache_hits}")
    
    # 緩存命中率應該 > 0
    assert cache_hits > 0
```

---

### 5.2 性能測試

```python
def test_performance_scaling():
    """測試不同服務數量下的性能"""
    import time
    
    results = []
    
    for n in [5, 10, 15, 20, 30]:
        services = [f'service_{i}' for i in range(n)]
        
        def v(subset):
            return len(subset) * 0.3
        
        calculator = ShapleyValueCalculator()
        
        start_time = time.time()
        shapley_values = calculator.calculate_shapley_values(services, v)
        elapsed_time = time.time() - start_time
        
        results.append({
            'num_services': n,
            'time_seconds': elapsed_time,
            'method': '精確' if n <= 10 else '近似'
        })
        
        print(f"n={n}: {elapsed_time:.2f}s ({results[-1]['method']})")
    
    # 驗證：n > 10 時應該使用近似算法，時間不應該指數增長
    for i in range(1, len(results)):
        if results[i]['num_services'] > 10:
            time_ratio = results[i]['time_seconds'] / results[i-1]['time_seconds']
            # 近似算法的時間增長應該是線性的（< 2 倍）
            assert time_ratio < 3, f"時間增長過快: {time_ratio}x"
```

---

### 5.3 精準度測試（在實際數據集上）

```python
def test_accuracy_on_real_data():
    """在 Online Boutique 數據集上測試"""
    import pandas as pd
    from RCAEval.e2e.cpg import cpg
    
    # 載入測試數據
    data_path = "data/online-boutique/adservice_cpu/metrics.csv"
    data = pd.read_csv(data_path)
    
    # 真實根因
    ground_truth = "adservice"
    
    # 運行 CPG（使用 Shapley Value）
    result = cpg(data, inject_time=None, dataset="online-boutique")
    
    # 評估
    ranks = result['ranks']
    
    # 計算 AC@K
    ac_at_1 = 1 if ranks[0] == ground_truth else 0
    ac_at_3 = 1 if ground_truth in ranks[:3] else 0
    ac_at_5 = 1 if ground_truth in ranks[:5] else 0
    
    print(f"AC@1: {ac_at_1}, AC@3: {ac_at_3}, AC@5: {ac_at_5}")
    print(f"Top 5 排名: {ranks[:5]}")
    
    # 期望：至少在 Top-3 內
    assert ac_at_3 == 1, f"根因 {ground_truth} 不在 Top-3 內"
```

---

## 階段六：優化與調整

### 6.1 採樣策略優化

**問題**：蒙特卡羅採樣次數如何選擇？

**實驗設計**：
```python
def optimize_sampling_strategy():
    """找出最佳採樣次數（平衡精確度和效率）"""
    services = [f's{i}' for i in range(20)]
    
    def v(subset):
        return sum(hash(s) % 100 for s in subset) / len(subset) if subset else 0
    
    # 精確計算（作為基準）- 但 20 個服務太多，改用 10 個
    small_services = services[:10]
    calculator = ShapleyValueCalculator()
    exact_values = calculator._exact_shapley(small_services, v)
    
    # 測試不同採樣次數
    sampling_nums = [10, 50, 100, 500, 1000, 2000, 5000]
    
    for num_samples in sampling_nums:
        errors = []
        times = []
        
        for _ in range(5):  # 重複 5 次
            start = time.time()
            approx_values = calculator._approximate_shapley(
                small_services, v, num_samples=num_samples
            )
            elapsed = time.time() - start
            times.append(elapsed)
            
            # 計算誤差
            error = np.mean([abs(exact_values[s] - approx_values[s]) 
                            for s in small_services])
            errors.append(error)
        
        avg_error = np.mean(errors)
        avg_time = np.mean(times)
        
        print(f"採樣 {num_samples:5d} 次: "
              f"誤差 {avg_error:.4f}, 時間 {avg_time:.3f}s")
    
    # 結論：選擇誤差 < 5% 且時間合理的最小採樣次數
```

**預期結果**：
- 100 次：快速但誤差大（~10%）
- 1000 次：平衡點（誤差 ~3-5%，時間可接受）
- 5000 次：高精度但較慢（誤差 ~1%）

**建議配置**：
```python
SAMPLING_CONFIG = {
    'fast': 100,      # 快速模式
    'balanced': 1000, # 平衡模式（默認）
    'accurate': 5000  # 高精度模式
}
```

---

### 6.2 緩存策略優化

**LRU 緩存實現**：
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size=10000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            # 移動到末尾（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                # 移除最舊的項目
                self.cache.popitem(last=False)
```

**整合到 ShapleyValueCalculator**：
```python
class ShapleyValueCalculator:
    def __init__(self, cache_size=10000):
        self.cache = LRUCache(max_size=cache_size)
        # ...
    
    def _get_subset_value(self, subset, system_anomaly_function):
        cache_key = tuple(sorted(subset))
        
        cached_value = self.cache.get(cache_key)
        if cached_value is not None:
            self.stats['cache_hits'] += 1
            return cached_value
        
        self.stats['cache_misses'] += 1
        value = system_anomaly_function(subset)
        self.cache.put(cache_key, value)
        
        return value
```

---

## 階段七：評估與對比

### 7.1 全面評估計畫

**數據集**：
1. Online Boutique（25 個故障場景）
2. Sock Shop（類似規模）
3. Train Ticket（類似規模）

**對比方法**：
- Baseline: PageRank
- Proposed: Shapley Value
- Ablation: 不同系統異常函數設計

**評估指標**：
```python
def evaluate_rca_method(method_name, data_path, ground_truth):
    """評估 RCA 方法"""
    data = pd.read_csv(data_path)
    
    result = cpg(data, inject_time=None, dataset="online-boutique")
    ranks = result['ranks']
    
    # AC@K
    ac_at_k = {}
    for k in [1, 3, 5, 10]:
        ac_at_k[f'AC@{k}'] = 1 if ground_truth in ranks[:k] else 0
    
    # Rank
    rank = ranks.index(ground_truth) + 1 if ground_truth in ranks else len(ranks)
    
    # Reciprocal Rank
    rr = 1.0 / rank
    
    return {
        'method': method_name,
        'ground_truth': ground_truth,
        'rank': rank,
        'reciprocal_rank': rr,
        **ac_at_k
    }
```

**統計分析**：
```python
def compare_methods():
    """比較 Shapley Value vs PageRank"""
    
    # 收集所有測試案例的結果
    shapley_results = []
    pagerank_results = []
    
    for test_case in all_test_cases:
        # ... 運行兩種方法 ...
        shapley_results.append(result_shapley)
        pagerank_results.append(result_pagerank)
    
    # 統計比較
    print("Shapley Value vs PageRank")
    print(f"平均 AC@1: {np.mean([r['AC@1'] for r in shapley_results]):.2f} vs "
          f"{np.mean([r['AC@1'] for r in pagerank_results]):.2f}")
    print(f"平均 AC@3: {np.mean([r['AC@3'] for r in shapley_results]):.2f} vs "
          f"{np.mean([r['AC@3'] for r in pagerank_results]):.2f}")
    
    # 顯著性檢驗（配對 t 檢驗）
    from scipy.stats import ttest_rel
    
    shapley_ac3 = [r['AC@3'] for r in shapley_results]
    pagerank_ac3 = [r['AC@3'] for r in pagerank_results]
    
    t_stat, p_value = ttest_rel(shapley_ac3, pagerank_ac3)
    print(f"配對 t 檢驗: t={t_stat:.3f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("結論：Shapley Value 顯著優於 PageRank")
    else:
        print("結論：兩種方法無顯著差異")
```

---

### 7.2 可解釋性分析

**案例研究**：
```python
def analyze_interpretability(test_case):
    """分析 Shapley Value 的可解釋性"""
    
    # 運行 CPG
    result = cpg(test_case['data'], ...)
    
    # 獲取詳細的 Shapley 分解
    shapley_values = result.get('shapley_values', {})
    
    print(f"案例: {test_case['name']}")
    print(f"真實根因: {test_case['ground_truth']}")
    print("\nShapley Value 排名:")
    
    sorted_services = sorted(shapley_values.items(), 
                            key=lambda x: x[1], reverse=True)
    
    for i, (service, value) in enumerate(sorted_services[:5], 1):
        print(f"{i}. {service}: {value:.4f}")
        
        # 解釋：該服務的邊際貢獻
        print(f"   解釋: 當加入 {service} 時，系統異常分數平均增加 {value:.2%}")
    
    # 對比：真實根因的 Shapley Value
    if test_case['ground_truth'] in shapley_values:
        ground_truth_value = shapley_values[test_case['ground_truth']]
        print(f"\n真實根因 {test_case['ground_truth']} 的 Shapley Value: "
              f"{ground_truth_value:.4f}")
```

**可解釋性維度**：
1. **公平性**：每個服務的貢獻是公平分配的
2. **邊際性**：明確顯示「有你」vs「沒你」的差異
3. **可驗證性**：總和等於全系統異常分數（效率性公理）

---

## 階段八：文檔與總結

### 8.1 技術文檔

**內容大綱**：

```markdown
# Shapley Value 根因分析技術文檔

## 1. 理論背景
- 合作賽局理論
- Shapley Value 定義與性質
- 在 RCA 中的應用

## 2. 算法設計
- 精確計算算法
- 蒙特卡羅近似算法
- 系統異常函數設計

## 3. 實現細節
- ShapleyValueCalculator 類
- 緩存機制
- 性能優化

## 4. 使用指南
- 基本用法
- 參數配置
- 降級機制

## 5. 評估結果
- 精準度對比
- 性能分析
- 案例研究

## 6. 未來改進方向
- KernelSHAP 集成
- 動態採樣策略
- 增量計算
```

---

### 8.2 研究總結報告

**關鍵貢獻**：
1. 首次將 Shapley Value 應用於微服務 RCA
2. 設計了適合系統診斷的系統異常函數
3. 實現了高效的近似算法，支持大規模服務集

**實驗結果總結**：
- AC@1 提升: X% → Y%
- AC@3 提升: X% → Y%
- 可解釋性：提供每個服務的邊際貢獻度量

**學術價值**：
- 跨領域創新（XAI + 系統診斷）
- 堅實的理論基礎（賽局理論）
- 實用性（處理組合爆炸、緩存優化）

---

## 時間規劃

| 階段 | 任務 | 預計時間 |
|------|------|---------|
| 一 | 理論研究與背景調查 | 3-5 天 |
| 二 | 系統設計 | 2-3 天 |
| 三 | 核心實現 | 5-7 天 |
| 四 | 整合到 CPG | 2-3 天 |
| 五 | 測試與驗證 | 3-5 天 |
| 六 | 優化與調整 | 2-3 天 |
| 七 | 評估與對比 | 3-5 天 |
| 八 | 文檔與總結 | 2-3 天 |
| **總計** | | **22-34 天** |

---

## 風險與對策

### 風險 1：組合爆炸導致計算不可行
**對策**：
- 實現自動切換到近似算法
- 設置計算時間限制
- 優化緩存策略

### 風險 2：近似算法精確度不足
**對策**：
- 調整採樣次數
- 實驗驗證誤差範圍
- 提供精確模式選項

### 風險 3：系統異常函數設計不當
**對策**：
- 多種設計方案對比
- Ablation study
- 引入領域知識

### 風險 4：精準度提升不顯著
**對策**：
- 分析失敗案例
- 改進系統異常函數
- 結合貝葉斯網路

---

## 參考文獻

1. Shapley, L. S. (1953). "A value for n-person games"
2. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions" (SHAP)
3. Castro, J., et al. (2009). "Polynomial calculation of the Shapley value based on sampling"
4. Chen, J., et al. (2018). "L-Shapley and C-Shapley: Efficient model interpretation for structured data"

---

## 附錄

### A. Shapley Value 數學推導

詳細推導過程...

### B. 算法複雜度分析

時間和空間複雜度證明...

### C. 實驗數據詳細結果

完整的評估數據表格...
