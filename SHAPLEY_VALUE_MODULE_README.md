# Shapley Value 模組使用指南

## 概述

`shapley_value.py` 是一個獨立的模組，提供基於合作賽局理論的公平貢獻分配算法，用於根因分析中的服務貢獻度量化。

## 主要功能

### 1. ShapleyValueCalculator 類
- **精確計算**：適用於小規模服務集合（≤10個服務）
- **蒙特卡羅近似**：適用於大規模服務集合
- **LRU 緩存**：提升計算效率
- **統計信息**：提供計算性能指標

### 2. 輔助函數
- `get_service_priority()`: 服務優先級分數計算
- `extract_base_service_name()`: 服務名稱提取
- `create_system_anomaly_function()`: 系統異常評估函數
- `calculate_shapley_contributions()`: 便捷計算函數

## 快速開始

### 基本使用

```python
from RCAEval.e2e.shapley_value import calculate_shapley_contributions

# 創建事件數據
events = [
    MockEvent('serviceA_w0', 0.8),  # 高異常分數
    MockEvent('serviceB_w0', 0.3),  # 低異常分數
    MockEvent('serviceC_w0', 0.2)   # 低異常分數
]

services = ['serviceA', 'serviceB', 'serviceC']

# 計算 Shapley Values
shapley_values = calculate_shapley_contributions(services, events)

# 找出根因
root_cause = max(shapley_values, key=shapley_values.get)
print(f"根因: {root_cause}")
```

### 高級使用

```python
from RCAEval.e2e.shapley_value import (
    ShapleyValueCalculator,
    create_system_anomaly_function
)

# 創建計算器
calculator = ShapleyValueCalculator(cache_size=1000)

# 創建系統異常函數
system_anomaly_func = create_system_anomaly_function(events)

# 計算 Shapley Values
shapley_values = calculator.calculate_shapley_values(
    services, 
    system_anomaly_func,
    max_services_for_exact=5
)

# 獲取統計信息
stats = calculator.get_statistics()
print(f"緩存命中率: {stats['cache_hit_rate']:.2%}")
```

## API 參考

### ShapleyValueCalculator

#### `__init__(cache_size: int = 10000)`
初始化計算器
- `cache_size`: 緩存大小限制

#### `calculate_shapley_values(services, system_anomaly_function, max_services_for_exact=10)`
計算所有服務的 Shapley Value
- `services`: 服務名稱列表
- `system_anomaly_function`: 系統異常評估函數
- `max_services_for_exact`: 使用精確計算的最大服務數量

#### `get_statistics()`
返回計算統計信息
- 緩存命中率、計算時間、緩存大小等

### 輔助函數

#### `get_service_priority(service_name: str) -> float`
根據服務名稱返回優先級分數
- Level 1 (0.6): 上游入口或核心服務
- Level 2 (0.5): 中間業務邏輯服務  
- Level 3 (0.4): 下游基礎設施或輔助服務

#### `extract_base_service_name(service_name: str) -> str`
從帶窗口標識的服務名提取基礎服務名
- `'serviceA_w0'` -> `'serviceA'`
- `'gateway_w2'` -> `'gateway'`

#### `create_system_anomaly_function(all_events) -> callable`
創建系統異常評估函數
- 返回一個函數，用於計算服務子集的系統異常分數

#### `calculate_shapley_contributions(services, all_events, max_services_for_exact=10) -> Dict[str, float]`
便捷函數：計算服務的 Shapley Value 貢獻度
- 自動創建系統異常函數和計算器
- 返回歸一化的 Shapley Values

## 使用示例

運行完整的使用示例：

```bash
python3 shapley_usage_example.py
```

## 理論基礎

### Shapley Value 公式

```
φᵢ = Σ [|S|!(n-|S|-1)! / n!] × [v(S∪{i}) - v(S)]
```

其中：
- `φᵢ`: 服務 i 的 Shapley Value
- `S`: 不包含服務 i 的所有可能聯盟
- `v(S)`: 聯盟 S 的價值（系統異常分數）

### 優良性質

1. **效率性**: 所有 Shapley Values 之和等於總價值
2. **對稱性**: 對稱的服務獲得相同的 Shapley Value
3. **虛擬性**: 不貢獻價值的服務獲得 0
4. **可加性**: 多個遊戲的 Shapley Value 可以相加

## 性能特點

- **精確計算**: O(2^n × n) 時間複雜度
- **蒙特卡羅近似**: O(M × n) 時間複雜度，M 為採樣次數
- **緩存優化**: LRU 緩存避免重複計算
- **自適應算法**: 根據服務數量自動選擇計算方法

## 注意事項

1. 服務數量 > 10 時建議使用蒙特卡羅近似
2. 緩存大小可根據內存情況調整
3. 系統異常函數的設計影響計算結果
4. 服務優先級權重已優化，減少對結果的影響

## 與 CPG 框架集成

該模組已集成到 CPG 框架中，可以通過以下方式使用：

```python
from RCAEval.e2e.cpg import cpg

# 使用 Shapley Value 方法
result = cpg(data, use_shapley=True)

# 使用原始 PageRank 方法
result = cpg(data, use_shapley=False)
```

