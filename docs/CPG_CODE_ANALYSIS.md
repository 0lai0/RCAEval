# CPG 程序代码解析文档

## 概述

CPG（Causal Propagation Graph，无监督多模态事件驱动因果传播框架）是一个创新的微服务根因分析方法，专门设计用于实时、流式处理多模态监控数据。本文档详细解析CPG的代码实现。

## 核心架构

### 1. 系统架构图

```
原始多模态数据 → 原子事件化 → 滑窗聚合 → 异常检测 → 因果图构建 → 根因排序
    ↓              ↓          ↓        ↓         ↓           ↓
Metrics/Logs/   AtomicEvent  AggEvent  POT      BFS+TE     PageRank
Traces         Extraction   Merge     Detector  Analysis   + Score
```

### 2. 主要类结构

#### 2.1 数据结构类

```python
@dataclass
class AtomicEvent:
    """原子事件 E = (t, N, T, D)"""
    t: int          # Unix毫秒时间戳
    N: str          # 服务节点标识
    T: str          # 事件类型
    D: Dict[str, Any]  # 事件载荷

@dataclass
class AggregatedEvent:
    """聚合事件 AE = (t_agg, N, F)"""
    t: int          # 聚合窗口最大时间戳
    N: str          # 服务节点
    F: np.ndarray   # 特征向量
```

**设计理念**：
- `AtomicEvent`：表示系统中的最小事件单位，统一了Metrics、Logs、Traces三种数据源
- `AggregatedEvent`：将时间窗口内的原子事件聚合为高维特征向量，便于后续分析

#### 2.2 核心组件类

```python
class DrainLogParser:
    """简化的Drain日志解析器"""
    def parse(self, log_message: str) -> str:
        # 使用正则表达式提取日志模板
        # 实际应用建议使用完整的Drain3库
```

**功能**：将非结构化日志转换为结构化模板，支持：
- IP地址掩码：`192.168.1.1` → `<IP>`
- 数字掩码：`12345` → `<NUM>`
- 十六进制掩码：`0xDEADBEEF` → `<HEX>`

```python
class POTAnomalyDetector:
    """POT (Peaks Over Threshold) 异常检测器"""
    def detect(self, score: float) -> Optional[float]:
        # 使用广义帕累托分布进行异常检测
        # 自动选择阈值，无需人工干预
```

**核心算法**：
1. **阈值选择**：在50-95百分位之间选择最优阈值
2. **GPD拟合**：使用最大似然估计拟合广义帕累托分布
3. **异常判断**：基于POT理论计算动态阈值

```python
class TransferEntropyCalculator:
    """传递熵计算器"""
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        # 计算从x到y的信息传递量
        # 量化因果关系强度
```

**数学原理**：
```
TE(X→Y) = ∑ p(y_t, y_{t-1}, x_{t-1}) * log[p(y_t|y_{t-1}, x_{t-1}) / p(y_t|y_{t-1})]
```

## 算法流程详解

### 第一步：原子事件化

#### Metrics事件提取
```python
def _extract_atomic_events_from_metrics(self, metrics_df: pd.DataFrame):
    for _, row in metrics_df.iterrows():
        t = int(row['time'] * 1000)  # 转换为毫秒
        for col in metric_cols:
            svc = col.split('_')[0]  # 提取服务名
            metric = col.split('_', 1)[1]  # 提取指标名
            
            # 滑窗标准化
            normalized_value = (value - mean) / (std + ε)
            
            event = AtomicEvent(
                t=t, N=svc, T=f"METRIC_{metric}", 
                D={"value": normalized_value}
            )
```

**关键特性**：
- 实时标准化：使用滑动窗口进行Z-score标准化
- 服务识别：从列名自动提取服务和指标信息
- 时间统一：统一转换为毫秒时间戳

#### Logs事件提取
```python
def _extract_atomic_events_from_logs(self, logs_df: pd.DataFrame):
    for _, row in logs_df.iterrows():
        t = int(str(row['timestamp'])[:13])  # 纳秒时间戳转毫秒
        svc = row['container_name'].split('-')[0]  # 提取服务名
        
        template_id = self.log_parser.parse(row['message'])
        
        event = AtomicEvent(
            t=t, N=svc, T=f"LOG_{template_id}", 
            D={"count": 1}
        )
```

**处理策略**：
- 模板提取：使用Drain算法识别日志模式
- 服务映射：从容器名提取服务标识
- 计数聚合：相同模板的日志进行计数

#### Traces事件提取
```python
def _extract_atomic_events_from_traces(self, traces_df: pd.DataFrame):
    t0 = int(start_time)
    t1 = int(start_time + duration)
    
    # 开始事件
    start_event = AtomicEvent(
        t=t0, N=svc, T=f"TRACE_START_{operation}", D={}
    )
    
    # 结束事件  
    end_event = AtomicEvent(
        t=t1, N=svc, T=f"TRACE_END_{operation}", 
        D={"duration": duration}
    )
```

**双事件模型**：
- START事件：记录调用开始时间
- END事件：记录调用结束时间和持续时间
- 支持分布式追踪的完整生命周期分析

### 第二步：多模态滑窗聚合

#### 聚合算法
```python
def _aggregate_events(self, events: List[AtomicEvent]):
    # 按服务分组
    service_events = defaultdict(list)
    
    # 滑窗聚合
    for service, svc_events in service_events.items():
        buffer = []
        window_start = None
        
        for event in svc_events:
            if event.t - window_start <= self.agg_window:
                buffer.append(event)
            else:
                # 生成聚合事件
                agg_event = self._merge_events(buffer, service)
                # 开始新窗口
                buffer = [event]
```

#### 特征向量构建
```python
def _merge_events(self, events: List[AtomicEvent], service: str):
    # 特征向量维度: 3*M + L + 4
    # M = 指标数量, L = 日志模板数量, 4 = Trace特征
    
    # Metrics特征 (mean, max, last)
    for metric in metrics:
        F[idx:idx+3] = [mean(values), max(values), last(values)]
    
    # Logs特征 (count)
    for template in log_templates:
        F[metrics_dim + i] = log_counts[template]
    
    # Traces特征 (start_count, end_count, total_duration, avg_duration)
    F[metrics_dim + logs_dim:] = [starts, ends, total_dur, avg_dur]
```

**特征设计原理**：
- **Metrics特征**：均值反映趋势，最大值捕获峰值，最后值表示当前状态
- **Logs特征**：模板计数反映异常日志的频率变化
- **Traces特征**：调用次数和延迟统计反映服务调用模式

### 第三步：POT异常检测

#### 阈值选择算法
```python
def detect(self, score: float):
    # 1. 候选阈值生成
    percentiles = np.linspace(50, 95, 10)
    
    # 2. GPD拟合质量评估
    for p in percentiles:
        threshold = np.percentile(buffer, p)
        excesses = [x - threshold for x in buffer if x > threshold]
        
        sigma, xi = self._fit_gpd(excesses)  # MLE估计
        ks_stat = kolmogorov_smirnov_test(excesses, sigma, xi)
        
        if ks_stat < best_ks:
            best_threshold = threshold
    
    # 3. 最终阈值计算
    T = u* + σ/ξ * [(n/k*(1-α))^(-ξ) - 1]
```

**数学模型**：
- **广义帕累托分布**：`F(x) = 1 - (1 + ξx/σ)^(-1/ξ)`
- **POT理论**：超过高阈值的极值遵循GPD分布
- **自适应阈值**：基于历史数据自动调整检测敏感度

### 第四步：局部因果图构建

#### BFS因果发现
```python
def _build_causal_graph(self, symptom_event, all_events):
    nodes = {symptom_event}
    queue = Queue([symptom_event])
    
    while not queue.empty():
        current = queue.get()
        
        # 查找上游服务
        upstream_services = service_deps[current.N]
        
        for upstream_svc in upstream_services:
            candidates = find_candidates_in_time_window(
                upstream_svc, current.t - lookback_window, current.t
            )
            
            for candidate in candidates:
                te = transfer_entropy(candidate.F, current.F)
                if te > threshold:
                    add_causal_edge(candidate, current, te)
```

**算法特点**：
- **局部搜索**：只在异常事件的时空邻域内构建因果图
- **服务拓扑感知**：利用微服务依赖关系指导搜索方向
- **传递熵量化**：使用信息论方法量化因果关系强度

#### 传递熵计算
```python
def calculate(self, x: np.ndarray, y: np.ndarray):
    # 1. 数据离散化
    x_disc = pd.cut(x, bins=self.bins)
    y_disc = pd.cut(y, bins=self.bins)
    
    # 2. 构建时间序列
    y_present = y_disc[lag:]
    y_past = y_disc[:-lag]  
    x_past = x_disc[:-lag]
    
    # 3. 计算条件概率
    p_xyz = joint_probability(y_present, y_past, x_past)
    p_yz = joint_probability(y_present, y_past)
    
    # 4. 传递熵计算
    te = sum(p_xyz * log(p_y_given_yz_x / p_y_given_yz))
```

### 第五步：根因排序

#### PageRank算法
```python
def _calculate_root_cause_scores(self, nodes, edges):
    G = nx.DiGraph()
    
    # 构建加权有向图
    for source, target, weight in edges:
        G.add_edge(source.N, target.N, weight=weight)
    
    # PageRank计算
    pagerank_scores = nx.pagerank(G, weight='weight')
    
    return sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
```

**排序策略**：
- **图结构权重**：传递熵值作为边权重
- **PageRank中心性**：识别因果图中的关键节点
- **多因子融合**：结合拓扑结构和因果强度

## 性能优化

### 1. 时间复杂度分析
- **原子事件化**：O(N_raw)，与原始数据量线性相关
- **滑窗聚合**：O(N_event)，与事件数量线性相关  
- **POT检测**：O(N_event × W)，W为窗口大小
- **因果图构建**：O(|V| × K × log K)，V为节点数，K为候选数
- **根因排序**：O(|E| + |V|log|V|)，标准图算法复杂度

### 2. 空间复杂度优化
- **滑动窗口**：固定大小缓冲区，避免内存无限增长
- **事件压缩**：聚合事件减少存储需求
- **增量计算**：流式处理避免全量数据加载

### 3. 实时性保证
- **事件驱动**：异常触发才构建因果图
- **局部搜索**：限制搜索范围和深度
- **并行计算**：多个异常事件可并行处理

## 配置参数说明

### 核心参数
```python
CPGFramework(
    agg_window=5000,        # 聚合窗口(ms)
    anomaly_window=1000,    # 异常检测窗口大小
    causal_threshold=0.1,   # 因果关系阈值
    lookback_window=30000,  # 回溯窗口(ms)  
    top_k=5                 # 因果候选数量
)
```

### 参数调优指南
- **agg_window**：越大越平滑，越小越敏感
- **anomaly_window**：影响POT检测的稳定性
- **causal_threshold**：控制因果关系的严格程度
- **lookback_window**：平衡检测精度和计算开销
- **top_k**：限制候选数量，提高计算效率

## 扩展性设计

### 1. 数据源扩展
```python
# 新增数据源只需实现对应的事件提取方法
def _extract_atomic_events_from_new_source(self, data):
    # 返回AtomicEvent列表
    pass
```

### 2. 异常检测器扩展
```python
class CustomAnomalyDetector:
    def detect(self, score: float) -> Optional[float]:
        # 实现自定义异常检测逻辑
        pass
```

### 3. 因果分析扩展
```python
class CustomCausalAnalyzer:
    def calculate_causality(self, x, y) -> float:
        # 实现自定义因果关系计算
        pass
```

## 错误处理和容错

### 1. 数据质量处理
- 缺失值填充：前向填充 + 零填充
- 异常值处理：基于统计分布的离群点检测
- 数据类型转换：自动类型推断和转换

### 2. 算法容错
- GPD拟合失败：回退到简单统计方法
- 图构建失败：返回基于异常分数的排序
- 传递熵计算异常：使用相关性作为替代

### 3. 系统鲁棒性
- 内存保护：限制缓冲区大小
- 超时机制：避免长时间计算阻塞
- 异常恢复：优雅降级，保证基本功能

## 总结

CPG框架通过创新的事件驱动架构和多模态融合技术，实现了：

1. **无监督学习**：无需标注数据和人工阈值设定
2. **实时处理**：流式架构支持毫秒级响应
3. **多模态融合**：统一处理Metrics、Logs、Traces
4. **因果推理**：基于信息论的严格因果关系量化
5. **自适应性**：POT理论实现动态阈值调整

该实现在保持算法先进性的同时，充分考虑了工程实践的需求，为微服务系统的智能运维提供了强有力的技术支撑。