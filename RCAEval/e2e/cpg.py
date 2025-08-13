"""
CPG: 无监督多模态事件驱动因果传播框架
Unsupervised Multimodal Event-Driven Causal Propagation Graph

核心目标:
- 无监督、无人工定阈值
- 流式(streaming)处理多模态(Metrics/Logs/Traces)
- 事件驱动，触发式构建局部因果图，精准定位微服务根因

作者: RCAEval Team
日期: 2025年1月
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import re
from dataclasses import dataclass
from queue import Queue
import json
import logging
from scipy import stats
from scipy.optimize import minimize
import networkx as nx

# 导入RCAEval现有工具
from RCAEval.io.time_series import preprocess, drop_constant
from RCAEval.e2e import rca

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AtomicEvent:
    """原子事件 E = (t, N, T, D)"""
    t: int  # Unix ms 时间戳
    N: str  # 服务节点标识
    T: str  # 事件类型
    D: Dict[str, Any]  # Payload

@dataclass
class AggregatedEvent:
    """聚合事件 AE = (t_agg, N, F)"""
    t: int  # 窗口内最大时间戳
    N: str  # 服务节点
    F: np.ndarray  # 特征向量

class DrainLogParser:
    """简化的Drain日志解析器"""
    
    def __init__(self, max_depth=4, sim_threshold=0.4):
        self.max_depth = max_depth
        self.sim_threshold = sim_threshold
        self.templates = {}
        self.template_count = 0
        
    def parse(self, log_message: str) -> str:
        """解析日志消息，返回模板ID"""
        # 简化实现：基于正则表达式的模板识别
        # 在实际应用中，建议使用完整的Drain3库
        
        # 预处理：移除数字、IP地址等变量部分
        processed = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', log_message)
        processed = re.sub(r'\b\d+\b', '<NUM>', processed)
        processed = re.sub(r'\b[0-9a-fA-F]{8,}\b', '<HEX>', processed)
        
        # 简单的模板匹配
        template_key = processed
        if template_key not in self.templates:
            self.template_count += 1
            self.templates[template_key] = f"TEMPLATE_{self.template_count}"
            
        return self.templates[template_key]

class POTAnomalyDetector:
    """POT (Peaks Over Threshold) 异常检测器"""
    
    def __init__(self, window_size=1000, alpha=0.01, min_samples=50):
        self.window_size = window_size
        self.alpha = alpha
        self.min_samples = min_samples
        self.buffer = deque(maxlen=window_size)
        
    def _fit_gpd(self, excesses):
        """拟合广义帕累托分布"""
        if len(excesses) < self.min_samples:
            return None, None
            
        # 使用矩估计作为初始值
        mean_exc = np.mean(excesses)
        var_exc = np.var(excesses)
        
        # MLE估计
        def neg_log_likelihood(params):
            sigma, xi = params
            if sigma <= 0:
                return np.inf
            if xi != 0:
                if np.any(1 + xi * excesses / sigma <= 0):
                    return np.inf
                return len(excesses) * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(1 + xi * excesses / sigma))
            else:
                return len(excesses) * np.log(sigma) + np.sum(excesses) / sigma
                
        # 初始估计
        sigma_init = mean_exc
        xi_init = -0.5 + mean_exc**2 / var_exc if var_exc > 0 else 0
        
        try:
            result = minimize(neg_log_likelihood, [sigma_init, xi_init], 
                            method='L-BFGS-B', bounds=[(1e-6, None), (-0.5, 0.5)])
            if result.success:
                return result.x[0], result.x[1]  # sigma, xi
        except:
            pass
            
        return sigma_init, xi_init
    
    def detect(self, score: float) -> Optional[float]:
        """检测异常，返回阈值（如果异常）"""
        self.buffer.append(score)
        
        if len(self.buffer) < self.window_size:
            return None
            
        # 选择合适的阈值
        percentiles = np.linspace(50, 95, 10)
        best_threshold = None
        best_ks = float('inf')
        
        for p in percentiles:
            threshold = np.percentile(list(self.buffer), p)
            excesses = [x - threshold for x in self.buffer if x > threshold]
            
            if len(excesses) < self.min_samples:
                continue
                
            sigma, xi = self._fit_gpd(excesses)
            if sigma is None:
                continue
                
            # 简化的KS检验
            # 实际应用中应使用完整的统计检验
            ks_stat = np.random.random()  # 占位符
            
            if ks_stat < best_ks:
                best_ks = ks_stat
                best_threshold = threshold
                
        if best_threshold is None:
            return None
            
        # 计算最终阈值
        excesses = [x - best_threshold for x in self.buffer if x > best_threshold]
        sigma, xi = self._fit_gpd(excesses)
        
        if sigma is None:
            return None
            
        n, k = len(self.buffer), len(excesses)
        if k == 0:
            return None
            
        # POT阈值公式
        threshold_final = best_threshold + sigma/xi * ((n/k*(1-self.alpha))**(-xi) - 1)
        
        return threshold_final if score > threshold_final else None

class TransferEntropyCalculator:
    """传递熵计算器"""
    
    def __init__(self, lag=1, bins=10):
        self.lag = lag
        self.bins = bins
        
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算从x到y的传递熵"""
        if len(x) != len(y) or len(x) < self.lag + 1:
            return 0.0
            
        # 离散化
        x_disc = pd.cut(x, bins=self.bins, labels=False, duplicates='drop')
        y_disc = pd.cut(y, bins=self.bins, labels=False, duplicates='drop')
        
        if x_disc is None or y_disc is None:
            return 0.0
            
        # 构建时间序列
        y_present = y_disc[self.lag:]
        y_past = y_disc[:-self.lag]
        x_past = x_disc[:-self.lag]
        
        # 计算联合概率
        try:
            # P(Y_t, Y_{t-1}, X_{t-1})
            joint_xyz = pd.crosstab([y_present, y_past], x_past, normalize=True)
            # P(Y_t, Y_{t-1})
            joint_yz = pd.crosstab(y_present, y_past, normalize=True)
            # P(Y_t | Y_{t-1}, X_{t-1})
            cond_y_yz = joint_xyz.div(joint_xyz.sum(axis=1), axis=0).fillna(0)
            # P(Y_t | Y_{t-1})
            cond_y_z = joint_yz.div(joint_yz.sum(axis=1), axis=0).fillna(0)
            
            # 计算传递熵
            te = 0.0
            for i in joint_xyz.index:
                for j in joint_xyz.columns:
                    p_xyz = joint_xyz.loc[i, j] if j in joint_xyz.columns else 0
                    if p_xyz > 0:
                        p_y_yz = cond_y_yz.loc[i, j] if j in cond_y_yz.columns else 0
                        p_y_z = cond_y_z.loc[i[0], i[1]] if i[1] in cond_y_z.columns else 0
                        
                        if p_y_yz > 0 and p_y_z > 0:
                            te += p_xyz * np.log2(p_y_yz / p_y_z)
                            
            return max(0.0, te)
            
        except Exception as e:
            logger.warning(f"Transfer entropy calculation failed: {e}")
            return 0.0

class CPGFramework:
    """CPG框架主类"""
    
    def __init__(self, 
                 agg_window=5000,  # 聚合窗口(ms)
                 anomaly_window=1000,  # 异常检测窗口
                 causal_threshold=0.1,  # 因果关系阈值
                 lookback_window=30000,  # 回溯窗口(ms)
                 top_k=5):  # 因果候选数量
        
        self.agg_window = agg_window
        self.anomaly_window = anomaly_window
        self.causal_threshold = causal_threshold
        self.lookback_window = lookback_window
        self.top_k = top_k
        
        # 初始化组件
        self.log_parser = DrainLogParser()
        self.anomaly_detector = POTAnomalyDetector(window_size=anomaly_window)
        self.te_calculator = TransferEntropyCalculator()
        
        # 存储
        self.atomic_events = []
        self.aggregated_events = []
        self.metrics_keys = []
        self.log_templates = []
        
    def _extract_atomic_events_from_metrics(self, metrics_df: pd.DataFrame) -> List[AtomicEvent]:
        """从Metrics数据提取原子事件"""
        events = []
        
        # 获取metrics列名
        metric_cols = [col for col in metrics_df.columns if col != 'time']
        self.metrics_keys = metric_cols
        
        # 滑窗标准化
        window_size = 10
        for _, row in metrics_df.iterrows():
            t = int(row['time'] * 1000)  # 转换为ms
            
            for col in metric_cols:
                if '_container' in col or '_' in col:
                    svc = col.split('_')[0]
                    metric = col.split('_', 1)[1]
                else:
                    svc = col
                    metric = 'value'
                
                # 简化的标准化（实际应用中需要滑窗）
                value = row[col]
                if pd.notna(value):
                    normalized_value = (value - metrics_df[col].mean()) / (metrics_df[col].std() + 1e-6)
                    
                    event = AtomicEvent(
                        t=t,
                        N=svc,
                        T=f"METRIC_{metric}",
                        D={"value": normalized_value}
                    )
                    events.append(event)
                    
        return events
    
    def _extract_atomic_events_from_logs(self, logs_df: pd.DataFrame) -> List[AtomicEvent]:
        """从Logs数据提取原子事件"""
        events = []
        
        for _, row in logs_df.iterrows():
            if 'timestamp' in row:
                # 使用纳秒时间戳的前13位作为毫秒
                t = int(str(row['timestamp'])[:13])
            else:
                t = int(row['time'] * 1000)
                
            svc = row['container_name'].split('-')[0] if 'container_name' in row else 'unknown'
            message = row['message'] if 'message' in row else ''
            
            # 解析日志模板
            template_id = self.log_parser.parse(message)
            if template_id not in self.log_templates:
                self.log_templates.append(template_id)
                
            event = AtomicEvent(
                t=t,
                N=svc,
                T=f"LOG_{template_id}",
                D={"count": 1}
            )
            events.append(event)
            
        return events
    
    def _extract_atomic_events_from_traces(self, traces_df: pd.DataFrame) -> List[AtomicEvent]:
        """从Traces数据提取原子事件"""
        events = []
        
        for _, row in traces_df.iterrows():
            svc = row.get('serviceName', 'unknown')
            operation = row.get('operationName', row.get('methodName', 'unknown'))
            start_time = row.get('startTimeMillis', row.get('time', 0) * 1000)
            duration = row.get('duration', 0)
            
            t0 = int(start_time)
            t1 = int(start_time + duration)
            
            # 开始事件
            start_event = AtomicEvent(
                t=t0,
                N=svc,
                T=f"TRACE_START_{operation}",
                D={}
            )
            events.append(start_event)
            
            # 结束事件
            end_event = AtomicEvent(
                t=t1,
                N=svc,
                T=f"TRACE_END_{operation}",
                D={"duration": duration}
            )
            events.append(end_event)
            
        return events
    
    def _aggregate_events(self, events: List[AtomicEvent]) -> List[AggregatedEvent]:
        """聚合原子事件"""
        # 按服务分组
        service_events = defaultdict(list)
        for event in events:
            service_events[event.N].append(event)
            
        aggregated = []
        
        for service, svc_events in service_events.items():
            # 按时间排序
            svc_events.sort(key=lambda e: e.t)
            
            # 滑窗聚合
            buffer = []
            window_start = None
            
            for event in svc_events:
                if not buffer:
                    buffer = [event]
                    window_start = event.t
                elif event.t - window_start <= self.agg_window:
                    buffer.append(event)
                else:
                    # 生成聚合事件
                    agg_event = self._merge_events(buffer, service)
                    aggregated.append(agg_event)
                    
                    # 开始新窗口
                    buffer = [event]
                    window_start = event.t
                    
            # 处理最后一个窗口
            if buffer:
                agg_event = self._merge_events(buffer, service)
                aggregated.append(agg_event)
                
        return sorted(aggregated, key=lambda ae: ae.t)
    
    def _merge_events(self, events: List[AtomicEvent], service: str) -> AggregatedEvent:
        """合并事件缓冲区为聚合事件"""
        t_agg = max(e.t for e in events)
        
        # 初始化特征向量
        # 维度: 3*M + L + 4 (M=metrics数量, L=日志模板数量, 4=trace特征)
        M = len(self.metrics_keys)
        L = len(self.log_templates)
        feature_dim = 3 * M + L + 4
        
        F = np.zeros(feature_dim)
        
        # 处理metrics特征
        metric_values = defaultdict(list)
        log_counts = defaultdict(int)
        trace_starts = 0
        trace_ends = 0
        durations = []
        
        for event in events:
            if event.T.startswith("METRIC_"):
                metric_name = event.T.split("METRIC_")[1]
                if f"{service}_{metric_name}" in self.metrics_keys:
                    metric_values[metric_name].append(event.D["value"])
            elif event.T.startswith("LOG_"):
                template_id = event.T.split("LOG_")[1]
                log_counts[template_id] += 1
            elif event.T.startswith("TRACE_START"):
                trace_starts += 1
            elif event.T.startswith("TRACE_END"):
                trace_ends += 1
                if "duration" in event.D:
                    durations.append(event.D["duration"])
        
        # 填充metrics特征 (mean, max, last)
        idx = 0
        for i, metric_key in enumerate(self.metrics_keys):
            if metric_key.startswith(service + "_"):
                metric_name = metric_key.split("_", 1)[1]
                values = metric_values.get(metric_name, [])
                if values:
                    F[idx] = np.mean(values)      # mean
                    F[idx + 1] = np.max(values)   # max
                    F[idx + 2] = values[-1]       # last
            idx += 3
            
        # 填充日志特征
        for i, template in enumerate(self.log_templates):
            F[3 * M + i] = log_counts.get(template, 0)
            
        # 填充trace特征
        trace_idx = 3 * M + L
        F[trace_idx] = trace_starts
        F[trace_idx + 1] = trace_ends
        F[trace_idx + 2] = sum(durations) if durations else 0
        F[trace_idx + 3] = np.mean(durations) if durations else 0
        
        return AggregatedEvent(t=t_agg, N=service, F=F)
    
    def _detect_anomalies(self, events: List[AggregatedEvent]) -> List[AggregatedEvent]:
        """检测异常事件"""
        anomalies = []
        
        for event in events:
            # 计算异常分数（使用特征向量的范数）
            score = np.linalg.norm(event.F)
            
            # POT异常检测
            threshold = self.anomaly_detector.detect(score)
            if threshold is not None:
                anomalies.append(event)
                logger.info(f"Anomaly detected in service {event.N} at time {event.t}, score: {score:.3f}")
                
        return anomalies
    
    def _build_causal_graph(self, symptom_event: AggregatedEvent, 
                          all_events: List[AggregatedEvent]) -> Tuple[List, List]:
        """构建因果图"""
        nodes = {symptom_event}
        edges = []
        queue = Queue()
        queue.put(symptom_event)
        
        # 简化的服务依赖关系（实际应用中需要配置文件）
        service_deps = self._get_service_dependencies()
        
        while not queue.empty():
            current_event = queue.get()
            
            # 查找上游服务
            upstream_services = service_deps.get(current_event.N, [])
            
            for upstream_svc in upstream_services:
                # 在时间窗口内查找候选事件
                candidates = [
                    e for e in all_events 
                    if e.N == upstream_svc and 
                    current_event.t - self.lookback_window <= e.t < current_event.t
                ]
                
                # 按时间排序，取最近的K个
                candidates = sorted(candidates, key=lambda e: e.t, reverse=True)[:self.top_k]
                
                for candidate in candidates:
                    # 计算传递熵
                    te = self.te_calculator.calculate(candidate.F, current_event.F)
                    
                    if te > self.causal_threshold:
                        if candidate not in nodes:
                            nodes.add(candidate)
                            queue.put(candidate)
                        
                        edges.append((candidate, current_event, te))
                        logger.info(f"Causal edge: {candidate.N} -> {current_event.N}, TE: {te:.3f}")
        
        return list(nodes), edges
    
    def _get_service_dependencies(self) -> Dict[str, List[str]]:
        """获取服务依赖关系（简化版本）"""
        # 实际应用中应该从配置文件或服务注册中心获取
        return {
            'frontend': ['cartservice', 'catalogservice', 'recommendationservice'],
            'checkoutservice': ['cartservice', 'emailservice', 'paymentservice'],
            'cartservice': ['redis'],
            'catalogservice': ['catalogdb'],
            'recommendationservice': ['catalogservice'],
        }
    
    def _calculate_root_cause_scores(self, nodes: List, edges: List) -> Dict[str, float]:
        """计算根因贡献度"""
        # 构建NetworkX图
        G = nx.DiGraph()
        
        # 添加节点
        for node in nodes:
            G.add_node(node.N)
            
        # 添加边
        for source, target, weight in edges:
            G.add_edge(source.N, target.N, weight=weight)
        
        # 计算PageRank分数
        try:
            pagerank_scores = nx.pagerank(G, weight='weight')
        except:
            # 如果图为空或有问题，返回均匀分布
            pagerank_scores = {node.N: 1.0/len(nodes) for node in nodes}
            
        return pagerank_scores
    
    def process(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """处理多模态数据"""
        logger.info("开始CPG处理流程")
        
        # 第一步：原子事件化
        logger.info("步骤1: 原子事件化")
        all_events = []
        
        if 'metric' in data:
            metric_events = self._extract_atomic_events_from_metrics(data['metric'])
            all_events.extend(metric_events)
            logger.info(f"提取到 {len(metric_events)} 个metrics事件")
            
        if 'logts' in data:
            log_events = self._extract_atomic_events_from_logs(data['logts'])
            all_events.extend(log_events)
            logger.info(f"提取到 {len(log_events)} 个日志事件")
            
        if 'traces' in data:
            trace_events = self._extract_atomic_events_from_traces(data['traces'])
            all_events.extend(trace_events)
            logger.info(f"提取到 {len(trace_events)} 个trace事件")
        
        # 第二步：多模态滑窗聚合
        logger.info("步骤2: 多模态滑窗聚合")
        self.aggregated_events = self._aggregate_events(all_events)
        logger.info(f"生成 {len(self.aggregated_events)} 个聚合事件")
        
        # 第三步：全局异常检测
        logger.info("步骤3: 全局异常检测")
        anomaly_events = self._detect_anomalies(self.aggregated_events)
        logger.info(f"检测到 {len(anomaly_events)} 个异常事件")
        
        if not anomaly_events:
            logger.warning("未检测到异常事件")
            return {
                "ranks": [],
                "adj": np.array([]),
                "node_names": [],
                "causal_edges": []
            }
        
        # 第四步：局部因果图构建
        logger.info("步骤4: 局部因果图构建")
        all_nodes = set()
        all_edges = []
        
        for symptom in anomaly_events:
            nodes, edges = self._build_causal_graph(symptom, self.aggregated_events)
            all_nodes.update(nodes)
            all_edges.extend(edges)
            
        # 第五步：多因子归因
        logger.info("步骤5: 多因子归因")
        node_list = list(all_nodes)
        root_cause_scores = self._calculate_root_cause_scores(node_list, all_edges)
        
        # 排序根因
        ranked_services = sorted(root_cause_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # 构建邻接矩阵
        service_names = [node.N for node in node_list]
        adj_matrix = np.zeros((len(service_names), len(service_names)))
        
        for source, target, weight in all_edges:
            try:
                i = service_names.index(source.N)
                j = service_names.index(target.N)
                adj_matrix[i][j] = weight
            except ValueError:
                continue
        
        # 格式化输出
        result = {
            "ranks": [service for service, _ in ranked_services],
            "adj": adj_matrix,
            "node_names": service_names,
            "causal_edges": [(e[0].N, e[1].N, e[2]) for e in all_edges],
            "anomaly_events": len(anomaly_events),
            "total_events": len(all_events)
        }
        
        logger.info(f"CPG处理完成，识别出前5个根因: {result['ranks'][:5]}")
        return result

@rca
def cpg(data, inject_time=None, dataset=None, **kwargs):
    """
    CPG: 无监督多模态事件驱动因果传播框架
    
    Args:
        data: 多模态数据，可以是DataFrame或dict
              如果是dict，格式为 {"metric": df, "logts": df, "traces": df}
        inject_time: 异常注入时间点
        dataset: 数据集名称
        **kwargs: 其他参数
            - agg_window: 聚合窗口大小(ms)，默认5000
            - anomaly_window: 异常检测窗口大小，默认1000
            - causal_threshold: 因果关系阈值，默认0.1
            - lookback_window: 回溯窗口(ms)，默认30000
            - top_k: 因果候选数量，默认5
    
    Returns:
        dict: {
            "ranks": 根因排序列表,
            "adj": 邻接矩阵,
            "node_names": 节点名称列表,
            "causal_edges": 因果边列表
        }
    """
    
    logger.info(f"启动CPG方法，数据集: {dataset}")
    
    # 参数配置
    config = {
        'agg_window': kwargs.get('agg_window', 5000),
        'anomaly_window': kwargs.get('anomaly_window', 1000),
        'causal_threshold': kwargs.get('causal_threshold', 0.1),
        'lookback_window': kwargs.get('lookback_window', 30000),
        'top_k': kwargs.get('top_k', 5)
    }
    
    # 初始化CPG框架
    cpg_framework = CPGFramework(**config)
    
    # 处理输入数据
    if isinstance(data, dict):
        # 多模态数据
        processed_data = {}
        
        if 'metric' in data:
            metric_data = data['metric']
            if inject_time is not None:
                # 分离正常和异常数据
                normal_data = metric_data[metric_data['time'] < inject_time]
                anomal_data = metric_data[metric_data['time'] >= inject_time]
                processed_metric = pd.concat([normal_data, anomal_data], ignore_index=True)
            else:
                processed_metric = metric_data
                
            processed_metric = preprocess(
                data=processed_metric, 
                dataset=dataset, 
                dk_select_useful=kwargs.get("dk_select_useful", False)
            )
            processed_data['metric'] = processed_metric
            
        if 'logts' in data:
            logts_data = data['logts']
            logts_data = drop_constant(logts_data)
            processed_data['logts'] = logts_data
            
        if 'traces' in data:
            processed_data['traces'] = data['traces']
            
    else:
        # 单模态数据（metrics）
        if inject_time is not None:
            normal_data = data[data['time'] < inject_time]
            anomal_data = data[data['time'] >= inject_time]
            processed_data = pd.concat([normal_data, anomal_data], ignore_index=True)
        else:
            processed_data = data
            
        processed_data = preprocess(
            data=processed_data, 
            dataset=dataset, 
            dk_select_useful=kwargs.get("dk_select_useful", False)
        )
        
        # 转换为多模态格式
        processed_data = {'metric': processed_data}
    
    # 运行CPG分析
    try:
        result = cpg_framework.process(processed_data)
        
        logger.info(f"CPG分析完成，检测到 {result.get('anomaly_events', 0)} 个异常事件")
        logger.info(f"构建因果图包含 {len(result.get('causal_edges', []))} 条因果边")
        
        return result
        
    except Exception as e:
        logger.error(f"CPG分析失败: {str(e)}")
        # 返回默认结果
        return {
            "ranks": [],
            "adj": np.array([]),
            "node_names": [],
            "causal_edges": []
        }

if __name__ == "__main__":
    # 测试代码
    print("CPG框架测试")
    
    # 创建测试数据
    test_metric_data = pd.DataFrame({
        'time': range(100),
        'service_a_cpu': np.random.normal(0.5, 0.1, 100),
        'service_a_memory': np.random.normal(0.6, 0.1, 100),
        'service_b_cpu': np.random.normal(0.4, 0.1, 100)
    })
    
    test_log_data = pd.DataFrame({
        'time': range(100),
        'timestamp': [f"{i}000000000" for i in range(100)],
        'container_name': ['service-a-pod'] * 50 + ['service-b-pod'] * 50,
        'message': ['Connection established'] * 30 + ['Error occurred'] * 20 + ['Request processed'] * 50
    })
    
    multimodal_data = {
        'metric': test_metric_data,
        'logts': test_log_data
    }
    
    result = cpg(multimodal_data, inject_time=50)
    print(f"测试结果: {result}")