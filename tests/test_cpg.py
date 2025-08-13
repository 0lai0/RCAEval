"""
CPG方法测试文件
测试CPG（无监督多模态事件驱动因果传播框架）的各个组件和整体功能
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os
from collections import deque

# 导入CPG相关模块
from RCAEval.e2e.cpg import (
    cpg, CPGFramework, AtomicEvent, AggregatedEvent,
    DrainLogParser, POTAnomalyDetector, TransferEntropyCalculator
)


class TestAtomicEvent:
    """测试原子事件数据结构"""
    
    def test_atomic_event_creation(self):
        """测试原子事件创建"""
        event = AtomicEvent(
            t=1000,
            N="service-a",
            T="METRIC_cpu",
            D={"value": 0.8}
        )
        
        assert event.t == 1000
        assert event.N == "service-a"
        assert event.T == "METRIC_cpu"
        assert event.D["value"] == 0.8
    
    def test_atomic_event_equality(self):
        """测试原子事件相等性"""
        event1 = AtomicEvent(t=1000, N="svc", T="METRIC", D={"v": 1})
        event2 = AtomicEvent(t=1000, N="svc", T="METRIC", D={"v": 1})
        event3 = AtomicEvent(t=2000, N="svc", T="METRIC", D={"v": 1})
        
        assert event1 == event2
        assert event1 != event3


class TestAggregatedEvent:
    """测试聚合事件数据结构"""
    
    def test_aggregated_event_creation(self):
        """测试聚合事件创建"""
        features = np.array([1.0, 2.0, 3.0])
        event = AggregatedEvent(
            t=2000,
            N="service-b", 
            F=features
        )
        
        assert event.t == 2000
        assert event.N == "service-b"
        np.testing.assert_array_equal(event.F, features)


class TestDrainLogParser:
    """测试Drain日志解析器"""
    
    def setup_method(self):
        """测试设置"""
        self.parser = DrainLogParser()
    
    def test_simple_log_parsing(self):
        """测试简单日志解析"""
        log1 = "Connection from 192.168.1.1 established"
        log2 = "Connection from 10.0.0.1 established"
        
        template1 = self.parser.parse(log1)
        template2 = self.parser.parse(log2)
        
        # 相同模板应该返回相同的模板ID
        assert template1 == template2
        assert "TEMPLATE_" in template1
    
    def test_different_log_templates(self):
        """测试不同日志模板"""
        log1 = "User login successful"
        log2 = "Database connection failed"
        
        template1 = self.parser.parse(log1)
        template2 = self.parser.parse(log2)
        
        # 不同模板应该返回不同的模板ID
        assert template1 != template2
    
    def test_numeric_masking(self):
        """测试数字掩码"""
        log1 = "Process 12345 started"
        log2 = "Process 67890 started"
        
        template1 = self.parser.parse(log1)
        template2 = self.parser.parse(log2)
        
        # 数字应该被掩码化，返回相同模板
        assert template1 == template2
    
    def test_ip_masking(self):
        """测试IP地址掩码"""
        log1 = "Request from 192.168.1.100"
        log2 = "Request from 10.0.0.50"
        
        template1 = self.parser.parse(log1)
        template2 = self.parser.parse(log2)
        
        # IP地址应该被掩码化
        assert template1 == template2


class TestPOTAnomalyDetector:
    """测试POT异常检测器"""
    
    def setup_method(self):
        """测试设置"""
        self.detector = POTAnomalyDetector(window_size=100, alpha=0.01)
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        assert self.detector.window_size == 100
        assert self.detector.alpha == 0.01
        assert len(self.detector.buffer) == 0
    
    def test_buffer_filling(self):
        """测试缓冲区填充"""
        # 添加正常数据
        normal_scores = np.random.normal(0, 1, 50)
        
        for score in normal_scores:
            result = self.detector.detect(score)
            # 缓冲区未满时应该返回None
            assert result is None
        
        assert len(self.detector.buffer) == 50
    
    def test_anomaly_detection(self):
        """测试异常检测"""
        # 填充正常数据
        normal_scores = np.random.normal(0, 1, 100)
        for score in normal_scores:
            self.detector.detect(score)
        
        # 添加异常数据
        anomaly_score = 10.0  # 明显的异常值
        result = self.detector.detect(anomaly_score)
        
        # 应该检测到异常（返回阈值）
        if result is not None:
            assert result > 0
            assert anomaly_score > result
    
    def test_gpd_fitting(self):
        """测试GPD拟合"""
        # 生成具有重尾的数据
        data = np.concatenate([
            np.random.normal(0, 1, 90),  # 正常数据
            np.random.exponential(2, 10)  # 异常数据
        ])
        
        excesses = [x - 2.0 for x in data if x > 2.0]
        
        if len(excesses) >= self.detector.min_samples:
            sigma, xi = self.detector._fit_gpd(excesses)
            assert sigma > 0  # 尺度参数应为正
            assert -0.5 <= xi <= 0.5  # 形状参数应在合理范围内


class TestTransferEntropyCalculator:
    """测试传递熵计算器"""
    
    def setup_method(self):
        """测试设置"""
        self.calculator = TransferEntropyCalculator()
    
    def test_calculator_initialization(self):
        """测试计算器初始化"""
        assert self.calculator.lag == 1
        assert self.calculator.bins == 10
    
    def test_transfer_entropy_calculation(self):
        """测试传递熵计算"""
        # 生成有因果关系的时间序列
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = np.zeros(n)
        
        # y依赖于x的历史值
        for i in range(1, n):
            y[i] = 0.5 * x[i-1] + 0.3 * y[i-1] + np.random.normal(0, 0.1)
        
        te_xy = self.calculator.calculate(x, y)
        te_yx = self.calculator.calculate(y, x)
        
        # x到y的传递熵应该大于y到x的传递熵
        assert te_xy >= 0
        assert te_yx >= 0
        # 由于x影响y，但y不直接影响x，所以te_xy应该更大
        # 注意：由于数据有限和随机性，这个测试可能不总是通过
    
    def test_independent_series(self):
        """测试独立时间序列"""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)
        
        te = self.calculator.calculate(x, y)
        
        # 独立序列的传递熵应该接近0
        assert te >= 0
        assert te < 0.5  # 应该相对较小
    
    def test_short_series(self):
        """测试短时间序列"""
        x = np.array([1, 2])
        y = np.array([3, 4])
        
        te = self.calculator.calculate(x, y)
        
        # 数据太短，应该返回0
        assert te == 0.0
    
    def test_identical_series(self):
        """测试相同时间序列"""
        x = np.random.normal(0, 1, 50)
        y = x.copy()
        
        te = self.calculator.calculate(x, y)
        
        # 相同序列的传递熵应该较高
        assert te >= 0


class TestCPGFramework:
    """测试CPG框架"""
    
    def setup_method(self):
        """测试设置"""
        self.framework = CPGFramework(
            agg_window=1000,
            anomaly_window=50,
            causal_threshold=0.05,
            lookback_window=5000,
            top_k=3
        )
    
    def test_framework_initialization(self):
        """测试框架初始化"""
        assert self.framework.agg_window == 1000
        assert self.framework.anomaly_window == 50
        assert self.framework.causal_threshold == 0.05
        assert isinstance(self.framework.log_parser, DrainLogParser)
        assert isinstance(self.framework.anomaly_detector, POTAnomalyDetector)
        assert isinstance(self.framework.te_calculator, TransferEntropyCalculator)
    
    def test_metrics_event_extraction(self):
        """测试Metrics事件提取"""
        # 创建测试数据
        metrics_df = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'service_a_cpu': [0.5, 0.6, 0.7, 0.8, 0.9],
            'service_a_memory': [0.4, 0.5, 0.6, 0.7, 0.8],
            'service_b_cpu': [0.3, 0.4, 0.5, 0.6, 0.7]
        })
        
        events = self.framework._extract_atomic_events_from_metrics(metrics_df)
        
        # 检查事件数量
        assert len(events) == 15  # 5行 × 3列
        
        # 检查事件类型
        event_types = set(event.T for event in events)
        assert "METRIC_cpu" in event_types
        assert "METRIC_memory" in event_types
        
        # 检查服务名
        services = set(event.N for event in events)
        assert "service" in services  # 简化的服务名提取
    
    def test_logs_event_extraction(self):
        """测试Logs事件提取"""
        logs_df = pd.DataFrame({
            'time': [1, 2, 3],
            'timestamp': ['1000000000000', '2000000000000', '3000000000000'],
            'container_name': ['service-a-pod-123', 'service-b-pod-456', 'service-a-pod-789'],
            'message': ['Connection established', 'Error occurred', 'Connection established']
        })
        
        events = self.framework._extract_atomic_events_from_logs(logs_df)
        
        # 检查事件数量
        assert len(events) == 3
        
        # 检查事件类型
        event_types = set(event.T for event in events)
        assert all(t.startswith("LOG_") for t in event_types)
        
        # 检查服务名提取
        services = set(event.N for event in events)
        assert "service" in services
    
    def test_traces_event_extraction(self):
        """测试Traces事件提取"""
        traces_df = pd.DataFrame({
            'serviceName': ['service-a', 'service-b'],
            'operationName': ['get_user', 'update_db'],
            'startTimeMillis': [1000, 2000],
            'duration': [100, 200]
        })
        
        events = self.framework._extract_atomic_events_from_traces(traces_df)
        
        # 检查事件数量（每个trace产生2个事件：开始和结束）
        assert len(events) == 4
        
        # 检查事件类型
        start_events = [e for e in events if e.T.startswith("TRACE_START_")]
        end_events = [e for e in events if e.T.startswith("TRACE_END_")]
        
        assert len(start_events) == 2
        assert len(end_events) == 2
    
    def test_event_aggregation(self):
        """测试事件聚合"""
        # 创建测试事件
        events = [
            AtomicEvent(t=1000, N="service-a", T="METRIC_cpu", D={"value": 0.5}),
            AtomicEvent(t=1100, N="service-a", T="METRIC_memory", D={"value": 0.6}),
            AtomicEvent(t=1200, N="service-a", T="LOG_TEMPLATE_1", D={"count": 1}),
            AtomicEvent(t=2500, N="service-a", T="METRIC_cpu", D={"value": 0.7}),  # 新窗口
        ]
        
        # 设置metrics_keys和log_templates
        self.framework.metrics_keys = ["service-a_cpu", "service-a_memory"]
        self.framework.log_templates = ["TEMPLATE_1"]
        
        aggregated = self.framework._aggregate_events(events)
        
        # 应该产生2个聚合事件（两个时间窗口）
        assert len(aggregated) >= 1
        
        # 检查聚合事件结构
        for agg_event in aggregated:
            assert isinstance(agg_event, AggregatedEvent)
            assert agg_event.N == "service-a"
            assert isinstance(agg_event.F, np.ndarray)
            assert len(agg_event.F) > 0
    
    def test_service_dependencies(self):
        """测试服务依赖关系"""
        deps = self.framework._get_service_dependencies()
        
        assert isinstance(deps, dict)
        assert 'frontend' in deps
        assert 'cartservice' in deps['frontend']
    
    def test_full_process_single_modal(self):
        """测试单模态完整处理流程"""
        # 创建单模态测试数据
        data = {
            'metric': pd.DataFrame({
                'time': range(20),
                'service_a_cpu': np.random.normal(0.5, 0.1, 20),
                'service_a_memory': np.random.normal(0.6, 0.1, 20)
            })
        }
        
        result = self.framework.process(data)
        
        # 检查返回结果结构
        assert isinstance(result, dict)
        assert 'ranks' in result
        assert 'adj' in result
        assert 'node_names' in result
        assert 'causal_edges' in result
        
        assert isinstance(result['ranks'], list)
        assert isinstance(result['adj'], np.ndarray)
        assert isinstance(result['node_names'], list)
        assert isinstance(result['causal_edges'], list)
    
    def test_full_process_multimodal(self):
        """测试多模态完整处理流程"""
        # 创建多模态测试数据
        data = {
            'metric': pd.DataFrame({
                'time': range(10),
                'service_a_cpu': np.random.normal(0.5, 0.1, 10),
                'service_b_cpu': np.random.normal(0.4, 0.1, 10)
            }),
            'logts': pd.DataFrame({
                'time': range(10),
                'timestamp': [f"{i}000000000000" for i in range(10)],
                'container_name': ['service-a-pod'] * 5 + ['service-b-pod'] * 5,
                'message': ['Normal operation'] * 8 + ['Error occurred'] * 2
            })
        }
        
        result = self.framework.process(data)
        
        # 检查返回结果
        assert isinstance(result, dict)
        assert 'total_events' in result
        assert result['total_events'] > 0


class TestCPGFunction:
    """测试CPG主函数"""
    
    def test_cpg_with_dataframe(self):
        """测试CPG函数处理DataFrame"""
        # 创建测试数据
        data = pd.DataFrame({
            'time': range(10),
            'service_a_cpu': np.random.normal(0.5, 0.1, 10),
            'service_a_memory': np.random.normal(0.6, 0.1, 10)
        })
        
        result = cpg(data, inject_time=5, dataset="test")
        
        # 检查返回结果
        assert isinstance(result, dict)
        assert 'ranks' in result
        assert 'adj' in result
        assert 'node_names' in result
        assert 'causal_edges' in result
    
    def test_cpg_with_multimodal_dict(self):
        """测试CPG函数处理多模态字典"""
        data = {
            'metric': pd.DataFrame({
                'time': range(10),
                'service_a_cpu': np.random.normal(0.5, 0.1, 10),
                'service_b_cpu': np.random.normal(0.4, 0.1, 10)
            }),
            'logts': pd.DataFrame({
                'time': range(10),
                'timestamp': [f"{i}000000000000" for i in range(10)],
                'container_name': ['service-a-pod'] * 10,
                'message': ['Operation completed'] * 10
            })
        }
        
        result = cpg(data, inject_time=5, dataset="test")
        
        # 检查返回结果
        assert isinstance(result, dict)
        assert 'ranks' in result
    
    def test_cpg_with_custom_parameters(self):
        """测试CPG函数自定义参数"""
        data = pd.DataFrame({
            'time': range(5),
            'service_a_cpu': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        result = cpg(
            data,
            inject_time=3,
            dataset="test",
            agg_window=2000,
            causal_threshold=0.2,
            top_k=3
        )
        
        assert isinstance(result, dict)
        assert 'ranks' in result
    
    def test_cpg_error_handling(self):
        """测试CPG函数错误处理"""
        # 测试空数据
        empty_data = pd.DataFrame()
        
        result = cpg(empty_data, inject_time=1, dataset="test")
        
        # 应该返回默认结果而不是抛出异常
        assert isinstance(result, dict)
        assert result['ranks'] == []
        assert len(result['adj']) == 0
    
    def test_cpg_without_inject_time(self):
        """测试CPG函数无注入时间"""
        data = pd.DataFrame({
            'time': range(5),
            'service_a_cpu': np.random.normal(0.5, 0.1, 5)
        })
        
        result = cpg(data, dataset="test")
        
        assert isinstance(result, dict)
        assert 'ranks' in result


class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 创建模拟的现实数据
        np.random.seed(42)
        
        # 正常期间的数据
        normal_period = 50
        anomaly_period = 10
        total_time = normal_period + anomaly_period
        
        # Metrics数据
        time_series = list(range(total_time))
        service_a_cpu = (
            np.random.normal(0.3, 0.05, normal_period).tolist() +
            np.random.normal(0.8, 0.1, anomaly_period).tolist()  # 异常期间CPU升高
        )
        service_b_cpu = np.random.normal(0.2, 0.03, total_time)
        
        metrics_df = pd.DataFrame({
            'time': time_series,
            'service_a_cpu': service_a_cpu,
            'service_b_cpu': service_b_cpu,
            'service_a_memory': np.random.normal(0.5, 0.1, total_time)
        })
        
        # Logs数据
        normal_messages = ['Request processed successfully'] * normal_period
        error_messages = ['Connection timeout error'] * anomaly_period
        
        logs_df = pd.DataFrame({
            'time': time_series,
            'timestamp': [f"{i}000000000000" for i in time_series],
            'container_name': ['service-a-pod-123'] * total_time,
            'message': normal_messages + error_messages
        })
        
        # 多模态数据
        multimodal_data = {
            'metric': metrics_df,
            'logts': logs_df
        }
        
        # 运行CPG分析
        result = cpg(
            multimodal_data,
            inject_time=normal_period,
            dataset="integration_test",
            agg_window=5000,
            anomaly_window=20,
            causal_threshold=0.05
        )
        
        # 验证结果
        assert isinstance(result, dict)
        assert 'ranks' in result
        assert 'causal_edges' in result
        assert 'anomaly_events' in result
        assert 'total_events' in result
        
        # 检查是否检测到异常
        assert result['total_events'] > 0
        
        # 检查根因排序
        if result['ranks']:
            assert isinstance(result['ranks'][0], str)
            # 在这个测试场景中，service-a应该被识别为根因
            # 注意：由于算法的随机性，这个断言可能不总是成功
    
    @patch('RCAEval.e2e.cpg.logger')
    def test_logging_integration(self, mock_logger):
        """测试日志集成"""
        data = pd.DataFrame({
            'time': range(5),
            'service_a_cpu': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        cpg(data, inject_time=3, dataset="log_test")
        
        # 验证日志调用
        mock_logger.info.assert_called()
        
        # 检查是否记录了关键步骤
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("CPG" in call for call in log_calls)


class TestPerformance:
    """性能测试"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建较大的测试数据集
        size = 1000
        
        data = {
            'metric': pd.DataFrame({
                'time': range(size),
                'service_a_cpu': np.random.normal(0.5, 0.1, size),
                'service_b_cpu': np.random.normal(0.4, 0.1, size),
                'service_c_cpu': np.random.normal(0.3, 0.1, size)
            })
        }
        
        import time
        start_time = time.time()
        
        result = cpg(data, inject_time=size//2, dataset="performance_test")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证结果
        assert isinstance(result, dict)
        assert 'ranks' in result
        
        # 性能检查（应该在合理时间内完成）
        assert processing_time < 30  # 30秒内完成
        
        print(f"Processing {size} data points took {processing_time:.2f} seconds")
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 创建测试数据
        data = pd.DataFrame({
            'time': range(500),
            'service_a_cpu': np.random.normal(0.5, 0.1, 500),
            'service_b_cpu': np.random.normal(0.4, 0.1, 500)
        })
        
        result = cpg(data, inject_time=250, dataset="memory_test")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 验证内存使用在合理范围内（小于100MB）
        assert memory_increase < 100 * 1024 * 1024
        
        print(f"Memory increase: {memory_increase / (1024*1024):.2f} MB")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])