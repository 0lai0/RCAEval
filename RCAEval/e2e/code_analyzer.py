import re
import pandas as pd

class StackTraceAnalyzer:
    """Stack trace 分析器，提取 code-level 指標"""
    
    def __init__(self):
        # 常見的錯誤模式
        self.error_patterns = {
            'overflow': r'OverflowException|Value was either too large or too small',
            'null_reference': r'NullReferenceException|Object reference not set',
            'argument_exception': r'ArgumentException|Parameter.*invalid',
            'timeout': r'TimeoutException|Request timeout',
            'connection': r'ConnectionException|Connection.*failed',
            'memory': r'OutOfMemoryException|Insufficient memory',
            'division_by_zero': r'DivideByZeroException|Division by zero',
            'index_out_of_range': r'IndexOutOfRangeException|Index was outside'
        }
        
        # 程式碼檔案和行號模式
        self.code_location_pattern = r'at\s+([^.]+)\.([^.]+)\.([^(]+)\([^)]*\)\s+in\s+([^:]+):line\s+(\d+)'
        
    def parse_stack_trace(self, stack_trace):
        """解析 stack trace 提取錯誤信息"""
        if pd.isna(stack_trace) or not isinstance(stack_trace, str):
            return {}
            
        result = {
            'error_type': 'unknown',
            'service': 'unknown',
            'file': 'unknown',
            'line': 0,
            'method': 'unknown',
            'severity': 0.5
        }
        
        # 檢測錯誤類型
        for error_type, pattern in self.error_patterns.items():
            if re.search(pattern, stack_trace, re.IGNORECASE):
                result['error_type'] = error_type
                # 設定嚴重性分數
                severity_map = {
                    'overflow': 0.9,
                    'memory': 0.95,
                    'null_reference': 0.8,
                    'timeout': 0.7,
                    'connection': 0.6,
                    'argument_exception': 0.5,
                    'division_by_zero': 0.85,
                    'index_out_of_range': 0.75
                }
                result['severity'] = severity_map.get(error_type, 0.5)
                break
        
        # 提取程式碼位置
        match = re.search(self.code_location_pattern, stack_trace)
        if match:
            result['service'] = match.group(1).lower()
            result['method'] = match.group(3)
            result['file'] = match.group(4).split('/')[-1]  # 只取檔名
            result['line'] = int(match.group(5))
            
        return result
    
    def analyze_log_patterns(self, logs_df, inject_time):
        """分析日誌模式，分離正常日誌和 stack traces"""
        if logs_df is None or logs_df.empty:
            return {}, {}
            
        # 分離正常時期和異常時期的日誌
        normal_logs = logs_df[logs_df['timestamp'] < inject_time * 1_000_000]
        anomaly_logs = logs_df[logs_df['timestamp'] >= inject_time * 1_000_000]
        
        # 分離一般日誌和 stack traces
        normal_simple_logs = normal_logs[normal_logs['message'].apply(lambda x: len(str(x).split("\n")) == 1)]
        normal_stack_traces = normal_logs[normal_logs['message'].apply(lambda x: len(str(x).split("\n")) > 1)]
        
        anomaly_simple_logs = anomaly_logs[anomaly_logs['message'].apply(lambda x: len(str(x).split("\n")) == 1)]
        anomaly_stack_traces = anomaly_logs[anomaly_logs['message'].apply(lambda x: len(str(x).split("\n")) > 1)]
        
        # 轉換成時間序列
        log_features = {}
        stack_features = {}
        
        # 處理每個服務的日誌
        services = logs_df['container_name'].unique()
        
        for service in services:
            # 正常日誌計數
            normal_svc_logs = normal_simple_logs[normal_simple_logs['container_name'] == service]
            anomaly_svc_logs = anomaly_simple_logs[anomaly_simple_logs['container_name'] == service]
            
            # Stack trace 計數和分析
            normal_svc_stacks = normal_stack_traces[normal_stack_traces['container_name'] == service]
            anomaly_svc_stacks = anomaly_stack_traces[anomaly_stack_traces['container_name'] == service]
            
            # 計算日誌特徵
            log_features[f"{service}_log_count"] = len(anomaly_svc_logs) - len(normal_svc_logs)
            log_features[f"{service}_error_rate"] = len(anomaly_svc_logs[anomaly_svc_logs['message'].str.contains('error|fail|exception', case=False, na=False)]) / max(len(anomaly_svc_logs), 1)
            
            # 計算 stack trace 特徵
            stack_features[f"{service}_stack_count"] = len(anomaly_svc_stacks)
            
            # 分析 stack traces 內容
            error_severity = 0
            error_types_found = []
            for _, row in anomaly_svc_stacks.iterrows():
                parsed = self.parse_stack_trace(row['message'])
                error_severity += parsed['severity']
                if parsed['error_type'] != 'unknown':
                    error_types_found.append(parsed['error_type'])
                    
            if error_types_found:
                print(f"[Code-Level] {service}: {set(error_types_found)}")
                
            stack_features[f"{service}_error_severity"] = error_severity / max(len(anomaly_svc_stacks), 1)
            stack_features[f"{service}_has_code_error"] = 1 if len(anomaly_svc_stacks) > 0 else 0
            
        return log_features, stack_features 