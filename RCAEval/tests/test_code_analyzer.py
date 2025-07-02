import unittest
import pandas as pd
from RCAEval.e2e.code_analyzer import StackTraceAnalyzer

class TestStackTraceAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = StackTraceAnalyzer()
        
    def test_parse_stack_trace(self):
        # 測試一個典型的 stack trace
        stack_trace = """System.NullReferenceException: Object reference not set to an instance of an object.
            at Service.UserService.GetUser() in /app/UserService.cs:line 123"""
        
        result = self.analyzer.parse_stack_trace(stack_trace)
        
        # 驗證解析結果
        self.assertEqual(result['error_type'], 'null_reference')
        self.assertEqual(result['service'], 'service')
        self.assertEqual(result['file'], 'UserService.cs')
        self.assertEqual(result['line'], 123)
        self.assertEqual(result['method'], 'GetUser')
        self.assertGreater(result['severity'], 0.5)  # null reference 應該有較高的嚴重性
        
    def test_analyze_log_patterns(self):
        # 創建測試用的日誌數據
        logs_data = {
            'timestamp': [
                1000000,  # 正常時期
                1000000,  # 正常時期
                2000000,  # 異常時期
                2000000   # 異常時期
            ],
            'container_name': [
                'service1',
                'service1',
                'service1',
                'service1'
            ],
            'message': [
                'Normal log message',
                'System.NullReferenceException: Object reference not set\nat Service1.Method() in File.cs:line 100',
                'Error occurred',
                'System.OverflowException: Value too large\nat Service1.Process() in Process.cs:line 200'
            ]
        }
        logs_df = pd.DataFrame(logs_data)
        
        # 設定注入時間點（區分正常和異常時期）
        inject_time = 1.5  # 1.5 * 1_000_000 = 1500000
        
        log_features, stack_features = self.analyzer.analyze_log_patterns(logs_df, inject_time)
        
        # 驗證日誌特徵
        self.assertIn('service1_log_count', log_features)
        self.assertIn('service1_error_rate', log_features)
        
        # 驗證 stack trace 特徵
        self.assertIn('service1_stack_count', stack_features)
        self.assertIn('service1_error_severity', stack_features)
        self.assertIn('service1_has_code_error', stack_features)
        
        # 驗證具體的計數
        self.assertEqual(stack_features['service1_stack_count'], 1)  # 異常期間有1個 stack trace
        self.assertEqual(stack_features['service1_has_code_error'], 1)  # 有檢測到程式碼錯誤
        
    def test_empty_logs(self):
        # 測試空的日誌數據
        empty_df = pd.DataFrame()
        log_features, stack_features = self.analyzer.analyze_log_patterns(empty_df, 1.0)
        
        # 驗證返回空字典
        self.assertEqual(log_features, {})
        self.assertEqual(stack_features, {})
        
    def test_error_patterns(self):
        # 測試各種錯誤模式
        error_cases = {
            'overflow': 'System.OverflowException: Value was too large',
            'null_reference': 'System.NullReferenceException: Object reference not set',
            'timeout': 'System.TimeoutException: Request timeout occurred',
            'memory': 'System.OutOfMemoryException: Insufficient memory',
            'division_by_zero': 'System.DivideByZeroException: Division by zero occurred'
        }
        
        for error_type, stack_trace in error_cases.items():
            result = self.analyzer.parse_stack_trace(stack_trace)
            self.assertEqual(result['error_type'], error_type)

if __name__ == '__main__':
    unittest.main() 