#!/usr/bin/env python3
"""
離線GA調參腳本 - CPG閾值優化
根據數據集特徵自動尋找最佳參數
"""

import numpy as np
import pandas as pd
import json
import argparse
import logging
from typing import Dict, Any, Tuple
import networkx as nx
from RCAEval.e2e.cpg import CPGFramework
from RCAEval.io.time_series import preprocess, drop_constant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GAThresholdTuner:
    """遺傳算法閾值調參器"""
    
    def __init__(self, iterations=50, population_size=20):
        self.iterations = iterations
        self.population_size = population_size
        
    def tune_thresholds(self, data: Dict[str, pd.DataFrame], 
                       dataset_name: str = "unknown") -> Dict[str, float]:
        """
        對給定數據集優化CPG閾值參數
        
        Args:
            data: 多模態數據字典
            dataset_name: 數據集名稱
            
        Returns:
            最佳參數字典
        """
        logger.info(f"開始GA調參，數據集: {dataset_name}")
        
        # 初始化CPG框架（使用percentile模式，關閉p-value測試以降低成本）
        framework = CPGFramework(
            agg_window=5000,
            threshold_mode='percentile',
            p_value_threshold=None  # 關閉p-value測試
        )
        
        # 提取並聚合事件
        all_events = []
        if 'metric' in data:
            metric_events = framework._extract_atomic_events_from_metrics(data['metric'])
            all_events.extend(metric_events)
            logger.info(f"提取到 {len(metric_events)} 個metrics事件")
            
        if 'logts' in data:
            log_events = framework._extract_atomic_events_from_logs(data['logts'])
            all_events.extend(log_events)
            logger.info(f"提取到 {len(log_events)} 個日誌事件")
            
        if 'traces' in data:
            trace_events = framework._extract_atomic_events_from_traces(data['traces'])
            all_events.extend(trace_events)
            logger.info(f"提取到 {len(trace_events)} 個trace事件")
        
        if not all_events:
            logger.error("未提取到任何事件")
            return self._default_params()
            
        # 聚合事件
        aggregated_events = framework._aggregate_events(all_events)
        n = len(aggregated_events)
        logger.info(f"生成 {n} 個聚合事件")
        
        if n == 0:
            return self._default_params()
        
        # 根據數據大小設定目標異常比例
        if n <= 1000:
            target_anomaly_ratio = 0.08
        elif n >= 5000:
            target_anomaly_ratio = 0.02
        else:
            # 線性插值
            t = (n - 1000) / (5000 - 1000)
            target_anomaly_ratio = 0.08 * (1 - t) + 0.02 * t
            
        logger.info(f"目標異常比例: {target_anomaly_ratio:.3f}")
        
        # 預計算異常分數
        scores = framework._compute_anomaly_scores(aggregated_events)
        score_values = np.array([s for _, s in scores])
        
        if len(score_values) == 0:
            return self._default_params()
            
        # 計算數據變異性
        cv = np.std(score_values) / (np.mean(np.abs(score_values)) + 1e-6)
        logger.info(f"分數變異係數: {cv:.3f}")
        
        # GA優化
        best_fitness = -1e9
        best_params = (0.95, 0.90)  # 默認值
        
        logger.info(f"開始GA優化，迭代次數: {self.iterations}")
        
        # 初始化種群
        population = []
        for _ in range(self.population_size):
            individual = (
                np.random.uniform(0.80, 0.99),  # anomaly_percentile
                np.random.uniform(0.60, 0.98)   # causal_percentile
            )
            population.append(individual)
        
        # 進化過程
        for generation in range(self.iterations // self.population_size + 1):
            # 評估種群
            fitness_scores = []
            for anomaly_perc, causal_perc in population:
                fitness = self._evaluate_params(
                    anomaly_perc, causal_perc, 
                    scores, aggregated_events, framework,
                    target_anomaly_ratio, cv, n
                )
                fitness_scores.append((fitness, (anomaly_perc, causal_perc)))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = (anomaly_perc, causal_perc)
                    logger.info(f"世代 {generation+1}: 新最佳 fitness={fitness:.4f}, "
                              f"anomaly_perc={anomaly_perc:.3f}, causal_perc={causal_perc:.3f}")
            
            # 選擇和交叉
            fitness_scores.sort(reverse=True)  # 按適應度排序
            survivors = [individual for _, individual in fitness_scores[:self.population_size//2]]
            
            # 生成新種群
            new_population = survivors.copy()
            while len(new_population) < self.population_size:
                # 選擇父母
                parent1 = np.random.choice(len(survivors))
                parent2 = np.random.choice(len(survivors))
                
                # 交叉
                child_anomaly = (survivors[parent1][0] + survivors[parent2][0]) / 2
                child_causal = (survivors[parent1][1] + survivors[parent2][1]) / 2
                
                # 突變
                if np.random.random() < 0.1:  # 10%突變率
                    child_anomaly += np.random.normal(0, 0.02)
                    child_causal += np.random.normal(0, 0.02)
                
                # 限制範圍
                child_anomaly = np.clip(child_anomaly, 0.80, 0.99)
                child_causal = np.clip(child_causal, 0.60, 0.98)
                
                new_population.append((child_anomaly, child_causal))
            
            population = new_population
        
        result = {
            'anomaly_percentile': best_params[0],
            'causal_percentile': best_params[1],
            'dataset': dataset_name,
            'data_size': n,
            'target_anomaly_ratio': target_anomaly_ratio,
            'best_fitness': best_fitness,
            'coefficient_variation': cv
        }
        
        logger.info(f"GA優化完成，最佳參數: {result}")
        return result
    
    def _evaluate_params(self, anomaly_perc: float, causal_perc: float,
                        scores, aggregated_events, framework,
                        target_ratio: float, cv: float, n: int) -> float:
        """評估參數組合的適應度"""
        try:
            # 1. 異常檢測評估
            score_values = [s for _, s in scores]
            threshold_score = np.percentile(score_values, anomaly_perc * 100.0)
            anomaly_events = [e for e, s in scores if s >= threshold_score]
            
            actual_ratio = len(anomaly_events) / max(1, n)
            ratio_score = 1.0 - abs(actual_ratio - target_ratio) / max(target_ratio, 1e-6)
            ratio_score = max(0.0, ratio_score)
            
            # 2. 因果邊評估
            edges_count = 0
            service_deps = framework._get_service_dependencies()
            
            for symptom in anomaly_events[:10]:  # 限制檢查數量以控制成本
                try:
                    upstream_services = service_deps.get(symptom.N, [])
                    for upstream_svc in upstream_services:
                        candidates = [
                            e for e in aggregated_events
                            if e.N == upstream_svc and 
                            symptom.t - framework.lookback_window <= e.t < symptom.t
                        ]
                        candidates = sorted(candidates, key=lambda e: e.t, reverse=True)[:framework.top_k]
                        
                        if not candidates:
                            continue
                            
                        te_values = []
                        for candidate in candidates:
                            te = framework.te_calculator.calculate(candidate.F, symptom.F)
                            te_values.append(te)
                        
                        if te_values:
                            thr = np.percentile(te_values, causal_perc * 100.0)
                            edges_count += sum(1 for te in te_values if te >= thr)
                except Exception:
                    continue
            
            edge_score = min(edges_count / 20.0, 1.0)  # 標準化到[0,1]
            
            # 3. 圖結構評估（模組化）
            modularity_score = 0.0
            if edges_count > 0:
                try:
                    # 簡化的模組化估計
                    modularity_score = min(edges_count / 50.0, 0.5)  # 假設適度的模組化
                except Exception:
                    modularity_score = 0.0
            
            # 4. 懲罰項
            penalty = 0.0
            if edges_count == 0:
                penalty = 0.5  # 嚴重懲罰無邊情況
            elif actual_ratio < 0.001:
                penalty = 0.3  # 懲罰異常數過少
            elif actual_ratio > 0.15:
                penalty = 0.2  # 懲罰異常數過多
            
            # 綜合適應度函數
            fitness = (0.3 * edge_score + 
                      0.3 * ratio_score + 
                      0.2 * modularity_score + 
                      0.2 * min(cv, 1.0) - 
                      penalty)
            
            return fitness
            
        except Exception as e:
            logger.warning(f"參數評估失敗: {str(e)}")
            return -1.0
    
    def _default_params(self) -> Dict[str, float]:
        """返回默認參數"""
        return {
            'anomaly_percentile': 0.95,
            'causal_percentile': 0.90,
            'dataset': 'unknown',
            'data_size': 0,
            'target_anomaly_ratio': 0.05,
            'best_fitness': 0.0,
            'coefficient_variation': 0.0
        }

def load_dataset(dataset_path: str) -> Dict[str, pd.DataFrame]:
    """載入數據集"""
    # 這裡需要根據實際的數據集格式進行調整
    # 示例實現
    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
            return {'metric': df}
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data_dict = json.load(f)
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, list):
                    result[key] = pd.DataFrame(value)
                elif isinstance(value, dict):
                    result[key] = pd.DataFrame.from_dict(value)
            return result
        else:
            logger.error(f"不支援的文件格式: {dataset_path}")
            return {}
    except Exception as e:
        logger.error(f"載入數據集失敗: {str(e)}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='CPG閾值GA調參工具')
    parser.add_argument('--dataset', required=True, help='數據集文件路徑')
    parser.add_argument('--output', default='cpg_tuned_params.json', help='輸出參數文件')
    parser.add_argument('--iterations', type=int, default=50, help='GA迭代次數')
    parser.add_argument('--name', default='unknown', help='數據集名稱')
    
    args = parser.parse_args()
    
    # 載入數據
    logger.info(f"載入數據集: {args.dataset}")
    data = load_dataset(args.dataset)
    
    if not data:
        logger.error("數據集載入失敗")
        return
    
    # 預處理數據
    if 'metric' in data:
        try:
            data['metric'] = preprocess(data['metric'], dataset=args.name)
        except Exception as e:
            logger.warning(f"metrics預處理失敗: {str(e)}")
    
    if 'logts' in data:
        try:
            data['logts'] = drop_constant(data['logts'])
        except Exception as e:
            logger.warning(f"logs預處理失敗: {str(e)}")
    
    # GA調參
    tuner = GAThresholdTuner(iterations=args.iterations)
    best_params = tuner.tune_thresholds(data, args.name)
    
    # 保存結果
    with open(args.output, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"調參完成，結果已保存至: {args.output}")
    print(f"\n推薦參數:")
    print(f"anomaly_percentile: {best_params['anomaly_percentile']:.3f}")
    print(f"causal_percentile: {best_params['causal_percentile']:.3f}")
    print(f"適用數據大小: {best_params['data_size']}")
    print(f"適應度分數: {best_params['best_fitness']:.3f}")

if __name__ == "__main__":
    main()
