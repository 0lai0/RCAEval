#!/usr/bin/env python3
"""
批量CPG調參腳本
對所有數據集運行基因演算法找出最佳參數
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CPGBatchTuner:
    """CPG批量調參器"""
    
    def __init__(self, datasets_dir: str = "data", output_dir: str = "tuned_params"):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 數據集配置
        self.dataset_configs = {
            'online-boutique': {
                'has_metrics': True,
                'has_logs': True, 
                'has_traces': True,
                'inject_times': [1659726540, 1659726600, 1659726660]  # 示例注入時間
            },
            'sock-shop': {
                'has_metrics': True,
                'has_logs': True,
                'has_traces': False,
                'inject_times': [1659726540, 1659726600]
            },
            'train-ticket': {
                'has_metrics': True,
                'has_logs': True,
                'has_traces': True,
                'inject_times': [1659726540, 1659726600, 1659726660]
            }
        }
    
    def find_datasets(self) -> List[str]:
        """查找所有可用的數據集"""
        datasets = []
        for dataset_name in self.dataset_configs.keys():
            dataset_path = self.datasets_dir / dataset_name
            if dataset_path.exists():
                datasets.append(dataset_name)
                logger.info(f"找到數據集: {dataset_name}")
            else:
                logger.warning(f"數據集不存在: {dataset_path}")
        
        # 也檢查其他目錄
        for subdir in self.datasets_dir.iterdir():
            if subdir.is_dir() and subdir.name not in datasets:
                # 檢查是否有metrics數據
                if any(f.name.endswith('.csv') or f.name.endswith('.json') for f in subdir.iterdir()):
                    datasets.append(subdir.name)
                    logger.info(f"發現新數據集: {subdir.name}")
        
        return datasets
    
    def prepare_dataset_for_tuning(self, dataset_name: str) -> Dict[str, Any]:
        """準備數據集用於調參"""
        dataset_path = self.datasets_dir / dataset_name
        config = self.dataset_configs.get(dataset_name, {
            'has_metrics': True,
            'has_logs': False,
            'has_traces': False,
            'inject_times': []
        })
        
        data_files = {}
        
        # 查找metrics文件
        if config.get('has_metrics', True):
            metrics_files = list(dataset_path.glob('*metric*.csv')) + \
                           list(dataset_path.glob('*metric*.json')) + \
                           list(dataset_path.glob('metrics.csv'))
            if metrics_files:
                data_files['metrics'] = str(metrics_files[0])
            else:
                # 尋找任何CSV文件
                csv_files = list(dataset_path.glob('*.csv'))
                if csv_files:
                    data_files['metrics'] = str(csv_files[0])
        
        # 查找logs文件
        if config.get('has_logs', False):
            log_files = list(dataset_path.glob('*log*.csv')) + \
                       list(dataset_path.glob('*log*.json'))
            if log_files:
                data_files['logs'] = str(log_files[0])
        
        # 查找traces文件
        if config.get('has_traces', False):
            trace_files = list(dataset_path.glob('*trace*.csv')) + \
                         list(dataset_path.glob('*trace*.json'))
            if trace_files:
                data_files['traces'] = str(trace_files[0])
        
        return {
            'name': dataset_name,
            'files': data_files,
            'config': config
        }
    
    def run_ga_tuning(self, dataset_info: Dict[str, Any], 
                     iterations: int = 100) -> Dict[str, Any]:
        """對單個數據集運行GA調參"""
        dataset_name = dataset_info['name']
        logger.info(f"開始調參數據集: {dataset_name}")
        
        try:
            # 創建臨時數據文件（JSON格式）
            temp_data_file = self.output_dir / f"{dataset_name}_temp_data.json"
            
            # 載入並合併數據
            combined_data = {}
            
            if 'metrics' in dataset_info['files']:
                metrics_file = dataset_info['files']['metrics']
                if metrics_file.endswith('.csv'):
                    df = pd.read_csv(metrics_file)
                    combined_data['metric'] = df.to_dict('records')
                elif metrics_file.endswith('.json'):
                    with open(metrics_file, 'r') as f:
                        combined_data['metric'] = json.load(f)
            
            if 'logs' in dataset_info['files']:
                logs_file = dataset_info['files']['logs']
                if logs_file.endswith('.csv'):
                    df = pd.read_csv(logs_file)
                    combined_data['logts'] = df.to_dict('records')
                elif logs_file.endswith('.json'):
                    with open(logs_file, 'r') as f:
                        combined_data['logts'] = json.load(f)
            
            if 'traces' in dataset_info['files']:
                traces_file = dataset_info['files']['traces']
                if traces_file.endswith('.csv'):
                    df = pd.read_csv(traces_file)
                    combined_data['traces'] = df.to_dict('records')
                elif traces_file.endswith('.json'):
                    with open(traces_file, 'r') as f:
                        combined_data['traces'] = json.load(f)
            
            # 保存臨時數據文件
            with open(temp_data_file, 'w') as f:
                json.dump(combined_data, f)
            
            # 運行GA調參
            output_file = self.output_dir / f"{dataset_name}_best_params.json"
            
            cmd = [
                'python', 'ga_tune.py',
                '--dataset', str(temp_data_file),
                '--output', str(output_file),
                '--iterations', str(iterations),
                '--name', dataset_name
            ]
            
            logger.info(f"執行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                # 讀取結果
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        best_params = json.load(f)
                    
                    logger.info(f"數據集 {dataset_name} 調參完成")
                    logger.info(f"最佳參數: anomaly_percentile={best_params.get('anomaly_percentile', 'N/A'):.3f}, "
                              f"causal_percentile={best_params.get('causal_percentile', 'N/A'):.3f}")
                    
                    # 清理臨時文件
                    temp_data_file.unlink(missing_ok=True)
                    
                    return best_params
                else:
                    logger.error(f"調參輸出文件不存在: {output_file}")
            else:
                logger.error(f"調參失敗: {result.stderr}")
            
        except Exception as e:
            logger.error(f"數據集 {dataset_name} 調參出錯: {str(e)}")
        
        # 清理臨時文件
        if temp_data_file.exists():
            temp_data_file.unlink()
        
        return {}
    
    def evaluate_params(self, dataset_name: str, params: Dict[str, Any]) -> Dict[str, float]:
        """評估參數在數據集上的表現"""
        try:
            # 使用調參後的參數運行CPG
            cmd = [
                'python', 'main.py',
                '--method', 'cpg',
                '--dataset', dataset_name,
                '--anomaly_percentile', str(params.get('anomaly_percentile', 0.95)),
                '--causal_percentile', str(params.get('causal_percentile', 0.90))
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # 解析評估結果
                output_lines = result.stdout.split('\n')
                metrics = {}
                
                for line in output_lines:
                    if 'Avg@5-' in line:
                        parts = line.split(':')
                        if len(parts) == 2:
                            metric_name = parts[0].strip().replace('Avg@5-', '')
                            metric_value = float(parts[1].strip())
                            metrics[metric_name] = metric_value
                
                return metrics
            else:
                logger.error(f"評估失敗: {result.stderr}")
                
        except Exception as e:
            logger.error(f"評估參數時出錯: {str(e)}")
        
        return {}
    
    def run_batch_tuning(self, max_workers: int = 2, iterations: int = 100) -> Dict[str, Any]:
        """批量運行所有數據集的調參"""
        datasets = self.find_datasets()
        
        if not datasets:
            logger.error("未找到任何數據集")
            return {}
        
        logger.info(f"準備調參 {len(datasets)} 個數據集: {datasets}")
        
        all_results = {}
        
        # 準備數據集信息
        dataset_infos = []
        for dataset_name in datasets:
            dataset_info = self.prepare_dataset_for_tuning(dataset_name)
            if dataset_info['files']:
                dataset_infos.append(dataset_info)
            else:
                logger.warning(f"數據集 {dataset_name} 沒有找到有效數據文件")
        
        # 並行調參（限制並行數以避免資源耗盡）
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_dataset = {
                executor.submit(self.run_ga_tuning, dataset_info, iterations): dataset_info['name']
                for dataset_info in dataset_infos
            }
            
            for future in as_completed(future_to_dataset):
                dataset_name = future_to_dataset[future]
                try:
                    best_params = future.result()
                    if best_params:
                        all_results[dataset_name] = best_params
                        
                        # 評估參數效果
                        logger.info(f"評估數據集 {dataset_name} 的調參效果")
                        evaluation = self.evaluate_params(dataset_name, best_params)
                        all_results[dataset_name]['evaluation'] = evaluation
                        
                except Exception as e:
                    logger.error(f"數據集 {dataset_name} 處理失敗: {str(e)}")
        
        # 保存總結報告
        summary_file = self.output_dir / "tuning_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"批量調參完成，結果保存至: {summary_file}")
        
        # 生成最佳參數建議
        self.generate_recommendations(all_results)
        
        return all_results
    
    def generate_recommendations(self, results: Dict[str, Any]):
        """生成參數建議報告"""
        recommendations_file = self.output_dir / "parameter_recommendations.md"
        
        with open(recommendations_file, 'w') as f:
            f.write("# CPG參數調優建議報告\n\n")
            f.write("## 各數據集最佳參數\n\n")
            
            # 按數據集大小分類
            small_datasets = []
            medium_datasets = []
            large_datasets = []
            
            for dataset_name, result in results.items():
                data_size = result.get('data_size', 0)
                if data_size < 1000:
                    small_datasets.append((dataset_name, result))
                elif data_size < 5000:
                    medium_datasets.append((dataset_name, result))
                else:
                    large_datasets.append((dataset_name, result))
            
            f.write("### 小型數據集 (<1000事件)\n")
            self._write_dataset_recommendations(f, small_datasets)
            
            f.write("\n### 中型數據集 (1000-5000事件)\n")
            self._write_dataset_recommendations(f, medium_datasets)
            
            f.write("\n### 大型數據集 (>5000事件)\n")
            self._write_dataset_recommendations(f, large_datasets)
            
            # 通用建議
            f.write("\n## 通用參數建議\n\n")
            
            all_anomaly_percs = [r.get('anomaly_percentile', 0.95) for r in results.values() if 'anomaly_percentile' in r]
            all_causal_percs = [r.get('causal_percentile', 0.90) for r in results.values() if 'causal_percentile' in r]
            
            if all_anomaly_percs and all_causal_percs:
                avg_anomaly = np.mean(all_anomaly_percs)
                avg_causal = np.mean(all_causal_percs)
                
                f.write(f"- **平均最佳異常百分位**: {avg_anomaly:.3f}\n")
                f.write(f"- **平均最佳因果百分位**: {avg_causal:.3f}\n")
                f.write(f"- **異常百分位範圍**: {min(all_anomaly_percs):.3f} - {max(all_anomaly_percs):.3f}\n")
                f.write(f"- **因果百分位範圍**: {min(all_causal_percs):.3f} - {max(all_causal_percs):.3f}\n")
        
        logger.info(f"參數建議報告已生成: {recommendations_file}")
    
    def _write_dataset_recommendations(self, f, datasets):
        """寫入數據集建議"""
        for dataset_name, result in datasets:
            f.write(f"- **{dataset_name}**:\n")
            f.write(f"  - 數據大小: {result.get('data_size', 'N/A')}\n")
            f.write(f"  - 異常百分位: {result.get('anomaly_percentile', 'N/A'):.3f}\n")
            f.write(f"  - 因果百分位: {result.get('causal_percentile', 'N/A'):.3f}\n")
            f.write(f"  - 適應度分數: {result.get('best_fitness', 'N/A'):.3f}\n")
            
            evaluation = result.get('evaluation', {})
            if evaluation:
                avg_score = np.mean(list(evaluation.values()))
                f.write(f"  - 平均評分: {avg_score:.3f}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='CPG批量調參工具')
    parser.add_argument('--datasets-dir', default='data', help='數據集根目錄')
    parser.add_argument('--output-dir', default='tuned_params', help='輸出目錄')
    parser.add_argument('--max-workers', type=int, default=2, help='並行工作數')
    parser.add_argument('--iterations', type=int, default=100, help='每個數據集的GA迭代次數')
    parser.add_argument('--datasets', nargs='+', help='指定要調參的數據集（可選）')
    
    args = parser.parse_args()
    
    tuner = CPGBatchTuner(args.datasets_dir, args.output_dir)
    
    if args.datasets:
        # 只調參指定的數據集
        logger.info(f"調參指定數據集: {args.datasets}")
        results = {}
        for dataset_name in args.datasets:
            dataset_info = tuner.prepare_dataset_for_tuning(dataset_name)
            if dataset_info['files']:
                result = tuner.run_ga_tuning(dataset_info, args.iterations)
                if result:
                    results[dataset_name] = result
                    evaluation = tuner.evaluate_params(dataset_name, result)
                    results[dataset_name]['evaluation'] = evaluation
        
        # 保存結果
        summary_file = Path(args.output_dir) / "tuning_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        tuner.generate_recommendations(results)
    else:
        # 調參所有數據集
        results = tuner.run_batch_tuning(args.max_workers, args.iterations)
    
    logger.info("批量調參完成！")

if __name__ == "__main__":
    main()
