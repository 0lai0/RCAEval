#!/usr/bin/env python3
import subprocess
import re
import pandas as pd
import shutil
import os
from datetime import datetime

# Batch-run orginTest.py for multiple methods and datasets and collect metrics
# Cleans output/results/ before each run

def save_individual_results(all_results, method, dataset, timestamp, output_dir):
    """Save individual experiment results in the requested format"""
    # Filter results for current method and dataset
    current_results = [r for r in all_results if r['Method'] == method and r['Dataset'] == dataset]
    
    if not current_results:
        return
    
    # Create method-specific subdirectory
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # Create filename: dataset_method_timestamp
    filename = f"{dataset}_{method}_{timestamp}.txt"
    filepath = os.path.join(method_dir, filename)
    
    # Format the table as requested
    lines = []
    lines.append("Metric Type\tprecision@1\tprecision@3\tprecision@5\tavg@5")
    
    # Define the order of metrics to display
    metric_order = [
        ('Overall Performance', 'overall'),
        ('CPU Faults', 'cpu'),
        ('MEM Faults', 'mem'),
        ('DISK Faults', 'disk'),
        ('SOCKET Faults', 'socket'),
        ('DELAY Faults', 'delay'),
        ('LOSS Faults', 'loss')
    ]
    
    for display_name, key in metric_order:
        # Find the result for this fault type
        result = next((r for r in current_results if r['Fault Type'] == key), None)
        if result:
            line = f"{display_name}\t{result['Precision@1']:.4f}\t{result['Precision@3']:.4f}\t{result['Precision@5']:.4f}\t{result['Avg@5']:.4f}"
        else:
            line = f"{display_name}\t0.0000\t0.0000\t0.0000\t0.0000"
        lines.append(line)
    
    # Save to file
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Results saved to: {filepath}")

def save_comprehensive_comparison(df, summary_dir, timestamp):
    """Save a comprehensive comparison table for all datasets and methods"""
    # Create comparison table for each dataset
    datasets = df['Dataset'].unique()
    
    comparison_lines = []
    comparison_lines.append("RE Dataset Comparison Results")
    comparison_lines.append("=" * 80)
    comparison_lines.append("")
    
    for dataset in datasets:
        comparison_lines.append(f"Dataset: {dataset.upper()}")
        comparison_lines.append("-" * 40)
        
        # Get data for this dataset
        dataset_data = df[df['Dataset'] == dataset]
        
        # Create table header
        comparison_lines.append("Metric Type\tprecision@1\tprecision@3\tprecision@5\tavg@5-baro\tavg@5-cpg")
        
        # Group by fault type
        fault_types = ['overall', 'cpu', 'mem', 'disk', 'socket', 'delay', 'loss']
        fault_names = ['Overall Performance', 'CPU Faults', 'MEM Faults', 'DISK Faults', 
                      'SOCKET Faults', 'DELAY Faults', 'LOSS Faults']
        
        for fault_type, fault_name in zip(fault_types, fault_names):
            baro_data = dataset_data[(dataset_data['Fault Type'] == fault_type) & (dataset_data['Method'] == 'baro')]
            cpg_data = dataset_data[(dataset_data['Fault Type'] == fault_type) & (dataset_data['Method'] == 'cpg')]
            
            if not baro_data.empty and not cpg_data.empty:
                baro_row = baro_data.iloc[0]
                cpg_row = cpg_data.iloc[0]
                line = f"{fault_name}\t{baro_row['Precision@1']:.4f}\t{baro_row['Precision@3']:.4f}\t{baro_row['Precision@5']:.4f}\t{baro_row['Avg@5']:.4f}\t{cpg_row['Avg@5']:.4f}"
            else:
                line = f"{fault_name}\t0.0000\t0.0000\t0.0000\t0.0000\t0.0000"
            
            comparison_lines.append(line)
        
        comparison_lines.append("")
        comparison_lines.append("")
    
    # Save comprehensive comparison
    comparison_file = os.path.join(summary_dir, f"comprehensive_comparison_{timestamp}.txt")
    with open(comparison_file, 'w') as f:
        f.write('\n'.join(comparison_lines))
    
    print(f"Comprehensive comparison saved to: {comparison_file}")

def run_and_parse_orgin_experiments():
    # For testing, use a smaller subset first
    experiments = [
        ('baro', 'online-boutique'),
        # Uncomment below for full RE dataset experiments
        # ('baro', 're1-ob'), ('cpg', 're1-ob'),
        # ('baro', 're1-ss'), ('cpg', 're1-ss'),
        # ('baro', 're1-tt'), ('cpg', 're1-tt'),
        # ('baro', 're2-ob'), ('cpg', 're2-ob'),
        # ('baro', 're2-ss'), ('cpg', 're2-ss'),
        # ('baro', 're2-tt'), ('cpg', 're2-tt'),
        # ('baro', 're3-ob'), ('cpg', 're3-ob'),
        # ('baro', 're3-ss'), ('cpg', 're3-ss'),
        # ('baro', 're3-tt'), ('cpg', 're3-tt'),
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "orginTest_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results_dir = "output/results"

    results = []
    print(f"Running {len(experiments)} experiments via orginTest.py...")
    for method, dataset in experiments:
        print(f"\n--- {method} on {dataset} ---")
        
        # Clean output/results/ directory before each run
        if os.path.exists(results_dir):
            print(f"Cleaning {results_dir}...")
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        cmd = ["python", "orginTest.py", "--method", method, "--dataset", dataset]
        print(f"Command: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            out = proc.stdout
            
            # Parse the metrics table from orginTest.py output
            # Look for the formatted metrics table
            parsing_metrics = False
            current_fault_type = None
            
            for line in out.splitlines():
                # Look for the metrics table section
                if "Performance Metrics for" in line and method.upper() in line:
                    parsing_metrics = True
                    continue
                
                # Stop parsing when we hit the separator line
                if parsing_metrics and "=" * 80 in line:
                    break
                
                # Parse metrics lines
                if parsing_metrics and "\t" in line and not line.startswith("Metric Type"):
                    parts = line.split("\t")
                    if len(parts) >= 5:
                        fault_type = parts[0].strip()
                        precision_1 = float(parts[1].strip())
                        precision_3 = float(parts[2].strip())
                        precision_5 = float(parts[3].strip())
                        avg_5 = float(parts[4].strip())
                        
                        # Map fault type names to standard format
                        fault_mapping = {
                            'Overall Performance': 'overall',
                            'CPU Faults': 'cpu',
                            'MEM Faults': 'mem',
                            'DISK Faults': 'disk',
                            'SOCKET Faults': 'socket',
                            'DELAY Faults': 'delay',
                            'LOSS Faults': 'loss'
                        }
                        
                        if fault_type in fault_mapping:
                            mapped_fault = fault_mapping[fault_type]
                            results.append({
                                'Method': method,
                                'Dataset': dataset,
                                'Fault Type': mapped_fault,
                                'Precision@1': precision_1,
                                'Precision@3': precision_3,
                                'Precision@5': precision_5,
                                'Avg@5': avg_5
                            })
            
            # Save individual experiment results to orginTest_results
            save_individual_results(results, method, dataset, timestamp, output_dir)
            
            print("Experiment completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running {method} on {dataset}: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")

    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)
    
    # Save CSV files to a separate summary directory
    summary_dir = "run_experiments_orgin"
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save long format
    csv_long = os.path.join(summary_dir, f"summary_long_{timestamp}.csv")
    df.to_csv(csv_long, index=False, encoding='utf-8-sig')
    print(f"\nSaved long-format results to {csv_long}")

    # Pivot to wide format for Avg@5
    comp = df.pivot_table(index=['Dataset'], columns='Method', values='Avg@5')
    comp.reset_index(inplace=True)
    csv_wide = os.path.join(summary_dir, f"summary_wide_{timestamp}.csv")
    comp.to_csv(csv_wide, index=False, encoding='utf-8-sig')
    print(f"Saved wide-format results to {csv_wide}")
    
    # Also save a comprehensive comparison table
    save_comprehensive_comparison(df, summary_dir, timestamp)

    # Define colors for terminal output
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

    # Print detailed comparison report
    print("\n\n" + "="*80)
    print("         方法橫向比較報告 (Comparison Report) - orginTest.py Results")
    print("="*80)
    
    # Group by dataset and fault type for detailed display
    detailed_df = df.pivot_table(
        index=['Dataset', 'Fault Type'],
        columns='Method',
        values='Avg@5'
    ).reset_index()
    
    # Create detailed comparison with all metrics
    print(f"{'Dataset':<12} {'Fault Type':<15} {'Avg@5_baro':>12} {'Avg@5_cpg':>12}")
    print("-" * 80)
    
    current_dataset = None
    for _, row in detailed_df.iterrows():
        if row['Dataset'] != current_dataset:
            if current_dataset is not None:
                print(Colors.YELLOW + "-" * 80 + Colors.RESET)
            current_dataset = row['Dataset']
        
        val_baro = row.get('baro', 0)
        val_cpg = row.get('cpg', 0)
        baro_str = f"{val_baro:>12.4f}"
        cpg_str = f"{val_cpg:>12.4f}"
        
        if val_baro > val_cpg:
            baro_str = Colors.GREEN + baro_str + Colors.RESET
        elif val_cpg > val_baro:
            cpg_str = Colors.GREEN + cpg_str + Colors.RESET
        
        print(f"{row['Dataset']:<12} {row['Fault Type']:<15} {baro_str} {cpg_str}")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)

if __name__ == '__main__':
    run_and_parse_orgin_experiments()

