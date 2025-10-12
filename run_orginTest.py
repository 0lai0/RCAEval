#!/usr/bin/env python3
"""
Batch script to run baro and cpg methods using orginTest.py
- Cleans output directory before each run
- Saves results to orginTest_results/ with timestamp naming
- Supports multiple datasets
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
import argparse
import json
import glob

def clean_output_directory():
    """Remove all files in the output directory"""
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleaned output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

def run_experiment(method, dataset, test_mode=False):
    """Run a single experiment with the given method and dataset"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"Running {method} on {dataset}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*60}")
    
    # Clean output directory before running
    clean_output_directory()
    
    # Build command
    cmd = [
        sys.executable, "orginTest.py",
        "--method", method,
        "--dataset", dataset
    ]
    
    if test_mode:
        cmd.append("--test")
        print("Running in test mode (limited data)")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"{method} on {dataset} completed successfully")
            
            # Move results to orginTest_results with timestamp naming
            move_results_to_final_location(method, dataset, timestamp)
            
            return True
        else:
            print(f"{method} on {dataset} failed")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{method} on {dataset} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"{method} on {dataset} crashed: {e}")
        return False

def extract_metrics_from_results(result_dir, method):
    """Extract performance metrics from experiment results"""
    results_path = os.path.join(result_dir, "results")
    if not os.path.exists(results_path):
        return None
    
    # Import required modules for evaluation
    try:
        from RCAEval.benchmark.evaluation import Evaluator
        from RCAEval.classes.graph import Node
        from RCAEval.utility import load_json
    except ImportError as e:
        print(f"Warning: Could not import evaluation modules: {e}")
        return None
    
    # Get all result files
    result_files = glob.glob(os.path.join(results_path, "*.json"))
    
    if not result_files:
        return None
    
    # Initialize evaluators for different fault types
    evaluators = {
        'all': Evaluator(),
        'cpu': Evaluator(),
        'mem': Evaluator(),
        'disk': Evaluator(),
        'socket': Evaluator(),
        'delay': Evaluator(),
        'loss': Evaluator()
    }
    
    # Process each result file
    for rp in result_files:
        try:
            data = load_json(rp)
            if "error" in data:
                continue
            
            # Extract service and fault info from filename
            filename = os.path.basename(rp)
            parts = filename.replace('.json', '').split('_')
            if len(parts) < 3:
                continue
            
            service = parts[0]
            fault_type = parts[1]
            case = parts[2]
            
            for i, ranks in data.items():
                # Filter out IP address format strings
                s_ranks = []
                for x in ranks:
                    service_name = x.split("_")[0].replace("-db", "")
                    # Check if it's IP address format
                    is_ip_format = (
                        (service_name.startswith("192-168-") and service_name.count("-") >= 4) or
                        (service_name.count("-") >= 4 and all(part.isdigit() for part in service_name.split("-")))
                    )
                    if not is_ip_format:
                        s_ranks.append(Node(service_name, "unknown"))
                
                # Remove duplicates
                if s_ranks:
                    old_s_ranks = s_ranks.copy()
                    s_ranks = [old_s_ranks[0]] + [
                        old_s_ranks[i] for i in range(1, len(old_s_ranks))
                        if old_s_ranks[i] not in old_s_ranks[:i]
                    ]
                
                # Add to appropriate evaluators
                evaluators['all'].add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                
                if fault_type in evaluators:
                    evaluators[fault_type].add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                
                # Map delay and loss to appropriate evaluators
                if fault_type == "delay":
                    evaluators['delay'].add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                elif fault_type == "loss":
                    evaluators['loss'].add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    
        except Exception as e:
            print(f"Warning: Error processing {rp}: {e}")
            continue
    
    # Calculate metrics for each fault type
    metrics = {}
    for fault_type, evaluator in evaluators.items():
        if evaluator.num > 0:  # Only calculate if there are cases
            metrics[fault_type] = {
                'precision@1': evaluator.accuracy(1) if evaluator.accuracy(1) is not None else 0.0,
                'precision@3': evaluator.accuracy(3) if evaluator.accuracy(3) is not None else 0.0,
                'precision@5': evaluator.accuracy(5) if evaluator.accuracy(5) is not None else 0.0,
                'avg@5': evaluator.average(5) if evaluator.average(5) is not None else 0.0
            }
    
    return metrics

def format_metrics_table(metrics_data):
    """Format metrics data into the requested table format"""
    if not metrics_data:
        return "No metrics data available"
    
    # Create table header
    header = "Metric Type\tprecision@1\tprecision@3\tprecision@5\tavg@5"
    
    lines = [header]
    
    # Define the order of metrics to display
    metric_order = [
        ('Overall Performance', 'all'),
        ('CPU Faults', 'cpu'),
        ('MEM Faults', 'mem'),
        ('DISK Faults', 'disk'),
        ('SOCKET Faults', 'socket'),
        ('DELAY Faults', 'delay'),
        ('LOSS Faults', 'loss')
    ]
    
    for display_name, key in metric_order:
        if key in metrics_data:
            metrics = metrics_data[key]
            line = f"{display_name}\t{metrics['precision@1']:.4f}\t{metrics['precision@3']:.4f}\t{metrics['precision@5']:.4f}\t{metrics['avg@5']:.4f}"
            lines.append(line)
        else:
            line = f"{display_name}\t0.0000\t0.0000\t0.0000\t0.0000"
            lines.append(line)
    
    return "\n".join(lines)

def save_metrics_to_file(metrics_data, method, dataset, timestamp, output_dir):
    """Save formatted metrics to a text file"""
    formatted_table = format_metrics_table(metrics_data)
    
    # Create metrics file name
    metrics_filename = f"{method}_{dataset}_metrics_{timestamp}.txt"
    metrics_path = os.path.join(output_dir, metrics_filename)
    
    # Save to file
    with open(metrics_path, 'w') as f:
        f.write(formatted_table)
    
    print(f"Metrics saved to: {metrics_path}")
    return metrics_path

def move_results_to_final_location(method, dataset, timestamp):
    """Move results from output/ to orginTest_results/ with proper naming"""
    output_dir = "output"
    results_dir = "orginTest_results"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create method-specific subdirectory
    method_dir = os.path.join(results_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # Define target directory name: method_dataset_timestamp
    target_dir_name = f"{method}_{dataset}_{timestamp}"
    target_dir = os.path.join(method_dir, target_dir_name)
    
    if os.path.exists(output_dir):
        # Move the entire output directory to the target location
        shutil.move(output_dir, target_dir)
        print(f"Results moved to: {target_dir}")
        
        # Extract and display metrics
        print(f"\nExtracting metrics for {method} on {dataset}...")
        metrics_data = extract_metrics_from_results(target_dir, method)
        
        if metrics_data:
            # Display formatted metrics table
            formatted_table = format_metrics_table(metrics_data)
            print(f"\nPerformance Metrics for {method.upper()} on {dataset.upper()}:")
            print("="*80)
            print(formatted_table)
            print("="*80)
            
            # Save metrics to file
            save_metrics_to_file(metrics_data, method, dataset, timestamp, target_dir)
        else:
            print(f"Warning: Could not extract metrics from results")
        
        # Create a new empty output directory for next run
        os.makedirs(output_dir, exist_ok=True)
    else:
        print(f"No output directory found to move")

def generate_summary_metrics(results):
    """Generate a summary metrics file combining all successful experiments"""
    summary_lines = []
    
    # Add header
    summary_lines.append("EXPERIMENT SUMMARY METRICS")
    summary_lines.append("="*80)
    summary_lines.append("")
    
    # Process each successful result
    for result in results:
        if result['success']:
            method = result['method']
            dataset = result['dataset']
            
            # Find the most recent metrics file for this method-dataset combination
            method_dir = os.path.join("orginTest_results", method)
            if os.path.exists(method_dir):
                # Find all directories matching the pattern
                pattern = f"{method}_{dataset}_*"
                matching_dirs = []
                for item in os.listdir(method_dir):
                    if item.startswith(f"{method}_{dataset}_"):
                        matching_dirs.append(item)
                
                if matching_dirs:
                    # Get the most recent one
                    latest_dir = sorted(matching_dirs)[-1]
                    metrics_file = os.path.join(method_dir, latest_dir, f"{method}_{dataset}_metrics_{latest_dir.split('_')[-1]}.txt")
                    
                    if os.path.exists(metrics_file):
                        summary_lines.append(f"Method: {method.upper()} | Dataset: {dataset.upper()} | Timestamp: {latest_dir.split('_')[-1]}")
                        summary_lines.append("-" * 80)
                        
                        with open(metrics_file, 'r') as f:
                            content = f.read().strip()
                            summary_lines.append(content)
                        
                        summary_lines.append("")
                        summary_lines.append("")
    
    # Save summary file
    summary_filename = f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    summary_path = os.path.join("orginTest_results", summary_filename)
    
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    
    print(f"Summary metrics saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch script for running RCA experiments")
    parser.add_argument("--methods", nargs="+", default=["baro", "cpg"], 
                       help="Methods to run (default: baro cpg)")
    parser.add_argument("--datasets", nargs="+", 
                       default=["online-boutique", "sock-shop-1", "train-ticket"],
                       help="Datasets to run (default: online-boutique sock-shop-1 train-ticket)")
    parser.add_argument("--test", action="store_true", 
                       help="Run in test mode with limited data")
    parser.add_argument("--single", action="store_true",
                       help="Run only one method-dataset combination")
    
    args = parser.parse_args()
    
    print("RCAEval Batch Experiment Runner")
    print("="*60)
    print(f"Methods: {args.methods}")
    print(f"Datasets: {args.datasets}")
    print(f"Test mode: {args.test}")
    print(f"Total combinations: {len(args.methods) * len(args.datasets)}")
    
    if args.single:
        print("Single run mode: will stop after first completion")
    
    # Track results
    results = []
    total_experiments = len(args.methods) * len(args.datasets)
    current_experiment = 0
    
    start_time = datetime.now()
    
    # Run experiments
    for method in args.methods:
        for dataset in args.datasets:
            current_experiment += 1
            print(f"\nProgress: {current_experiment}/{total_experiments}")
            
            success = run_experiment(method, dataset, args.test)
            results.append({
                "method": method,
                "dataset": dataset,
                "success": success,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            })
            
            if args.single:
                print("Single run mode: stopping after first experiment")
                break
        
        if args.single:
            break
    
    # Generate summary metrics file
    generate_summary_metrics(results)
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {duration}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    
    print(f"\nDetailed Results:")
    for result in results:
        status = "Successful" if result['success'] else "Failed"
        print(f"  {status} {result['method']} on {result['dataset']}")
    
    print(f"\nResults saved in: orginTest_results/")
    print("="*60)

if __name__ == "__main__":
    main()
