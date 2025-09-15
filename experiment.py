#!/usr/bin/env python3
"""
RCAEval Experiment Script

A research experiment script for executing and evaluating various RCA algorithms
on standardized datasets. This script provides comprehensive evaluation metrics
and supports multiple microservice system datasets.

Key Features:
- Supports 20+ RCA algorithms (baro, cpg, circa, cloudranger, etc.)
- Multiple standardized datasets (online-boutique, sock-shop, train-ticket, RE1/2/3)
- Automated data preprocessing and fault injection handling
- Dual-level evaluation (service-level and metric-level)
- NO CACHING - Fresh execution every time
"""

import argparse
import glob
import json
import os
import shutil
import warnings
from datetime import datetime
from os.path import basename, dirname, join

# Turn off all warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.classes.graph import Node
from RCAEval.utility import (
    dump_json,
    is_py38,
    is_py310,
    load_json,
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
    download_re1_dataset,
    download_re2_dataset,
    download_re3_dataset,
)


def import_rca_algorithms():
    """Import RCA algorithms based on Python version"""
    if is_py310():
        from RCAEval.e2e import (
            baro,
            causalrca,
            circa,
            cloudranger,
            cmlp_pagerank,
            dummy,
            e_diagnosis,
            easyrca,
            fci_pagerank,
            fci_randomwalk,
            ges_pagerank,
            granger_pagerank,
            granger_randomwalk,
            lingam_pagerank,
            lingam_randomwalk,
            micro_diag,
            microcause,
            microrank,
            mscred,
            nsigma,
            ntlr_pagerank,
            ntlr_randomwalk,
            pc_pagerank,
            pc_randomwalk,
            run,
            tracerca,
            cpg_adaptive,
            cpg,
        )
        return {
            'baro': baro, 'causalrca': causalrca, 'circa': circa, 'cloudranger': cloudranger,
            'cmlp_pagerank': cmlp_pagerank, 'dummy': dummy, 'e_diagnosis': e_diagnosis,
            'easyrca': easyrca, 'fci_pagerank': fci_pagerank, 'fci_randomwalk': fci_randomwalk,
            'ges_pagerank': ges_pagerank, 'granger_pagerank': granger_pagerank,
            'granger_randomwalk': granger_randomwalk, 'lingam_pagerank': lingam_pagerank,
            'lingam_randomwalk': lingam_randomwalk, 'micro_diag': micro_diag,
            'microcause': microcause, 'microrank': microrank, 'mscred': mscred,
            'nsigma': nsigma, 'ntlr_pagerank': ntlr_pagerank, 'ntlr_randomwalk': ntlr_randomwalk,
            'pc_pagerank': pc_pagerank, 'pc_randomwalk': pc_randomwalk, 'run': run,
            'tracerca': tracerca, 'cpg_adaptive': cpg_adaptive, 'cpg': cpg
        }
    elif is_py38():
        from RCAEval.e2e import dummy, e_diagnosis, ht, rcd, mmrcd
        return {
            'dummy': dummy, 'e_diagnosis': e_diagnosis, 'ht': ht, 'rcd': rcd, 'mmrcd': mmrcd
        }
    else:
        raise RuntimeError("Please use Python 3.8 or 3.10")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RCAEval Experiment Script - Execute and evaluate RCA algorithms on standardized datasets"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        required=True,
        help="Choose an RCA algorithm method (e.g., baro, cpg, circa, nsigma)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Choose a dataset for evaluation",
        choices=[
            "online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket",
            "re1-ob", "re1-ss", "re1-tt", "re2-ob", "re2-ss", "re2-tt", 
            "re3-ob", "re3-ss", "re3-tt"
        ]
    )
    parser.add_argument(
        "--length", 
        type=int, 
        default=20, 
        help="Time series length in minutes for analysis (RQ4 research)"
    )
    parser.add_argument(
        "--tdelta", 
        type=int, 
        default=0, 
        help="Simulate delay in anomaly detection (seconds)"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Perform smoke test (run only 2 cases per dataset)"
    )
    
    return parser.parse_args()


def download_dataset(dataset_name):
    """Download the specified dataset"""
    print(f"Ensuring dataset '{dataset_name}' is available...")
    
    if "online-boutique" in dataset_name or "re1-ob" in dataset_name:
        download_online_boutique_dataset()
    elif "sock-shop-1" in dataset_name:
        download_sock_shop_1_dataset()
    elif "sock-shop-2" in dataset_name or "re1-ss" in dataset_name:
        download_sock_shop_2_dataset()
    elif "train-ticket" in dataset_name or "re1-tt" in dataset_name:
        download_train_ticket_dataset()
    elif "re2" in dataset_name:
        download_re2_dataset()
    elif "re3" in dataset_name:
        download_re3_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_path(dataset_name):
    """Get the file system path for the dataset"""
    dataset_map = {
        "online-boutique": "data/online-boutique",
        "sock-shop-1": "data/sock-shop-1", 
        "sock-shop-2": "data/sock-shop-2",
        "train-ticket": "data/train-ticket",
        "re1-ob": "data/online-boutique",
        "re1-ss": "data/sock-shop-2",
        "re1-tt": "data/train-ticket",
        "re2-ob": "data/RE2/RE2-OB",
        "re2-ss": "data/RE2/RE2-SS",
        "re2-tt": "data/RE2/RE2-TT",
        "re3-ob": "data/RE3/RE3-OB",
        "re3-ss": "data/RE3/RE3-SS",
        "re3-tt": "data/RE3/RE3-TT"
    }
    return dataset_map[dataset_name]


def find_data_files(dataset_path, test_mode=False):
    """Find all data files in the dataset"""
    # First try to find data.csv files
    data_paths = list(glob.glob(os.path.join(dataset_path, "**/data.csv"), recursive=True))
    
    # If no data.csv files found, look for simple_metrics.csv
    if not data_paths:
        data_paths = list(glob.glob(os.path.join(dataset_path, "**/simple_metrics.csv"), recursive=True))
    
    if not data_paths:
        raise ValueError(f"No data files found in {dataset_path}")
    
    # For smoke testing, only use first 2 files
    if test_mode:
        data_paths = data_paths[:2]
        print(f"Smoke test mode: Using {len(data_paths)} data files")
    else:
        print(f"Found {len(data_paths)} data files for analysis")
    
    return sorted(data_paths)


def determine_sli(data_path, service_name, data_columns):
    """Determine Service Level Indicator (SLI) based on dataset and available columns"""
    if "my-sock-shop" in data_path or "fse-ss" in data_path:
        sli = "front-end_cpu"
        if f"{service_name}_latency" in data_columns:
            sli = f"{service_name}_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service_name}_lat_90" in data_columns:
            sli = f"{service_name}_lat_90"
    elif "train-ticket" in data_path or "fse-tt" in data_path or "RE2-TT" in data_path or "RE3-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
        if f"{service_name}_latency" in data_columns:
            sli = f"{service_name}_latency"
    elif "online-boutique" in data_path or "fse-ob" in data_path or "RE2-OB" in data_path or "RE2-SS" in data_path or "RE3-OB" in data_path or "RE3-SS" in data_path:
        sli = "frontend_latency"
        if f"{service_name}_latency" in data_columns:
            sli = f"{service_name}_latency"
        elif "frontend_1" in data_columns:
            sli = "frontend_1"
    else:
        raise ValueError(f"SLI determination not implemented for dataset: {data_path}")
    
    return sli


def preprocess_data(data, data_path):
    """Preprocess the time series data"""
    # Remove latency-50 columns, keep only latency-90
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    
    # Special handling for mm-tt dataset
    if "mm-tt" in data_path:
        time_col = data["time"]
        data = data.loc[:, data.columns.str.startswith("ts-")]
        data["time"] = time_col
    
    # Handle infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values
    data = data.fillna(method="ffill")
    data = data.fillna(0)
    
    # Rename latency-90 columns to latency
    data = data.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in data.columns
            if c.endswith("_latency-90")
        }
    )
    
    return data


def process_single_case(data_path, algorithm_func, args):
    """Process a single test case"""
    # NO CACHING - Always fresh execution
    run_args = argparse.Namespace()
    run_args.root_path = os.getcwd()
    run_args.data_path = data_path
    
    # Convert length from minutes to seconds (divided by 2 for metrics sampling rate)
    data_length = args.length * 60 // 2
    
    data_dir = dirname(data_path)
    
    # Parse service and metric from directory structure
    service, metric = basename(dirname(dirname(data_path))).split("_", 1)
    case = basename(dirname(data_path))
    
    print(f"Processing: {service}_{metric}_{case}")
    
    # Load and preprocess data - NO CACHING
    data = pd.read_csv(data_path)
    data = preprocess_data(data, data_path)
    
    # Read fault injection time
    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip()) + args.tdelta
    
    # Extract normal and anomalous periods
    normal_df = data[data["time"] < inject_time].tail(data_length)
    anomal_df = data[data["time"] >= inject_time].head(data_length)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)
    
    # Determine SLI
    sli = determine_sli(data_path, service, data.columns)
    
    # Execute RCA algorithm - NO CACHING
    try:
        start_time = datetime.now()
        
        result = algorithm_func(
            data,
            inject_time,
            dataset=args.dataset,
            anomalies=None,
            dk_select_useful=False,
            sli=sli,
            verbose=False,
            n_iter=len(data.columns) - 1,  # Exclude time column
            args=run_args,
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        root_causes = result.get("ranks", [])
        
        return {
            "service": service,
            "metric": metric, 
            "case": case,
            "root_causes": root_causes,
            "execution_time": execution_time,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        print(f"ERROR processing {data_path}: {str(e)}")
        return {
            "service": service,
            "metric": metric,
            "case": case,
            "root_causes": [],
            "execution_time": 0,
            "success": False,
            "error": str(e)
        }


def map_fault_code_to_type(fault_code):
    """Map fault codes (f1, f2, etc.) to actual fault types"""
    # Common fault code mappings for RE datasets
    fault_mapping = {
        # RE3 and RE2 dataset fault codes
        "f1": "cpu",
        "f2": "mem", 
        "f3": "delay",
        "f4": "disk",
        # Additional mappings for RE3
        "f3_1": "delay",  # Special case for RE3
        # Direct fault type names (for other datasets)
        "cpu": "cpu",
        "mem": "mem",
        "memory": "mem",
        "delay": "delay",
        "latency": "delay",
        "loss": "loss",
        "disk": "disk",
        "diskio": "disk",
        "socket": "socket",
        "io": "disk",
    }
    
    # Handle both lowercase and uppercase
    fault_lower = fault_code.lower()
    mapped_type = fault_mapping.get(fault_lower, fault_lower)
    
    print(f"DEBUG: Mapped fault code '{fault_code}' to fault type '{mapped_type}'")
    return mapped_type


def evaluate_results(results):
    """Evaluate algorithm performance using standardized metrics"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Initialize evaluators for different fault types
    evaluators = {
        "all": {"service": Evaluator(), "metric": Evaluator()},
        "cpu": {"service": Evaluator(), "metric": Evaluator()},
        "mem": {"service": Evaluator(), "metric": Evaluator()},
        "delay": {"service": Evaluator(), "metric": Evaluator()},
        "loss": {"service": Evaluator(), "metric": Evaluator()},
        "disk": {"service": Evaluator(), "metric": Evaluator()},
        "socket": {"service": Evaluator(), "metric": Evaluator()},
    }
    
    successful_cases = [r for r in results if r["success"]]
    failed_cases = [r for r in results if not r["success"]]
    
    print(f"Successful cases: {len(successful_cases)}")
    print(f"Failed cases: {len(failed_cases)}")
    
    if failed_cases:
        print("\nFailed cases:")
        for case in failed_cases:
            print(f"  - {case['service']}_{case['metric']}_{case['case']}: {case['error']}")
    
    # Debug: Print some sample cases to understand the structure
    if successful_cases:
        print(f"\nSample cases structure:")
        for i, result in enumerate(successful_cases[:3]):
            print(f"  Case {i+1}: service='{result['service']}', metric='{result['metric']}', case='{result['case']}'")
    
    # Process successful cases
    for result in successful_cases:
        service = result["service"]
        fault_code = result["metric"]  # This is the fault code (f1, f2, etc.)
        root_causes = result["root_causes"]
        
        if not root_causes:
            continue
        
        # Map fault code to actual fault type
        fault_type = map_fault_code_to_type(fault_code)
            
        # Service-level evaluation (extract service names)
        s_ranks = []
        for rc in root_causes:
            parts = rc.split("_")
            service_name = parts[0].replace("-db", "")
            s_ranks.append(Node(service_name, "unknown"))
        
        # Remove duplicates while preserving order
        seen = set()
        s_ranks_dedup = []
        for node in s_ranks:
            if node not in seen:
                s_ranks_dedup.append(node)
                seen.add(node)
        
        # Metric-level evaluation (full metric names)
        f_ranks = []
        for rc in root_causes:
            parts = rc.split("_")
            if len(parts) >= 2:
                service_name = parts[0]
                metric_name = parts[1]
            else:
                service_name = parts[0]
                metric_name = "unknown"
            f_ranks.append(Node(service_name, metric_name))
        
        # Add to evaluators
        service_answer = Node(service, "unknown")
        
        # Map fault types for metric-level evaluation
        metric_answer_map = {
            "cpu": Node(service, "cpu"),
            "mem": Node(service, "mem"),
            "delay": Node(service, "latency"),
            "loss": Node(service, "latency"),
            "disk": Node(service, "diskio"),
            "socket": Node(service, "socket"),
        }
        metric_answer = metric_answer_map.get(fault_type, Node(service, fault_type))
        
        # Add to overall evaluator
        evaluators["all"]["service"].add_case(ranks=s_ranks_dedup, answer=service_answer)
        evaluators["all"]["metric"].add_case(ranks=f_ranks, answer=metric_answer)
        
        # Add to fault-specific evaluator
        if fault_type in evaluators:
            evaluators[fault_type]["service"].add_case(ranks=s_ranks_dedup, answer=service_answer)
            evaluators[fault_type]["metric"].add_case(ranks=f_ranks, answer=metric_answer)
    
    # Print evaluation results
    print(f"\n{'Fault Type':<12} {'Top@1':<8} {'Top@3':<8} {'Top@5':<8} {'Avg@5':<8}")
    print("-" * 50)
    
    for fault_type, evaluator_pair in evaluators.items():
        if fault_type == "all":
            continue
            
        s_eval = evaluator_pair["service"]
        avg5 = s_eval.average(5)
        
        if avg5 is not None:
            fault_name = "DISK" if fault_type == "disk" else fault_type.upper()
            print(f"{fault_name:<12} {s_eval.accuracy(1):<8.2f} {s_eval.accuracy(3):<8.2f} {s_eval.accuracy(5):<8.2f} {avg5:<8.2f}")
    
    # Calculate and display average execution time
    total_time = sum(r["execution_time"] for r in successful_cases)
    avg_speed = total_time / len(successful_cases) if successful_cases else 0
    print(f"\nAverage execution time per case: {avg_speed:.2f} seconds")
    
    return evaluators


def save_experiment_output(results, args, output_dir="experiment_output"):
    """Save experiment results to JSON file - NO CACHING"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = join(output_dir, f"{args.method}_{args.dataset}_{timestamp}.json")
    
    # Prepare output data
    output_data = {
        "experiment_info": {
            "method": args.method,
            "dataset": args.dataset,
            "length": args.length,
            "tdelta": args.tdelta,
            "test_mode": args.test,
            "timestamp": timestamp,
            "total_cases": len(results),
            "successful_cases": len([r for r in results if r["success"]]),
            "failed_cases": len([r for r in results if not r["success"]])
        },
        "results": results
    }
    
    # Save to file
    dump_json(filename=output_file, data=output_data)
    print(f"\nExperiment results saved to: {output_file}")
    
    return output_file


def main():
    """Main experiment execution function"""
    print("RCAEval Experiment Script")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Import algorithms
    algorithms = import_rca_algorithms()
    
    # Validate method
    if args.method not in algorithms:
        raise ValueError(f"Method '{args.method}' not available. Available methods: {list(algorithms.keys())}")
    
    algorithm_func = algorithms[args.method]
    
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Time series length: {args.length} minutes")
    print(f"Anomaly detection delay: {args.tdelta} seconds")
    print(f"Test mode: {args.test}")
    
    # Download dataset
    download_dataset(args.dataset)
    
    # Get dataset path and find data files
    dataset_path = get_dataset_path(args.dataset)
    data_files = find_data_files(dataset_path, args.test)
    
    # Process all cases - NO CACHING
    print(f"\nProcessing {len(data_files)} cases...")
    results = []
    
    start_time = datetime.now()
    
    for data_path in tqdm(data_files, desc="Processing cases"):
        result = process_single_case(data_path, algorithm_func, args)
        results.append(result)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per case: {total_time/len(results):.2f} seconds")
    
    # Evaluate results
    evaluators = evaluate_results(results)
    
    # Save results - NO CACHING
    output_file = save_experiment_output(results, args)
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
