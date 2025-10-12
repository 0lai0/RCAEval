#!/usr/bin/env python3
"""
Test version of run_main_nocache.py to verify functionality
"""

import subprocess
import os
import shutil
from datetime import datetime

def test_single_experiment():
    """Test a single experiment to verify the parsing and saving works"""
    
    method = "baro"
    dataset = "online-boutique"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Testing {method} on {dataset}")
    
    # Clean output directory
    results_dir = "output/results"
    if os.path.exists(results_dir):
        print(f"Cleaning {results_dir}...")
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiment
    cmd = ["python", "orginTest.py", "--method", method, "--dataset", dataset, "--test"]
    print(f"Command: {' '.join(cmd)}")
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = proc.stdout
        
        print("="*60)
        print("EXPERIMENT OUTPUT:")
        print("="*60)
        print(out)
        print("="*60)
        
        # Parse metrics
        results = []
        parsing_metrics = False
        
        for line in out.splitlines():
            if "Performance Metrics for" in line and method.upper() in line:
                parsing_metrics = True
                continue
            
            if parsing_metrics and "=" * 80 in line:
                break
            
            if parsing_metrics and "\t" in line and not line.startswith("Metric Type"):
                parts = line.split("\t")
                if len(parts) >= 5:
                    fault_type = parts[0].strip()
                    precision_1 = float(parts[1].strip())
                    precision_3 = float(parts[2].strip())
                    precision_5 = float(parts[3].strip())
                    avg_5 = float(parts[4].strip())
                    
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
        
        print(f"\nParsed {len(results)} metrics:")
        for result in results:
            print(f"  {result['Fault Type']}: Avg@5={result['Avg@5']:.4f}")
        
        # Save results
        save_test_results(results, method, dataset, timestamp)
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")

def save_test_results(results, method, dataset, timestamp):
    """Save test results in the requested format"""
    
    output_dir = "orginTest_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create method-specific subdirectory
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # Create filename: dataset_method_timestamp
    filename = f"{dataset}_{method}_{timestamp}.txt"
    filepath = os.path.join(method_dir, filename)
    
    # Format the table as requested
    lines = []
    lines.append("Metric Type\tprecision@1\tprecision@3\tprecision@5\tavg@5")
    
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
        result = next((r for r in results if r['Fault Type'] == key), None)
        if result:
            line = f"{display_name}\t{result['Precision@1']:.4f}\t{result['Precision@3']:.4f}\t{result['Precision@5']:.4f}\t{result['Avg@5']:.4f}"
        else:
            line = f"{display_name}\t0.0000\t0.0000\t0.0000\t0.0000"
        lines.append(line)
    
    # Save to file
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nResults saved to: {filepath}")
    
    # Display the saved content
    print("\nSaved content:")
    print("="*60)
    with open(filepath, 'r') as f:
        print(f.read())
    print("="*60)

if __name__ == "__main__":
    test_single_experiment()
