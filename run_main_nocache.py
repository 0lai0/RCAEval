#!/usr/bin/env python3
import subprocess
import re
import pandas as pd
from io import StringIO
from datetime import datetime
import os

# Batch-run main.py for multiple methods and datasets and collect Avg@5 metrics

def run_and_parse_main_experiments():
    experiments = [
        ('baro', 're1-ob'), ('cpg', 're1-ob'),
        ('baro', 're1-ss'), ('cpg', 're1-ss'),
        ('baro', 're1-tt'), ('cpg', 're1-tt'),
        ('baro', 're2-ob'), ('cpg', 're2-ob'),
        ('baro', 're2-ss'), ('cpg', 're2-ss'),
        ('baro', 're2-tt'), ('cpg', 're2-tt'),
        ('baro', 're3-ob'), ('cpg', 're3-ob'),
        ('baro', 're3-ss'), ('cpg', 're3-ss'),
        ('baro', 're3-tt'), ('cpg', 're3-tt'),
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "run_experiments_main"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    print(f"Running {len(experiments)} experiments via main_nocache.py...")
    for method, dataset in experiments:
        print(f"\n--- {method} on {dataset} ---")
        cmd = ["python", "main_nocache.py", "--method", method, "--dataset", dataset]
        print(f"Command: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            out = proc.stdout
            # Extract lines: Avg@5-<FAULT>: <value>
            for line in out.splitlines():
                m = re.match(r"Avg@5-([A-Z]+):\s*([0-9.]+)", line)
                if m:
                    fault, avg5 = m.groups()
                    results.append({
                        'Method': method,
                        'Dataset': dataset,
                        'Fault Type': fault.lower(),
                        'Avg@5': float(avg5)
                    })
        except subprocess.CalledProcessError as e:
            print(f"Error running {method} on {dataset}: {e}")

    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)
    # Save long format
    csv_long = os.path.join(output_dir, f"summary_long_{timestamp}.csv")
    df.to_csv(csv_long, index=False, encoding='utf-8-sig')
    print(f"Saved long-format results to {csv_long}")

    # Pivot to wide format
    comp = df.pivot_table(index=['Dataset'], columns='Method', values='Avg@5')
    comp.reset_index(inplace=True)
    csv_wide = os.path.join(output_dir, f"summary_wide_{timestamp}.csv")
    comp.to_csv(csv_wide, index=False, encoding='utf-8-sig')
    print(f"Saved wide-format results to {csv_wide}")

    # Define colors for terminal output
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

    # Print detailed comparison report like run_experiments.py
    print("\n\n" + "="*70)
    print("         方法橫向比較報告 (Comparison Report)")
    print("="*70)
    
    # Group by dataset and fault type for detailed display
    detailed_df = df.pivot_table(
        index=['Dataset', 'Fault Type'],
        columns='Method',
        values='Avg@5'
    ).reset_index()
    
    # Create detailed comparison with all metrics
    print(f"{'Dataset':<10} {'Fault Type':<12} {'Avg@5_baro':>12} {'Avg@5_cpg':>12}")
    print("-" * 70)
    
    current_dataset = None
    for _, row in detailed_df.iterrows():
        if row['Dataset'] != current_dataset:
            if current_dataset is not None:
                print(Colors.YELLOW + "-" * 70 + Colors.RESET)
            current_dataset = row['Dataset']
        
        val_baro = row.get('baro', 0)
        val_cpg = row.get('cpg', 0)
        baro_str = f"{val_baro:>12.2f}"
        cpg_str = f"{val_cpg:>12.2f}"
        
        if val_baro > val_cpg:
            baro_str = Colors.GREEN + baro_str + Colors.RESET
        elif val_cpg > val_baro:
            cpg_str = Colors.GREEN + cpg_str + Colors.RESET
        
        print(f"{row['Dataset']:<10} {row['Fault Type']:<12} {baro_str} {cpg_str}")

if __name__ == '__main__':
    run_and_parse_main_experiments()
