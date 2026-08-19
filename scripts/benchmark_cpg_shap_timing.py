#!/usr/bin/env python3
"""Benchmark CPG-Shapley per-case inference time on RE1/RE2 datasets.

Runs orginTest.py for each dataset, logs output, cleans output/results
between runs, and produces a LaTeX-ready timing summary table.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "output" / "results"
LOG_DIR = REPO_ROOT / "output" / "cpg_shap_timing_logs"
SUMMARY_JSON = REPO_ROOT / "output" / "cpg_shap_timing_summary.json"
SUMMARY_TXT = REPO_ROOT / "output" / "cpg_shap_timing_summary.txt"

DATASETS = [
    "re1-ob",
    "re1-ss",
    "re1-tt",
    "re2-ob",
    "re2-ss",
    "re2-tt",
]

SYSTEM_SUFFIX = {
    "ob": "OB (Online Boutique)",
    "ss": "SS (Sock Shop)",
    "tt": "TT (Train Ticket)",
}


def clean_results() -> None:
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_summary(stdout: str) -> dict:
    avg_match = re.search(r"Average speed:\s*([\d.]+)s per case", stdout)
    total_match = re.search(r"Total cases:\s*(\d+)", stdout)
    success_match = re.search(r"Successful cases:\s*(\d+)", stdout)
    return {
        "avg_seconds_per_case": float(avg_match.group(1)) if avg_match else None,
        "total_cases": int(total_match.group(1)) if total_match else None,
        "successful_cases": int(success_match.group(1)) if success_match else None,
    }


def run_dataset(dataset: str, python_exe: str) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{dataset}_{timestamp}.log"

    clean_results()

    cmd = [python_exe, "orginTest.py", "--method", "cpg_shap", "--dataset", dataset]
    started = datetime.now()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    finished = datetime.now()
    elapsed = (finished - started).total_seconds()

    log_content = (
        f"# command: {' '.join(cmd)}\n"
        f"# started: {started.isoformat()}\n"
        f"# finished: {finished.isoformat()}\n"
        f"# wall_clock_seconds: {elapsed:.2f}\n"
        f"# return_code: {proc.returncode}\n"
        f"\n===== STDOUT =====\n{proc.stdout}\n"
        f"\n===== STDERR =====\n{proc.stderr}\n"
    )
    log_path.write_text(log_content, encoding="utf-8")

    parsed = parse_summary(proc.stdout)
    result = {
        "dataset": dataset,
        "re": dataset.split("-")[0],
        "system": dataset.split("-")[1],
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "return_code": proc.returncode,
        "wall_clock_seconds": round(elapsed, 2),
        **parsed,
    }

    clean_results()
    return result


def format_latex_row(label: str, ob: float | None, ss: float | None, tt: float | None) -> str:
    def fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    return f"  {label} & {fmt(ob)} & {fmt(ss)} & {fmt(tt)} \\\\"


def build_summary_table(results: list[dict]) -> str:
    by_key = {(r["re"], r["system"]): r for r in results}
    lines = [
        "CPG-Shapley per-case inference time (seconds)",
        "=" * 60,
        "",
        "LaTeX table rows (columns: OB | SS | TT):",
        "",
    ]

    for re in ("re1", "re2"):
        ob = by_key.get((re, "ob"), {}).get("avg_seconds_per_case")
        ss = by_key.get((re, "ss"), {}).get("avg_seconds_per_case")
        tt = by_key.get((re, "tt"), {}).get("avg_seconds_per_case")
        label = f"CPG-Shapley ({re.upper()})"
        lines.append(format_latex_row(label, ob, ss, tt))
        lines.append("")

    lines.extend(
        [
            "Plain summary:",
            "",
            "| RE   | System | Avg s/case | Total cases | Wall clock (s) | Log |",
            "|------|--------|------------|-------------|----------------|-----|",
        ]
    )
    for r in results:
        avg = r.get("avg_seconds_per_case")
        avg_s = f"{avg:.2f}" if avg is not None else "N/A"
        lines.append(
            f"| {r['re'].upper()} | {r['system'].upper()} | {avg_s} | "
            f"{r.get('total_cases', 'N/A')} | {r.get('wall_clock_seconds', 'N/A')} | "
            f"{r['log_path']} |"
        )

    lines.extend(["", "Paper-ready rows:", ""])
    for re in ("re1", "re2"):
        ob = by_key.get((re, "ob"), {}).get("avg_seconds_per_case")
        ss = by_key.get((re, "ss"), {}).get("avg_seconds_per_case")
        tt = by_key.get((re, "tt"), {}).get("avg_seconds_per_case")

        def fmt(v: float | None) -> str:
            return f"{v:.2f}" if v is not None else "【待填】"

        lines.append(f"{re.upper()}:")
        lines.append(f"  - ob（秒）：{fmt(ob)}")
        lines.append(f"  - ss（秒）：{fmt(ss)}")
        lines.append(f"  - tt（秒）：{fmt(tt)}")
        lines.append(f"  CPG-Shapley  & {fmt(ob)} & {fmt(ss)} & {fmt(tt)} \\\\")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CPG-Shapley timing on RE1/RE2")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Datasets to benchmark (default: all RE1/RE2)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip datasets already present in summary JSON with return_code=0",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if args.resume and SUMMARY_JSON.exists():
        prior = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        for item in prior.get("results", []):
            if item.get("return_code") == 0 and item.get("avg_seconds_per_case") is not None:
                existing[item["dataset"]] = item

    results: list[dict] = []
    for dataset in args.datasets:
        if dataset in existing:
            print(f"[skip] {dataset} (already benchmarked)")
            results.append(existing[dataset])
            continue

        print(f"[run] {dataset} ...", flush=True)
        result = run_dataset(dataset, args.python)
        results.append(result)
        avg = result.get("avg_seconds_per_case")
        print(
            f"[done] {dataset}: "
            f"avg={avg:.2f}s/case, wall={result['wall_clock_seconds']}s, "
            f"code={result['return_code']}, log={result['log_path']}",
            flush=True,
        )
        if result["return_code"] != 0:
            print(f"[warn] {dataset} failed; see log for details", flush=True)

    # Keep stable ordering
    order = {d: i for i, d in enumerate(DATASETS)}
    results.sort(key=lambda r: order.get(r["dataset"], 999))

    summary = {
        "method": "cpg_shap",
        "generated_at": datetime.now().isoformat(),
        "results": results,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    table = build_summary_table(results)
    SUMMARY_TXT.write_text(table, encoding="utf-8")

    print()
    print(table)
    print()
    print(f"Summary JSON: {SUMMARY_JSON}")
    print(f"Summary text: {SUMMARY_TXT}")

    failed = [r for r in results if r["return_code"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
