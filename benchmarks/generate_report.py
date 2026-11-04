#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  ServerlessLLM Benchmark Report Generator                                   #
# ---------------------------------------------------------------------------- #

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def load_results(
    results_dir: str, model_name: str, num_replicas: int, benchmark_type: str
) -> Dict:
    """Load benchmark results from JSON files."""
    results = {}

    for format_type in ["sllm", "safetensors"]:
        filename = (
            f"{model_name}_{format_type}_{num_replicas}_{benchmark_type}.json"
        )
        filename = filename.replace("/", "_")
        filepath = os.path.join(results_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                results[format_type] = json.load(f)
        else:
            print(f"Warning: {filepath} not found")

    return results


def _percentile(values: List[float], pct: float) -> float:
    """The pct-th percentile (pct in 0..100) of a list of numbers.

    Plain version: sort the samples, then p99 is "slower than all but the
    worst 1% of loads". When the percentile falls between two samples we
    blend them (standard linear interpolation).
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # rank = fractional position in the sorted list; low/high bracket it
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def calculate_stats(data: List[Dict]) -> Dict:
    """Calculate statistics from benchmark data."""
    if not data:
        return {}

    loading_times = [item["loading_time"] for item in data]
    throughputs = [item.get("throughput", 0) for item in data]

    return {
        "count": len(loading_times),
        "avg_loading_time": sum(loading_times) / len(loading_times),
        "min_loading_time": min(loading_times),
        "max_loading_time": max(loading_times),
        "std_loading_time": (
            sum(
                (x - sum(loading_times) / len(loading_times)) ** 2
                for x in loading_times
            )
            / len(loading_times)
        )
        ** 0.5,
        # tail percentiles: what the slow loads look like, not just the average
        "p50_loading_time": _percentile(loading_times, 50),
        "p90_loading_time": _percentile(loading_times, 90),
        "p95_loading_time": _percentile(loading_times, 95),
        "p99_loading_time": _percentile(loading_times, 99),
        "p999_loading_time": _percentile(loading_times, 99.9),
        "avg_throughput": sum(throughputs) / len(throughputs)
        if throughputs
        else 0,
    }


def generate_text_report(
    model_name: str, num_replicas: int, benchmark_type: str, results: Dict
) -> str:
    """Generate text summary report."""
    report = []

    report.append("=" * 60)
    report.append("ServerlessLLM Benchmark Results")
    report.append("=" * 60)
    report.append(f"Model: {model_name}")
    report.append(f"Replicas: {num_replicas}")
    report.append(f"Benchmark Type: {benchmark_type}")
    report.append("")

    if not results:
        report.append("No results found!")
        return "\n".join(report)

    # Calculate statistics
    stats = {}
    for format_type, data in results.items():
        stats[format_type] = calculate_stats(data)

    # Results table
    report.append("Loading Time Comparison:")
    report.append("-" * 60)
    report.append(
        f"{'Format':<15} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12} {'Std Dev':<10}"
    )
    report.append("-" * 60)

    sllm_stats = stats.get("sllm", {})
    safetensors_stats = stats.get("safetensors", {})

    if sllm_stats:
        report.append(
            f"{'SLLM':<15} "
            f"{sllm_stats['avg_loading_time']:<12.3f} "
            f"{sllm_stats['min_loading_time']:<12.3f} "
            f"{sllm_stats['max_loading_time']:<12.3f} "
            f"{sllm_stats['std_loading_time']:<10.3f}"
        )

    if safetensors_stats:
        report.append(
            f"{'SafeTensors':<15} "
            f"{safetensors_stats['avg_loading_time']:<12.3f} "
            f"{safetensors_stats['min_loading_time']:<12.3f} "
            f"{safetensors_stats['max_loading_time']:<12.3f} "
            f"{safetensors_stats['std_loading_time']:<10.3f}"
        )

    report.append("-" * 60)

    # Loading-time tail: the slow-case numbers (p50/p99/p99.9) the average hides.
    report.append("")
    report.append("Loading-time tail (seconds):")
    report.append("-" * 60)
    report.append(
        f"{'Format':<15} {'p50':<10} {'p99':<10} {'p99.9':<10} {'p99/avg':<10}"
    )
    report.append("-" * 60)
    for label, s in (("SLLM", sllm_stats), ("SafeTensors", safetensors_stats)):
        if s:
            # p99/avg > 1 means the worst loads are slower than the headline average
            ratio = (
                s["p99_loading_time"] / s["avg_loading_time"]
                if s.get("avg_loading_time")
                else float("nan")
            )
            report.append(
                f"{label:<15} "
                f"{s['p50_loading_time']:<10.3f} "
                f"{s['p99_loading_time']:<10.3f} "
                f"{s['p999_loading_time']:<10.3f} "
                f"{ratio:<10.2f}"
            )
    report.append("-" * 60)
    report.append("")

    # Speedup calculation
    if sllm_stats and safetensors_stats:
        speedup = (
            safetensors_stats["avg_loading_time"]
            / sllm_stats["avg_loading_time"]
        )
        report.append(f"SLLM Speedup: {speedup:.2f}x faster than SafeTensors")
        report.append("")

    # Inference throughput
    if (
        sllm_stats.get("avg_throughput", 0) > 0
        or safetensors_stats.get("avg_throughput", 0) > 0
    ):
        report.append("Inference Performance:")
        report.append("-" * 60)
        if sllm_stats.get("avg_throughput", 0) > 0:
            report.append(
                f"SLLM Avg Throughput: {sllm_stats['avg_throughput']:.2f} tokens/s"
            )
        if safetensors_stats.get("avg_throughput", 0) > 0:
            report.append(
                f"SafeTensors Avg Throughput: {safetensors_stats['avg_throughput']:.2f} tokens/s"
            )
        report.append("")

    report.append("=" * 60)

    return "\n".join(report)


def generate_json_summary(
    model_name: str, num_replicas: int, benchmark_type: str, results: Dict
) -> Dict:
    """Generate JSON summary."""
    stats = {}
    for format_type, data in results.items():
        stats[format_type] = calculate_stats(data)

    summary = {
        "model_name": model_name,
        "num_replicas": num_replicas,
        "benchmark_type": benchmark_type,
        "statistics": stats,
    }

    if stats.get("sllm") and stats.get("safetensors"):
        summary["speedup"] = (
            stats["safetensors"]["avg_loading_time"]
            / stats["sllm"]["avg_loading_time"]
        )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--num-replicas", type=int, required=True)
    parser.add_argument("--benchmark-type", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument(
        "--output-file", type=str, default="./results/summary.txt"
    )
    parser.add_argument("--generate-plots", type=str, default="false")

    args = parser.parse_args()

    # Load results
    results = load_results(
        args.results_dir,
        args.model_name,
        args.num_replicas,
        args.benchmark_type,
    )

    if not results:
        print("No results found to generate report")
        return

    # Generate text report
    text_report = generate_text_report(
        args.model_name, args.num_replicas, args.benchmark_type, results
    )

    # Save to file
    with open(args.output_file, "w") as f:
        f.write(text_report)

    # Print to console
    print(text_report)

    # Generate JSON summary
    json_summary = generate_json_summary(
        args.model_name, args.num_replicas, args.benchmark_type, results
    )
    json_output = args.output_file.replace(".txt", ".json")
    with open(json_output, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"\nReports saved:")
    print(f"  Text: {args.output_file}")
    print(f"  JSON: {json_output}")

    # Generate plots if requested
    if args.generate_plots.lower() == "true":
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Simple comparison plot
            formats = list(results.keys())
            avg_times = [
                calculate_stats(results[f])["avg_loading_time"] for f in formats
            ]

            plt.figure(figsize=(8, 6))
            plt.bar(formats, avg_times)
            plt.ylabel("Average Loading Time (s)")
            plt.title(f"Loading Time Comparison - {args.model_name}")

            plot_file = args.output_file.replace(".txt", ".png")
            plt.savefig(plot_file)
            print(f"  Plot: {plot_file}")
        except ImportError:
            print("Warning: matplotlib/seaborn not available, skipping plots")


if __name__ == "__main__":
    main()
