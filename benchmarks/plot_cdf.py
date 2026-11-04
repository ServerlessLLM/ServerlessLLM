#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  ServerlessLLM Loading-Latency CDF Plotter                                   #
#  Plots the full distribution of per-trial loading times (SLLM vs SafeTensors) #
#  so the tail is visible, not just the mean.                                   #
# ---------------------------------------------------------------------------- #

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def load_times(results_dir, model, fmt, n, test):
    fn = f"{model}_{fmt}_{n}_{test}.json".replace("/", "_")
    fp = os.path.join(results_dir, fn)
    if not os.path.exists(fp):
        print(f"Warning: {fp} not found")
        return []
    with open(fp) as f:
        return [d["loading_time"] for d in json.load(f) if "loading_time" in d]


def add_cdf(ax, times, label):
    # CDF = "what fraction of loads finished in <= x seconds": sort the times,
    # then the i-th point sits at height (i+1)/n. A long flat top-right = a tail.
    ordered = sorted(times)
    if not ordered:
        return
    fraction = [(i + 1) / len(ordered) for i in range(len(ordered))]
    ax.plot(ordered, fraction, marker=".", label=f"{label} (n={len(ordered)})")


def main():
    p = argparse.ArgumentParser(
        description="Plot loading-time CDF (SLLM vs SafeTensors)"
    )
    p.add_argument("--model-name", required=True)
    p.add_argument("--num-replicas", type=int, required=True)
    p.add_argument("--benchmark-type", default="random")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--output-file", default="./results/loading_cdf.png")
    a = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 6))
    for fmt in ("sllm", "safetensors"):
        add_cdf(
            ax,
            load_times(
                a.results_dir,
                a.model_name,
                fmt,
                a.num_replicas,
                a.benchmark_type,
            ),
            fmt,
        )
    ax.set_xlabel("Loading time (s)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Loading-time CDF — {a.model_name} ({a.benchmark_type})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(a.output_file, dpi=200, bbox_inches="tight")
    print(f"Saved {a.output_file}")


if __name__ == "__main__":
    main()
