import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pattern = "{model_name}_{model_format}_{num_repeats}_{test_name}.json"


def get_args():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model names to show in the plot.",
    )
    parser.add_argument(
        "--test-name",
        type=str,
        required=True,
        choices=["random", "cached"],
        help="Name of the test.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        required=True,
        help="Number of repeats for the benchmark.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to load results from.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="loading_latency.png",
        help="Output filename for the plot.",
    )
    return parser.parse_args()


def load_results(models, model_format, num_repeats, test_name, results_dir):
    """Load results from files and check for the expected number of repeats."""
    all_results = {}
    for model in models:
        model = model.replace("/", "_")
        filename = pattern.format(
            model_name=model,
            model_format=model_format,
            num_repeats=num_repeats,
            test_name=test_name,
        )
        filename = os.path.join(results_dir, filename)
        with open(filename) as f:
            results = json.load(f)
            if len(results) != num_repeats:
                print(
                    f"Error: Expected {num_repeats} repeats, but found {len(results)} in {filename}."
                )
                exit(1)
        all_results[model] = results
    return all_results


def print_statistics(sllm_results, safetensors_results):
    """Compute and print statistics for each model."""
    print("\n" + "=" * 80)
    print("ServerlessLLM Benchmark Statistics")
    print("=" * 80)

    for model_name in sllm_results.keys():
        # Extract loading times
        sllm_times = [
            result["loading_time"] for result in sllm_results[model_name]
        ]
        safetensors_times = [
            result["loading_time"] for result in safetensors_results[model_name]
        ]

        # Compute statistics
        sllm_avg = np.mean(sllm_times)
        sllm_min = np.min(sllm_times)
        sllm_max = np.max(sllm_times)
        sllm_std = np.std(sllm_times)

        safetensors_avg = np.mean(safetensors_times)
        safetensors_min = np.min(safetensors_times)
        safetensors_max = np.max(safetensors_times)
        safetensors_std = np.std(safetensors_times)

        # Compute speedup
        if sllm_avg > 0:
            speedup = safetensors_avg / sllm_avg
        else:
            print(f"Warning: SLLM average loading time is zero for model {model_name}. Speedup is set to infinity.")
            speedup = float('inf')

        # Display model name
        display_name = model_name.replace("_", "/")
        print(f"\nModel: {display_name}")
        print("-" * 80)
        print(
            f"{'Format':<15} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12} {'Std Dev':<12}"
        )
        print("-" * 80)
        print(
            f"{'SLLM':<15} {sllm_avg:<12.3f} {sllm_min:<12.3f} {sllm_max:<12.3f} {sllm_std:<12.3f}"
        )
        print(
            f"{'SafeTensors':<15} {safetensors_avg:<12.3f} {safetensors_min:<12.3f} {safetensors_max:<12.3f} {safetensors_std:<12.3f}"
        )
        print("-" * 80)
        print(f"SLLM Speedup: {speedup:.2f}x faster than SafeTensors")

    print("=" * 80 + "\n")


def create_dataframe(sllm_results, safetensors_results):
    """Convert results list to pandas DataFrame."""
    sllm_loading_latency = []
    safetensors_loading_latency = []
    sllm_model_list = []
    safetensors_model_list = []
    for model_name, results in sllm_results.items():
        for result in results:
            sllm_loading_latency.append(result["loading_time"])
            sllm_model_list.append(model_name)
        for result in safetensors_results[model_name]:
            safetensors_loading_latency.append(result["loading_time"])
            safetensors_model_list.append(model_name)
    sllm_model_list = [model.split("_")[1] for model in sllm_model_list]
    safetensors_model_list = [
        model.split("_")[1] for model in safetensors_model_list
    ]
    df = pd.DataFrame(
        {
            "Model": sllm_model_list + safetensors_model_list,
            "System": ["Sllm"] * len(sllm_model_list)
            + ["SafeTensors"] * len(safetensors_model_list),
            "Loading Time": list(sllm_loading_latency)
            + list(safetensors_loading_latency),
        }
    )
    return df


def plot_results(df, sllm_results, safetensors_results, output_filename):
    """Plot the results using horizontal bar chart with speedup annotations."""
    plt.style.use("default")

    # Get unique models
    models = list(sllm_results.keys())
    n_models = len(models)

    # Calculate figure height based on number of models
    fig_height = max(4, 2 + n_models * 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_height), facecolor="white")

    # Get colors from Seaborn Paired palette (reversed: SLLM gets darker blue)
    paired_colors = sns.color_palette("Paired", n_colors=2)
    colors = [
        paired_colors[1],
        paired_colors[0],
    ]  # Swap: SLLM=dark, SafeTensors=light

    # Prepare data for horizontal bars
    bar_height = 0.35
    group_spacing = 1.2  # Reduced spacing between model groups

    for i, model_name in enumerate(models):
        sllm_times = [r["loading_time"] for r in sllm_results[model_name]]
        safe_times = [
            r["loading_time"] for r in safetensors_results[model_name]
        ]
        sllm_mean = np.mean(sllm_times)
        safe_mean = np.mean(safe_times)
        speedup = safe_mean / sllm_mean

        y_sllm = i * group_spacing
        y_safe = i * group_spacing + 0.45

        # Draw bars
        ax.barh(
            y_sllm,
            sllm_mean,
            height=bar_height,
            color=colors[0],
            edgecolor="white",
            linewidth=1.5,
            label="SLLM" if i == 0 else "",
        )
        ax.barh(
            y_safe,
            safe_mean,
            height=bar_height,
            color=colors[1],
            edgecolor="white",
            linewidth=1.5,
            label="SafeTensors" if i == 0 else "",
        )

        # Model name above the bars
        model_label = (
            model_name.split("_")[1] if "_" in model_name else model_name
        )
        ax.text(
            0,
            y_safe + 0.28,
            model_label,
            fontsize=11,
            weight="600",
            fontfamily="sans-serif",
            va="bottom",
            ha="left",
        )

        # Bidirectional arrow on SLLM bar line
        ax.annotate(
            "",
            xy=(safe_mean, y_sllm),
            xytext=(sllm_mean, y_sllm),
            arrowprops=dict(arrowstyle="<->", lw=2, color="#374151"),
        )

        # Speedup text
        arrow_mid_x = (sllm_mean + safe_mean) / 2
        ax.text(
            arrow_mid_x,
            y_sllm,
            f"{speedup:.1f}x",
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            fontfamily="sans-serif",
            color="#374151",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#374151",
                linewidth=1.5,
            ),
        )

    # Remove y-axis labels (model names are now above bars)
    ax.set_yticks([])

    ax.set_xlabel(
        "Average Loading Time (s)",
        fontsize=12,
        weight="500",
        fontfamily="sans-serif",
    )
    ax.set_title(
        "Model Loading Performance",
        fontsize=16,
        weight="600",
        pad=20,
        fontfamily="sans-serif",
        loc="center",
    )

    # Legend in upper right
    ax.legend(loc="upper right", frameon=True, fontsize=11)

    # Styling
    ax.grid(axis="x", alpha=0.15, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color("#D1D5DB")

    # Let matplotlib auto-determine x-axis limit based on data
    ax.autoscale(axis="x", tight=False)
    ax.margins(x=0.05)  # Add 5% margin

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = get_args()

    models = args.models
    test_name = args.test_name
    num_repeats = args.num_repeats
    results_dir = args.results_dir
    output_filename = args.output_filename

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory {results_dir} does not exist.")

    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sllm_results = load_results(
        models, "sllm", num_repeats, test_name, results_dir
    )
    safetensors_results = load_results(
        models, "safetensors", num_repeats, test_name, results_dir
    )

    # Print statistics
    print_statistics(sllm_results, safetensors_results)

    df = create_dataframe(sllm_results, safetensors_results)
    plot_results(df, sllm_results, safetensors_results, output_filename)


if __name__ == "__main__":
    main()
