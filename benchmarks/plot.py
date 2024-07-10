import argparse
import json
import os

import matplotlib.pyplot as plt
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
        choices=["random", "single"],
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


def plot_results(df, output_filename):
    """Plot the results using seaborn's boxplot for loading latency."""
    # Set the style for the plot
    sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.2)

    # Create a box plot using seaborn
    plt.figure(figsize=(12, 8))
    # box_plot = sns.boxplot(x='Model', y='Loading Time', hue='System', data=df, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black"})
    # use bar plot instead of box plot
    bar_plot = sns.barplot(x="Model", y="Loading Time", hue="System", data=df)

    # Customize the plot
    plt.title("Model Loading Latency", fontsize=16, weight="bold")
    plt.xlabel("Model")
    plt.ylabel("Loading Time (s)")
    plt.legend(title="System")

    # Remove top and right spines for better aesthetics
    sns.despine(trim=True)

    # Save the plot as an image file
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)


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
    df = create_dataframe(sllm_results, safetensors_results)
    plot_results(df, output_filename)


if __name__ == "__main__":
    main()
