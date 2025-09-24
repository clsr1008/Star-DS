import os
import json
import argparse
import matplotlib.pyplot as plt

BENCHMARK_LIST = ["aime24x8", "aime25x8", "amc23x8", "math500", "minerva_math", "olympiadbench"]

def collect_metrics(results_dir, benchmark_name, max_step=None):
    steps = []
    accs = []

    # Iterate through each global_step_* folder
    for step_folder in sorted(os.listdir(results_dir), key=lambda x: int(x.split('_')[-1])):
        step_num = int(step_folder.split('_')[-1])
        if max_step is not None and step_num > max_step:
            continue  # Skip steps beyond max_step

        step_path = os.path.join(results_dir, step_folder)
        if not os.path.isdir(step_path):
            continue

        benchmark_path = os.path.join(step_path, benchmark_name)
        if not os.path.isdir(benchmark_path):
            continue

        # Find the metrics.json file
        metrics_files = [f for f in os.listdir(benchmark_path) if f.endswith("_metrics.json")]
        if not metrics_files:
            continue

        metrics_file = os.path.join(benchmark_path, metrics_files[0])
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            if 'acc' in metrics:
                accs.append(metrics['acc'])
                steps.append(step_num)

    return steps, accs


def plot_metrics_multiple(benchmark_name, results_dirs, labels, max_step=None):
    plt.figure(figsize=(10,6))

    for results_dir, label in zip(results_dirs, labels):
        steps, accs = collect_metrics(results_dir, benchmark_name, max_step)
        if steps:
            plt.plot(steps, accs, marker='o', label=label)
        else:
            print(f"No metrics found for '{label}' in {results_dir}")

    plt.xlabel("Training Epoch")
    plt.ylabel("Accuracy (%)")
    title = f"Accuracy vs Epoch for Benchmark '{benchmark_name}'"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot to data/plot directory
    save_dir = "data/plot"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{benchmark_name}_accuracy_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to {save_path}")

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Accuracy vs Step for a Benchmark")
    parser.add_argument("--results_dirs", type=str, nargs='+', required=True,
                        help="Paths to the results folders (space-separated)")
    parser.add_argument("--labels", type=str, nargs='+', required=True,
                        help="Labels for each experiment (space-separated, must match number of dirs)")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name or 'all'")
    parser.add_argument("--max_step", type=int, default=None, help="Maximum step to visualize")
    args = parser.parse_args()

    if len(args.results_dirs) != len(args.labels):
        raise ValueError("Number of results_dirs must match number of labels")

    if args.benchmark.lower() == "all":
        for benchmark in BENCHMARK_LIST:
            plot_metrics_multiple(benchmark, args.results_dirs, args.labels, args.max_step)
    else:
        plot_metrics_multiple(args.benchmark, args.results_dirs, args.labels, args.max_step)
