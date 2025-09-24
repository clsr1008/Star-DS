import os
import json
import argparse
import pandas as pd

BENCHMARK_LIST = ["aime24x8", "aime25x8", "amc23x8", "math500", "minerva_math", "olympiadbench"]

def collect_max_metric(results_dir, benchmark_name, max_step=None):
    max_acc = None
    best_step = None

    # 遍历每个 global_step_* 文件夹
    for step_folder in sorted(os.listdir(results_dir), key=lambda x: int(x.split('_')[-1])):
        step_num = int(step_folder.split('_')[-1])
        if max_step is not None and step_num > max_step:
            continue

        step_path = os.path.join(results_dir, step_folder)
        if not os.path.isdir(step_path):
            continue

        benchmark_path = os.path.join(step_path, benchmark_name)
        if not os.path.isdir(benchmark_path):
            continue

        # 找到 metrics.json 文件
        metrics_files = [f for f in os.listdir(benchmark_path) if f.endswith("_metrics.json")]
        if not metrics_files:
            continue

        metrics_file = os.path.join(benchmark_path, metrics_files[0])
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            if 'acc' in metrics:
                acc = metrics['acc']
                if max_acc is None or acc > max_acc:
                    max_acc = acc
                    best_step = step_num

    return best_step, max_acc


def report_max_metrics(benchmark_name, results_dirs, labels, max_step=None, output_file=None):
    results = {}
    for results_dir, label in zip(results_dirs, labels):
        best_step, max_acc = collect_max_metric(results_dir, benchmark_name, max_step)
        if max_acc is not None:
            results[label] = {
                "best_step": best_step,
                "max_acc": max_acc
            }
            print(f"[{label}] Benchmark={benchmark_name}: Max acc={max_acc:.2f} at step {best_step}")
        else:
            print(f"[{label}] No metrics found for {benchmark_name}")

    # 保存结果
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved results to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect maximum accuracy for a Benchmark")
    parser.add_argument("--results_dirs", type=str, nargs='+', required=True,
                        help="Paths to the results folders (space-separated)")
    parser.add_argument("--labels", type=str, nargs='+', required=True,
                        help="Labels for each experiment (space-separated, must match number of dirs)")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name or 'all'")
    parser.add_argument("--max_step", type=int, default=None, help="Maximum step to consider")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save results as JSON")

    args = parser.parse_args()

    if len(args.results_dirs) != len(args.labels):
        raise ValueError("Number of results_dirs must match number of labels")

    if args.benchmark.lower() == "all":
        all_results = {}
        for benchmark in BENCHMARK_LIST:
            results = report_max_metrics(benchmark, args.results_dirs, args.labels, args.max_step)
            all_results[benchmark] = results
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ Saved all benchmarks results to {args.output_file}")

        # ===== 新增：输出表格 =====
        table = {}
        for benchmark, results in all_results.items():
            row = {}
            for label in args.labels:
                if label in results and results[label].get("max_acc") is not None:
                    row[label] = results[label]["max_acc"]
                else:
                    row[label] = None
            table[benchmark] = row

        df = pd.DataFrame.from_dict(table, orient="index")
        df.index.name = "Benchmark"

        # ===== 新增：加一行均值 =====
        mean_row = df.mean(numeric_only=True)
        df.loc["Average"] = mean_row
        print("\n=== Results Table (Max Accuracy) ===")
        print(df.to_string(float_format=lambda x: f"{x:.2f}" if pd.notnull(x) else "NA"))
        output_csv = "data/plot/results_table.csv"
        df.to_csv(output_csv, float_format="%.2f")
    else:
        report_max_metrics(args.benchmark, args.results_dirs, args.labels, args.max_step, args.output_file)
