import json
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def compare_scores(file1, file2, show_n=10):
    """
    Compare two JSON score files and print statistics and sample comparisons
    """
    # Load files
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    df1 = pd.DataFrame(data1).set_index("index")
    df2 = pd.DataFrame(data2).set_index("index")

    # Align by index
    merged = df1.join(df2, lsuffix="_1", rsuffix="_2")

    results = {}
    for field in ["uncertainty_score", "reward_variance", "num_correct"]:
        col1 = f"{field}_1"
        col2 = f"{field}_2"

        mse = mean_squared_error(merged[col1], merged[col2])
        corr, _ = pearsonr(merged[col1], merged[col2])

        results[field] = {"mse": mse, "pearson_corr": corr}

    print("\n=== Difference Statistics ===")
    for field, stats in results.items():
        print(f"{field}: MSE={stats['mse']:.6f}, Pearson Corr={stats['pearson_corr']:.4f}")

    print(f"\n=== Sample Comparisons (first {show_n}) ===")
    print(merged.head(show_n).to_string(float_format=lambda x: f"{x:.6f}"))

    return merged, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two JSON score files")
    parser.add_argument("--file1", type=str, required=True, help="Path to first JSON file")
    parser.add_argument("--file2", type=str, required=True, help="Path to second JSON file")

    args = parser.parse_args()

    compare_scores(args.file1, args.file2)
