# view_dataset.py
import pandas as pd
import argparse
import json


def pretty_print_row(row):
    """Pretty print a single dataset row."""
    print(f"📌 index: {row.get('extra_info', {}).get('index', 'N/A')}  |  data source: {row['data_source']}")
    print(f"ability: {row['ability']}")

    print("\n--- Prompt ---")
    try:
        prompt_data = json.loads(row['prompt']) if isinstance(row['prompt'], str) else row['prompt']
        print(json.dumps(prompt_data, ensure_ascii=False, indent=2))
    except Exception:
        print(row['prompt'])

    print("\n--- Reward Model ---")
    try:
        reward_data = json.loads(row['reward_model']) if isinstance(row['reward_model'], str) else row['reward_model']
        print(json.dumps(reward_data, ensure_ascii=False, indent=2))
    except Exception:
        print(row['reward_model'])

    print("\n--- Extra Info ---")
    try:
        extra_data = json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row['extra_info']
        print(json.dumps(extra_data, ensure_ascii=False, indent=2))
    except Exception:
        print(row['extra_info'])
    print("=" * 60)


def main(file_path, start, end):
    print(f"Loading dataset: {file_path}")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return

    print("\n===== Dataset Info =====")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)

    # Display the specified range
    print(f"\n===== Display rows {start} to {end} (structured view) =====")
    for i in range(start, min(end, len(df))):
        row = df.iloc[i].to_dict()
        pretty_print_row(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View parquet dataset content (pretty structured output)")
    parser.add_argument("--file", type=str, required=True, help="Path to the parquet file")
    parser.add_argument("--start", type=int, default=0, help="Starting row (inclusive)")
    parser.add_argument("--end", type=int, default=5, help="Ending row (exclusive)")
    args = parser.parse_args()

    main(args.file, args.start, args.end)
