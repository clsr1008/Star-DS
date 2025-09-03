# view_dataset.py
import pandas as pd
import argparse
import json

def pretty_print_row(row):
    """美观打印一行数据"""
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
    print(f"正在加载数据集: {file_path}")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"❌ 无法读取 {file_path}: {e}")
        return

    print("\n===== 数据集信息 =====")
    print(f"数据行数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    print("\n数据类型:")
    print(df.dtypes)

    # 取指定范围
    print(f"\n===== 显示第 {start} 行到第 {end} 行（结构化显示） =====")
    for i in range(start, min(end, len(df))):
        row = df.iloc[i].to_dict()
        pretty_print_row(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看 parquet 数据集内容（美观结构化输出）")
    parser.add_argument("--file", type=str, required=True, help="parquet 文件路径")
    parser.add_argument("--start", type=int, default=0, help="起始行（包含）")
    parser.add_argument("--end", type=int, default=5, help="结束行（不包含）")
    args = parser.parse_args()

    main(args.file, args.start, args.end)
