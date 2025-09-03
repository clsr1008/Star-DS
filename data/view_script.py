from datasets import load_from_disk


def main():
    data_path = "data/math.8k"  # 你的数据目录
    dataset = load_from_disk(data_path)

    print(f"数据集大小: {len(dataset)}")
    print("字段名:", dataset.column_names)

    # 打印前5条样本
    for i in range(min(5, len(dataset))):
        print(f"\n第{i + 1}条样本:")
        print(dataset[i])


if __name__ == "__main__":
    main()
