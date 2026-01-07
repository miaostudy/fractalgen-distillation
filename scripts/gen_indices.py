import os
import argparse
from torchvision.datasets.folder import find_classes

def main():
    parser = argparse.ArgumentParser(description="Get ImageNet class indices from a list of class names.")
    parser.add_argument('--data_path', default='/data/wlf/datasets/imagenet', type=str,
                        help='Path to the ImageNet dataset (root folder containing train/val)')
    parser.add_argument('--class_list_path', default='misc/class_nette.txt', type=str,
                        help='Path to the text file containing class names (one per line)')
    parser.add_argument('--output_path', default='misc/class_nette_indices.txt', type=str,
                        help='Path to save the resulting indices')
    args = parser.parse_args()

    # 1. 确定训练集路径
    # ImageNet的标准结构通常是 data_path/train/n01440764/
    train_dir = os.path.join(args.data_path, 'train')
    if not os.path.exists(train_dir):
        print(f"Warning: '{train_dir}' not found. Trying to find classes directly in '{args.data_path}'")
        train_dir = args.data_path

    if not os.path.exists(train_dir):
        print(f"Error: Dataset path '{train_dir}' does not exist.")
        return

    print(f"Reading ImageNet classes from: {train_dir}")

    # 2. 获取ImageNet的类名到索引的映射
    # find_classes 会扫描文件夹名称并按字母顺序排序，这与 torchvision.datasets.ImageFolder 的行为一致
    try:
        classes, class_to_idx = find_classes(train_dir)
    except Exception as e:
        print(f"Error finding classes: {e}")
        return

    print(f"Found {len(classes)} classes in the dataset.")

    # 3. 读取目标类名列表
    if not os.path.exists(args.class_list_path):
        print(f"Error: Class list file not found at '{args.class_list_path}'")
        return

    with open(args.class_list_path, 'r') as f:
        # 读取每一行，去除空白字符
        query_classes = [line.strip() for line in f if line.strip()]

    print(f"Looking up indices for {len(query_classes)} classes from '{args.class_list_path}'...")

    # 4. 查找索引并保存
    found_indices = []
    missing_classes = []

    for qc in query_classes:
        # 注意：这里的 'class name' 指的是文件夹名称（Synset ID，如 n01440764）
        # 如果您的文本文件里是英文名称（如 'tench'），则需要额外的映射文件
        if qc in class_to_idx:
            found_indices.append(str(class_to_idx[qc]))
        else:
            missing_classes.append(qc)
            found_indices.append("-1") # 未找到用 -1 占位，或者您可以选择跳过

    if missing_classes:
        print(f"Warning: {len(missing_classes)} classes were not found in the dataset (e.g., {missing_classes[:3]}...)")

    # 5. 保存结果
    with open(args.output_path, 'w') as f:
        # 这里将索引保存为一行一个，或者如果您想要逗号分隔，可以改用 join
        # f.write(", ".join(found_indices))
        for idx in found_indices:
            f.write(f"{idx}\n")

    print(f"Successfully saved {len(found_indices)} indices to '{args.output_path}'")

if __name__ == '__main__':
    main()