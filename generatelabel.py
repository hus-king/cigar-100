import os

# 数据集根目录
root_dirs = {
    'train': './cifar100_images_train',
    'test': './cifar100_images_test'
}

# 获取类别列表（从 train 文件夹中读取）
class_names = sorted(os.listdir(root_dirs['train']))

# 构建类别名到 id 的映射（ID 从 0 开始）
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# 生成或替换标签函数
def generate_yolo_labels(data_dir, class_map):
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing class: {class_name}")
        image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]

        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(class_path, f"{base_name}.txt")

            # ✅ 即使文件存在也重新写入
            class_id = class_map[class_name] - 1
            with open(txt_path, 'w') as f:  # 使用 'w' 模式会覆盖已有文件
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

            print(f"Generated/Updated: {txt_path}")

# 执行生成
for split, dir_path in root_dirs.items():
    print(f"\n=== Processing {split} ===")
    generate_yolo_labels(dir_path, class_to_idx)

print("\n✅ 所有标签文件已生成或更新完毕！")