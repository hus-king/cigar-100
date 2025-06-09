import os

# 图像数据集路径（训练集或测试集都可以，结构一样）
dataset_path = './cifar100_images_train'  # 也可以是 cifar100_images_test

# 输出文件路径
output_file = 'class_name_to_id.txt'

# 获取所有类别的名字（即子文件夹名），并按字母排序
class_names = sorted(os.listdir(dataset_path))

# 过滤掉非目录的文件（可选）
class_names = [name for name in class_names if os.path.isdir(os.path.join(dataset_path, name))]

# 写入文件
with open(output_file, 'w') as f:
    for idx, name in enumerate(class_names):
        f.write(f"{idx}\t{name}\n")

print(f"✅ 类别映射文件已生成：{output_file}")
print("示例内容：")
with open(output_file, 'r') as f:
    print(f.read()[:200])  # 打印前几行作为示例