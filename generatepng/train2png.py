import pickle
import numpy as np
import os
from PIL import Image

# 路径设置
data_dir = './cifar-100-python/'
output_dir = './cifar100_images'

# 加载 meta 获取类别名称
meta = pickle.load(open(os.path.join(data_dir, 'meta'), 'rb'))
label_names = meta['fine_label_names']
label_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in label_names]

# 加载训练数据
def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

train_data = unpickle(os.path.join(data_dir, 'train'))

X_train = train_data[b'data']
y_train = train_data[b'fine_labels']

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
for label_name in label_names:
    os.makedirs(os.path.join(output_dir, label_name), exist_ok=True)

# 图像还原函数
def reshape_image(img_vector):
    r = img_vector[:1024].reshape(32, 32)
    g = img_vector[1024:2048].reshape(32, 32)
    b = img_vector[2048:].reshape(32, 32)
    return np.stack([r, g, b], axis=2)

# 保存图像
for i, (img_vector, label) in enumerate(zip(X_train, y_train)):
    class_name = label_names[label]
    img = reshape_image(img_vector)
    img_path = os.path.join(output_dir, class_name, f'{i}.png')
    Image.fromarray(img).save(img_path)

print("✅ CIFAR-100 图像已按类别保存完毕")