import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# 类别数量（根据你的数据集调整）
num_classes = 16  # 与权重文件保持一致

# 加载模型结构
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, num_classes)

# 加载训练好的权重
model.load_state_dict(torch.load("resnet18_custom.pth", map_location=torch.device('cpu')))
model = model.cpu()  # 使用 CPU
model.eval()

# 图像预处理（必须与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 类别索引到类别名称的映射
class_names = [
    "camel",
    "chair",
    "dolphin",
    "hamster",
    "lion",
    "maple_tree",
    "orange",
    "orchid",
    "pickup_truck",
    "pine",
    "rabbit",
    "skyscraper",
    "squirrel",
    "tractor",
    "turtle",
    "willow_tree"
]

# 推理函数
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 不再移到 GPU
    with torch.no_grad():
        output = model(image)
        predicted_idx = torch.argmax(output, dim=1).item()
    return predicted_idx

# 示例调用
if __name__ == "__main__":
    test_dir = "./test"
    for fname in os.listdir(test_dir):
        if fname.lower().endswith(".png"):
            image_path = os.path.join(test_dir, fname)
            predicted_class = predict_image(image_path)
            print(f"{fname}: {class_names[predicted_class]}")