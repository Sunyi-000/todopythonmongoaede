from flask import Flask, request, render_template, jsonify
from PIL import Image
from torchvision import models, transforms
import torch

# 假设您的模型文件位于以下路径
model_path = r'D:\APP\VS code\Microsoft VS Code\model.pth'

# 创建模型实例（这里以 ResNet50 为例）
model = models.resnet50(pretrained=False)

# 修改全连接层的输出单元数以匹配您的类别数
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 假设您有两个类别

# 加载模型状态字典
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# 将模型设置为评估模式
model.eval()

# 预处理转换
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

app = Flask(__name__)

@app.route('/')
def index():
    # 渲染 HTML 模板，用户可以上传图片
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 处理 POST 请求，进行图片预测
    if request.method == 'POST':
        image_file = request.files['image']
        image = Image.open(image_file)
        image = preprocess(image)
        image = image.unsqueeze(0)  # 添加批次维度

        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities).item()

        return jsonify({'prediction': '狗' if prediction == 0 else '猫'})
    return jsonify({'error': 'Invalid request.'})

if __name__ == '__main__':
    app.run(debug=True)