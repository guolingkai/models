

import torch
import torchvision

# 加载或定义你的PyTorch模型
model = torchvision.models.resnet18(pretrained=True)

# 示例输入数据（根据你的模型和任务调整）
example_input = torch.rand(1, 3, 224, 224)

# 导出模型为ONNX格式
torch.onnx.export(model, example_input, "your_model.onnx", verbose=True, input_names=["input"], output_names=["output"])

