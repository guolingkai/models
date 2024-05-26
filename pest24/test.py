import torch
import torch.nn
import onnx

model = torch.load('pest24_yolov5_bld.pt')
model.eval()

input_names = ['input']
output_names = ['output']

# x = torch.randn(1, 3, 32, 32, requires_grad=True)
x = torch.randn(1, 3, 640, 640, requires_grad=True)

torch.onnx.export(model, x, 'best.onnx', input_names=input_names, output_names=output_names, verbose='True')

