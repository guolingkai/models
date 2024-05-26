
import torch
model = torch.load('yolov5s.pt')
model.eval()
input_names = ['input']
output_names = ['output']
# x = torch.randn(1,3,512,512,requires_grad=True)
x = torch.randn(1,3,640,640,requires_grad=True)
torch.onnx.export(model, x, 'HyClsNet(traced).onnx', input_names=input_names, output_names=output_names, verbose='True')

