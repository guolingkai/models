

import torch
# torch_model = torch.load("pest24_yolov5_bld.pt") # pytorch模型加载
torch_model = torch.load("yolov5s.pt", strict=False) # pytorch模型加载

batch_size = 1  #批处理大小
input_shape = (3,244,244)   #输入数据

# set the model to inference mode
torch_model.eval()

x = torch.randn(batch_size,*input_shape)		# 生成张量
export_onnx_file = "test.onnx"					# 目的ONNX文件名
torch.onnx.export(torch_model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],		# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},		# 批处理变量
                                    "output":{0:"batch_size"}})



