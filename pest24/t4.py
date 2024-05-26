

from ultralytics import YOLO

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO('pest24_yolov5_bld.pt')
model.export(format = "onnx")


