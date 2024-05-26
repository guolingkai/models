
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因


model = YOLO("pest24/pest24_yolov5_bld.pt")
model.export(format="onnx")


