import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

from PIL import Image
import io
app = Flask(__name__)
model = YOLO('checkpoint/test.pt')
# 框（rectangle）可视化配置
bbox_color = (0, 0, 255)             # 框的 BGR 颜色
bbox_thickness = 2                   # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size':1,         # 字体大小
    'font_thickness':2,    # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-10,        # Y 方向，文字偏移距离，向下为正
}

def process_frame(img_bgr):
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''

    # 记录该帧开始处理的时间
    start_time = time.time()

    results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果

    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)

    # 预测框的 xyxy 坐标
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')

    # 关键点的 xy 坐标
    # bboxes_keypoints = results[0].keypoints.cpu().numpy().astype('uint32')

    for idx in range(num_bbox):  # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[0]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                              bbox_labelstr['font_thickness'])
    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)
    num_bbox_string = 'beanNumber  {:.2f}'.format(num_bbox)
    # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    FPS_string = 'FPS  {:.2f}'.format(FPS) # 写在画面上的字符串
    FPS_string = num_bbox_string + FPS_string
    img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)

    return img_bgr

def process_frameimg(img_bgr):
    #加载你的pt文件
    model = YOLO('checkpoint/test.pt')
    #把图片存到一个路径
    img_bgr.save("before.jpg")
    img_path = "before.jpg"
    results = model(img_path)
    img_bgr = cv2.imread(img_path)
    plt.imshow(img_bgr[:, :, ::-1])
    num_bbox = len(results[0].boxes.cls)
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    # 框（rectangle）可视化配置
    bbox_color = (0, 0, 255)  # 框的 BGR 颜色
    bbox_thickness = 6  # 框的线宽

    # 框类别文字
    bbox_labelstr = {
        'font_size': 4,  # 字体大小
        'font_thickness': 10,  # 字体粗细
        'offset_x': 0,  # X 方向，文字偏移距离，向右为正
        'offset_y': -80,  # Y 方向，文字偏移距离，向下为正
    }
    for idx in range(num_bbox):  # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[0]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                              bbox_labelstr['font_thickness'])

    return img_bgr
def generate_frames():
    # 获取摄像头，0为电脑默认摄像头，1为外接摄像头
    cap = cv2.VideoCapture(0)

    # 拍照
    time.sleep(3)  # 运行本代码后等几秒拍照
    # 从摄像头捕获一帧画面
    success, frame = cap.read()

    cap.release()  # 关闭摄像头
    cv2.destroyAllWindows()  # 关闭图像窗口
    img_bgr = process_frame(frame)
    # 将处理后的帧转换为JPEG格式
    ret, buffer = cv2.imencode('.jpg', img_bgr)

    frame_data = buffer.tobytes()

    # 使用生成器生成视频帧
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_image', methods=['POST'])
def generate_image():

    image_path = 'true'


    return render_template('test.html', image=image_path)

#上传图片检测
from flask import Flask, request, send_file
from PIL import Image

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    image = Image.open(file)

    # 在这里进行图像处理的操作，比如调整大小、应用滤镜等

    img_bgr = process_frameimg(image)  # 这里暂时将处理后的图片设置为原图
    # 将处理后的帧转换为JPEG格式
    ret, buffer = cv2.imencode('.jpg', img_bgr)
    frame_data = buffer.tobytes()
    def save_bytes_as_image(data, output_path):
        # 将字节数据转换为图像对象
        image = Image.open(io.BytesIO(data))

        # 保存图像为 jpg 文件
        image.save(output_path, 'JPEG')

    output_path = 'output.jpg'  # 图像保存路径

    save_bytes_as_image(frame_data, output_path)

    return send_file('output.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
# import onnx
#
# # 读取 ONNX 模型
# onnx_model = onnx.load('checkpoint/best.onnx')
#
# # 检查模型格式是否正确
# onnx.checker.check_model(onnx_model)
#
# print('无报错，onnx模型载入成功')
#
# print(onnx.helper.printable_graph(onnx_model.graph))