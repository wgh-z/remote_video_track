# 远程鼠标点击显示追踪目标
# import pyautogui
from flask import Flask, render_template, Response, request
import io
import requests
import cv2 as cv
import numpy as np
from PIL import Image
from yolov5_tracker import Tracker


app = Flask(__name__)

# 获取屏幕尺寸
# screenWidth, screenHeight = pyautogui.size()

l_rate, r_rate = None, None
show_id = dict()

@app.route('/')
def index():
    return render_template('index.html')


# 单击左键
@app.route('/left')  # 参数要与html相同
def leftpointer():
    global l_rate
    x = float(request.args["xrate"])  # 接收客户端传来的参数
    y = float(request.args["yrate"])
    l_rate = (x, y)
    print('left==', l_rate)
    return "success"


# # 单击右键
# @app.route('/right')
# def rightpointer():
#     global r_rate
#     x = float(request.args["xrate"])  # 接收客户端传来的参数
#     y = float(request.args["yrate"])
#     r_rate = (x, y)
#     print('right==', r_rate)
#     return "success"


# 双击左键
@app.route('/double')  # 参数要与html相同
def doubleleftpointer():
    global r_rate
    x = float(request.args["xrate"])  # 接收客户端传来的参数
    y = float(request.args["yrate"])
    r_rate = (x, y)
    print('right==', r_rate)
    return "success"


# # 按下
# @app.route('/down')
# def down():
#     global l_rate
#     x = float(request.args["xrate"])  # 接收客户端传来的参数
#     y = float(request.args["yrate"])
#     # pyautogui.mouseDown(x, y)   # 鼠标按下
#     l_rate = (x, y)
#     print('left==', x, y)
#     return "success"


# # 拖动
# @app.route('/move')
# def move():
#     x = int(float(request.args["xrate"]) * vidie_w)  # 接收客户端传来的参数
#     y = int(float(request.args["yrate"]) * video_h)
#     # pyautogui.moveTo(x, y)  # 拖动响应
#     print('move==', x, y)
#     return "success"


# # 释放
# @app.route('/up')
# def up():
#     global x, y
#     x = float(request.args["xrate"])  # 接收客户端传来的参数
#     y = float(request.args["yrate"])
#     # pyautogui.mouseUp()    # 鼠标释放
#     print('up==', x, y)
#     return "success"


def gen():
    global l_rate, r_rate, show_id
    weitght = './weights/yolov5m.pt'
    StrongSort = './weights/osnet_x0_25_msmt17.pth'
    imgsz = [640, 640]
    
    tracker = Tracker(yolo_weights=weitght,
                    #   reid_weights=StrongSort,
                      tracking_method='bytetrack',
                      imgsz=imgsz,
                      view_img=False,
                      save_txt=False,
                      save_csv=False,
                      save_conf=False,
                      save_crop=False,
                    #   nosave=True,
                    #   classes=[0],
                      line_thickness=2,
                    #   vid_stride=3,
                      device='0'
                      # half=True,
                      )

    imgs = requests.get('http://127.0.0.1:3001/video_feed', stream=True)
    for img in imgs.iter_content(chunk_size=809600):
        # 找到图片数据的开始和结束位置
        start = img.find(b'\xff\xd8')
        end = img.find(b'\xff\xd9')

        # 提取图片数据
        img = img[start:end+2]
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv.imdecode(img, cv.IMREAD_COLOR)

        frame, show_id = tracker(img, show_id, l_rate, r_rate)
        l_rate, r_rate = None, None

        # frame = cv.resize(frame, (1280, 720))

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + frame)


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002, debug=True, threaded=True)
