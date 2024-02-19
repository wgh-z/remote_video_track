# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
import time
import numpy as np
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from trackers.multi_tracker_zoo import create_tracker
from processing.regional_judgment import point_in_rect
from timer import Timer


show_id = []  # None为全部显示
show_cls = None  # None为全部显示
# click_point = None
ori_size = None
show_size = None
points = []
l_point = None  # 左键点击的点
r_point = None  # 右键点击的点

class Tracker:
    def __init__(
        self,
        yolo_weights=WEIGHTS / "yolov5s.pt",
        data=ROOT / "data/coco128.yaml",  # dataset.yaml path
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='bytetrack',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        show_fps=False,  # show fps
        show_size=(1280, 720),  # show video size
        bs=1,
        ):
        # init params
        self.yolo_weights = yolo_weights
        # self.source = source
        # self.data = data
        # self.reid_weights = reid_weights
        # self.tracking_method = tracking_method
        # self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        # self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_csv = save_csv
        self.save_conf = save_conf
        self.save_crop = save_crop
        # self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        # self.update = update
        # self.project = project
        # self.name = name
        # self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        # self.half = half
        # self.dnn = dnn
        # self.vid_stride = vid_stride
        self.show_fps = show_fps
        # self.show_size = show_size
        self.bs = bs
        
        self.save_img = not nosave
        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        if not nosave:
            (self.save_dir / "labels" if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # Define the path for the CSV file
        self.csv_path = self.save_dir / "predictions.csv"

        # yolo model
        device = select_device(device)
        self.model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # tracker
        self.tracker_list = []
        for i in range(bs):
            tracker = create_tracker(tracking_method, reid_weights, device, half)
            self.tracker_list.append(tracker, )
            if hasattr(tracker, 'model'):
                if hasattr(tracker.model, 'warmup'):
                    tracker.model.warmup()
        self.outputs = [None] * bs
        self.curr_frames, self.prev_frames = [None] * bs, [None] * bs

        # init inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else 1, 3, *self.imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device), Profile(device=device))

        self.timer = Timer(300)

    # def update(self, det, im0):
    #     self.outputs = self.tracker.update(det, im0)
    #     return self.outputs

    # def __del__(self):
    #     if hasattr(self.tracker, 'model'):
    #         if hasattr(self.tracker.model, 'close'):
    #             self.tracker.model.close()

    # Create or append to the CSV file
    def write_to_csv(self, image_name, prediction, confidence):
        data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not self.csv_path.is_file():
                writer.writeheader()
            writer.writerow(data)

    @smart_inference_mode()
    def __call__(self, im0s, show_id:dict, l_rate=None, r_rate=None):
        w, h = im0s.shape[1], im0s.shape[0]
        l_point = (int(w * l_rate[0]), int(h * l_rate[1])) if l_rate is not None else None
        r_point = (int(w * r_rate[0]), int(h * r_rate[1])) if r_rate is not None else None
        
        im = letterbox(im0s, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if self.model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with self.dt[1]:
            # visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if visualize else False
            visualize = False
            if self.model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = self.model(image, augment=self.augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, self.model(image, augment=self.augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = self.model(im, augment=self.augment, visualize=visualize)
        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            self.seen += 1
            # p, im0, frame = path, im0.copy(), getattr(dataset, "frame", 0)
            im0 = im0s

            p = Path('test')  # to Path
            save_path = str(self.save_dir / p.name)  # im.jpg
            # txt_path = str(self.save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            txt_path = str(self.save_dir / "labels" / p.stem)
            
            # s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
   
            if hasattr(self.tracker_list[i], 'tracker') and hasattr(self.tracker_list[i].tracker, 'camera_update'):
                if self.prev_frames[i] is not None and self.curr_frames[i] is not None:  # camera motion compensation
                    self.tracker_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                with self.dt[3]:
                    self.outputs[i] = self.tracker_list[i].update(det.cpu(), im0)   # <================================================添加其他检测器
                    id_set = set([output[4] for output in self.outputs[i]])
                    show_id = self.timer(id_set, show_id)
                # Write results
                if len(self.outputs[i]) > 0:
                    for output, conf in zip(self.outputs[i], det[:, 4]):
                        xyxy = output[0:4]
                        id = output[4]
                        cls = output[5]

                        if l_point is not None and id not in show_id:
                            if point_in_rect(l_point, xyxy):
                                # show_id.append(id)
                                show_id = self.timer.add_delay(show_id, id)
                                l_point = None
                        if r_point is not None:
                            if point_in_rect(r_point, xyxy):
                                try:
                                    # show_id.remove(id)
                                    del show_id[id]
                                except:
                                    pass
                                r_point = None

                        # c = int(cls)  # integer class
                        # label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if self.save_csv:
                            self.write_to_csv(p.name, label, confidence_str)

                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        # if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                        if id in show_id.keys():  # 显示指定id的目标
                            # if id in show_id or show_id == []:  # 显示指定id的目标
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if self.hide_labels else (f"{id} {self.names[c]}" if self.hide_conf else f"{id} {self.names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / "crops" / self.names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if self.view_img:
                if platform.system() == "Linux" and p not in self.windows:
                    self.windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                # im0 = cv2.resize(im0, show_size)

                # if self.show_fps:
                #     d_fps = (d_fps + (self.vid_stride / (time.time() - t1))) / 2
                #     im0 = cv2.putText(im0, "fps= %.2f" % (d_fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), line_thickness)  # 显示fps

                cv2.imshow('track', im0)
                cv2.waitKey(1)  # 1 millisecond

            return im0, show_id

            # # Save results (image with detections)
            # if self.save_img:
            #     if dataset.mode == "image":
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if self.vid_path[i] != save_path:  # new video
            #             self.vid_path[i] = save_path
            #             if isinstance(self.vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            #         vid_writer[i].write(im0)

        # # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")


if __name__ == "__main__":
    weitght = './weights/yolov5m.pt'
    # weitght = './weights/yolov5s.onnx'  # 89.3ms cpu
    # weitght = './weights/yolov5s_simplify.onnx'  # 88.3ms cpu
    # weitght = './weights/yolov5s_fp16.onnx'  # 125.5ms cpu
    # weitght = './weights/yolov5s_openvino_model'  # 39.3ms gpu
    # weitght = './weights/yolov5m_openvino_model'  # 91.1ms gpu
    # weitght = './weights/yolov5s_int8_openvino_model'  # 40.1ms cpu
    # weitght = './weights/yolov5m_int8_openvino_model'  # 83.9ms cpu

    StrongSort = './weights/osnet_x0_25_msmt17.pth'

    source = 'people.mp4'
    imgsz = [640, 640]
    l_point = None
    r_point = None
    
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
    
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = tracker(frame)
        cv2.imshow('track', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
