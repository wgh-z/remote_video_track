# import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
from processing.read_excel import read

# 画布尺寸
ori_w = 1536
ori_h = 864

size = 16


# data_frameA = pd.read_excel(r'多路图像导入配置表.xlsx', usecols='A', header=0, keep_default_na=False)
# path_list = data_frameA.values.tolist()
#
# while [''] in path_list:
#     path_list.remove([''])


# print(path_list)

# 创建无信号图像
def creat_no_signal(new_w, new_h):
    '''
    :param new_w: 宫格中单幅图宽度
    :param new_h: 宫格中单幅图高度
    :return: 无信号图
    '''
    no_signal_img = Image.new('RGB', (new_w, new_h), (0, 0, 0))
    font = ImageFont.truetype(font='font/simhei.ttf', size=25)
    draw = ImageDraw.Draw(no_signal_img)  # 绘图声明
    label = '未添加视频'
    label_size = draw.textsize(label, font)
    label = label.encode('utf-8')

    left = int((new_w - label_size[0]) / 2)
    top = int((new_h - label_size[1]) / 2)
    location = (left, top)
    draw.text(location, str(label, 'UTF-8'), fill=(255, 255, 255), font=font)
    del draw

    no_signal_img = np.array(no_signal_img)
    no_signal_img = cv.rectangle(no_signal_img, (0, 0), (new_w - 1, new_h - 1), (0, 0, 255), 1)

    return no_signal_img


# 拼接图像
def concat(path_list, new_w, new_h, size):
    '''
    :param path_list: 图像、视频路径列表
    :param new_w: 宫格中单幅图宽度
    :param new_h: 宫格中单幅图高度
    :param size: 宫格数
    :return: 拼接后的整张宫格图像
    '''
    imgs = []
    # imgs_new = []
    for f in path_list:
        if '.jpg' in f[0]:
            img = cv.imread(f[0])
            img = cv.resize(img, (new_w, new_h), cv.INTER_CUBIC)
            img = cv.rectangle(img, (0, 0), (new_w - 1, new_h - 1), (0, 0, 255), 1)
            imgs.append(img)  # 原视频帧
        # imgs_new.append(img_new)  # 缩放后视频帧

    no_signal_img = creat_no_signal(new_w, new_h)
    # 添加无信号图
    for j in range(size - len(path_list)):
        imgs.append(no_signal_img)
        # imgs_new.append(no_signal_img)

    # 宫格排列
    if size == 1:  # 1宫格
        img_concat = imgs[0]
    elif size == 4:  # 4宫格
        img0 = np.concatenate(imgs[0:2], 1)  # 沿1轴横向拼接
        img1 = np.concatenate(imgs[2:4], 1)
        img_concat = np.concatenate([img0, img1], 0)  # 沿0轴纵向拼接
    elif size == 9:  # 9宫格
        img0 = np.concatenate(imgs[0:3], 1)  # 沿1轴横向拼接
        img1 = np.concatenate(imgs[3:6], 1)
        img2 = np.concatenate(imgs[6:], 1)
        img_concat = np.concatenate([img0, img1, img2], 0)  # 沿0轴纵向拼接
    elif size == 16:  # 16宫格
        img0 = np.concatenate(imgs[0:4], 1)  # 沿1轴横向拼接
        img1 = np.concatenate(imgs[4:8], 1)
        img2 = np.concatenate(imgs[8:12], 1)
        img3 = np.concatenate(imgs[12:16], 1)
        img_concat = np.concatenate([img0, img1, img2, img3], 0)  # 沿0轴纵向拼接

    return img_concat


def image(size, path):
    scale = int(size ** 0.5)
    new_w = int(ori_w / scale)
    new_h = int(ori_h / scale)
    path_list = read(path, 'A')  # 视频路径
    model = read(path, 'B')  # 视频检测模型
    weight = read(path, 'C')  # 视频检测权重
    frame_rate = read(path, 'D')  # 视频检测帧率
    image = concat(path_list, new_w, new_h, size)
    return image, frame_rate


def split_img(image, size):
    cut_img = []
    scale = int(size ** 0.5)
    new_w = int(ori_w / scale)
    new_h = int(ori_h / scale)
    for y in range(scale):
        for x in range(scale):
            cut_img.append(image[y * new_h:(y + 1) * new_h, x * new_w:(x + 1) * new_w])
    # cv.imshow("concat.jpg", image)
    # for i, img in zip(range(size), cut_img[:len(path_list)]):
    #     cv.imshow(str(i) + ".jpg", img)
    return cut_img

def show_ori(size, path, column):
    path_list = read(path, column)
    imgs = []
    for f in path_list:
        if '.jpg' in f:
            img = cv.imread(f[0])
            # img = cv.resize(img, (new_w, new_h), cv.INTER_CUBIC)
            # img = cv.rectangle(img, (0, 0), (new_w - 1, new_h - 1), (0, 0, 255), 1)
            imgs.append(img)  # 原视频帧
    no_signal_img = creat_no_signal(1920, 1080)
    # 添加无信号图
    for j in range(size - len(path_list)):
        imgs.append(no_signal_img)
    return imgs


if __name__ == '__main__':
    img = image(16, r'/多路图像导入配置表.xlsx', 'A')
    cv.imshow('isac', img)
    cv.waitKey(0)  # 无限期显示窗口
