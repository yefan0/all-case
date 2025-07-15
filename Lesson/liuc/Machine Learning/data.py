import os
import cv2
import numpy as np
import torch


def load_data(data_path, fig_size, gray_img=True):
    """读取图片和标签"""
    images = []
    labels = []
    for label in os.listdir(data_path):
        image_dir = os.path.join(data_path, label)
        if not label.isdigit() or not os.path.isdir(image_dir):  # 非数据文件夹
            continue

        img_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        if len(img_files) < 2:  # 去除图片数量小于2的个体
            continue

        for image_name in img_files:
            image_path = os.path.join(image_dir, image_name)
            if gray_img:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
            if image is None:
                print(f"Warning: Image {image_path} could not be read.")
                continue
            image = cv2.resize(image, (fig_size, fig_size))
            if gray_img:
                image = np.expand_dims(image, 0)
            images.append(image)
            labels.append(int(label))

    images = np.array(images) / 255.0  # 归一化
    return images, np.array(labels)


def load_test_data(data_path, fig_size, gray_img=True):
    """加载测试数据"""
    images = []
    labels = []
    for file_name in os.listdir(data_path):
        if not file_name.endswith('.jpg'):
            continue
        label = file_name.split('_')[0]
        image_path = os.path.join(data_path, file_name)
        if gray_img:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
        image = cv2.resize(image, (fig_size, fig_size))
        if gray_img:
            image = np.expand_dims(image, 0)
        images.append(image)
        labels.append(label)
    images = torch.tensor(np.array(images, dtype=np.float32))
    return images/255.0, np.array(labels)
