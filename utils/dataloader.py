import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
from utils import *
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, image_transform=None, label_transform=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index].strip()
        label_path = annotation_line.split()[0]
        
        # 将 Windows 路径转换成 Linux 路径
        label_path = label_path.replace("\\", "/")

        # 使用 os.path.join 和 os.path.basename 生成 image 路径
        image_filename = os.path.basename(label_path).replace("label", "image")
        image_path = os.path.join(os.path.dirname(label_path), "..", "image", image_filename)
        image_path = os.path.normpath(image_path)  # 统一路径格式
        
        # 检测路径是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        image = Image.open(image_path)
        label = Image.open(label_path)

        if label.mode != 'L':
            label = label.convert('L')

        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0, 1]))
        if self.label_transform is not None:
            label = self.label_transform(label)
        label = torch.from_numpy(np.array(label))

        return image, label

    def __len__(self):
        return self.length

