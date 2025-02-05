import json
import numpy as np
import random
from PIL import Image, ImageFilter
from torchvision import transforms
from collections import OrderedDict
import pickle
import torch
import cv2
import os

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def convert_image_to_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image

class CustomToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        if isinstance(image, Image.Image):
            return self.to_tensor(image)
        else:
            return image

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def sample_related_index(matrix, idx):
    related_indices = np.where(matrix[idx] == 1)[0]
    if len(related_indices) == 0:
        print(idx, matrix[idx])
        return None
    sampled_index = related_indices[random.randrange(len(related_indices))]
    return sampled_index

def blend_cloud_mask(img, mask, alpha_factor=0.3):
    img_size = img.size
    mask = mask.resize(img_size, Image.LANCZOS)
    # 提取云层mask的Alpha通道
    cloud_mask_np = np.array(mask)
    alpha_channel = cloud_mask_np[:, :, 3] / 255.0

    # 调整Alpha通道，控制云层的透明度
    alpha_channel = np.clip(alpha_channel * alpha_factor, 0, 1)

    # 提取云层mask的RGB通道`
    cloud_rgb = cloud_mask_np[:, :, :3]
    # 将遥感图片转换为numpy数组
    remote_sensing_np = np.array(img)
    # 执行Alpha融合
    result_np = (
                cloud_rgb * alpha_channel[:, :, None] + remote_sensing_np * (1 - alpha_channel[:, :, None])).astype(
        np.uint8)
    # 将结果转换回PIL图像
    result_image = Image.fromarray(result_np)
    return result_image

def blur_image(img, r=5.0):
    blurred_image = img.filter(ImageFilter.GaussianBlur(radius=r))  # radius是模糊半径，可根据需要调整
    return blurred_image

def BEN_blend_cloud_mask(img, mask, alpha_factor=0.3):
    img_size = img.shape[1:]  # (W, H)
    mask = mask.resize(img_size, Image.LANCZOS)
    cloud_mask_np = np.array(mask)
    alpha_channel = cloud_mask_np[:, :, 3] / 255.0
    alpha_channel = np.clip(alpha_channel * alpha_factor, 0, 1)
    cloud_rgb = cloud_mask_np[:, :, :3]
    result_np = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        remote_sensing_np = img[i, :, :]
        result_np[i, :, :] = (
            cloud_rgb[:, :, i % 3] * alpha_channel + remote_sensing_np * (1 - alpha_channel)).astype(np.uint8)

    return result_np

def BEN_blur_image(img, r=5.0):
    blurred_img = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        channel_img = Image.fromarray(img[i, :, :], mode='L')
        blurred_channel = channel_img.filter(ImageFilter.GaussianBlur(radius=r))
        blurred_img[i, :, :] = np.array(blurred_channel)
    return blurred_img

class Uint16ToFloatTransform:
    def __call__(self, tensor):
        return tensor.float()

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

def load_pikle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# 选择空闲GPU
def select_GPU(threshold_ratio=0.5):
    available_gpus = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        for i in range(num_gpus):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = total_memory - allocated_memory

            # 计算空闲内存比例
            free_ratio = free_memory / total_memory
            if free_ratio >= threshold_ratio:
                available_gpus.append(i)

    if len(available_gpus) > 1:
        device = torch.device(f"cuda:{available_gpus[0]}")
    elif len(available_gpus) == 1:
        device = torch.device(f"cuda:{available_gpus[0]}")
    else:
        device = torch.device("cpu")
    return device, available_gpus

# 失真
def add_gaussian_noise(image, distortion_level):
    """为图像添加高斯噪声"""
    image_array = np.array(image).astype(np.float32) / 255.0  # 转为 [0, 1] 范围
    mean = 0
    std_dev = distortion_level * 0.05  # 根据失真程度调整标准差
    noise = np.random.normal(mean, std_dev, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 1) * 255  # 加噪并恢复到 [0, 255]
    return Image.fromarray(noisy_image.astype(np.uint8))

def apply_motion_blur(image, distortion_level):
    """为图像添加运动模糊效果"""
    kernel_size = int(distortion_level * 20) + 1  # 根据失真程度调整卷积核大小
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # 确保为奇数
    return image.filter(ImageFilter.BoxBlur(kernel_size // 2))  # 使用 BoxBlur 模拟模糊

def apply_geometric_distortion(image, distortion_level):
    """为图像添加几何失真"""
    image_array = np.array(image)
    rows, cols, channels = image_array.shape

    # 定义仿射变换的源点和目标点
    distortion_factor = distortion_level * 0.05
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    dst_points = np.float32([
        [0, 0],
        [cols - 1 + np.random.uniform(-distortion_factor * cols, distortion_factor * cols), 0],
        [0, rows - 1 + np.random.uniform(-distortion_factor * rows, distortion_factor * rows)]
    ])

    # 计算仿射变换矩阵
    matrix = cv2.getAffineTransform(src_points, dst_points)
    distorted_array = cv2.warpAffine(image_array, matrix, (cols, rows))

    return Image.fromarray(distorted_array)

def distortion_image(image, level):
    # 添加高斯噪声
    noisy_image = add_gaussian_noise(image, level)
    # 添加运动模糊
    blurred_image = apply_motion_blur(noisy_image, level)
    # 添加几何失真
    distorted_image = apply_geometric_distortion(blurred_image, level)
    return distorted_image

# logger只打印主进程
def logger_print(rank, logger, msg):
    if rank is None:
        logger.info(msg)
        return

    if logger and rank == 0:
        logger.info(msg)
