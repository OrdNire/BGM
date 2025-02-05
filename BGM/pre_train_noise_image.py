from dataset.dataset_DLRSD import DLRSD
import pickle as pkl
import os
from PIL import Image
from utils.func import blur_image, blend_cloud_mask, sample_related_index
import re
import random
import numpy as np
import torch
from tqdm import tqdm

cloud_mask_dir = "/mnt/data/jiangfanzhuo/custom_cloud_mask"

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def split_filename(filename):
    # 使用正则表达式查找第一个数字的位置
    match = re.search(r'(\d+)', filename)
    if match:
        index = match.start()
        # 分割文件名
        name_part = filename[:index]
        number_part = filename[index:]
        return name_part, number_part
    else:
        # 如果没有找到数字，返回原文件名
        return filename, ''

def construct_noise(img):
    # 合成噪声
    noise_type = random.random()
    if noise_type < 0.5:
        blur_ratio = random.uniform(1, 8)
        noise_img = blur_image(img, r=blur_ratio)
        noise_level = blur_ratio / 8.0
    else:
        mask_num = random.randint(0, 9999)
        cloud_ratio = random.uniform(0, 1)
        while cloud_ratio == 0:
            cloud_ratio = random.uniform(0, 1)
        noise_level = cloud_ratio
        mask = Image.open(os.path.join(cloud_mask_dir, f"cloud_{mask_num}.png"))
        noise_img = blend_cloud_mask(img, mask, alpha_factor=cloud_ratio)
    return noise_img, noise_level

if __name__ == '__main__':
    data_root = "./data/"
    meta_file = "DLRSD_retrieval_noise0.pkl"
    save_path = "/mnt/data/jiangfanzhuo/train_noise_data/DLRSD"

    meta_dir = os.path.join(data_root, "DLRSD")
    data_dir = os.path.join(data_root, "DLRSD/DLRSD", "UCMerced_LandUse/UCMerced_LandUse/Images")
    with open(os.path.join(meta_dir, meta_file), 'rb') as fin:
        gnd = pkl.load(fin)

    for i in tqdm(range(len(gnd["imlist"]))):
        im_fn = gnd["imlist"][i]
        name_part, num_part = split_filename(im_fn)
        im_path = os.path.join(
            data_dir, name_part, im_fn)
        im = Image.open(im_path)
        im = im.convert("RGB")
        noise_img, noise_level = construct_noise(im)
        img_save_dir = os.path.join(save_path, name_part)
        os.makedirs(img_save_dir, exist_ok=True)
        noise_img_tensor = np.array(noise_img)
        data_to_save = {
            'noise_img': noise_img_tensor,  # 噪声图像
            'noise_level': noise_level  # 噪声级别
        }
        torch.save(data_to_save, os.path.join(img_save_dir, im_fn.replace('.tif', '.pth')))
