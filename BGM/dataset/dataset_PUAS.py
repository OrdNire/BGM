import os
import random
import re

import numpy as np
import torch.utils.data
from PIL import Image

import pickle as pkl

from utils.func import blur_image, blend_cloud_mask, sample_related_index
# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

cloud_mask_dir = "/mnt/data/jiangfanzhuo/custom_cloud_mask"

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

class PlanetUAS(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, meta_path, data_path, split, transform, meta_file):
        assert os.path.exists(
            meta_path), "Meta path '{}' not found".format(meta_path)
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._meta_path, self._data_path, self._split, self._transform = meta_path, data_path, split, transform
        self._meta_file = meta_file
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        self._db = []
        with open(os.path.join(self._meta_path, self._meta_file), 'rb') as fin:
            gnd = pkl.load(fin)
            self.match_matrix = gnd["match_matrix"]
            if self._split == 'query':
                for i in range(len(gnd["qimlist"])):
                    im_fn = gnd["qimlist"][i]
                    im_path = os.path.join(
                        self._data_path, im_fn + ".jpg")
                    # import pdb
                    # pdb.set_trace()
                    self._db.append({"im_path": im_path, "qclass": gnd['qclasses'][i], "idx": i})

            elif self._split == 'gallery':
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    im_path = os.path.join(
                        self._data_path, im_fn + ".jpg")
                    self._db.append({"im_path": im_path, "class": gnd["imclasses"][i], "idx": i})

            elif self._split == 'db':
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    im_path = os.path.join(
                        self._data_path, im_fn + ".jpg")
                    self._db.append({"im_path": im_path, "class": gnd["imclasses"][i], "idx": i})

    def _construct_noise(self, img):
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

    def _load_tgt_img(self, idx):
        try:
            im = Image.fromarray(np.array(Image.open(self._db[idx]["im_path"]))[:, :, :3])
        except:
            print('error: ', self._db[idx]["im_path"])
        im = self._transform(im)
        return im

    def __getitem__(self, index):
        # Load the image
        batch = {}
        try:
            im = Image.fromarray(np.array(Image.open(self._db[index]["im_path"]))[:, :, :3])
        except:
            print('error: ', self._db[index]["im_path"])

        if self._split == "db":
            noise_im, noise_level = self._construct_noise(im)
            noise_im = self._transform(noise_im)
            batch["noise_im"] = noise_im
            batch["noise_level"] = noise_level

            target_idx = sample_related_index(self.match_matrix, index)
            batch["tgt_im"] = self._load_tgt_img(target_idx)
            batch["tgt_idx"] = target_idx
            batch["tgt_onehot_label"] = np.asarray(self._db[target_idx]['class']['onehot_label']).astype(float)


        im = self._transform(im)
        batch["im"] = im
        batch["idx"] = self._db[index]['idx']
        if "class" in self._db[index].keys():
            batch["onehot_label"] = np.asarray(self._db[index]['class']['onehot_label']).astype(float)
        else:
            batch["onehot_label"] = np.asarray(self._db[index]['qclass']['onehot_label']).astype(float)
        return batch

    def __len__(self):
        return len(self._db)