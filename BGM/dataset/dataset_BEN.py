import os
import random
import re

import numpy as np
import torch.utils.data
from PIL import Image
from skimage.transform import resize
from osgeo import gdal

import pickle as pkl

from utils.func import blur_image, blend_cloud_mask, sample_related_index, BEN_blur_image, BEN_blend_cloud_mask
# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
MAX_VALUE = 20566

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

def get_BigEarthNet(img_path) -> np.ndarray:
    """Get image data from BigEarthNet dataset
    Args:
        img_path (str): image path
    Returns:
        numpy: flatten image data
    """
    patch_name = img_path.split('/')[-1]
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    # get band image data
    tif_img = []
    for band_name in band_names:
        tif_path = img_path + '/' + patch_name + '_' + band_name + '.tif'
        band_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
        if not band_ds:
            print("can't open " + patch_name)
            break
        raster_band = band_ds.GetRasterBand(1)
        band_data = np.array(raster_band.ReadAsArray(), dtype='uint16')
        # interpolate the image to (120,120)
        if band_data.shape[0] != 120:
            band_data = resize(band_data, (120, 120), preserve_range=True, order=3)
        tif_img.append(band_data)
    tif_img = np.array(tif_img, dtype='uint16')
    return tif_img

class BEN(torch.utils.data.Dataset):
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
                    parts = str(im_fn).rsplit('_', 2)
                    im_path = os.path.join(
                        self._data_path, parts[0], im_fn)
                    # import pdb
                    # pdb.set_trace()
                    self._db.append({"im_path": im_path, "qclass": gnd['qclasses'][i], "idx": i})

            elif self._split == 'gallery':
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    parts = str(im_fn).rsplit('_', 2)
                    im_path = os.path.join(
                        self._data_path, parts[0], im_fn)
                    self._db.append({"im_path": im_path, "class": gnd["imclasses"][i], "idx": i})

            elif self._split == 'db':
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    parts = str(im_fn).rsplit('_', 2)
                    im_path = os.path.join(
                        self._data_path, parts[0], im_fn)
                    self._db.append({"im_path": im_path, "class": gnd["imclasses"][i], "idx": i})

    def _load_img(self, idx):
        patch_name = self._db[idx]["im_path"].split('/')[-1]
        img_path = self._db[idx]["im_path"] + '/' + patch_name + ".pth"
        im_state = torch.load(img_path)
        return im_state

    def __getitem__(self, index):
        # Load the image
        batch = {}
        try:
            im_state = self._load_img(index)
        except:
            print('error: ', self._db[index]["im_path"])

        if self._split == "db":
            noise_im = im_state["noise_im"].reshape([12, 120, 120])
            noise_level = im_state["noise_level"] / MAX_VALUE
            noise_im = self._transform(torch.from_numpy(noise_im))
            batch["noise_im"] = noise_im
            batch["noise_level"] = noise_level

            target_idx = sample_related_index(self.match_matrix, index)
            tgt_im_state = self._load_img(target_idx)
            tgt_im = tgt_im_state["im"].reshape([12, 120, 120]) / MAX_VALUE
            batch["tgt_im"] = self._transform(torch.from_numpy(tgt_im))
            batch["tgt_idx"] = target_idx
            batch["tgt_onehot_label"] = np.asarray(self._db[target_idx]['class']['onehot_label']).astype(float)

        im = im_state["im"].reshape([12, 120, 120]) / MAX_VALUE
        im = self._transform(torch.from_numpy(im))
        batch["im"] = im
        batch["idx"] = self._db[index]['idx']
        if "class" in self._db[index].keys():
            batch["onehot_label"] = np.asarray(self._db[index]['class']['onehot_label']).astype(float)
        else:
            batch["onehot_label"] = np.asarray(self._db[index]['qclass']['onehot_label']).astype(float)
        return batch

    def __len__(self):
        return len(self._db)

class BEN_cache(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, meta_path, data_path, split, meta_file):
        assert os.path.exists(
            meta_path), "Meta path '{}' not found".format(meta_path)
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._meta_path, self._data_path, self._split = meta_path, data_path, split
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
                    parts = str(im_fn).rsplit('_', 2)
                    im_path = os.path.join(
                        self._data_path, parts[0], im_fn)
                    # import pdb
                    # pdb.set_trace()
                    self._db.append({"im_path": im_path, "qclass": gnd['qclasses'][i], "idx": i})

            elif self._split == 'gallery':
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    parts = str(im_fn).rsplit('_', 2)
                    im_path = os.path.join(
                        self._data_path, parts[0], im_fn)
                    self._db.append({"im_path": im_path, "class": gnd["imclasses"][i], "idx": i})

            elif self._split == 'db':
                for i in range(len(gnd["imlist"])):
                    im_fn = gnd["imlist"][i]
                    parts = str(im_fn).rsplit('_', 2)
                    im_path = os.path.join(
                        self._data_path, parts[0], im_fn)
                    self._db.append({"im_path": im_path, "class": gnd["imclasses"][i], "idx": i})



    def _construct_noise(self, img):
        # 合成噪声
        noise_type = random.random()
        if noise_type:
            blur_ratio = random.uniform(1, 8)
            noise_img = BEN_blur_image(img, r=blur_ratio)
            noise_level = blur_ratio / 8.0
        # if noise_type < 0.5:
        #     blur_ratio = random.uniform(1, 8)
        #     noise_img = BEN_blur_image(img, r=blur_ratio)
        #     noise_level = blur_ratio / 8.0
        # else:
        #     mask_num = random.randint(0, 9999)
        #     cloud_ratio = random.uniform(0, 1)
        #     while cloud_ratio == 0:
        #         cloud_ratio = random.uniform(0, 1)
        #     noise_level = cloud_ratio
        #     mask = Image.open(os.path.join(cloud_mask_dir, f"cloud_{mask_num}.png"))
        #     noise_img = BEN_blend_cloud_mask(img, mask, alpha_factor=cloud_ratio)
        return noise_img.astype('uint16'), noise_level


    def __getitem__(self, index):
        # Load the image
        batch = {}
        try:
            im = get_BigEarthNet(self._db[index]["im_path"])
        except:
            print('error: ', self._db[index]["im_path"])

        # if self._split == "db":
        #     noise_im, noise_level = self._construct_noise(im)
        #     batch["noise_im"] = noise_im
        #     batch["noise_level"] = noise_level

        if self._split == "db":
            noise_im, noise_level = self._construct_noise(im)
            batch["noise_im"] = noise_im
            batch["noise_level"] = noise_level

        batch["im"] = im
        batch["idx"] = self._db[index]['idx']
        if "class" in self._db[index].keys():
            batch["onehot_label"] = np.asarray(self._db[index]['class']['onehot_label']).astype(float)
        else:
            batch["onehot_label"] = np.asarray(self._db[index]['qclass']['onehot_label']).astype(float)
        return batch

    def __len__(self):
        return len(self._db)
