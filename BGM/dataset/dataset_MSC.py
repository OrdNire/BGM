import os
import random
import re

import numpy as np
import torch.utils.data
from PIL import Image

import pickle as pkl
from utils.func import blur_image, blend_cloud_mask, sample_related_index
import pandas as pd
from tqdm import tqdm

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
from tqdm import tqdm

cloud_mask_dir = "/mnt/data/jiangfanzhuo/custom_cloud_mask"

class MSC(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, meta_path, data_path, noise_data_path, split, transform, meta_file):
        assert os.path.exists(
            meta_path), "Meta path '{}' not found".format(meta_path)
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._meta_path, self._data_path, self._split, self._transform = meta_path, data_path, split, transform
        self._meta_file = meta_file
        self._noise_data_path = noise_data_path
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
                im_fn = gnd["qimlist"][i] + ".jpg"
                im_path = os.path.join(
                    self._data_path, im_fn)
                self._db.append({"im_path": im_path, "class": gnd['qclasses'][i], "idx": i})

        elif self._split == 'gallery':
            for i in range(len(gnd["imlist"])):
                im_fn = gnd["imlist"][i] + ".jpg"
                im_path = os.path.join(
                    self._data_path, im_fn)
                self._db.append({"im_path": im_path, "class": gnd["imclasses"][i], "idx": i})

        elif self._split == 'db':
            for i in range(len(gnd["imlist"])):
                im_fn = gnd["imlist"][i] + ".jpg"
                im_path = os.path.join(
                    self._data_path, im_fn)
                self._db.append({"im_path": im_path, "class": gnd["imclasses"][i],  "idx": i})

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
            im = Image.open(self._db[idx]["im_path"])
            im = im.convert("RGB")
        except:
            print('error: ', self._db[idx]["im_path"])
        im = self._transform(im)
        return im

    def __getitem__(self, index):
        # Load the image
        batch = {}
        try:
            im = Image.open(self._db[index]["im_path"])
            im = im.convert("RGB")
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
        batch["onehot_label"] = np.asarray(self._db[index]['class']['onehot_label']).astype(float)
        return batch


    def __len__(self):
        return len(self._db)

class MSC_BLIP(torch.utils.data.Dataset):
    def __init__(self, data_path, processor):
        self.annotation_file = os.path.join(data_path, "labels.csv")
        self.image_path = os.path.join(data_path, "images/images")
        self._data_path = data_path
        self.processor = processor
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        self._db = []
        multi_label_df = pd.read_csv(self.annotation_file, header=0, index_col=0)
        self.label2idx = {label: idx for (idx, label) in enumerate(multi_label_df.columns)}
        self.idx2label = {idx: label for (idx, label) in enumerate(multi_label_df.columns)}
        for file in tqdm(os.listdir(self.image_path)):
            img = Image.open(os.path.join(self.image_path, file)).convert("RGB")
            encoding = self.processor(images=img, padding="max_length", return_tensors="pt")
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            encoding["label"] = [self.idx2label[idx] for idx, v in enumerate(multi_label_df.loc[str(file).split('.')[0]].tolist()) if v == 1]
            encoding["filename"] = file
            self._db.append(encoding)
            img.close()

    def __getitem__(self, index):
        return self._db[index]

    def __len__(self):
        return len(self._db)

def MSC_collate_fn_blip(batch, processor):
    processed_batch = {}
    for key in batch[0].keys():
        if key == "filename":
            processed_batch[key] = [example[key] for example in batch]
        elif key != "label":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            prompt = "includes the words:"
            batch_text = [prompt + ','.join(example["label"]) + '.' for example in batch]
            text_inputs = processor.tokenizer(
                batch_text, padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
            processed_batch["label"] = [example[key] for example in batch]
    return processed_batch