import os
import random
import re
import pandas as pd

import numpy as np
import torch.utils.data
from PIL import Image

import pickle as pkl

from utils.func import blur_image, blend_cloud_mask, sample_related_index, load_pikle
# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

cloud_mask_dir = "/mnt/data/jiangfanzhuo/custom_cloud_mask"
noise_db_dir = "/mnt/data/jiangfanzhuo/train_noise_data/DLRSD"

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

class DLRSD(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, meta_path, data_path, noise_data_path, split, transform, meta_file, pre_transform=None):
        assert os.path.exists(
            meta_path), "Meta path '{}' not found".format(meta_path)
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._meta_path, self._data_path, self._split, self._transform = meta_path, data_path, split, transform
        self._meta_file = meta_file
        self._pre_transform = pre_transform
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
                im_fn = gnd["qimlist"][i]
                name_part, num_part = split_filename(im_fn)
                im_path = os.path.join(
                    self._data_path, name_part, im_fn)
                # import pdb
                # pdb.set_trace()
                im = self._load_image(im_path)
                if self._pre_transform:
                    im = self._pre_transform(im)
                self._db.append({"im": im, "qclss": gnd['qclasses'][i], "idx": i})

        elif self._split == 'gallery':
            for i in range(len(gnd["gimlist"])):
                im_fn = gnd["gimlist"][i]
                name_part, num_part = split_filename(im_fn)
                if gnd["sim_count"][i] < 1.0:
                    im_path = os.path.join(
                        self._noise_data_path, name_part, im_fn
                    )
                else:
                    im_path = os.path.join(
                        self._data_path, name_part, im_fn)
                im = self._load_image(im_path)
                if self._pre_transform:
                    im = self._pre_transform(im)
                self._db.append({"im": im, "class": gnd["gclasses"][i], "idx": i})

        elif self._split == 'db':
            for i in range(len(gnd["imlist"])):
                im_fn = gnd["imlist"][i]
                name_part, num_part = split_filename(im_fn)
                im_path = os.path.join(
                    self._data_path, name_part, im_fn)
                im = self._load_image(im_path)
                noise_im, noise_level = self._construct_noise(im)
                if self._pre_transform:
                    im = self._pre_transform(im)
                    noise_im = self._pre_transform(noise_im)
                self._db.append({"im": im, "noise_im": noise_im, "noise_level": noise_level, "class": gnd["imclasses"][i], "idx": i})

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

    def _load_image(self, im_path):
        try:
            im = Image.open(im_path)
            im = im.convert("RGB")
        except:
            print('error: ', im_path)
        return im

    def __getitem__(self, index):
        # Load the image
        batch = {}
        im = self._db[index]["im"]

        if self._split == "db":
            noise_im, noise_level = self._db[index]["noise_im"], self._db[index]["noise_level"]
            noise_im = self._transform(noise_im)
            batch["noise_im"] = noise_im
            batch["noise_level"] = noise_level

            target_idx = sample_related_index(self.match_matrix, index)
            batch["tgt_im"] = self._transform(self._db[target_idx]["im"])
            batch["tgt_onehot_label"] = np.asarray(self._db[target_idx]['class']['onehot_label']).astype(float)

        im = self._transform(im)
        batch["im"] = im
        batch["idx"] = self._db[index]['idx']
        if "class" in self._db[index].keys():
            batch["onehot_label"] = np.asarray(self._db[index]['class']['onehot_label']).astype(float)
        else:
            batch["onehot_label"] = np.asarray(self._db[index]['qclss']['onehot_label']).astype(float)
        return batch

    def __len__(self):
        return len(self._db)


class DLRSD_W_noise(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, meta_path, data_path, noise_data_path, split, transform, meta_file, pre_transform=None):
        assert os.path.exists(
            meta_path), "Meta path '{}' not found".format(meta_path)
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._meta_path, self._data_path, self._split, self._transform = meta_path, data_path, split, transform
        self._meta_file = meta_file
        self._pre_transform = pre_transform
        self._noise_data_path = noise_data_path
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        self._db = []
        self.q_db = []
        with open(os.path.join(self._meta_path, self._meta_file), 'rb') as fin:
            gnd = pkl.load(fin)
        self.match_matrix = gnd["match_matrix"]
        if self._split == 'query':
            for i in range(len(gnd["qimlist"])):
                im_fn = gnd["qimlist"][i]
                name_part, num_part = split_filename(im_fn)
                im_path = os.path.join(
                    self._data_path, name_part, im_fn)
                # import pdb
                # pdb.set_trace()
                im = self._load_image(im_path)
                if self._pre_transform:
                    im = self._pre_transform(im)
                self.q_db.append({"im": im, "qclss": gnd['qclasses'][i], "idx": i})

        elif self._split == 'gallery':
            for i in range(len(gnd["gimlist"])):
                im_fn = gnd["gimlist"][i]
                name_part, num_part = split_filename(im_fn)
                if gnd["sim_count"][i] < 1.0:
                    im_path = os.path.join(
                        self._noise_data_path, name_part, im_fn
                    )
                else:
                    im_path = os.path.join(
                        self._data_path, name_part, im_fn)
                im = self._load_image(im_path)
                if self._pre_transform:
                    im = self._pre_transform(im)
                self.q_db.append({"im": im, "class": gnd["gclasses"][i], "idx": i})

        elif self._split == 'db':
            for i in range(len(gnd["gimlist"])):
                im_fn = gnd["gimlist"][i]
                name_part, num_part = split_filename(im_fn)
                if gnd["sim_count"][i] < 1.0: # Noise
                    im_path = os.path.join(
                        self._noise_data_path, name_part, im_fn
                    )

                    im = self._load_image(im_path)
                    if self._pre_transform:
                        im = self._pre_transform(im)
                else:
                    im_path = os.path.join(
                        self._data_path, name_part, im_fn)
                    im = self._load_image(im_path)
                    noise_im, noise_level = self._construct_noise(im)
                    if self._pre_transform:
                        im = self._pre_transform(im)
                        noise_im = self._pre_transform(noise_im)
                    self.q_db.append({"im": im, "noise_im": noise_im, "noise_level": noise_level, "class": gnd["gclasses"][i], "idx": i})
                    # query, query_noise,
                self._db.append({"im": im, "class": gnd["gclasses"][i], "idx": i, "noise_level": 1 - gnd["sim_count"][i]})

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

    def _load_image(self, im_path):
        try:
            im = Image.open(im_path)
            im = im.convert("RGB")
        except:
            print('error: ', im_path)
        return im

    def __getitem__(self, index):
        # Load the image
        batch = {}
        query_im = self.q_db[index]["im"]

        if self._split == "db":
            query_noise_im, query_noise_level = self.q_db[index]["noise_im"], self.q_db[index]["noise_level"]
            query_noise_im = self._transform(query_noise_im)
            batch["query_noise_im"] = query_noise_im
            batch["query_noise_level"] = query_noise_level

            target_idx = sample_related_index(self.match_matrix, index)
            batch["tgt_im"] = self._transform(self._db[target_idx]["im"])
            batch["tgt_noise_level"] = self._db[target_idx]["noise_level"]
            batch["tgt_onehot_label"] = np.asarray(self._db[target_idx]['class']['onehot_label']).astype(float)

        query_im = self._transform(query_im)
        batch["query_im"] = query_im
        batch["idx"] = self.q_db[index]['idx']
        if "class" in self.q_db[index].keys():
            batch["onehot_label"] = np.asarray(self.q_db[index]['class']['onehot_label']).astype(float)
        else:
            batch["onehot_label"] = np.asarray(self.q_db[index]['qclss']['onehot_label']).astype(float)
        return batch

    def __len__(self):
        return len(self.q_db)


class DLRSD_BLIP(torch.utils.data.Dataset):
    def __init__(self, data_path, processor):
        self.annotation_file = os.path.join(data_path, "multi-labels.xlsx")
        self.image_path = os.path.join(data_path, "UCMerced_LandUse/UCMerced_LandUse/Images")
        self._data_path = data_path
        self.processor = processor
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        self._db = []
        multi_label_df = pd.read_excel(self.annotation_file, header=0, index_col=0)
        self.label2idx = {label: idx for (idx, label) in enumerate(multi_label_df.columns)}
        self.idx2label = {idx: label for (idx, label) in enumerate(multi_label_df.columns)}
        for class_name in os.listdir(self.image_path):
            class_path = os.path.join(self.image_path, class_name)
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                for file in files:
                    img = Image.open(os.path.join(class_path, file)).convert("RGB")
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

def DLRSD_collate_fn_blip(batch, processor):
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

class DLRSD_CLIP(torch.utils.data.Dataset):
    def __init__(self, data_path, processor):
        self.image_path = os.path.join(data_path, "UCMerced_LandUse/UCMerced_LandUse/Images")
        self._data_path = data_path
        self.processor = processor
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        self._db = []
        for class_name in os.listdir(self.image_path):
            class_path = os.path.join(self.image_path, class_name)
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                for file in files:
                    img = Image.open(os.path.join(class_path, file)).convert("RGB")
                    encoding = self.processor(images=img, padding="max_length", return_tensors="pt")
                    encoding = {k: v.squeeze() for k, v in encoding.items()}
                    self._db.append({"encoding": encoding, "filename": file})
                    img.close()

    def __getitem__(self, index):
        return self._db[index]

    def __len__(self):
        return len(self._db)

class CGCommon_T2I(torch.utils.data.Dataset):

    def __init__(self, image_path, datas, split, transform, pre_transform=None):
        self.image_path, self.datas, self.split = image_path, datas, split
        self.pre_transform = pre_transform
        self.transform = transform
        self._construct_db()

    def _construct_db(self):
        self.db = []

        if self.split == "query":
            for idx, data in enumerate(self.datas):
                label_onehot = np.asarray(data["label"]["onehot"])
                for cap in data["captions"]:
                    self.db.append({"cap": cap, "label": label_onehot, "idx": idx})
        elif self.split == "db":
            for idx, data in enumerate(self.datas):
                im_fn = os.path.join(self.image_path, data["filename"])
                im = self._load_image(im_fn)
                if self.pre_transform is not None:
                    im = self.pre_transform(im)
                label_onehot = np.asarray(data["label"]["onehot"])
                self.db.append({
                    "im": im, "cap_list": data["captions"], "label": label_onehot,
                    "noise_files": data["noise_file"], "idx": idx
                })

    def _load_image(self, im_path):
        try:
            im = Image.open(im_path)
            im = im.convert("RGB")
        except:
            print('error: ', im_path)
        return im

    def __getitem__(self, index):
        # Load the image
        batch = {}

        data = self.db[index]
        if self.split == "query":
            batch = data
        elif self.split == "db":
            batch["im"] = self.transform(data["im"])
            noise_info = random.choice(data["noise_files"])
            noise_im = self._load_image(os.path.join(self.image_path, noise_info["filename"]))
            if self.pre_transform is not None:
                noise_im = self.pre_transform(noise_im)
            batch["noise_im"] = self.transform(noise_im)
            batch["noise_level"] = noise_info["level"]
            batch["cap"] = random.choice(data["cap_list"])
            batch["label"] = data["label"]
            batch["idx"] = data['idx']
        return batch

    def __len__(self):
        return len(self.db)

def collate_fn_T2I(batch, tokenizer):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "cap":
            processed_batch[key] = torch.stack([torch.tensor(example[key]) for example in batch], dim=0)
        else:
            tmp_dict = {}
            for example in batch:
                encodings = tokenizer(example[key], padding="max_length", truncation=True, return_tensors="pt", max_length=150)
                for k, v in encodings.items():
                    tmp_dict.setdefault(k, []).append(v.squeeze(0))
            for k, v in tmp_dict.items():
                tmp_dict[k] = torch.stack(v, dim=0)
            processed_batch["cap"] = tmp_dict
    return processed_batch

if __name__ == '__main__':
    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.insert(0, parent_dir)
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer, BertModel
    from functools import partial
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # RGB
    scale_size = 224
    tokenizer = BertTokenizer.from_pretrained("/home/jiangfanzhuo/SpaceIR/data/bert-base-uncased")
    pre_trainTransform = transforms.Compose([transforms.ToTensor(),
                                             normTransform])
    trainTransform = transforms.Compose([transforms.RandomResizedCrop((scale_size, scale_size)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(degrees=30)
                                         ])
    dataset = DLRSD_T2I("../data/DLRSD", "../data/DLRSD/DLRSD/UCMerced_LandUse/UCMerced_LandUse/Images", None, "db",
                        transform=trainTransform, meta_file="DLRSD_retrieval_noise0.pkl", pre_transform=pre_trainTransform)
    loader = DataLoader(dataset, batch_size=3, collate_fn=partial(collate_fn_T2I, tokenizer=tokenizer))
    for batch in loader:
        print(batch)
        print("done.")