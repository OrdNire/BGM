import torch
import numpy as np
import os
import random
from PIL import Image

class CGCommon_T2I_L(torch.utils.data.Dataset):

    def __init__(self, image_path, datas, split, transform, datas2=None, noise_rate=4):
        self.image_path, self.datas, self.split = image_path, datas, split
        self.transform = transform
        self.datas2 = datas2
        self.noise_rate = noise_rate
        assert noise_rate <= 4, "invalid noise rate."
        self._construct_db()

    def _construct_db(self):
        self.db = []
        # for noise
        self.imgid2level = {}
        if self.split == "query":
            for idx, data in enumerate(self.datas):
                label_onehot = np.asarray(data["label"]["onehot"])
                for cap in data["captions"]:
                    self.db.append({"cap": cap, "label": label_onehot, "idx": idx})
        elif self.split == "db":
            for idx, data in enumerate(self.datas):
                im_fn = os.path.join(self.image_path, data["filename"])
                label_onehot = np.asarray(data["label"]["onehot"])
                self.db.append({
                    "im": im_fn, "cap_list": data["captions"], "label": label_onehot,
                    "noise_files": data["noise_file"], "idx": idx
                })
        elif self.split == "noise_db":
            n = 0
            self.noise_img_idx_list = []
            for idx, data in enumerate(self.datas):
                im_fn = os.path.join(self.image_path, data["filename"])
                self.db.append({
                    "im": im_fn, "level": 0, "idx": n, "label": np.asarray(data["label"]["onehot"])
                })
                self.imgid2level[n] = 0
                n += 1
                self.noise_img_idx_list.append(idx)
                for cnt, noise_info in enumerate(data["noise_file"]):
                    if cnt >= self.noise_rate:
                        break
                    noise_fn = os.path.join(self.image_path, noise_info["filename"])
                    self.db.append({
                        "im": noise_fn, "level": noise_info["level"], "idx": n, "label": np.asarray(data["label"]["onehot"])
                    })
                    self.imgid2level[n] = noise_info["level"]
                    n += 1
                    self.noise_img_idx_list.append(idx)
        elif self.split == "memory":
            image_id = 0
            for idx, data in enumerate(self.datas):
                im_fn = os.path.join(self.image_path, data["filename"])
                label_onehot = np.asarray(data["label"]["onehot"])
                self.db.append({
                    "im": im_fn, "label": label_onehot, "idx": image_id
                })
                image_id += 1
            for idx, data in enumerate(self.datas2):
                im_fn = os.path.join(self.image_path, data["filename"])
                label_onehot = np.asarray(data["label"]["onehot"])
                self.db.append({
                    "im": im_fn, "label": label_onehot, "idx": image_id
                })
                image_id += 1

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
            im = self._load_image(data["im"])
            batch["im"] = self.transform(im)
            noise_info = random.choice(data["noise_files"])
            noise_im = self._load_image(os.path.join(self.image_path, noise_info["filename"]))
            batch["noise_im"] = self.transform(noise_im)
            batch["noise_level"] = noise_info["level"]
            batch["cap"] = random.choice(data["cap_list"])
            batch["label"] = data["label"]
            batch["idx"] = data['idx']
        elif self.split == "noise_db":
            im = self._load_image(data["im"])
            batch["im"] = self.transform(im)
            batch["level"] = data["level"]
            batch["idx"] = data["idx"]
            batch["label"] = data["label"]
        elif self.split == "memory":
            im = self._load_image(data["im"])
            batch["im"] = self.transform(im)
            batch["label"] = data["label"]
            batch["idx"] = data['idx']
        return batch

    def __len__(self):
        return len(self.db)

class CGCommon_T2I(torch.utils.data.Dataset):

    def __init__(self, image_path, datas, split, transform, pre_transform=None, noise_rate=4):
        self.image_path, self.datas, self.split = image_path, datas, split
        self.pre_transform = pre_transform
        self.transform = transform
        self.noise_rate = noise_rate
        assert noise_rate <= 20, "invalid noise rate."
        self._construct_db()

    def _construct_db(self):
        self.db = []

        # for noise
        self.imgid2level = {}
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
        elif self.split == "noise_db":
            n = 0
            self.noise_img_idx_list = []
            for idx, data in enumerate(self.datas):
                im_fn = os.path.join(self.image_path, data["filename"])
                im = self._load_image(im_fn)
                if self.pre_transform is not None:
                    im = self.pre_transform(im)
                self.db.append({
                    "im": im, "level": 0, "idx": n
                })
                self.imgid2level[n] = 0
                n += 1
                self.noise_img_idx_list.append(idx)
                for cnt, noise_info in enumerate(data["noise_file"]):
                    if cnt >= self.noise_rate:
                        break
                    noise_fn = os.path.join(self.image_path, noise_info["filename"])
                    noise_im = self._load_image(noise_fn)
                    if self.pre_transform is not None:
                        noise_im = self.pre_transform(noise_im)
                    self.db.append({
                        "im": noise_im, "level": noise_info["level"], "idx": n
                    })
                    self.imgid2level[n] = noise_info["level"]
                    n += 1
                    self.noise_img_idx_list.append(idx)

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
        elif self.split == "noise_db":
            batch["im"] = self.transform(data["im"])
            batch["level"] = data["level"]
            batch["idx"] = data["idx"]
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