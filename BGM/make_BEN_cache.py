from dataset.dataset_BEN import BEN_cache, get_BigEarthNet
import os
import torch
from tqdm import tqdm
import numpy as np
import pickle

if __name__ == '__main__':

    # tag = '_subset_'
    # split = "db"
    # npmem_file = "data/BigEarthNet-19" + '/' + "BEN" + tag + split + '.dat'
    # noise_npmem_file = "data/BigEarthNet-19" + '/' + "BEN" + tag + "noise_" + split + '.dat'
    #
    # meta_path = "/home/jiangfanzhuo/SpaceIR/data/BigEarthNet-19"
    # data_path = "/home/jiangfanzhuo/SpaceIR/data/BigEarthNet-19/BigEarthNet-19/BigEarthNet-S2"
    # meta_file = "BEN_sub_retrieval.pkl"
    #
    # if os.path.exists(npmem_file) == False:
    #     # create npmem file
    #     print("Start to create " + npmem_file + "\n")
    #     count = np.prod([12,120,120])
    #     dataset = BEN_cache(meta_path, data_path, split, meta_file)
    #     dl = torch.utils.data.DataLoader(
    #         dataset,
    #         num_workers=32,
    #         shuffle=False,
    #         pin_memory=False,
    #         batch_size=128,
    #         drop_last=False
    #     )
    #     n = len(dl.dataset._db)
    #     dtype = 'uint16'
    #     fp = np.memmap(npmem_file, dtype=dtype, mode='w+', shape=(n, count))
    #     if split == "db":
    #         noise_fp = np.memmap(noise_npmem_file, dtype=dtype, mode='w+', shape=(n, count))
    #     noise_level_dict = {}
    #     for batch in tqdm(dl):
    #         img_data = batch["im"]
    #         indices = batch["idx"]
    #         if split == "db":
    #             noise_img_data = batch["noise_im"]
    #             noise_level = batch["noise_level"]
    #         if type(img_data) == torch.Tensor:
    #             img_data = img_data.numpy().astype(dtype)
    #         for cur_i, i in enumerate(indices):
    #             fp[i, :] = img_data[cur_i].reshape(-1)
    #             if split == "db":
    #                 noise_fp[i, :] = noise_img_data[cur_i].reshape(-1)
    #                 noise_level_dict[i.item()] = float(noise_level[cur_i].item())
    #         fp.flush()
    #         if split == "db":
    #             noise_fp.flush()
    #
    #     if split == "db":
    #         print(noise_level_dict)
    #         with open(f'BEN_subset_db.pkl', 'wb') as f:
    #             pickle.dump(noise_level_dict, f)
    #
    #     # check if data is correct
    #     flag_create_npmem = True
    #     byte = 2 if dtype == 'uint16' else 1
    #     check_inds = np.random.randint(low=0, high=n, size=5)
    #     for index in range(n):
    #         im = np.fromfile(npmem_file, count=count, dtype=dtype, offset=index * byte * count)
    #         if im.shape[0] != count:
    #             flag_create_npmem = False
    #             break
    #         if index in check_inds:
    #             im_tmp = get_BigEarthNet(dl.dataset._db[index]["im_path"]).reshape(-1)
    #             if np.sum(im_tmp - im) != 0:
    #                 flag_create_npmem = False
    #                 break
    #     if flag_create_npmem:
    #         print("Create " + npmem_file + " success!\n")
    #     else:
    #         os.remove(npmem_file)
    #         print("Create " + npmem_file + " failed, please run it again!\n")

##      way 2:
    tag = '_subset_'
    split = "db"

    meta_path = "/home/jiangfanzhuo/SpaceIR/data/BigEarthNet-19"
    data_path = "/home/jiangfanzhuo/SpaceIR/data/BigEarthNet-19/BigEarthNet-19/BigEarthNet-S2"
    meta_file = "BEN_sub_retrieval.pkl"

    count = np.prod([12,120,120])
    dataset = BEN_cache(meta_path, data_path, split, meta_file)
    dl = torch.utils.data.DataLoader(
        dataset,
        num_workers=32,
        shuffle=False,
        pin_memory=False,
        batch_size=128,
        drop_last=False
    )
    n = len(dl.dataset._db)
    dtype = 'uint16'
    for batch in tqdm(dl):
        indices = batch["idx"]
        for cur_i, i in enumerate(indices):
            img_state = {}
            img_state["im"] = batch["im"][cur_i].numpy().reshape(-1)
            if split == "db":
                img_state["noise_im"] = batch["noise_im"][cur_i].numpy().reshape(-1)
                img_state["noise_level"] = batch["noise_level"][cur_i].item()

            img_path = dl.dataset._db[i.item()]["im_path"]
            patch_name = img_path.split('/')[-1]
            save_path = img_path + '/' + patch_name + '.pth'
            torch.save(img_state, save_path)