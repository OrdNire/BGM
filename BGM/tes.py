import faiss
import numpy as np
import torch
import random

from utils.Log import Logger
from config_args import get_args_train_SIR
from load_data import get_data
from models.SpaceIR.SIR import SIR
from tqdm import tqdm
import argparse
import os
import torch.nn as nn
args = get_args_train_SIR(argparse.ArgumentParser())

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


logger = Logger(args.model_name, dir_path=".")
for arg, value in vars(args).items():
    logger.info(f"{arg}: {value}")
logger.info('Labels: {}'.format(args.num_labels))

# dataset
query_loader, gallery_loader, train_gallery_loader, db_loader = get_data(args)

# model
model = SIR(id_len=args.num_labels, voc_length=args.num_labels*args.distance_prc+1, num_labels=args.num_labels, distance_prc=args.distance_prc,
            enc_depth=args.layers, dec_depth=args.layers, num_heads=args.heads, drop_rate=args.dropout)

model_params = sum(p.numel() for p in model.parameters()) / 1e6
logger.info(f"Total parameters: {model_params:.2f}M")

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.cuda()

# label embedding
label_embedding = torch.load(os.path.join(args.meta_root, args.dataset, "label_embedding.pt"))

with torch.no_grad():
    for i, batch in enumerate(tqdm(db_loader)):
        img = batch["im"].float().cuda()
        expand_label_embedding = label_embedding.unsqueeze(0).repeat(img.shape[0], 1, 1)
        feature = model.module.forward(img, tgt=None, label_embedding=expand_label_embedding.cuda().detach(),
                                       stage="tokenizer")  # tgt: (num_labels, 2)
        if i == 0:
            feats = feature.cpu()
        else:
            feats = torch.cat((feats, feature.cpu()), 0)
    feats = feats.numpy()

# 假设你有一些特征数据，形状为 (N, dim)
N = feats.shape[0]  # 示例中的样本数量
dim = feats.shape[1]  # 示例中的特征维度
data = np.random.random((N, dim)).astype(np.float32)

dim = data.shape[1]
m = 4
k = 8
pq = faiss.ProductQuantizer(dim, 4, 2)
x_q=[]

pq.train(feats)
codes = pq.compute_codes(feats)
print(codes, codes.shape)

