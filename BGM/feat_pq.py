import faiss
import argparse, numpy as np
import os.path

import torch
import torch.nn as nn
import random

from utils.Log import Logger
from config_args import get_args_train_SIR
from load_data import get_data
from models.CTrans.CTran import CTranModel
from models.SpaceIR.SIR import SIR
from engine import tokenizer_train, tokenizer_eval, SIR_classification_train, SIR_indexing_train, SIR_retrieval_train, SIR_tokenizer
from custom_loss import LabelContrastiveLoss, LabelSmoothingCrossEntropy
import scheduler
from tqdm import tqdm
import torch.distributed as dist

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

@torch.no_grad()
def get_features(data):
    for i, batch in enumerate(tqdm(data)):
        img = batch["im"].float().cuda()
        expand_label_embedding = label_embedding.unsqueeze(0).repeat(img.shape[0], 1, 1)
        feature = model.module.forward(img, tgt=None, label_embedding=expand_label_embedding.cuda().detach(),
                                       stage="tokenizer")  # tgt: (num_labels, 2)
        if i == 0:
            feats = feature.cpu()
        else:
            feats = torch.cat((feats, feature.cpu()), 0)

    feats = feats.numpy()
    return feats

if __name__ == '__main__':
    # information
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

    ################### Tokenizer #################

    logger.info("======================== Tokenizer ========================")
    model.eval()
    print(args.id_len, args.cluster_num)
    # feats = get_features(db_loader)
    feats = np.random.random((1680, 768)).astype(np.float32)

    dim = feats.shape[1]
    m = 4
    k = args.cluster_num
    pq = faiss.ProductQuantizer(dim, m, 256)
    x_q = []

    pq.train(feats)
    codes = pq.compute_codes(feats)
    print(codes)
    codebook = pq.compute_codes(feats)
    logger.info(f"codebook shape: {codebook.shape} codes: {codebook}")
    logger.info("save codebook ...")
    save_dict = {
        'codebook': codebook
    }
    torch.save(save_dict, args.model_name + '/codebook.pth')
    ####################################################
