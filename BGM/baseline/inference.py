import os
import sys
import random
import torch
import numpy as np
import argparse
import pickle
from collections import OrderedDict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)
from config_args import get_args_baseAR_infer
from utils.Log import Logger
from load_data import get_data
from models.base.model import BaseAR
from models.CLIP.clip import clip
from engine import baseAR_inference, baseAR_inference_for166

args = get_args_baseAR_infer(argparse.ArgumentParser())

device = torch.device("cuda:0")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # information
    logger = Logger(args.name, dir_path=args.model_dir)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info('Labels: {}'.format(args.num_labels))

    # dataset
    query_loader, gallery_loader, train_gallery_loader, db_loader = get_data(args)
    logger.info(f"train gallery size: {len(train_gallery_loader.dataset)}")
    logger.info(f"query size: {len(query_loader.dataset)}")

    # load code, cfg
    with open(os.path.join(args.meta_root, args.dataset, args.code_file), 'rb') as f:
        code = pickle.load(f)
    mapping = code["mapping"]
    with open(os.path.join(args.meta_root, args.dataset, args.meta_file), 'rb') as f:
        cfg = pickle.load(f)

    id_length = mapping.shape[-1]
    voc_length = np.unique(mapping).shape[0]
    cfg["voc_length"] = voc_length

    # load model
    model = BaseAR(dec_depth=12, num_classes=voc_length, id_len=id_length)
    mm, preprocess = clip.load('ViT-B/16')
    mm = mm.to('cpu')
    mm = mm.type(torch.float32)
    model.encoder = mm.visual

    state = torch.load(args.model_path)

    # 处理 DataParallel 前缀
    state_dict = state['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove `module.` prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    model = model.to(device)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Total parameters: {model_params:.2f}M")

    # inference
    baseAR_inference_for166(args, model, query_loader, logger, code, cfg, device)