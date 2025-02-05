import argparse, numpy as np
import copy
import os.path

import torch
import torch.nn as nn
import random
import pickle

from utils.Log import Logger
from utils.func import select_GPU
from config_args import get_args_infer_SIRMM
from load_data_MM import get_data
from models.SpaceIR.SIR_MM import build_SIRMM
from engine import SIRMM_inference, SIRMM_inferenceV2, SIRMM_eval, SIRMM_eval_CGNoise

args = get_args_infer_SIRMM(argparse.ArgumentParser())

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
    logger = Logger(args.model_name, dir_path=".")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info('Labels: {}'.format(args.num_labels))

    # Get Device
    args.device, args.available_gpus = select_GPU()
    if len(args.available_gpus) >= 1:
        logger.info(f"User GPUs: {args.available_gpus}")
    else:
        logger.info(f"No free GPUs.")
        exit(0)

    # args.device = torch.device("cpu")

    # cfg
    # with open(os.path.join(args.meta_root, args.dataset, args.meta_file), 'rb') as f:
    #     cfg = pickle.load(f)
    args.voc_length = args.num_labels*args.distance_prc+1

    # dataset
    args.is_inference = True
    query_loader, gallery_loader = get_data(args)

    # 读取test_matrix
    args.map_matrix = np.load(os.path.join(args.dataroot, args.dataset, "map_matrix.npy"))
    args.ndcg_matrix = np.load(os.path.join(args.dataroot, args.dataset, "ndcg_matrix.npy"))

    # label embedding
    label_embedding = torch.load(os.path.join(args.meta_root, args.dataset, "label_embedding.pt"))

    # model
    model = build_SIRMM(args, label_embedding)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Total parameters: {model_params:.2f}M")

    if len(args.available_gpus) > 1:
        logger.info(f"DataParallel.")
        model = nn.DataParallel(model, device_ids=args.available_gpus).to(args.device)

    logger.info("load retrieval model weight....")
    state = torch.load(os.path.join(args.model_path, "retrieval_model.pth"))
    # print(f"best mR: {state['mR']}")
    model.load_state_dict(state["state_dict"], strict=False)

    model = model.to(args.device)

    ###################################### Inference ####################################
    if args.test_noise:
        mV = SIRMM_eval_CGNoise(args, model.module, query_loader, gallery_loader, logger)
        # mV = SIRMM_eval_CGNoise(args, model, query_loader, gallery_loader, logger)
    else:
        mAP = SIRMM_eval(args, model.module, query_loader, gallery_loader, logger, stage="inference")
