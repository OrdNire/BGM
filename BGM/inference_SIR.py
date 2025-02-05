import argparse, numpy as np
import os.path

import torch
import torch.nn as nn
import random
import pickle

from utils.Log import Logger
from config_args import get_args_train_SIR, get_args_infer_SIR
from load_data import get_data
from models.CTrans.CTran import CTranModel
from models.SpaceIR.SIR import SIR
from engine import tokenizer_train, tokenizer_eval, SIR_retrieval_train, SIR_inference
from custom_loss import LabelContrastiveLoss, LabelSmoothingCrossEntropy
import scheduler

args = get_args_infer_SIR(argparse.ArgumentParser())

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

    # cfg
    with open(os.path.join(args.meta_root, args.dataset, args.meta_file), 'rb') as f:
        cfg = pickle.load(f)
    args.voc_length = args.num_labels*args.distance_prc+1

    # dataset
    query_loader, gallery_loader, train_gallery_loader, db_loader = get_data(args)

    # model
    model = SIR(id_len=args.id_len, voc_length=2**args.k_bit, num_labels=args.num_labels, distance_prc=args.distance_prc,
                enc_depth=args.layers, dec_depth=args.layers, num_heads=args.heads, drop_rate=args.dropout)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Total parameters: {model_params:.2f}M")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    logger.info("load codebook ....")
    codebook = torch.load(args.codebook_path)

    logger.info("load model weight....")
    state = torch.load(args.model_path)
    model.load_state_dict(state["state_dict"])

    model = model.cuda()

    # label embedding
    label_embedding = torch.load(os.path.join(args.meta_root, args.dataset, "label_embedding.pt"))

    # criterion
    contrastive_criterion = LabelContrastiveLoss()
    seq2seq_criterion = LabelSmoothingCrossEntropy()

    classifier_criterion = torch.nn.BCEWithLogitsLoss()
    criterion_dic = {"contrastive_criterion": contrastive_criterion,
                     "classifier_criterion": classifier_criterion,
                     "seq2seq_criterion":seq2seq_criterion}

    ###################################### Inference ####################################

    SIR_inference(args, model, label_embedding, query_loader, gallery_loader, codebook, logger, cfg)