import argparse, numpy as np
import copy
import os.path

import torch
import torch.nn as nn
import random

from utils.Log import Logger
from config_args import get_args_train_SIR2
from load_data import get_data
from models.CTrans.CTran import CTranModel
from models.SpaceIR.SIR_exp2 import SIR
from engine import phase1_train_SIR2, phase1_eval_SIR2, phase2_train_SIR2
from custom_loss import LabelContrastiveLoss, LabelSmoothingCrossEntropy, FocalLoss, AsymmetricLossOptimized, GlobalContrastiveLoss, GlobalContrastiveLossV2
import scheduler
import torch.distributed as dist
import pickle

args = get_args_train_SIR2(argparse.ArgumentParser())

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

if __name__ == '__main__':
    # information
    logger = Logger(args.model_name, dir_path=".")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info('Labels: {}'.format(args.num_labels))

    # dataset
    query_loader, gallery_loader, train_gallery_loader, db_loader = get_data(args)

    # label embedding
    label_embedding = torch.load(os.path.join(args.meta_root, args.dataset, "label_embedding.pt"))

    # model
    model = SIR(num_labels=args.num_labels, distance_prc=args.distance_prc,
            enc_depth=args.layers, num_heads=args.heads, drop_rate=args.dropout, embed_dim=args.hidden_dim,
                label_embedding=label_embedding.detach())

    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Total parameters: {model_params:.2f}M")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    else:
        print(f"Using Single GPU! on {args.cuda}")

    model = model.cuda()
    model_to_use = model.module if hasattr(model, 'module') else model

    # criterion
    contrastive_criterion = LabelContrastiveLoss(label_embedding=label_embedding.cuda().detach())
    ce_loss = nn.CrossEntropyLoss()

    if args.class_loss == "FL":
        classifier_criterion = FocalLoss()
    elif args.class_loss == "ASL":
        classifier_criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    else:
        classifier_criterion = torch.nn.BCEWithLogitsLoss()
    criterion_dic = {"contrastive_criterion": contrastive_criterion,
                     "classifier_criterion": classifier_criterion,
                     "ce_loss":ce_loss}

    # criterion_dic = {"contrastive_criterion": contrastive_criterion,
    #                  "classifier_criterion": classifier_criterion,
    #                  "seq2seq_criterion": seq2seq_criterion}

    ###################################### Train ####################################

    ################### Phase1 #################
    if args.classification_resume:
        logger.info("load classification stage weight....")
        state = torch.load(args.classification_path)
        model.load_state_dict(state["state_dict"])
    else:
        # frozen decoder
        for param in model_to_use.decoder.parameters():
            param.requires_grad = False

        # optimizer
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=args.lr)  # , weight_decay=0.0004)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                        weight_decay=1e-4)

        lr_scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                                   t_initial=int(args.classification_epochs),
                                                   lr_min=1e-7,
                                                   warmup_t=int(args.classification_epochs * 0.1),
                                                   warmup_lr_init=1e-7,
                                                   cycle_decay=1
                                                   )

        best_mAP = 0
        best_epoch = 0
        for epoch in range(1, args.classification_epochs + 1):
            logger.info('======================== Classification {} ========================'.format(epoch))
            for param_group in optimizer.param_groups:
                logger.info('LR: {}'.format(param_group['lr']))

            batch_loss = phase1_train_SIR2(args, model, db_loader, optimizer, epoch, logger,
                            criterion_dic=criterion_dic,
                            scheduler=lr_scheduler)
            lr_scheduler.step(epoch)

            if (epoch >= 40) and (epoch % 10 == 0):
                mAP = phase1_eval_SIR2(args, model, query_loader)
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_epoch = epoch
                    logger.info(f"best mAP:{best_mAP} epoch: {best_epoch} save classification model...")
                    save_dict = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        "mAP": mAP,
                        'classification_loss': batch_loss
                    }
                    torch.save(save_dict, args.model_name + '/classification_model.pth')
                else:
                    logger.info(f"cur mAP: {mAP} best mAP:{best_mAP} epoch: {best_epoch}")
                if (epoch - best_epoch) >= 20:
                    logger.info(f"Epoch:[{epoch}/{args.classification_epochs + 1}] early stop.")
                    break

    ################### Retrieval #################
    indexing_model = copy.deepcopy(model)

    # optimizer
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr)  # , weight_decay=0.0004)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                    weight_decay=1e-4)

    lr_scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                            t_initial=int(args.retrieval_epochs),
                                            lr_min=1e-7,
                                            warmup_t=int(args.retrieval_epochs * 0.1),
                                            warmup_lr_init=1e-7,
                                            cycle_decay=1
                                            )

    best_loss = float("inf")
    for epoch in range(1, args.retrieval_epochs + 1):
        logger.info('======================== Retrieval {} ========================'.format(epoch))
        for param_group in optimizer.param_groups:
            logger.info('LR: {}'.format(param_group['lr']))

        batch_loss = phase2_train_SIR2(args, model, indexing_model, db_loader, optimizer, epoch, logger,
                            criterion_dic=criterion_dic,
                            scheduler=lr_scheduler)
        lr_scheduler.step(epoch)

        if (epoch % 20 == 0) and (batch_loss < best_loss):
            best_loss = batch_loss
            logger.info(f"save retrieval model... loss: {best_loss}")
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'retrieval_loss': best_loss
            }
            torch.save(save_dict, args.model_name+'/retrieval_model.pth')
