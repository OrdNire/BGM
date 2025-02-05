import argparse, numpy as np
import os.path

import torch
import torch.nn as nn
import random

from utils.Log import Logger
from config_args import get_args_train_SIR
from load_data import get_data
from models.CTrans.CTran import CTranModel
from models.SpaceIR.SIR_exp1 import SIR
from engine import tokenizer_train, tokenizer_eval, SIR_classification_train, SIR_indexing_train, SIR_retrieval_train, SIR_tokenizer, SIR_classification_eval
from custom_loss import LabelContrastiveLoss, LabelSmoothingCrossEntropy, FocalLoss, AsymmetricLossOptimized, GlobalContrastiveLoss, GlobalContrastiveLossV2, GlobalContrastiveLossV3
import scheduler
import torch.distributed as dist
import pickle
import time

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
    if args.dataset == "BigEarthNet-19":
        model = SIR(id_len=args.id_len, input_channels=12, num_classes=[args.distance_prc], num_labels=args.num_labels,
                    distance_prc=args.distance_prc,
                    enc_depth=args.layers, dec_depth=args.layers, num_heads=args.heads, drop_rate=args.dropout)
    else:
        model = SIR(id_len=args.id_len, num_classes=[args.distance_prc], num_labels=args.num_labels, distance_prc=args.distance_prc,
                enc_depth=args.layers, dec_depth=args.layers, num_heads=args.heads, drop_rate=args.dropout, embed_dim=args.hidden_dim,
                    label_embedding=label_embedding.detach(), decoder_type=args.decoder_type)

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
    seq2seq_criterion = LabelSmoothingCrossEntropy()
    global_contrastive_criterion = GlobalContrastiveLoss(beta=0.0)
    ce_loss = nn.CrossEntropyLoss()

    if args.class_loss == "FL":
        classifier_criterion = FocalLoss()
    elif args.class_loss == "ASL":
        classifier_criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    else:
        classifier_criterion = torch.nn.BCEWithLogitsLoss()
    criterion_dic = {"contrastive_criterion": contrastive_criterion,
                     "classifier_criterion": classifier_criterion,
                     "global_contrastive_criterion": global_contrastive_criterion,
                     "seq2seq_criterion":seq2seq_criterion,
                     "ce_loss":ce_loss}

    # criterion_dic = {"contrastive_criterion": contrastive_criterion,
    #                  "classifier_criterion": classifier_criterion,
    #                  "seq2seq_criterion": seq2seq_criterion}

    ###################################### Train ####################################
    train_start_time = time.time()

    ################### Classification #################
    if args.classification_resume:
        logger.info("load classification stage weight....")
        state = torch.load(args.classification_path)
        model.load_state_dict(state["state_dict"], strict=False)

        mAP = SIR_classification_eval(args, model, query_loader)
        logger.info(f"classification phase best mAP:{mAP}")

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

            batch_loss = SIR_classification_train(args, model, db_loader, optimizer, epoch, logger,
                            criterion_dic=criterion_dic,
                            scheduler=lr_scheduler)
            lr_scheduler.step(epoch)

            if (epoch >= 50) and (epoch % 10 == 0):
                mAP = SIR_classification_eval(args, model, query_loader)
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_epoch = epoch
                    logger.info(f"best mAP:{best_mAP} epoch: {best_epoch} save classification model...")
                    save_dict = {
                        'epoch': args.indexing_epochs,
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


    ####################################################

    ################### Indexing #################
    # frozen classification and hot deocder
    for param in model_to_use.decoder.parameters():
        param.requires_grad = True
    for param in model_to_use.encoder.parameters():
        param.requires_grad = False
    for param in model_to_use.classifier_layer.parameters():
        param.requires_grad = False
    for param in model_to_use.distance_layer.parameters():
        param.requires_grad = False
    for param in model_to_use.label_proj.parameters():
        param.requires_grad = False
    # for param in model_to_use.global_feature_layer.parameters():
    #     param.requires_grad = False

    if args.indexing_resume:
        logger.info("load indexing stage weight....")
        state = torch.load(args.indexing_path)
        model.load_state_dict(state["state_dict"])
    else:
        if args.indexing_epochs > 0:
            # optimizer
            if args.optim == 'adam':
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=args.lr)  # , weight_decay=0.0004)
            else:
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                            weight_decay=1e-4)

            lr_scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                                    t_initial=int(args.indexing_epochs),
                                                    lr_min=1e-7,
                                                    warmup_t=int(args.indexing_epochs * 0.1),
                                                    warmup_lr_init=1e-7,
                                                    cycle_decay=1
                                                    )

            for epoch in range(1, args.indexing_epochs + 1):
                logger.info('======================== Indexing {} ========================'.format(epoch))
                for param_group in optimizer.param_groups:
                    logger.info('LR: {}'.format(param_group['lr']))

                batch_loss = SIR_indexing_train(args, model, db_loader, optimizer, epoch, logger, None,
                                criterion_dic=criterion_dic,
                                scheduler=lr_scheduler)
                lr_scheduler.step(epoch)

            logger.info("save indexing model...")
            save_dict = {
                'epoch': args.indexing_epochs,
                'state_dict': model.state_dict(),
                'indexing_loss': batch_loss
            }
            torch.save(save_dict, args.model_name + '/indexing_model.pth')
    ####################################################

    ################### Retrieval #################

    # frozen classification
    for param in model_to_use.encoder.parameters():
        param.requires_grad = False
    for param in model_to_use.classifier_layer.parameters():
        param.requires_grad = False
    for param in model_to_use.distance_layer.parameters():
        param.requires_grad = False
    for param in model_to_use.label_proj.parameters():
        param.requires_grad = False
    # for param in model_to_use.global_feature_layer.parameters():
    #     param.requires_grad = False

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

        batch_loss = SIR_retrieval_train(args, model, db_loader, optimizer, epoch, logger, None,
                        criterion_dic=criterion_dic,
                        scheduler=lr_scheduler)
        lr_scheduler.step(epoch)

        if (epoch % 25 == 0) and (batch_loss < best_loss):
            best_loss = batch_loss
            logger.info(f"save retrieval model... loss: {best_loss}")
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'retrieval_loss': best_loss
            }
            torch.save(save_dict, args.model_name+'/retrieval_model.pth')
    total_time = time.time() - train_start_time
    if total_time < 60:
        logger.info(f'Total training time: {total_time:.2f} seconds')
    elif total_time < 3600:
        minutes = total_time // 60
        seconds = total_time % 60
        logger.info(f'Total training time: {minutes:.0f} minutes {seconds:.2f} seconds')
    else:
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        logger.info(f'Total training time: {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds')