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

if __name__ == '__main__':
    # information
    logger = Logger(args.model_name, dir_path=".")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info('Labels: {}'.format(args.num_labels))

    # dataset
    query_loader, gallery_loader, train_gallery_loader, db_loader = get_data(args)

    # model
    model = SIR(id_len=args.id_len, voc_length=2**args.k_bit, num_labels=args.num_labels, distance_prc=args.distance_prc,
                enc_depth=args.layers, dec_depth=args.layers, num_heads=args.heads, drop_rate=args.dropout)

    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Total parameters: {model_params:.2f}M")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.cuda()

    # label embedding
    label_embedding = torch.load(os.path.join(args.meta_root, args.dataset, "label_embedding.pt"))

    # criterion
    contrastive_criterion = LabelContrastiveLoss()
    seq2seq_criterion = LabelSmoothingCrossEntropy(pad_index=args.num_labels*args.distance_prc)

    classifier_criterion = torch.nn.BCEWithLogitsLoss()
    criterion_dic = {"contrastive_criterion": contrastive_criterion,
                     "classifier_criterion": classifier_criterion,
                     "seq2seq_criterion":seq2seq_criterion}

    ###################################### Train ####################################

    ################### Classification #################
    if True:
        logger.info("load classification stage weight....")
        state = torch.load(args.classification_path)
        model.load_state_dict(state["state_dict"])
    else:

        # frozen decoder
        for param in model.module.decoder.parameters():
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

        for epoch in range(1, args.classification_epochs + 1):
            logger.info('======================== Classification {} ========================'.format(epoch))
            for param_group in optimizer.param_groups:
                logger.info('LR: {}'.format(param_group['lr']))

            batch_loss = SIR_classification_train(args, model, label_embedding, db_loader, optimizer, epoch, logger,
                            criterion_dic=criterion_dic,
                            scheduler=lr_scheduler)
            lr_scheduler.step(epoch)

            if epoch % 50 == 0:
                logger.info("save classification model...")
                save_dict = {
                    'epoch': args.indexing_epochs,
                    'state_dict': model.state_dict(),
                    'indexing_loss': batch_loss
                }
                torch.save(save_dict, args.model_name + '/classification_model.pth')
    ####################################################

    ################### Tokenizer #################

    logger.info("======================== Tokenizer ========================")
    clean_codes, noise_codes = SIR_tokenizer(args, model, label_embedding, db_loader, logger)
    logger.info("save codes ...")
    codes = {
        'clean_codes': clean_codes,
        "noise_codes": noise_codes
    }
    torch.save(codes, args.model_name + '/codes.pth')
    ####################################################


    ################### Indexing #################
    # frozen classification and hot deocder
    for param in model.module.decoder.parameters():
        param.requires_grad = True
    for param in model.module.encoder.parameters():
        param.requires_grad = True
    for param in model.module.classifier_layer.parameters():
        param.requires_grad = True
    for param in model.module.distance_layer.parameters():
        param.requires_grad = True

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

            batch_loss = SIR_indexing_train(args, model, label_embedding, db_loader, optimizer, epoch, logger, codes,
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

    # # frozen classification
    # for param in model.module.encoder.parameters():
    #     param.requires_grad = False
    # for param in model.module.classifier_layer.parameters():
    #     param.requires_grad = False
    # for param in model.module.distance_layer.parameters():
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

        batch_loss = SIR_retrieval_train(args, model, label_embedding, db_loader, optimizer, epoch, logger, codes,
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