import argparse, numpy as np
import os.path

import torch
import torch.nn as nn
import random

from utils.Log import Logger
from config_args import get_args_tokenizer
from load_data import get_data
from models.CTrans.CTran import CTranModel
from models.SpaceIR.SIR import SIR
from engine import tokenizer_train, tokenizer_eval
from custom_loss import LabelContrastiveLoss
import scheduler


args = get_args_tokenizer(argparse.ArgumentParser())

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
    query_loader, gallery_loader, db_loader = get_data(args)
    
    # model
    model = CTranModel(args.num_labels,args.use_lmt,args.pos_emb,args.layers,args.heads,args.dropout,args.no_x_features)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Total parameters: {model_params:.2f}M")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.cuda()

    if args.freeze_backbone:
        for p in model.module.backbone.parameters():
            p.requires_grad = False
        for p in model.module.backbone.base_network.layer4.parameters():
            p.requires_grad = True

    # label embedding
    label_embedding = torch.load(os.path.join(args.meta_root, args.dataset, "label_embedding.pt"))

    # optimizer
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr)  # , weight_decay=0.0004)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                    weight_decay=1e-4)

    scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                                t_initial=10,
                                                lr_min=1e-6,
                                                warmup_t=2,
                                                warmup_lr_init=1e-7,
                                                cycle_decay=0.5
                                                )

    # criterion

    contrastive_criterion = LabelContrastiveLoss(args.beta)

    classifier_criterion = torch.nn.BCEWithLogitsLoss()
    criterion_dic = {"contrastive_criterion":contrastive_criterion,
                 "classifier_criterion":classifier_criterion}

    # best value
    best_mAP = 0
    
    for epoch in range(1,args.epochs+1):
        logger.info('======================== {} ========================'.format(epoch))
        for param_group in optimizer.param_groups:
            logger.info('LR: {}'.format(param_group['lr']))

        ################### Train #################
        tokenizer_train(args,model,label_embedding,db_loader,optimizer,epoch,logger,
                     criterion_dic=criterion_dic,
                     scheduler=scheduler)
        scheduler.step(epoch)

        if (epoch + 1) % 1 == 0:
            logger.info("Evaluation.")
            mAP = tokenizer_eval(args,model,label_embedding,query_loader,epoch,logger,
                         criterion_dic=criterion_dic)
            if mAP > best_mAP:
                logger.info("Save Model......")
                best_mAP = mAP
                save_dict = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'test_mAP': best_mAP
                }
                torch.save(save_dict, args.model_name+'/best_model.pt')
        
