import os
import sys
import random
import torch
import numpy as np
import argparse
import pickle
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)
from config_args import get_args_baseAR
from utils.Log import Logger
from load_data import get_data
from models.base.model import BaseAR
from models.CLIP.clip import clip
from utils.func import get_trainable_params
import scheduler
from custom_loss import LabelSmoothingCrossEntropy
from engine import baseAR_train

args = get_args_baseAR(argparse.ArgumentParser())

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
    logger = Logger(args.model_name, dir_path=args.save_path)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info('Labels: {}'.format(args.num_labels))

    # dataset
    query_loader, gallery_loader, train_gallery_loader, db_loader = get_data(args)
    logger.info(f"train gallery size: {len(train_gallery_loader.dataset)}")
    logger.info(f"db size: {len(db_loader.dataset)}")

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

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Total parameters: {model_params:.2f}M")

    # train criterion
    criterion = LabelSmoothingCrossEntropy()

    # optimzier
    optimizer = AdamW([
        {'params': get_trainable_params(model.module.decoder if torch.cuda.device_count() > 1 else model.decoder)},
        {'params': get_trainable_params(model.module.encoder if torch.cuda.device_count() > 1 else model.encoder), 'lr': 0.01 * args.lr}
                       ], lr=args.lr, betas=(0.9, 0.96), eps=1e-08, weight_decay=0.05)

    if args.indexing_epochs > 0:
        # lr scheduler
        lr_scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                                        t_initial=args.indexing_epochs,
                                                        lr_min=5e-7,
                                                        warmup_t=max(int(args.indexing_epochs * 0.1), 1),
                                                        warmup_lr_init=5e-8,
                                                        cycle_decay=1,
                                                        cycle_limit = 4,
                                                        )

        # train indexing
        logger.info(f"========= indexing stage =========")
        indexing_loss = []
        for epoch in range(args.indexing_epochs):
            running_loss = baseAR_train(args, model, train_gallery_loader, optimizer, epoch, logger, criterion,
                         code, cfg, device, scheduler=lr_scheduler, stage="indexing")
            indexing_loss.append(running_loss)
        state = {"net": model.state_dict(), "epoch": epoch + 1}
        torch.save(state, os.path.join(args.save_path, f"indexing_{epoch + 1}.pth"))

    # optimzier
    optimizer = AdamW([
        {'params': get_trainable_params(model.module.decoder if torch.cuda.device_count() > 1 else model.decoder)},
        {'params': get_trainable_params(model.module.encoder if torch.cuda.device_count() > 1 else model.encoder), 'lr': 0.01 * args.lr}
                       ], lr=args.lr, betas=(0.9, 0.96), eps=1e-08, weight_decay=0.05)
    # lr scheduler
    lr_scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                            t_initial=args.retrieval_epochs,
                                            lr_min=5e-7,
                                            warmup_t=int(args.retrieval_epochs * 0.1),
                                            warmup_lr_init=5e-8,
                                            cycle_decay=1,
                                            cycle_limit=4,
                                            )

    # train retrieval
    logger.info(f"========= retrieval stage =========")
    retrieval_loss = []
    for epoch in range(args.retrieval_epochs):
        running_loss = baseAR_train(args, model, db_loader, optimizer, epoch, logger, criterion,
                     code, cfg, device, scheduler=lr_scheduler, stage="retrieval")
        retrieval_loss.append(running_loss)
        if (epoch+1) % args.save_step == 0:
            state = {"net": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch + 1, "loss": running_loss}
            torch.save(state, os.path.join(args.save_path, f"ar_{epoch + 1}.pth"))

    logger.info("save running loss.")
    logger.info(f"indexing_loss: {indexing_loss}")
    logger.info(f"retrieval_loss: {retrieval_loss}")
    running_loss = {"indexing_loss": indexing_loss, "retrieval_loss": retrieval_loss}
    with open(os.path.join(args.save_path, f"running_loss.pkl"), 'wb') as file:
        pickle.dump(running_loss, file)