import argparse, numpy as np
import os.path

import torch
import torch.nn as nn
import random

from utils.Log import Logger
from utils.func import select_GPU, logger_print
from config_args import get_args_train_SIRMM
from load_data_MM import get_data
from models.SpaceIR.SIR_MM import build_SIRMM
from engine import SIRMM_train, SIRMM_eval
from custom_loss import LabelContrastiveLoss, FocalLoss, AsymmetricLossOptimized, GlobalContrastiveLoss, LabelSmoothingCrossEntropyV2, LabelContrastiveLossHardway
import scheduler
import torch.distributed as dist
import pickle
import time
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import threading

os.environ["NCCL_TIMEOUT"] = "3600"       # 设置 NCCL 超时时间
os.environ["NCCL_BLOCKING_WAIT"] = "1"    # 启用阻塞等待
os.environ["NCCL_DEBUG"] = "INFO"         # 启用调试日志

print(f"NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}")
print(f"NCCL_BLOCKING_WAIT: {os.environ.get('NCCL_BLOCKING_WAIT')}")

args = get_args_train_SIRMM(argparse.ArgumentParser())

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup():
    # 关闭分布式训练
    dist.destroy_process_group()

# def async_eval():
#     global best_map
#     avg_map = SIRMM_eval(args, model, query_data=query_loader, gallery_data=gallery_loader, logger=logger)
#     logger_print(rank, logger, f"current avg_mAP:{avg_map} best avg_mAP:{best_map}")
#     if avg_map > best_map:
#         best_map = avg_map
#         save_dict = {
#             'epoch': epoch,
#             'state_dict': model.state_dict(),
#             'mAP': best_map
#         }
#         torch.save(save_dict, args.model_name + '/retrieval_model.pth')

if __name__ == '__main__':

    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    args.rank = rank
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    # Seed
    set_seed(42 + rank)

    # information
    logger = None
    if rank == 0:
        logger = Logger(args.model_name, dir_path=".")
    for arg, value in vars(args).items():
        logger_print(rank, logger, f"{arg}: {value}")
    logger_print(rank, logger, 'Labels: {}'.format(args.num_labels))

    # Get Device
    torch.cuda.set_device(rank)
    args.device = torch.device(f"cuda:{rank}")
    print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")

    # dataset
    query_loader, gallery_loader, db_loader, sampler_db = get_data(args)

    # label embedding
    label_embedding = torch.load(os.path.join(args.meta_root, args.dataset, "label_embedding.pt"))

    # 读取recall matrix
    args.map_matrix = np.load(os.path.join(args.dataroot, args.dataset, "map_matrix.npy"))
    args.ndcg_matrix = np.load(os.path.join(args.dataroot, args.dataset, "ndcg_matrix.npy"))

    # # cfg
    # with open(os.path.join(args.meta_root, args.dataset, args.meta_file), 'rb') as f:
    #     cfg = pickle.load(f)

    # model
    model = build_SIRMM(args, label_embedding)

    # pretrain_load
    if args.pretrain_path != "None":
        model_dict = model.state_dict()
        checkpoint = torch.load(os.path.join(args.pretrain_path, "retrieval_model.pth"))
        pretrained_dict = checkpoint["state_dict"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and ("interactor" not in k or "classifier_layer" not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger_print(rank, logger, "Load pretrained weights.")

    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger_print(rank, logger, f"Total parameters: {model_params:.2f}M")

    model = model.to(args.device)
    model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    model_to_use = model.module if hasattr(model, 'module') else model

    # criterion
    contrastive_criterion = LabelContrastiveLoss(label_embedding=label_embedding.cuda().detach())
    # seq2seq_criterion = LabelSmoothingCrossEntropy(pad_index=args.distance_prc)
    seq2seq_criterion = LabelSmoothingCrossEntropyV2()
    global_contrastive_criterion = GlobalContrastiveLoss(beta=0.0)
    ce_loss = nn.CrossEntropyLoss()
    contrastive_hardway_criterion = LabelContrastiveLossHardway()

    if args.class_loss == "FL":
        classifier_criterion = FocalLoss()
    elif args.class_loss == "ASL":
        classifier_criterion = AsymmetricLossOptimized(gamma_neg=1, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    else:
        classifier_criterion = torch.nn.BCEWithLogitsLoss()
    criterion_dic = {"contrastive_criterion": contrastive_criterion,
                     "classifier_criterion": classifier_criterion,
                     "global_contrastive_criterion": global_contrastive_criterion,
                     "seq2seq_criterion":seq2seq_criterion,
                     "ce_loss":ce_loss,
                     "contrastive_hardway_criterion": contrastive_hardway_criterion}
    ###################################### Train ####################################
    train_start_time = time.time()

    # ################### Memory #################
    # args.mem_epochs = args.epochs // 3
    # # optimizer
    # if args.optim == 'adam':
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                                  lr=args.lr)  # , weight_decay=0.0004)
    # else:
    #     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
    #                                 weight_decay=1e-4)
    #
    # lr_scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
    #                                            t_initial=int(args.mem_epochs),
    #                                            lr_min=1e-7,
    #                                            warmup_t=int(args.mem_epochs * 0.1),
    #                                            warmup_lr_init=1e-7,
    #                                            cycle_decay=1
    #                                            )
    # for epoch in range(1, args.mem_epochs + 1):
    #     logger_print(rank, logger, '======================== Memory Training {} ========================'.format(epoch))
    #     for param_group in optimizer.param_groups:
    #         logger_print(rank, logger, 'LR: {}'.format(param_group['lr']))
    #
    #     sampler_mem.set_epoch(epoch)
    #
    #     batch_loss = SIRMM_train(args, model, mem_loader, optimizer, epoch, logger,
    #                              criterion_dic=criterion_dic,
    #                              scheduler=lr_scheduler,
    #                              stage="memory")
    #     lr_scheduler.step(epoch)
    #
    # if rank == 0:
    #     save_dict = {
    #         'epoch': args.mem_epochs,
    #         'state_dict': model.state_dict(),
    #     }
    #     torch.save(save_dict, args.model_name + '/memory_model.pth')
    # dist.barrier()

    ################### Train #################
    if args.train_resume:
        logger_print(rank, logger, "load train stage weight....")
        state = torch.load(args.train_path)
        model.load_state_dict(state["state_dict"], strict=False)
        mR = SIRMM_eval(args, model, query_data=query_loader, gallery_data=gallery_loader, logger=logger)
    else:
        # optimizer
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=args.lr) # , weight_decay=0.01
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                        weight_decay=1e-4)

        lr_scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                                   t_initial=int(args.epochs),
                                                   lr_min=1e-7,
                                                   warmup_t=int(args.epochs * 0.1),
                                                   warmup_lr_init=1e-7,
                                                   cycle_decay=1
                                                   )
        # val_value = float('inf')
        val_value = 0
        for epoch in range(1, args.epochs + 1):
            logger_print(rank, logger, '======================== Joint Training {} ========================'.format(epoch))
            for param_group in optimizer.param_groups:
                logger_print(rank, logger, 'LR: {}'.format(param_group['lr']))

            sampler_db.set_epoch(epoch)

            batch_loss = SIRMM_train(args, model, db_loader, optimizer, epoch, logger,
                            criterion_dic=criterion_dic,
                            scheduler=lr_scheduler,
                            stage="train")
            lr_scheduler.step(epoch)

            if (epoch % args.eval_step == 0):
                logger_print(rank, logger,
                             '======================== Validation {} ========================'.format(epoch))
                # sampler_eval.set_epoch(epoch)
                # val = SIRMM_train(args, model, eval_loader, optimizer, epoch, logger,
                #                       criterion_dic=criterion_dic,
                #                       scheduler=lr_scheduler,
                #                       stage="eval")

                if rank == 0:
                    # val = SIRMM_eval(args, model.module, query_data=query_loader, gallery_data=gallery_loader, logger=logger)
                    # logger_print(rank, logger, f"current value:{val} best value:{val_value}")
                    # save_dict = {
                    #     'epoch': epoch,
                    #     'state_dict': model.state_dict(),
                    #     'mAP': val
                    # }
                    # if val > val_value:
                    #     val_value = val
                    #     torch.save(save_dict, args.model_name + '/retrieval_model.pth')
                    # torch.save(save_dict, args.model_name + f'/retrieval_model_{epoch}.pth')
                    save_dict = {
                        'epoch': epoch,
                        'state_dict': model.state_dict()
                    }
                    torch.save(save_dict, args.model_name + '/retrieval_model.pth')
                    torch.save(save_dict, args.model_name + f'/retrieval_model_{epoch}.pth')
                    logger_print(rank, logger,
                                 '======================== Validation End ========================'.format(epoch))
                dist.barrier()

    total_time = time.time() - train_start_time
    if total_time < 60:
        logger_print(rank, logger, f'Total training time: {total_time:.2f} seconds')
    elif total_time < 3600:
        minutes = total_time // 60
        seconds = total_time % 60
        logger_print(rank, logger, f'Total training time: {minutes:.0f} minutes {seconds:.2f} seconds')
    else:
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        logger_print(rank, logger, f'Total training time: {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds')
    cleanup()