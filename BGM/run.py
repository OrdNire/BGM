import os
import sys
import time
import random
import argparse


if __name__ == '__main__':
    dataset = "DOTA"
    epochs = 30
    max_topK = 10
    abl_setting = "MT"
    distance_prc = 10
    contr_lambda = 0.1

    dist_launch = "CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 NCCL_TIMEOUT=6000 /home/###/.conda/envs/pt39/bin/python -W ignore -m torch.distributed.launch --master_port 12351 --nproc_per_node=2 " \
               "--nnodes=1"

    os.system(f"{dist_launch} "
                  f"--use_env train_SIRMM.py --dataset {dataset} --epochs {epochs} --max_topK {max_topK} --abl_setting {abl_setting} --contr_lambda {contr_lambda} --distance_prc {distance_prc}")