#!/bin/bash
set -e

#echo "Starting trian DLRSD"
#
#python main.py \
#      --meta_file "DLRSD_retrieval_noise1.0.pkl" \
#      --code_file "autom_id_DLRSD_3360.pkl" \
#      --dataset "DLRSD" \
#      --noise_rate 1.0 \
#      --indexing_epochs 50 \
#      --retrieval_epochs 150 \
#      --lr 3.5e-4 \
#
#echo "end trian DLRSD"

echo "Starting test"

python inference.py \
      --meta_file "DLRSD_retrieval_noise1.0.pkl"  \
      --code_file "autom_id_DLRSD_3360.pkl" \
      --dataset "DLRSD" \
      --model_dir "/home/jiangfanzhuo/SpaceIR/baseline/results/baseAR_DLRSD_50_150_0.00035_1.0" \
      --model_name "ar_150.pth" \
      --noise_rate 0  \
      --beam_size 100

echo "end test"
