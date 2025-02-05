#!/bin/bash
set -e

echo "Starting trian MSC"

python main.py \
      --meta_file "MSC_retrieval_noise0.pkl" \
      --code_file "autom_id_MultiScene-Clean_12600.pkl" \
      --dataset "MultiScene-Clean" \
      --noise_rate 0 \
      --indexing_epochs 50 \
      --retrieval_epochs 150 \
      --lr 3.5e-4 \

echo "end trian MSC"

#echo "Starting test MSC"
#
#python inference.py \
#      --meta_file "MSC_retrieval_noise0.pkl"  \
#      --code_file "autom_id_MultiScene-Clean_12600.pkl" \
#      --dataset "MultiScene-Clean" \
#      --model_dir "/home/jiangfanzhuo/SpaceIR/baseline/results/baseAR_MultiScene-Clean_100_300_0.0001_0" \
#      --model_name "ar_200.pth" \
#      --noise_rate 0  \
#      --beam_size 100
#
#echo "end test MSC"