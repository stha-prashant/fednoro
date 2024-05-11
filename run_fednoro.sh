#!/bin/bash -l


#SBATCH --account mvaal --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g

module purge
conda activate dplearning

num_clients=100
frac=0.1

python3 train_FedNoRo.py \
    --gpu=0 \
    --exp fednoro \
    --dataset=$dataset \
    --model=$architecture \
    --optimizer sgd \
    --rounds=$epochs \
    --n_clients=$num_clients \
    --frac=$frac \
    --local_ep=3 \
    --base_lr 0.01 \
    --batch_size=50 \
    --iid=0 \
    --alpha=$alpha \
    --seed=$seed \
    --non_iid_prob_class=0.7 \
    --level_n_system=$noisy_client_ratio \
    --level_n_lowerb=$minimum_noise \
    --s1 50 \
    --wandb