num_clients=100
frac=0.1
epochs=1000
noisy_client_ratio=0.4
minimum_noise=0.5
alpha=5

SEEDS=(1 2 3)

python3 train_FedNoRo.py \
    --gpu=0 \
    --exp fednoro \
    --dataset=pathmnist \
    --model=resnet18 \
    --optimizer sgd \
    --rounds=$epochs \
    --n_clients=$num_clients \
    --frac=$frac \
    --local_ep=3 \
    --base_lr 0.01 \
    --batch_size=100 \
    --iid=0 \
    --alpha=$alpha \
    --seed 1 \
    --non_iid_prob_class=0.7 \
    --level_n_system=$noisy_client_ratio \
    --level_n_lowerb=$minimum_noise \
    --s1 50 \