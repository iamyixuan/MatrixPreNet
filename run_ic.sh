#!/bin/zsh

set -e

train="True"
model_path="./experiments/FNN-DD_IC-3000-B8-lr0.0049460803423949-train-best-config/"

python train_model.py\
    --train $train\
    --trainer "unsupervised"\
    --model_type "PrecondCNN"\
    --optimizer_nm "Adam"\
    --loss_fn   "CNNMatCondNumberLoss"\
    --data_dir "./data/U1_DD_matrices.pt"\
    --data_name "U1_DD"\
    --in_dim 1792\
    --out_dim 960\
    --num_epochs 3000\
    --batch_size 128\
    --learning_rate 0.001\
    --hidden_dim 32\
    --in_ch 2\
    --out_ch 1\
    --kernel_size 3\
    --model_path $model_path\
    --additional_info "U1_mat_kappa_2"\
