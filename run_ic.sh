#!/bin/zsh

set -e

source ~/.zshrc

conda activate ml

train="True"
model_path="./experiments/FNN-DD_IC-3000-B128-lr0.0001-kappaLoss/model.pth"

python train_model.py\
    --train $train\
    --trainer "unsupervised"\
    --model_type "FNN"\
    --optimizer_nm "Adam"\
    --loss_fn   "MatConditionNumberLoss"\
    --data_dir "./data/DD_mat_IC_L.pt"\
    --data_name "DD_IC"\
    --in_dim 1792\
    --out_dim 960\
    --num_epochs 3000\
    --batch_size 128\
    --learning_rate 0.001\
    --hidden_dim 256\
    --in_ch 1\
    --out_ch 1\
    --kernel_size 7\
    --model_path $model_path\
    --additional_info "test-2"\
