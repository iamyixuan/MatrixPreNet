#!/bin/zsh

set -e

source ~/.zshrc

conda activate ml

python train_model.py\
    --trainer "supervised"\
    --model_type "linear_inverse"\
    --optimizer_nm "Adam"\
    --loss_fn "ComplexMSE"\
    --data_dir "./data/linear_inv_data.pkl"\
    --data "linear_inverse"\
    --num_epochs 1000\
    --batch_size 32\
    --learning_rate 0.01
    