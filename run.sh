#!/bin/zsh

set -e

source ~/.zshrc

conda activate ml

python train_model.py\
    --trainer "supervised"\
    --model_type "RNN"\
    --optimizer_nm "Adam"\
    --loss_fn "MSE"\
    --data_dir "./data/DD_CG_solutions.pkl"\
    --data_name "time_series"\
    --in_dim 256\
    --out_dim 256\
    --num_epochs 1000\
    --batch_size 32\
    --learning_rate 0.0001\
    --hidden_dim 256\
