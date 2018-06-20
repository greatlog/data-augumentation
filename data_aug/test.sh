#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python main.py \
    --output_dir /data/model_weights/luozx/data_aug/experiment_discrim/test1/ \
    --summary_dir /data/model_weights/luozx/data_aug/experiment_discrim/test1/log \
    --mode test \
    --pre_trained_model True \
    --vgg19_path /data/model_weights/luozx/data_aug/imagenet-vgg-verydeep-19.mat \
    --vgg19_layer conv5_4 \
    --checkpoint /data/model_weights/luozx/data_aug/experiment_discrim/1345001 \
    --input_size 512 \
    --batch_size 4 \
    --alpha 0.25 \
    --gamma 0.75 \
    --is_training True \
    --task combined_model \
    --learning_rate 1e-4 \
    --decay_steps 40000 \
    --decay_rate 0.9 \
    --stair False \
    --beta 0.9 \
    --display_freq 100 \
    --max_iter 10000000 \
    --save_freq 1000

