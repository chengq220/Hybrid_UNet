#!/usr/bin/env bash

python train.py \
--dataroot datasets/CVC-ClinicDB \
--name fix_kernel \
--arch test \
--loss_func dice \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 7 \
--batch_size 1 \
--lr 0.00001 \
--num_aug 20 \
--niter 125 \
--niter_decay 125 \
--export_folder predictions \
--pred datasets/CVC-ClinicDB/validation/1.png \
--label datasets/CVC-ClinicDB/GroundTruth/1.png \