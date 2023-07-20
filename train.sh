#!/usr/bin/env bash

python train.py \
--dataroot datasets/CVC-ClinicDB \
--name unet_avoid_non_manifold \
--arch unet \
--loss_func bce \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 4 \
--batch_size 1 \
--lr 0.00001 \
--num_aug 20 \
--niter 75 \
--niter_decay 75 \
--export_folder predictions \
--pred datasets/CVC-ClinicDB/validation/1.png \
--label datasets/CVC-ClinicDB/GroundTruth/1.png \