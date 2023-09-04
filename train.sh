#!/usr/bin/env bash

python train.py \
--dataroot datasets/CVC-ClinicDB \
--name test \
--arch test \
--loss_func dice \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 3 \
--batch_size 1 \
--lr 0.00001 \
--num_aug 20 \
--niter 175 \
--niter_decay 175 \
--export_folder predictions \
--pred datasets/CVC-ClinicDB/validation/1.png \
--label datasets/CVC-ClinicDB/GroundTruth/1.png \