#!/usr/bin/env bash

python train.py \
--dataroot datasets/CVC-ClinicDB \
--name profiling \
--arch test \
--loss_func dice \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 3 \
--batch_size 1 \
--lr 0.00002 \
--num_aug 20 \
--niter 100 \
--niter_decay 100 \
--export_folder predictions \
--pred datasets/CVC-ClinicDB/validation/1.png \
--label datasets/CVC-ClinicDB/GroundTruth/1.png \