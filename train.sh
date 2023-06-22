#!/usr/bin/env bash

python train.py \
--dataroot datasets/CVC-ClinicDB \
--name Medical_Imaging \
--arch hybrid \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 1 \
--batch_size 10 \
--lr 0.00001 \
--num_aug 20 \
--niter 125 \
--niter_decay 125 \
--export_folder predictions \
--pred datasets/CVC-ClinicDB/validation/1.png \
--label datasets/CVC-ClinicDB/GroundTruth/1.png \