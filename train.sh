#!/usr/bin/env bash

python train.py \
--dataroot datasets/CVC-ClinicDB \
--name Medical_Imaging_multgpus \
--arch hybrid \
--resize 64 64 \
--ncf 16 32 64 \
--gpu_ids 1 \
--batch_size 5 \
--lr 0.00001 \
--num_aug 20 \
--niter 125 \
--niter_decay 125 \
--export_folder predictions \
--pred datasets/CVC-ClinicDB/validation/1.png \
--label datasets/CVC-ClinicDB/GroundTruth/1.png \