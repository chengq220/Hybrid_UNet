#!/usr/bin/env bash

python predict.py \
--dataroot datasets/CVC-ClinicDB \
--name Medical_Imaging \
--arch unet \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 1 \
--pred datasets/CVC-ClinicDB/validation/1.png \
--label datasets/CVC-ClinicDB/GroundTruth/1.png \
--export_folder predction1
# --which_epoch 1000 \
