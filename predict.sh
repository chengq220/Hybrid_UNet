#!/usr/bin/env bash

python predict.py \
--dataroot datasets/CVC-ClinicDB \
--name Medical_Imaging_Dice \
--arch unet \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 4 \
--pred datasets/CVC-ClinicDB/validation/556.png \
--label datasets/CVC-ClinicDB/GroundTruth/556.png \
--export_folder manual_pred
