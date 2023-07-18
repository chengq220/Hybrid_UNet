#!/usr/bin/env bash

python predict.py \
--dataroot datasets/CVC-ClinicDB \
--name unet_with_splineatbottleneck \
--arch unet \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 4 \
--pred datasets/CVC-ClinicDB/train1/212.png \
--label datasets/CVC-ClinicDB/GroundTruth/212.png \
--export_folder manual_pred
