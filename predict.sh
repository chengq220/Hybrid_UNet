#!/usr/bin/env bash

python predict.py \
--dataroot datasets/CVC-ClinicDB \
--name fix_pool \
--arch test \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 4 \
--pred datasets/CVC-ClinicDB/train1/612.png \
--label datasets/CVC-ClinicDB/GroundTruth/612.png \
--export_folder manual_pred
