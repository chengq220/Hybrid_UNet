#!/usr/bin/env bash

python predict.py \
--dataroot datasets/CVC-ClinicDB \
--name fixed_pooling \
--arch hybrid \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 4 \
--pred datasets/CVC-ClinicDB/train1/1.png \
--label datasets/CVC-ClinicDB/GroundTruth/1.png \
--export_folder manual_pred
