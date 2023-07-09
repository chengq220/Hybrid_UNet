#!/usr/bin/env bash

python predict.py \
--dataroot datasets/CVC-ClinicDB \
--name Medical_imaging_spline \
--arch hybrid \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 1 \
--pred datasets/CVC-ClinicDB/validation/1.png \
--label datasets/CVC-ClinicDB/GroundTruth/1.png \
--export_folder manual_pred
