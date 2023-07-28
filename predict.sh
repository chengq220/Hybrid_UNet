#!/usr/bin/env bash

python predict.py \
--dataroot datasets/CVC-ClinicDB \
--name full_test_3_layer \
--arch test \
--resize 256 256 \
--ncf 64 128 256 512 1024 \
--gpu_ids 5 \
--pred datasets/CVC-ClinicDB/test/270.png \
--label datasets/CVC-ClinicDB/GroundTruth/270.png \
--export_folder manual_pred
