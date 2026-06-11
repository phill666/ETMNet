#!/usr/bin/env bash
set -e
python tools/threshold_calibration.py --config configs/magnetic_tile_etmnet.yaml --checkpoint checkpoints/etmnet_mtd.pth --mode both

