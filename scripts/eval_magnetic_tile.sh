#!/usr/bin/env bash
set -e
python tools/test.py --config configs/magnetic_tile_etmnet.yaml --checkpoint checkpoints/etmnet_mtd.pth

