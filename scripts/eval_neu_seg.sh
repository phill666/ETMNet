#!/usr/bin/env bash
set -e
python tools/test.py --config configs/neu_seg_etmnet.yaml --checkpoint checkpoints/etmnet_neu_seg.pth

