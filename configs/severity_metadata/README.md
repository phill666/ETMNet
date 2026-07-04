# Severity Metadata

This directory stores dataset-level severity metadata generated only from training-set
ground-truth masks.

The public repository does not include dataset images, masks, or fixed split files, so
real metadata JSON files are not fabricated here. Generate them after preparing each
dataset and split:

```powershell
python tools\build_severity_metadata.py --dataset-name mtd --mask-root <train-mask-root> --split-file <train-split-file> --out configs\severity_metadata\mtd.json
python tools\build_severity_metadata.py --dataset-name ksdd2 --mask-root <train-mask-root> --split-file <train-split-file> --out configs\severity_metadata\ksdd2.json
```

Protocol version: `revision_2026_07`.
