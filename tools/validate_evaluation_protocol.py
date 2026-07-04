import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from metrics.geometry_metrics import (
    boundary_distance_error,
    centroid_error,
    component_count,
    sample_geometry_record,
    skeleton_length,
)
from metrics.severity_metrics import (
    aggregate_severity,
    build_metadata_from_masks,
    severity_record,
)
from metrics.threshold_calibration import (
    THRESHOLD_CANDIDATES,
    add_measurement_normalization,
    select_diceopt,
    select_measurementopt,
)


ROOT = Path(__file__).resolve().parents[1]


def ok(name, condition, details=""):
    return {"name": name, "passed": bool(condition), "details": details}


def main():
    checks = []

    skel = np.zeros((1, 5), dtype=np.uint8)
    skel[0, :] = 1
    checks.append(ok("five horizontal skeleton pixels length is 4", abs(skeleton_length(skel, already_skeleton=True) - 4.0) < 1e-9))

    diag = np.eye(2, dtype=np.uint8)
    checks.append(ok("two diagonal skeleton pixels length is sqrt(2)", abs(skeleton_length(diag, already_skeleton=True) - np.sqrt(2.0)) < 1e-9))

    checks.append(ok("diagonal touching pixels are one 8-neighbor component", component_count(diag) == 1))

    two = np.zeros((3, 5), dtype=np.uint8)
    two[1, 1] = 1
    two[1, 4] = 1
    checks.append(ok("separated regions are two components", component_count(two) == 2))

    a = np.zeros((5, 5), dtype=np.uint8)
    b = np.zeros((5, 5), dtype=np.uint8)
    a[1, 1] = 1
    b[4, 4] = 1
    checks.append(ok("centroid distance for known coordinates", abs(centroid_error(a, b) - np.sqrt(18.0)) < 1e-9))

    empty = np.zeros((4, 3), dtype=np.uint8)
    gt = np.zeros((4, 3), dtype=np.uint8)
    gt[1:3, 1] = 1
    diag_penalty = np.sqrt(4 * 4 + 3 * 3)
    checks.append(ok("empty prediction BDE diagonal penalty", abs(boundary_distance_error(empty, gt) - diag_penalty) < 1e-6))
    checks.append(ok("empty prediction CE diagonal penalty", abs(centroid_error(empty, gt) - diag_penalty) < 1e-6))

    rec_empty_gt = sample_geometry_record(a, np.zeros_like(a))
    checks.append(ok("GT-empty sample has no geometry errors", rec_empty_gt["rae"] is None and rec_empty_gt["ce"] is None))

    masks = []
    for size in [1, 2, 3, 4]:
        m = np.zeros((8, 8), dtype=np.uint8)
        m[1:1 + size, 1:1 + size] = 1
        masks.append(m)
    metadata = build_metadata_from_masks(masks, dataset_name="synthetic")
    big = np.ones((8, 8), dtype=np.uint8)
    sev = severity_record(big, masks[0], metadata)
    checks.append(ok("severity scores are clipped to [0, 1]", 0.0 <= sev["pred_score"] <= 1.0 and 0.0 <= sev["gt_score"] <= 1.0))
    checks.append(ok("severity metadata has training quantiles", "q33_3" in metadata["quantiles"] and "q66_7" in metadata["quantiles"]))
    checks.append(ok("severity MAE uses continuous scores", aggregate_severity([sev])["Severity_MAE"] == abs(sev["pred_score"] - sev["gt_score"])))

    dice_rows = [{"threshold": 0.45, "Dice/F1": 0.8}, {"threshold": 0.55, "Dice/F1": 0.8}, {"threshold": 0.25, "Dice/F1": 0.7}]
    checks.append(ok("DiceOpt tie chooses smaller equally close to 0.50", select_diceopt(dice_rows)["threshold"] == 0.45))

    meas_rows = [{"threshold": 0.40, "RAE": 1, "BDE": 2, "CE": 3}, {"threshold": 0.60, "RAE": 1, "BDE": 2, "CE": 3}]
    norm = add_measurement_normalization(meas_rows)
    checks.append(ok("MeasurementOpt constant metrics do not produce NaN", all(row["measurement_score"] == 0.0 for row in norm)))
    checks.append(ok("MeasurementOpt tie chooses smaller equally close to 0.50", select_measurementopt(meas_rows)["threshold"] == 0.40))
    checks.append(ok("threshold candidates are fixed 0.05..0.95", THRESHOLD_CANDIDATES == [x / 100.0 for x in range(5, 100, 5)]))

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    checks.append(ok("README title is revised", readme.startswith("# ETMNet: An Edge-Guided Measurement Network")))
    banned = ["topology" + "-aware", "topology " + "preserving", "measurement" + "-oriented loss", "topological " + "supervision"]
    leftovers = []
    for path in [ROOT / "README.md", ROOT / "PROTOCOL.md"]:
        text = path.read_text(encoding="utf-8").lower()
        for pattern in banned:
            if re.search(pattern, text):
                leftovers.append("{}:{}".format(path.name, pattern))
    checks.append(ok("README/PROTOCOL contain no banned legacy claims", not leftovers, ", ".join(leftovers)))

    report = {
        "protocol_version": "revision_2026_07",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    out = ROOT / "runs" / "protocol_validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

