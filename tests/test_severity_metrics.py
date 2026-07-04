import unittest

import numpy as np

from metrics.severity_metrics import aggregate_severity, build_metadata_from_masks, severity_record


class SeverityMetricTests(unittest.TestCase):
    def test_metadata_and_continuous_mae(self):
        masks = []
        for size in [1, 2, 3, 4]:
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[:size, :size] = 1
            masks.append(mask)
        meta = build_metadata_from_masks(masks, dataset_name="synthetic")
        rec = severity_record(masks[-1], masks[0], meta)
        agg = aggregate_severity([rec])
        self.assertIn("q33_3", meta["quantiles"])
        self.assertGreaterEqual(rec["pred_score"], 0.0)
        self.assertLessEqual(rec["pred_score"], 1.0)
        self.assertAlmostEqual(agg["Severity_MAE"], abs(rec["pred_score"] - rec["gt_score"]))


if __name__ == "__main__":
    unittest.main()
