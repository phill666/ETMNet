import math
import unittest

import numpy as np

from metrics.geometry_metrics import boundary_distance_error, centroid_error, component_count, sample_geometry_record, skeleton_length


class GeometryMetricTests(unittest.TestCase):
    def test_skeleton_lengths(self):
        horizontal = np.ones((1, 5), dtype=np.uint8)
        self.assertAlmostEqual(skeleton_length(horizontal, already_skeleton=True), 4.0)
        diagonal = np.eye(2, dtype=np.uint8)
        self.assertAlmostEqual(skeleton_length(diagonal, already_skeleton=True), math.sqrt(2.0))

    def test_components(self):
        diagonal = np.eye(2, dtype=np.uint8)
        self.assertEqual(component_count(diagonal), 1)
        separated = np.zeros((3, 5), dtype=np.uint8)
        separated[1, 1] = 1
        separated[1, 4] = 1
        self.assertEqual(component_count(separated), 2)

    def test_centroid_and_empty_penalty(self):
        pred = np.zeros((4, 3), dtype=np.uint8)
        gt = np.zeros((4, 3), dtype=np.uint8)
        gt[1:3, 1] = 1
        diag = 5.0
        self.assertAlmostEqual(boundary_distance_error(pred, gt), diag, places=5)
        self.assertAlmostEqual(centroid_error(pred, gt), diag, places=5)

    def test_gt_empty_not_in_geometry_mean(self):
        pred = np.ones((3, 3), dtype=np.uint8)
        gt = np.zeros((3, 3), dtype=np.uint8)
        rec = sample_geometry_record(pred, gt)
        self.assertIsNone(rec["rae"])


if __name__ == "__main__":
    unittest.main()
