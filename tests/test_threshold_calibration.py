import unittest

from metrics.threshold_calibration import THRESHOLD_CANDIDATES, add_measurement_normalization, select_diceopt, select_measurementopt


class ThresholdCalibrationTests(unittest.TestCase):
    def test_candidates(self):
        self.assertEqual(THRESHOLD_CANDIDATES, [x / 100.0 for x in range(5, 100, 5)])

    def test_dice_tie_break(self):
        rows = [{"threshold": 0.45, "Dice/F1": 0.9}, {"threshold": 0.55, "Dice/F1": 0.9}]
        self.assertEqual(select_diceopt(rows)["threshold"], 0.45)

    def test_measurement_constant_no_nan_and_tie(self):
        rows = [{"threshold": 0.40, "RAE": 1, "BDE": 2, "CE": 3}, {"threshold": 0.60, "RAE": 1, "BDE": 2, "CE": 3}]
        norm = add_measurement_normalization(rows)
        self.assertTrue(all(row["measurement_score"] == 0.0 for row in norm))
        self.assertEqual(select_measurementopt(rows)["threshold"], 0.40)


if __name__ == "__main__":
    unittest.main()
