import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nir_v1_scientific_feature_extraction import mad, robust_slope


class ScientificFeatureMathTests(unittest.TestCase):
    def test_mad_is_primary_variability(self):
        self.assertEqual(mad([1, 2, 3, 100]), 1.0)

    def test_robust_slope_uses_time_in_seconds(self):
        slope = robust_slope([0, 1000, 2000, 3000], [1, 2, 3, 4])
        self.assertAlmostEqual(slope, 1.0, places=6)

    def test_invalid_values_are_not_silent_measurements(self):
        self.assertTrue(np.isnan(robust_slope([1], [2])))

    def test_plateau_preserves_zero_slope(self):
        self.assertEqual(robust_slope([0, 1000, 2000, 3000], [2, 2, 2, 2]), 0.0)

    def test_positive_and_negative_trends(self):
        self.assertGreater(robust_slope([0, 1000, 2000, 3000], [1, 2, 3, 4]), 0)
        self.assertLess(robust_slope([0, 1000, 2000, 3000], [4, 3, 2, 1]), 0)


if __name__ == "__main__":
    unittest.main()
