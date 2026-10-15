import unittest

import numpy as np

from gaussian_parser import VibrationalMode
from spectrum_processing import TheoreticalSettings, calculate_theoretical_spectrum, estimate_baseline


class SpectrumProcessingTests(unittest.TestCase):
    def test_calculated_spectrum_is_normalized_and_centered(self):
        settings = TheoreticalSettings(width=20.0, normalization="unit")
        modes = [VibrationalMode(1, 1000.0, 10.0, 2.0, 0.2)]

        x, y = calculate_theoretical_spectrum(modes, 500.0, 1500.0, settings, "ir", points=1001)

        self.assertAlmostEqual(x[int(np.argmax(y))], 1000.0)
        self.assertAlmostEqual(float(np.max(y)), 1.0)

    def test_linear_baseline_subtraction(self):
        x = np.arange(6, dtype=float)
        y = 2.0 * x + np.array([0.0, 1.0, 4.0, 1.0, 0.0, 0.0])

        result = estimate_baseline(x, y, "linear")

        self.assertAlmostEqual(result.corrected[0], 0.0)
        self.assertGreater(result.corrected[2], 3.0)

    def test_als_baseline_has_input_shape(self):
        x = np.linspace(0, 100, 200)
        y = 0.01 * x + np.exp(-0.5 * ((x - 50) / 4) ** 2)

        result = estimate_baseline(x, y, "als", smoothing=1e4)

        self.assertEqual(result.baseline.shape, y.shape)
        self.assertTrue(np.isfinite(result.baseline).all())


if __name__ == "__main__":
    unittest.main()
