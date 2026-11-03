#!/usr/bin/env python3
"""Unit tests for the hard-edge DBA ring analytic model."""

import unittest

import numpy as np

from generate_analytic import cell, describe


class DbaRingAnalyticTest(unittest.TestCase):
    def test_selected_ring_is_stable(self):
        result = describe(2.84)
        self.assertTrue(result["stable"])
        self.assertAlmostEqual(result["qx_principal"], 0.2298924308671466, places=13)
        self.assertAlmostEqual(result["qy_principal"], 0.3104636029797891, places=13)

    def test_six_cell_determinant(self):
        ring = np.linalg.matrix_power(cell(2.84), 6)
        self.assertAlmostEqual(np.linalg.det(ring), 1.0, places=11)


if __name__ == "__main__":
    unittest.main()
