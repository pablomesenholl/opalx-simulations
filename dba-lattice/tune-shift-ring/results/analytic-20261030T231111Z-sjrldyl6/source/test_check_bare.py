"""Unit checks of validation gates using the archived single-rank COF result."""
import copy
import json
from pathlib import Path
import unittest
from check_bare import compare


class BareComparisonTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = json.loads((root/'config.json').read_text())
        self.result = json.loads((Path(__file__).parent/'bare-reference.json').read_text())

    def test_recorded_proton_map(self):
        report = compare(self.config, self.result)
        self.assertLess(max(abs(v) for v in report['tune_difference']), 1e-9)
        self.assertGreater(report['opalx_branch_assigned_tunes'][1], 1.)

    def test_reject_wrong_map_and_unstable(self):
        bad = copy.deepcopy(self.result)
        bad['matrix'][0][1] *= -1
        with self.assertRaises(ValueError):
            compare(self.config, bad)
        bad = copy.deepcopy(self.result)
        bad['stability'] = 'NON_UNIT_CIRCLE'
        with self.assertRaises(ValueError):
            compare(self.config, bad)


if __name__ == '__main__':
    unittest.main()
