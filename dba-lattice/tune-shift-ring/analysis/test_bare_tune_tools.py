import json
from pathlib import Path
import unittest
import numpy as np
from bare_tune_tools import cell_optics, section_returns, phase_tune, section_frequency, laboratory_coordinates, cell_crossing_tune, matched_phase_difference


class BareTuneTests(unittest.TestCase):
    def test_interpolation_comparison_uses_matching_events(self):
        t = np.arange(65)/6
        first = dict(time=t[:-1]*1e-6, cells=np.arange(64)%6,
                     turns=t[:-1], phase=.77*t[:-1]+.2)
        second = dict(time=t*1e-6+1e-16, cells=np.arange(65)%6,
                      turns=t, phase=.770001*t+4.2)
        second['phase'][-1] += .1  # Extra terminal stencil is deliberately bad.
        difference, count = matched_phase_difference(first, second, 1e-6, 6)
        self.assertEqual(count, 64)
        self.assertAlmostEqual(difference, 1e-6, places=13)

    def test_exact_cell_optics(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
        self.assertAlmostEqual(cell_optics(config, 'x')['tune'], .7701075691328534, places=12)
        self.assertAlmostEqual(cell_optics(config, 'y')['tune'], 1.689536397020211, places=12)

    def test_section_returns_and_integer_tunes(self):
        period = 1.1797e-6
        time = np.arange(.003, 64.003, .0071)*period
        z = np.sin(2*np.pi*time/period)
        pz = np.cos(2*np.pi*time/period)
        returns = section_returns(time, z, pz, period)
        np.testing.assert_allclose(returns/period, np.arange(1, len(returns)+1), atol=1e-9, rtol=0)
        for tune in (.7701075691, 1.689536397):
            phase = 2*np.pi*tune*time/period + .2*np.sin(2*np.pi*time/period)
            amplitude, beta, alpha = 1e-5, 12., -0.3
            x = amplitude*np.cos(phase)
            xp = (-amplitude*np.sin(phase)-alpha*x)/beta
            result = phase_tune(time, x, xp, returns, beta, alpha)
            self.assertAlmostEqual(result['tune'], tune, places=9)
            np.testing.assert_allclose(result['half_tunes'], tune, atol=2e-9, rtol=0)

    def test_reject_missing_turns_and_zero_amplitude(self):
        t = np.linspace(0, 1, 100)
        with self.assertRaises(ValueError):
            section_returns(t, np.ones(100), np.ones(100), .1)
        with self.assertRaises(ValueError):
            phase_tune(t, np.zeros(100), np.zeros(100), np.arange(20), 1., 0.)

    def test_independent_frequency_fit_with_offset_and_phase(self):
        for tune in (.22989243, .310463603):
            signal = 1e-5*np.cos(2*np.pi*tune*np.arange(64)+.3)+2e-4
            self.assertAlmostEqual(section_frequency(signal), tune, places=8)

    def test_own_cell_crossings_and_laboratory_transform(self):
        period = 1e-6
        t = np.arange(.001,32,.0031)
        angle = 2*np.pi*t
        direction = np.column_stack([-np.sin(angle),np.zeros(len(t)),np.cos(angle)])
        horizontal = np.column_stack([np.cos(angle),np.zeros(len(t)),np.sin(angle)])
        reference = horizontal.copy()
        qx,qy = .770107569,1.689536397
        r = np.column_stack([1e-5*np.cos(2*np.pi*qx*t),1e-5*np.cos(2*np.pi*qy*t),np.zeros(len(t))])
        p = np.column_stack([-1e-5*np.sin(2*np.pi*qx*t),-1e-5*np.sin(2*np.pi*qy*t),np.ones(len(t))])
        lab_r,lab_p = laboratory_coordinates(reference,direction,r,p)
        np.testing.assert_allclose(lab_r,reference+horizontal*r[:,0,None]+np.array([0,1,0])*r[:,1,None],atol=1e-15)
        poses=[(np.array([np.cos(phi),0,np.sin(phi)]),-phi) for phi in np.arange(6)*np.pi/3]
        for axis, expected in [('x',qx),('y',qy)]:
            result=cell_crossing_tune(t*period,lab_r,lab_p,poses,period,1.,0.,axis,bend_radius=1.)
            self.assertAlmostEqual(result['tune'],expected,places=7)
            self.assertAlmostEqual(result['independent_tune'],expected,places=7)


if __name__ == '__main__':
    unittest.main()
