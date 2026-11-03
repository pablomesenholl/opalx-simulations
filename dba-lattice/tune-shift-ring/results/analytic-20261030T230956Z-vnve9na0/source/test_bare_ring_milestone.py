"""Pure preparation/analysis/process-lifecycle tests; never execute OPALX."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import pandas as pd

import bare_ring_milestone as study


class BareRingMilestoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = (study.ROOT/"bare-cof.in").read_text()
        cls.config = json.loads((study.ROOT/"config.json").read_text())
        cls.sections = study.d1_sections(cls.base)

    def make_cof(self):
        p0 = .07304823094265515
        matrix = np.zeros((4, 4))
        scale = np.diag([1., p0])
        for axis, start in (("x", 0), ("y", 2)):
            cell = study.section_optics(self.config, axis)["matrix"]
            matrix[start:start+2, start:start+2] = scale @ np.linalg.matrix_power(cell, 6) @ np.linalg.inv(scale)
        return dict(converged=True, matrix=matrix.tolist(), residual=[0.]*4, coordinates=[0.]*4,
                    position_m=self.sections[0]["origin_m"],
                    momentum_mc=(p0*np.array(self.sections[0]["normal"])).tolist())

    def make_events(self, turns=32):
        _, labels = study.make_probes(self.make_cof())
        rows = []
        for label in labels:
            axis = "x" if label["axis"] == "centre" else label["axis"]
            optics = study.section_optics(self.config, axis)
            for event in range(turns*6):
                section = self.sections[event % 6]
                theta = section["theta_rad"]
                horizontal = np.array([np.cos(theta), 0., -np.sin(theta)])
                basis = horizontal if axis == "x" else np.array([0., 1., 0.])
                phase = 2*np.pi*optics["tune"]*event/6 + .2
                x = label["amplitude_m"]*np.cos(phase)
                slope = (-label["amplitude_m"]*np.sin(phase)-optics["alpha"]*x)/optics["beta"]
                r = np.array(section["origin_m"])+x*basis
                p = .07304823094265515*(np.array(section["normal"])+slope*basis)
                rows.append(dict(particle_id=label["id"], plane_id=event % 6, turn=event//6,
                                 time_s=(event+1)*2e-7, status="crossing", bracket_start_time_s=(event+1)*2e-7-1e-11,
                                 bracket_end_time_s=(event+1)*2e-7+1e-11, fraction=.5, **dict(zip(("x_m", "y_m", "z_m"), r)),
                                 **dict(zip(("px_mc", "py_mc", "pz_mc"), p))))
        return pd.DataFrame(rows)

    def test_hash_json_and_unambiguous_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"a.json"
            study.write_json(path, dict(value=1))
            self.assertEqual(json.loads(path.read_text()), dict(value=1))
            self.assertEqual(len(study.sha256(path)), 64)
            with self.assertRaises(ValueError):
                study.write_json(path, dict(value=float("nan")))
        self.assertEqual(study.replace_one("DT=1", r"DT=\d", "DT=2"), "DT=2")
        with self.assertRaises(ValueError):
            study.replace_one("DT=1 DT=2", r"DT=\d", "DT=3")

    def test_sections_and_lattice_preserve_geometry(self):
        self.assertEqual(len(self.sections), 6)
        np.testing.assert_allclose(self.sections[0]["origin_m"], [-.5179491924311226, 0., 1.433012701892219])
        prefix = study.shifted_lattice(self.base)
        self.assertIn("LATTICE: RING = (C1_D1, C1_Q,", prefix)
        self.assertNotIn("BEAM1: BEAM", prefix)
        self.assertEqual([line for line in prefix.splitlines() if line.startswith("C")],
                         [line for line in self.base.splitlines() if line.startswith("C")])
        with self.assertRaises(ValueError):
            study.d1_sections(self.base.replace("C1_D1: DRIFT", "WRONG: DRIFT"))
        with self.assertRaises(ValueError):
            study.shifted_lattice("missing")

    def test_cof_and_tracking_decks(self):
        text = study.cof_deck(self.base, 3.125e-11, 1.5e-5)
        self.assertIn('TIMEINTEGRATOR="BORIS"', text)
        self.assertIn("ic: COF, SECTION={", text)
        self.assertIn("MAXSTEPS=2000000", text)
        self.assertEqual(text.count("ic: COF"), 1)
        self.assertNotIn("ENABLELINEARTRANSFERMAPS=TRUE", text)
        device = study.tracking_deck(self.base, 6.25e-11, 32, 9)
        self.assertIn("TYPE=NONE", device)
        self.assertIn("NPARTDIST=9", device)
        self.assertIn("TURNS=32", device)
        self.assertNotIn("ic: COF", device)
        self.assertNotIn("INITIALORBIT", device)
        for invalid in (lambda: study.cof_deck(self.base, -1., 1e-5),
                        lambda: study.tracking_deck(self.base, 1e-10, 0, 9)):
            with self.assertRaises(ValueError):
                invalid()

    def test_actual_closed_orbit_probe_launch(self):
        cof = self.make_cof()
        particles, labels = study.make_probes(cof)
        self.assertEqual(particles.shape, (9, 6))
        self.assertEqual([label["id"] for label in labels], list(range(9)))
        np.testing.assert_allclose(particles.mean(axis=0), np.r_[cof["position_m"], cof["momentum_mc"]], atol=1e-15)
        np.testing.assert_allclose(particles[:, 3:], np.tile(cof["momentum_mc"], (9, 1)), atol=0)
        self.assertEqual(labels[1]["axis"], "x")
        self.assertEqual(labels[3]["axis"], "y")
        with self.assertRaises(ValueError):
            study.make_probes(dict(cof, converged=False))

    def test_matrix_modes_are_independent_eigenphases(self):
        cof = self.make_cof()
        modes = study.matrix_tunes(cof, self.config)
        for axis in ("x", "y"):
            optics = study.section_optics(self.config, axis)
            self.assertGreater(optics["beta"], 0)
            self.assertAlmostEqual(np.linalg.det(optics["matrix"]), 1., places=13)
            self.assertAlmostEqual(modes[axis]["tune"], optics["tune"], places=12)
            self.assertLess(abs(modes[axis]["modulus_error"]), 1e-13)
        bad = np.array(cof["matrix"])
        bad[0, 2] = 1e-3
        with self.assertRaises(ValueError):
            study.matrix_tunes(dict(cof, matrix=bad), self.config)
        with self.assertRaises(ValueError):
            study.section_optics(dict(self.config, elements=[]), "x")

    def test_tune_extraction_recovers_integer_and_rejects_missing_planes(self):
        events = self.make_events()
        for particle, axis in ((1, "x"), (3, "y")):
            result = study.fit_probe_tune(events[events.particle_id == particle], self.sections,
                                          study.section_optics(self.config, axis), axis)
            expected = study.section_optics(self.config, axis)["tune"]
            self.assertAlmostEqual(result["tune"], expected, places=11)
            self.assertAlmostEqual(result["independent_tune"], expected, places=7)
            self.assertLess(abs(result["half_tunes"][1]-result["half_tunes"][0]), 1e-11)
        with self.assertRaises(ValueError):
            study.fit_probe_tune(events[(events.particle_id == 1) & (events.plane_id != 2)], self.sections,
                                 study.section_optics(self.config, "x"), "x")
        with self.assertRaises(ValueError):
            study.fit_probe_tune(events[events.particle_id == 0], self.sections,
                                 study.section_optics(self.config, "x"), "x")

    def test_endpoint_and_bitwise_passivity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"case.h5"
            with h5py.File(path, "w") as output:
                group = output.create_group("Step#2")
                group["id"] = [2, 0, 1]
                for coordinate in ("x", "y", "z", "px", "py", "pz"):
                    group[coordinate] = [2., 0., 1.]
                group.attrs["TIME"] = [1e-6]
                group.attrs["RefPartR"] = [1., 2., 3.]
                group.attrs["RefPartP"] = [4., 5., 6.]
            first = study.read_endpoint(path)
            np.testing.assert_array_equal(first["ids"], [0, 1, 2])
            self.assertTrue(study.compare_passivity(first, first)["passed"])
            second = dict(first, phase=first["phase"].copy())
            second["phase"][0, 0] = np.nextafter(0., 1.)
            self.assertFalse(study.compare_passivity(first, second)["passed"])
            second["phase"][0, 0] = -0.
            self.assertTrue(np.array_equal(first["phase"], second["phase"]))
            self.assertFalse(study.compare_passivity(first, second)["passed"])

    def test_raw_array_and_csv_comparison_preserves_signed_zero(self):
        self.assertFalse(study.arrays_bitwise_equal(np.array([0.]), np.array([-0.])))
        self.assertFalse(study.arrays_bitwise_equal(np.array([1.], dtype=np.float64), np.array([1.], dtype=np.float32)))
        self.assertFalse(study.arrays_bitwise_equal(np.array([1.]), np.array([[1.]])))
        self.assertTrue(study.arrays_bitwise_equal(np.array([-0., 1.]), np.array([-0., 1.])))
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/"a.csv", Path(directory)/"b.csv"
            header = b"particle_id,plane_id,turn,time_s,x_m\n"
            first.write_bytes(header+b"0,0,1,1e-6,0\n1,0,1,1e-6,1e-5\n")
            second.write_bytes(header+b"1,0,1,1e-6,1e-5\n0,0,1,1e-6,0\n2,0,1,1e-6,2e-5\n")
            self.assertTrue(study.raw_probe_records_equal(first, second, [0, 1]))
            second.write_bytes(header+b"1,0,1,1e-6,1e-5\n0,0,1,1e-6,-0\n")
            self.assertFalse(study.raw_probe_records_equal(first, second, [0, 1]))
            second.write_bytes(header+b"0,0,1,1e-6,0\n1,0,1,1e-6,0.00001\n")
            self.assertFalse(study.raw_probe_records_equal(first, second, [0, 1]))

    def test_record_validation_and_all_turn_central_orbit(self):
        events = self.make_events()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"events.csv"
            events.to_csv(path, index=False)
            actual = study.read_events(path, list(range(9)))
            result = study.central_orbit_errors(actual, self.sections, self.make_cof())
            self.assertLess(result["maximum_position_error_m"], 1e-14)
            self.assertLess(result["maximum_momentum_error_mc"], 1e-15)
            self.assertEqual(result["minimum_returns_per_plane"], 32)
            actual.loc[0, "bracket_start_time_s"] = actual.loc[0, "time_s"]
            actual.loc[0, "bracket_end_time_s"] = actual.loc[0, "time_s"]
            actual.to_csv(path, index=False)
            self.assertEqual(len(study.read_events(path, list(range(9)))), len(actual))
            central = actual[actual.particle_id == 0].index[8]
            actual.loc[central, "y_m"] += 1e-6
            result = study.central_orbit_errors(actual, self.sections, self.make_cof())
            self.assertGreater(result["maximum_position_error_m"], 9e-7)
            actual.loc[0, "status"] = "missing"
            actual.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                study.read_events(path, list(range(9)))
            actual.loc[0, "status"] = "crossing"
            actual.loc[0, "fraction"] = 2.
            actual.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                study.read_events(path, list(range(9)))

    def test_tangential_phase_slip_is_distinct_from_geometric_orbit_error(self):
        cof = self.make_cof()
        endpoint = dict(ids=np.array([0]), phase=np.array([[0., 0., -2e-8, 0., 0., .07304823094265515]]),
                        reference_r=np.array(cof["position_m"]), reference_p=np.array(cof["momentum_mc"]))
        result = study.endpoint_orbit_errors(endpoint, cof)
        self.assertEqual(result["transverse_particle_reference_distance_m"], 0.)
        self.assertEqual(result["particle_reference_distance_m"], 2e-8)
        self.assertEqual(result["simultaneous_longitudinal_slip_m"], -2e-8)
        self.assertLess(result["equivalent_flight_time_slip_s"], 0.)
        self.assertEqual(result["reference_return_distance_m"], 0.)
        endpoint["phase"][0, 0] = 2e-8
        result = study.endpoint_orbit_errors(endpoint, cof)
        self.assertGreater(result["transverse_particle_reference_distance_m"], study.ORBIT_TOLERANCE_M)
        endpoint["phase"][0, 3] = 2e-10
        self.assertGreater(study.endpoint_orbit_errors(endpoint, cof)["transverse_particle_reference_momentum_mc"], study.MOMENTUM_TOLERANCE_MC)

    def test_streamed_schedule_hash_excludes_wall_clock_and_keeps_all_steps(self):
        lines = [f"OPAL-X> * clock {i} Step {i} container[0] Turn 1, angle=21.308 [deg], t= {i*.0625:g} [ns], E=2.500 [MeV]\n" for i in range(2001)]
        first, second = {}, {}
        output = io.StringIO()
        study.filter_step_log(io.StringIO("header\n"+"".join(lines)+"footer\n"), output, first)
        study.filter_step_log(io.StringIO("".join(line.replace("clock", "other clock") for line in lines)), io.StringIO(), second)
        self.assertEqual(first["accepted_step_log_count"], 2001)
        self.assertEqual(first["accepted_step_log_sha256"], second["accepted_step_log_sha256"])
        self.assertEqual(len(study.numerical_step_records(output.getvalue())), 3)
        self.assertIn("header", output.getvalue())
        self.assertIn("footer", output.getvalue())
        study.filter_step_log(io.StringIO("".join(lines[:-1])), io.StringIO(), second)
        self.assertNotEqual(first["accepted_step_log_sha256"], second["accepted_step_log_sha256"])

    def test_execute_owns_success_and_timeout_without_opalx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root/"fake"
            fake.write_text("#!/bin/sh\nprintf 'Step 0 t= 1 [ns]\\nfinished\\n'\n")
            fake.chmod(0o755)
            result = study.execute(root, fake, {}, 5)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["accepted_step_log_count"], 1)
            self.assertTrue(result["child_reaped"])
            fake.write_text("#!/bin/sh\nexec sleep 2\n")
            with self.assertRaises(RuntimeError):
                study.execute(root, fake, {}, .01)
            result = json.loads((root/"run-status.json").read_text())
            self.assertEqual(result["status"], "timed_out")
            self.assertTrue(result["child_reaped"])

    def test_preparation_freezes_inputs_and_explicit_text_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root/"binary"
            fake.write_bytes(b"frozen binary")
            output = root/"study"
            with mock.patch.object(study, "snapshot_sources", return_value=({}, [])):
                actual = study.prepare(fake, output)
            self.assertEqual(actual, output)
            self.assertEqual((output/"opalx").read_bytes(), fake.read_bytes())
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(len(manifest["cases"]), 4)
            study.write_json(output/"cof-dt1-fd1/bare-cof.json", self.make_cof())
            manifest = study.prepare_device_cases(output)
            self.assertEqual(len(manifest["cases"]), 8)
            self.assertEqual((output/"device-coarse-all/case.in").read_text(), (output/"device-coarse-off/case.in").read_text())
            self.assertEqual((output/"device-coarse-all/particles.txt").read_bytes(), (output/"device-coarse-off/particles.txt").read_bytes())
            config = (output/"device-coarse-three/passive-probes.txt").read_text()
            self.assertIn("IDS 0 1 3\n", config)
            self.assertEqual(config.count("PLANE "), 6)
            self.assertFalse((output/"device-coarse-off/passive-probes.txt").exists())
            with self.assertRaises(ValueError):
                study.prepare(fake, output)

    def test_complete_analysis_requires_all_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root/"binary"
            fake.write_bytes(b"fake")
            output = root/"study"
            with mock.patch.object(study, "snapshot_sources", return_value=({}, [])):
                study.prepare(fake, output)
            for case in (output/name for name in ("cof-dt0-fd0", "cof-dt0-fd1", "cof-dt1-fd0", "cof-dt1-fd1")):
                study.write_json(case/"bare-cof.json", self.make_cof())
            manifest = study.prepare_device_cases(output)
            events = self.make_events()
            for case in manifest["cases"]:
                path = output/case["name"]
                study.write_json(path/"run-status.json", dict(status="completed", binary_sha256=manifest["binary_sha256"],
                                 accepted_step_log_count=100, accepted_step_log_sha256="same"))
                if case.get("ids"):
                    events[events.particle_id.isin(case["ids"])].to_csv(path/"passive-probes.csv", index=False)
            endpoint = dict(group="Step#1", ids=np.arange(9), phase=np.zeros((9, 6)), time=np.array([1e-6]),
                            reference_r=np.array(self.make_cof()["position_m"]), reference_p=np.array(self.make_cof()["momentum_mc"]))
            with mock.patch.object(study, "read_endpoint", return_value=endpoint):
                result = study.analyze(output)
            self.assertTrue(result["passed"], result["gates"])
            status = json.loads((output/"device-coarse-three/run-status.json").read_text())
            status["accepted_step_log_sha256"] = "different"
            study.write_json(output/"device-coarse-three/run-status.json", status)
            with mock.patch.object(study, "read_endpoint", return_value=endpoint):
                result = study.analyze(output)
            self.assertFalse(result["passed"])
            self.assertFalse(result["gates"]["device-coarse-three_passive"])

    def test_run_orchestration_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"opalx").write_bytes(b"fake")
            manifest = dict(status="prepared", binary_sha256=study.sha256(root/"opalx"), cases=[dict(name="cof", kind="cof")])
            study.write_json(root/"manifest.json", manifest)
            device = dict(cases=[dict(name="device", kind="device", environment={})])
            with mock.patch.object(study, "execute") as execute, mock.patch.object(study, "prepare_device_cases", return_value=device), mock.patch.object(study, "analyze", return_value=dict(passed=True)):
                self.assertTrue(study.run_study(root)["passed"])
                self.assertEqual(execute.call_count, 2)
            with mock.patch.object(study, "run_study", return_value=dict(passed=False)), mock.patch("builtins.print"):
                self.assertEqual(study.main(["--run", str(root)]), 1)
            with mock.patch.object(study, "analyze", return_value=dict(passed=True)), mock.patch("builtins.print"):
                self.assertEqual(study.main(["--analyze", str(root)]), 0)
            with mock.patch.object(study, "prepare", return_value=root), mock.patch("builtins.print"):
                self.assertEqual(study.main([]), 0)


if __name__ == "__main__":
    unittest.main()
