"""Reproducible bare DBA orbit/map/device validation with passive probes.

This is a study runner, not a public OPALX API. It freezes its executable and
source evidence, runs one MPI rank, and reports failing numerical gates without
changing tolerances. The physical model is the static magnetic DBA (no RF, no
space charge). Tune integers are identified using six equivalent straight
sections; the analytic model is only the independent comparison, not fit input.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import h5py
import numpy as np
import pandas as pd

from bare_tune_tools import cell_optics, section_frequency
from kv_model import plane_map
from run_gaussian_bare import snapshot_sources


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
TUNE_TOLERANCE = 2e-6
ORBIT_TOLERANCE_M = 1e-9
MOMENTUM_TOLERANCE_MC = 1e-10


def sha256(path):
    """Hash frozen input or output bytes for reproducibility."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    """Write finite JSON; missing evidence must not silently become NaN."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def replace_one(text, pattern, replacement):
    """Require one specific input replacement, never a silent stale deck edit."""
    result, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one match, found {count}: {pattern}")
    return result


def d1_sections(base, cells=6):
    """Parse equivalent field-free D1 midplanes without adding lattice elements."""
    sections = []
    for cell in range(1, cells + 1):
        command = re.search(rf"^C{cell}_D1:\s*DRIFT\b[^;]*;", base, re.MULTILINE)
        if command is None:
            raise ValueError(f"Missing C{cell}_D1")
        values = {}
        for name in ("X", "Y", "Z", "THETA", "PHI", "PSI", "L"):
            value = re.search(rf"\b{name}\s*=\s*([^,;]+)", command[0])
            if value is None:
                raise ValueError(f"Missing explicit {name} in {command[0]}")
            values[name] = float(value[1])
        if abs(values["PHI"]) + abs(values["PSI"]) > 1e-14 or values["L"] <= 0:
            raise ValueError("Expected a positive planar unrolled D1 drift")
        theta = values["THETA"]
        normal = np.array([np.sin(theta), 0., np.cos(theta)])
        origin = np.array([values[k] for k in ("X", "Y", "Z")]) + values["L"] / 2 * normal
        sections.append(dict(plane_id=cell-1, origin_m=origin.tolist(), normal=normal.tolist(),
                             theta_rad=theta, radius_m=.05))
    return sections


def shifted_lattice(base):
    """Start the ring's element order in D1, leaving its physical drift unbroken."""
    prefix = base.split("BEAM1: BEAM", 1)[0]
    match = re.search(r"^LATTICE:\s*RING\s*=\s*\(([^)]*)\);", prefix, re.MULTILINE)
    if match is None:
        raise ValueError("Missing explicit LATTICE RING")
    names = [name.strip() for name in match[1].split(",")]
    first = names.index("C1_D1")
    prefix = prefix[:match.start()] + "LATTICE: RING = (" + ", ".join(names[first:] + names[:first]) + ");" + prefix[match.end():]
    prefix = prefix.replace("ENABLELINEARTRANSFERMAPS=TRUE", "ENABLELINEARTRANSFERMAPS=FALSE")
    # These are dump controls only; they never alter physical stepping.
    prefix = re.sub(r"\bPSDUMPFREQ=\d+", "PSDUMPFREQ=1000000000", prefix)
    prefix = re.sub(r"\bSTATDUMPFREQ=\d+", "STATDUMPFREQ=1000000000", prefix)
    return "OPTION, STEPINFOFQ=1000000, REPARTFREQ=0;\n" + prefix


def cof_deck(base, dt, fd, integrator="BORIS"):
    """Find a fresh D1-centre orbit and four-dimensional finite-difference map."""
    if dt <= 0 or fd <= 0 or integrator not in ("BORIS", "DOP853"):
        raise ValueError("Positive DT/FD and supported reference integrator required")
    section = d1_sections(base)[0]
    tail = base[base.index("BEAM1: BEAM"):]
    tail = replace_one(tail, r"\bDT\s*=\s*[^,;]+", f"DT={dt:.17g}")
    tail = replace_one(tail, r'\bTIMEINTEGRATOR="[^"]+"', f'TIMEINTEGRATOR="{integrator}"')
    tail = replace_one(tail, r"\bMAXSTEPS=\d+", "MAXSTEPS=2000000")
    p0 = .07304823094265515
    steps = ",".join(f"{v:.17g}" for v in (fd, fd*p0, fd, fd*p0))
    tail = replace_one(tail, r"\bFDSTEP\s*=\s*\{[^}]+\}", "FDSTEP={" + steps + "}")
    pose = [*section["origin_m"], section["theta_rad"], 0., 0.]
    tail = replace_one(tail, r"^ic:\s*COF,", "ic: COF, SECTION={" + ",".join(f"{v:.17g}" for v in pose) + "},")
    return shifted_lattice(base) + tail


def make_probes(cof, amplitudes=(1e-5, 5e-6)):
    """Launch centre and +/- x/y offsets from the actual solved COF state."""
    if not cof["converged"] or any(a <= 0 for a in amplitudes):
        raise ValueError("A converged COF and positive probe amplitudes are required")
    position, momentum = np.array(cof["position_m"]), np.array(cof["momentum_mc"])
    direction = momentum / np.linalg.norm(momentum)
    if abs(direction[1]) > 1e-12:
        raise ValueError("This study requires a planar reference")
    horizontal = np.array([direction[2], 0., -direction[0]])
    rows, labels = [np.r_[position, momentum]], [dict(id=0, axis="centre", amplitude_m=0.)]
    for amplitude in amplitudes:
        for axis, basis in (("x", horizontal), ("y", np.array([0., 1., 0.]))):
            for sign in (1, -1):
                labels.append(dict(id=len(rows), axis=axis, amplitude_m=sign*amplitude))
                rows.append(np.r_[position + sign*amplitude*basis, momentum])
    return np.array(rows), labels


def tracking_deck(base, dt, turns, count):
    """Run absolute COF-derived probes through ordinary device tracking."""
    if dt <= 0 or turns < 1 or count < 1:
        raise ValueError("Positive tracking DT, turns, and count required")
    return shifted_lattice(base) + f'''
DIST: DISTRIBUTION, TYPE=FROMFILE, FNAME="particles.txt", NPARTDIST={count};
SOURCE: EMISSIONSOURCE, DISTRIBUTION=DIST;
SOURCES: EMISSIONSOURCELIST=(SOURCE);
ONE: BINNING, PARAMETER="GAMMAZ", MAXBINS=1, ADAPTIVEBINNING=FALSE, DUMPBINSFILE="NONE", TABLEPRINTFREQ=0;
FS: FIELDSOLVER, TYPE=NONE, BINS=ONE, NX=16, NY=16, NZ=16,
    PARFFTX=TRUE, PARFFTY=TRUE, PARFFTZ=TRUE, BCFFTX=OPEN, BCFFTY=OPEN, BCFFTZ=OPEN;
BEAM1: BEAM, PARTICLE=PROTON, CHARGE=1, NALLOC={count}, SOURCES=SOURCES, BCHARGE={count*1.602176634e-19:.17g};
TRACK, LINE=LATTICE, BEAM=BEAM1, DT={dt:.17g}, MAXSTEPS=10000000;
RUN, METHOD="PARALLEL", FIELDSOLVER=FS, TURNS={turns};
ENDTRACK;
QUIT;
'''


def section_optics(config, axis):
    """Conjugate the exact hard-edge cell map from B1 to the D1 midpoint."""
    optics = cell_optics(config, axis)
    transfer = np.eye(2)
    found = False
    for element in config["elements"]:
        length = element["length_m"]
        if element["name"] == "D1":
            length /= 2
            found = True
        transfer = plane_map(element[f"k{axis}_m2"], length) @ transfer
        if found:
            break
    if not found:
        raise ValueError("Missing D1 in analytic lattice")
    matrix = transfer @ optics["matrix"] @ np.linalg.inv(transfer)
    phase = optics["tune"] * 2*np.pi / config["cells"]
    return dict(matrix=matrix, beta=matrix[0, 1]/np.sin(phase),
                alpha=(matrix[0, 0]-matrix[1, 1])/(2*np.sin(phase)), tune=optics["tune"])


def matrix_tunes(cof, config):
    """Extract signed stable eigenphases; label integers explicitly from optics."""
    matrix = np.asarray(cof["matrix"], float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite four-dimensional COF matrix")
    if np.max(np.abs(matrix[:2, 2:])) > 1e-10 or np.max(np.abs(matrix[2:, :2])) > 1e-10:
        raise ValueError("This study requires uncoupled transverse modes")
    result = {}
    for axis, start in (("x", 0), ("y", 2)):
        block = matrix[start:start+2, start:start+2]
        eigenvalues = np.linalg.eigvals(block)
        chosen = eigenvalues[np.sign(eigenvalues.imag) == np.sign(block[0, 1])]
        if len(chosen) != 1 or abs(chosen[0].imag) < 1e-12:
            raise ValueError("No isolated stable mode")
        eigenvalue = chosen[0]
        if abs(abs(eigenvalue)-1) > 2e-5:
            raise ValueError("Non-unit eigenvalue modulus exceeds diagnostic tolerance")
        fractional = float((np.angle(eigenvalue)/(2*np.pi)) % 1)
        analytic = float(cell_optics(config, axis)["tune"])
        tune = fractional + round(analytic-fractional)
        result[axis] = dict(tune=tune, fractional_tune=fractional, analytic_tune=analytic,
                            modulus_error=float(abs(eigenvalue)-1), integer_branch_source="analytic cell phase")
    return result


def fit_probe_tune(events, sections, optics, axis):
    """Fit actual own-plane crossing coordinates, retaining integer phase advance."""
    events = events.sort_values("time_s").reset_index(drop=True)
    cells = len(sections)
    if len(events) < 8*cells:
        raise ValueError("Fewer than eight complete turns of observations")
    ids = events.plane_id.to_numpy(dtype=int)
    if np.any(np.diff(ids) % cells != 1):
        raise ValueError("Missing, duplicated, or out-of-order plane crossings")
    coordinate, slope = [], []
    for row in events.itertuples(index=False):
        section = sections[int(row.plane_id)]
        theta = section["theta_rad"]
        horizontal = np.array([np.cos(theta), 0., -np.sin(theta)])
        normal = np.array(section["normal"])
        r = np.array([row.x_m, row.y_m, row.z_m]) - section["origin_m"]
        p = np.array([row.px_mc, row.py_mc, row.pz_mc])
        if abs(r @ normal) > 2e-9 or p @ normal <= 0:
            raise ValueError("Invalid directed observation-plane event")
        basis = horizontal if axis == "x" else np.array([0., 1., 0.])
        coordinate.append(r @ basis)
        slope.append((p @ basis)/(p @ normal))
    coordinate, slope = np.array(coordinate), np.array(slope)
    normalized = coordinate + 1j*(optics["alpha"]*coordinate + optics["beta"]*slope)
    if np.min(abs(normalized)) < 1e-12:
        raise ValueError("Probe signal too small for tune extraction")
    phase = -np.unwrap(np.angle(normalized))/(2*np.pi)
    if np.max(abs(np.diff(phase))) >= .4:
        raise ValueError("Equivalent-cell sampling is too sparse or trajectory inaccurate")
    turns = np.arange(len(phase))/cells
    tune, offset = np.polyfit(turns, phase, 1)
    middle = len(phase)//2
    halves = [float(np.polyfit(turns[sl], phase[sl], 1)[0]) for sl in (slice(0, middle), slice(middle, None))]
    return dict(tune=float(tune), independent_tune=float(cells*section_frequency(coordinate)),
                half_tunes=halves, phase_residual_rms=float(np.std(phase-(tune*turns+offset))),
                event_count=len(events), energy_relative_span=float(np.ptp(np.sqrt(1+(events[["px_mc", "py_mc", "pz_mc"]].to_numpy()**2).sum(axis=1))-1)/(np.sqrt(1+np.sum(events[["px_mc", "py_mc", "pz_mc"]].iloc[0].to_numpy()**2))-1)))


def read_events(path, requested_ids):
    """Reject missing IDs, invalid brackets, and any non-crossing output rows."""
    events = pd.read_csv(path)
    columns = ("particle_id", "plane_id", "turn", "time_s", "x_m", "y_m", "z_m", "px_mc", "py_mc", "pz_mc",
               "bracket_start_time_s", "bracket_end_time_s", "fraction")
    if "status" not in events or not (events.status == "crossing").all():
        raise ValueError("Passive records contain missing or non-crossing observations")
    if not np.isfinite(events[list(columns)].to_numpy()).all() or set(events.particle_id.unique()) != set(requested_ids):
        raise ValueError("Invalid passive coordinates or unexpected/missing probe IDs")
    duration = events.bracket_end_time_s-events.bracket_start_time_s
    estimate = events.bracket_start_time_s + events.fraction*duration
    tolerance = 8*np.finfo(float).eps*np.maximum(abs(estimate), 1e-12)
    # Tiny accepted physical steps may share a rounded absolute timestamp.
    # Their spatial fraction remains meaningful; a decreasing timestamp is not.
    if (duration < 0).any() or ((events.fraction < 0) | (events.fraction > 1)).any() or (abs(estimate-events.time_s) > tolerance).any():
        raise ValueError("Invalid passive crossing time bracket or interpolation fraction")
    for _, group in events.groupby(["particle_id", "plane_id"]):
        if np.any(np.diff(group.sort_values("time_s").turn.to_numpy()) != 1):
            raise ValueError("Missing or duplicated counted plane returns")
    return events


def central_orbit_errors(events, sections, cof):
    """Compare every central crossing with the six symmetry-related COF states.

    This static DBA has six identical planar cells. Their D1-section maps and
    closed coordinates are related by rigid rotation about the vertical axis.
    No design-orbit assumption replaces the actual solved COF offset/momentum.
    """
    central = events[events.particle_id == 0]
    if set(central.plane_id.unique()) != set(range(len(sections))):
        raise ValueError("Central probe did not reach every observation plane")
    theta0 = sections[0]["theta_rad"]
    rotation0 = np.array([[np.cos(theta0), 0., np.sin(theta0)], [0., 1., 0.], [-np.sin(theta0), 0., np.cos(theta0)]])
    local_r = (np.array(cof["position_m"])-sections[0]["origin_m"]) @ rotation0
    local_p = np.array(cof["momentum_mc"]) @ rotation0
    position_error, momentum_error = [], []
    for row in central.itertuples(index=False):
        section = sections[int(row.plane_id)]
        theta = section["theta_rad"]
        rotation = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        expected_r = np.array(section["origin_m"])+rotation@local_r
        expected_p = rotation@local_p
        position_error.append(np.linalg.norm(np.array([row.x_m, row.y_m, row.z_m])-expected_r))
        momentum_error.append(np.linalg.norm(np.array([row.px_mc, row.py_mc, row.pz_mc])-expected_p))
    return dict(maximum_position_error_m=float(max(position_error)), maximum_momentum_error_mc=float(max(momentum_error)),
                minimum_returns_per_plane=int(central.groupby("plane_id").size().min()), observations=len(central))


def read_endpoint(path):
    """Read full final particle/reference state, including accepted-step metadata."""
    with h5py.File(path) as data:
        keys = sorted((key for key in data if re.fullmatch(r"Step#\d+", key)), key=lambda key: int(key[5:]))
        if not keys:
            raise ValueError("Missing H5 particle snapshots")
        group = data[keys[-1]]
        order = np.argsort(group["id"][:])
        return dict(group=keys[-1], ids=group["id"][:][order],
                    phase=np.column_stack([group[key][:][order] for key in ("x", "y", "z", "px", "py", "pz")]),
                    time=np.asarray(group.attrs["TIME"]), reference_r=np.asarray(group.attrs["RefPartR"]),
                    reference_p=np.asarray(group.attrs["RefPartP"]))


def compare_passivity(first, second):
    """Require bitwise identical numerical endpoint despite probe output changes."""
    checks = {key: arrays_bitwise_equal(first[key], second[key]) for key in ("ids", "phase", "time", "reference_r", "reference_p")}
    checks["same_output_step"] = first["group"] == second["group"]
    return dict(checks=checks, passed=all(checks.values()))


def arrays_bitwise_equal(first, second):
    """Compare array shape, dtype and raw logical-order bytes, including signed zero."""
    first, second = np.asarray(first), np.asarray(second)
    return first.shape == second.shape and first.dtype == second.dtype and first.tobytes(order="C") == second.tobytes(order="C")


def raw_probe_records_equal(first_path, second_path, selected_ids):
    """Compare exact header/CSV row bytes for common IDs without float parsing."""
    records = []
    headers = []
    selected_ids = set(selected_ids)
    for path in (first_path, second_path):
        lines = Path(path).read_bytes().splitlines(keepends=True)
        if not lines or not lines[0].startswith(b"particle_id,plane_id,turn,"):
            raise ValueError("Unexpected passive CSV header")
        headers.append(lines[0])
        keyed = []
        for line in lines[1:]:
            columns = line.split(b",")
            key = tuple(int(value) for value in columns[:3])
            if key[0] in selected_ids:
                keyed.append((key, line))
        if len({key for key, _ in keyed}) != len(keyed):
            raise ValueError("Duplicate passive particle/plane/turn rows")
        records.append(sorted(keyed))
    return headers[0] == headers[1] and records[0] == records[1]


def endpoint_orbit_errors(endpoint, cof):
    """Separate transverse orbit error from simultaneous longitudinal phase slip.

    A geometrically closed trajectory can have a slightly different numerical
    time of flight. Thus equal-time z displacement is retained as timing evidence,
    while geometric closure is tested independently at each directed section.
    The endpoint frame follows the reference, with z tangent to its momentum.
    """
    centre = np.flatnonzero(endpoint["ids"] == 0)
    if len(centre) != 1:
        raise ValueError("Missing or duplicated central probe")
    state = endpoint["phase"][centre[0]]
    position = state[:3]
    momentum = state[3:]
    p0 = np.linalg.norm(cof["momentum_mc"])
    speed = 299792458. * p0 / np.sqrt(1+p0*p0)
    return dict(particle_reference_distance_m=float(np.linalg.norm(position)),
                transverse_particle_reference_distance_m=float(np.linalg.norm(position[:2])),
                transverse_particle_reference_momentum_mc=float(np.linalg.norm(momentum[:2])),
                simultaneous_longitudinal_slip_m=float(position[2]),
                equivalent_flight_time_slip_s=float(position[2]/speed),
                reference_return_distance_m=float(np.linalg.norm(endpoint["reference_r"]-cof["position_m"])),
                reference_return_momentum_error_mc=float(np.linalg.norm(endpoint["reference_p"]-cof["momentum_mc"])))


def numerical_step_records(log):
    """Discard wall-clock prefixes but retain every printed accepted step/time."""
    return re.findall(r"\bStep\s+(\d+)[^\n]*?\bt\s*=\s*([+\-\d.eE]+)\s*\[([^]]+)\]", log)


def filter_step_log(source, target, result):
    """Hash every numeric step record while retaining one line per 1000 steps."""
    digest = hashlib.sha256()
    count = 0
    for line in source:
        records = numerical_step_records(line)
        if records:
            for record in records:
                digest.update((" ".join(record) + "\n").encode())
                count += 1
            if count == 1 or count % 1000 == 0:
                target.write(line)
        else:
            target.write(line)
    result.update(accepted_step_log_sha256=digest.hexdigest(), accepted_step_log_count=count,
                  schedule_hash_scope="Every emitted Step index and printed time/unit; wall-clock prefixes excluded. Printed times have OPALX log precision.")


def prepare_device_cases(study):
    """Use one freshly solved orbit for every DT and passive-selection case."""
    study = Path(study)
    manifest = json.loads((study/"manifest.json").read_text())
    base = (study/"bare-cof.in").read_text()
    cof = json.loads((study/"cof-dt1-fd1/bare-cof.json").read_text())
    sections = json.loads((study/"sections.json").read_text())
    particles, labels = make_probes(cof)
    write_json(study/"launch.json", dict(labels=labels, source_cof="cof-dt1-fd1/bare-cof.json",
                                        absolute_lab_state=particles.tolist()))
    specs = [("device-coarse-off", manifest["dt_s"], []),
             ("device-coarse-three", manifest["dt_s"], [0, 1, 3]),
             ("device-coarse-all", manifest["dt_s"], list(range(len(particles)))),
             ("device-fine-all", manifest["dt_s"]/2, list(range(len(particles))))]
    for name, dt, ids in specs:
        case = study/name
        case.mkdir()
        # One line per accepted step proves schedule invariance directly;
        # physical trajectories are compared independently in the final H5.
        deck = tracking_deck(base, dt, manifest["turns"], len(particles)).replace("STEPINFOFQ=1000000", "STEPINFOFQ=1")
        (case/"case.in").write_text(deck)
        np.savetxt(case/"particles.txt", particles[:, [0, 3, 1, 4, 2, 5]],
                   header=f"{len(particles)}\nx px y py z pz", comments="", fmt="%.17g")
        env = {}
        if ids:
            config = "OUTPUT passive-probes.csv\nIDS " + " ".join(str(i) for i in ids) + "\n"
            config += "".join("PLANE " + str(s["plane_id"]) + " " + " ".join(f"{v:.17g}" for v in [*s["origin_m"], *s["normal"]]) + "\n" for s in sections)
            (case/"passive-probes.txt").write_text(config)
            env["OPALX_TEST_PASSIVE_RING_PROBES"] = "passive-probes.txt"
        manifest["cases"].append(dict(name=name, kind="device", dt_s=dt, ids=ids, environment=env,
                                      input_sha256=sha256(case/"case.in"), particles_sha256=sha256(case/"particles.txt")))
        if ids:
            manifest["cases"][-1]["probe_config_sha256"] = sha256(case/"passive-probes.txt")
    manifest["status"] = "device_prepared"
    write_json(study/"manifest.json", manifest)
    return manifest


def analyze(study):
    """Report independent closure, matrix, DT, amplitude, and passivity gates."""
    study = Path(study)
    manifest = json.loads((study/"manifest.json").read_text())
    config = json.loads((study/"config.json").read_text())
    sections = json.loads((study/"sections.json").read_text())
    labels = json.loads((study/"launch.json").read_text())["labels"]
    maps, measurements, gates, observed_orbit = {}, [], {}, {}
    launch_cof = json.loads((study/"cof-dt1-fd1/bare-cof.json").read_text())
    for case in manifest["cases"]:
        directory = study/case["name"]
        status = json.loads((directory/"run-status.json").read_text())
        if status["status"] != "completed" or status["binary_sha256"] != manifest["binary_sha256"]:
            raise ValueError(f"Incomplete or different-binary case: {directory}")
        if sha256(directory/"case.in") != case["input_sha256"]:
            raise ValueError(f"Input changed after preparation: {directory}")
        if case["kind"] == "device" and sha256(directory/"particles.txt") != case["particles_sha256"]:
            raise ValueError(f"Particle launch changed after preparation: {directory}")
        if case["kind"] == "cof":
            cof = json.loads((directory/"bare-cof.json").read_text())
            modes = matrix_tunes(cof, config)
            maps[case["name"]] = dict(modes=modes, residual=cof["residual"], coordinates=cof["coordinates"])
            gates[case["name"]+"_closed"] = bool(cof["converged"] and max(abs(np.array(cof["residual"])[[0, 2]])) < ORBIT_TOLERANCE_M
                and max(abs(np.array(cof["residual"])[[1, 3]])) < MOMENTUM_TOLERANCE_MC)
        elif case["ids"]:
            if sha256(directory/"passive-probes.txt") != case["probe_config_sha256"]:
                raise ValueError(f"Probe configuration changed after preparation: {directory}")
            events = read_events(directory/"passive-probes.csv", case["ids"])
            errors = central_orbit_errors(events, sections, launch_cof)
            observed_orbit[case["name"]] = errors
            gates[case["name"]+"_orbit_all_sections_turns"] = errors["maximum_position_error_m"] < ORBIT_TOLERANCE_M and errors["maximum_momentum_error_mc"] < MOMENTUM_TOLERANCE_MC and errors["minimum_returns_per_plane"] >= manifest["turns"]-1
            if len(case["ids"]) == len(labels):
                for label in labels[1:]:
                    axis = label["axis"]
                    result = fit_probe_tune(events[events.particle_id == label["id"]], sections, section_optics(config, axis), axis)
                    measurements.append(dict(case=case["name"], **label, **result))
    baseline = maps["cof-dt1-fd1"]["modes"]
    for axis in ("x", "y"):
        tunes = [entry["modes"][axis]["tune"] for entry in maps.values()]
        gates[f"matrix_{axis}_dt_fd_convergence"] = bool(np.ptp(tunes) < TUNE_TOLERANCE)
        gates[f"matrix_{axis}_analytic_agreement"] = abs(baseline[axis]["tune"]-baseline[axis]["analytic_tune"]) < TUNE_TOLERANCE
    for row in measurements:
        row["matrix_tune"] = baseline[row["axis"]]["tune"]
        row["minus_matrix"] = row["tune"]-row["matrix_tune"]
    frame = pd.DataFrame(measurements)
    frame.to_csv(study/"tunes.csv", index=False)
    gates["device_tunes_match_matrix"] = bool(frame.minus_matrix.abs().max() < TUNE_TOLERANCE)
    gates["independent_frequency_agrees"] = bool((frame.tune-frame.independent_tune).abs().max() < TUNE_TOLERANCE)
    gates["half_record_agrees"] = max(abs(row["half_tunes"][0]-row["half_tunes"][1]) for row in measurements) < TUNE_TOLERANCE
    for axis in ("x", "y"):
        gates[f"device_{axis}_dt_amplitude_convergence"] = bool(np.ptp(frame[frame.axis == axis].tune) < TUNE_TOLERANCE)
    endpoints = {name: read_endpoint(study/name/"case.h5") for name in ("device-coarse-off", "device-coarse-three", "device-coarse-all", "device-fine-all")}
    passivity = {}
    off_steps = json.loads((study/"device-coarse-off/run-status.json").read_text())
    if not off_steps.get("accepted_step_log_count", 0):
        raise ValueError("No accepted-step records; schedule invariance is unproven")
    for name in ("device-coarse-three", "device-coarse-all"):
        result = compare_passivity(endpoints["device-coarse-off"], endpoints[name])
        other_steps = json.loads((study/name/"run-status.json").read_text())
        result["accepted_schedule_identical"] = all(off_steps[key] == other_steps.get(key) for key in ("accepted_step_log_sha256", "accepted_step_log_count"))
        result["accepted_step_records"] = off_steps["accepted_step_log_count"]
        result["passed"] &= result["accepted_schedule_identical"]
        passivity[name] = result
        gates[name+"_passive"] = result["passed"]
    # Compare original row bytes; decimal parsing must not erase any differences.
    gates["common_probe_records_identical"] = raw_probe_records_equal(study/"device-coarse-three/passive-probes.csv", study/"device-coarse-all/passive-probes.csv", [0, 1, 3])
    orbit = {}
    cof = json.loads((study/"cof-dt1-fd1/bare-cof.json").read_text())
    for name, endpoint in endpoints.items():
        if len(endpoint["ids"]) != len(labels):
            raise ValueError("Particle loss or missing central probe")
        orbit[name] = endpoint_orbit_errors(endpoint, cof)
        gates[name+"_reference_closure_and_transverse_orbit"] = max(orbit[name]["transverse_particle_reference_distance_m"], orbit[name]["reference_return_distance_m"]) < ORBIT_TOLERANCE_M and max(orbit[name]["reference_return_momentum_error_mc"], orbit[name]["transverse_particle_reference_momentum_mc"]) < MOMENTUM_TOLERANCE_MC
    coarse = orbit["device-coarse-all"]["simultaneous_longitudinal_slip_m"]
    fine = orbit["device-fine-all"]["simultaneous_longitudinal_slip_m"]
    phase_slip = dict(coarse_m=coarse, fine_m=fine, reduced_with_dt=abs(fine) < abs(coarse),
                      absolute_fine_over_coarse=None if coarse == 0 else abs(fine/coarse),
                      interpretation="Simultaneous longitudinal phase slip is a time-of-flight diagnostic, not geometric orbit nonclosure. Positive slip means ahead of the reference. The geometric 1 nm gate uses all own-plane crossings and transverse endpoint displacement.")
    report = dict(passed=all(gates.values()), gates=gates, maps=maps, measurements=measurements,
                  passivity=passivity, orbit=orbit, observed_orbit=observed_orbit, longitudinal_phase_slip=phase_slip,
                  binary_sha256=manifest["binary_sha256"],
                  scope=manifest["scope"], tune_tolerance=TUNE_TOLERANCE,
                  note="Only selected probes are observed. All particles use the same unchanged device integration. Host finite-difference matrices alone supply fractional phases; integer identification is independently verified by six-plane device records.")
    write_json(study/"summary.json", report)
    return report


def run_study(study, timeout=900):
    """Complete frozen COF cases then diagnostic-off/subset/all device cases."""
    study = Path(study).resolve()
    manifest = json.loads((study/"manifest.json").read_text())
    if manifest["status"] != "prepared" or sha256(study/"opalx") != manifest["binary_sha256"]:
        raise ValueError("Study is not fresh or frozen executable changed")
    for case in manifest["cases"]:
        execute(study/case["name"], study/"opalx", {}, timeout)
    manifest = prepare_device_cases(study)
    for case in manifest["cases"]:
        if case["kind"] == "device":
            execute(study/case["name"], study/"opalx", case["environment"], timeout)
    result = analyze(study)
    manifest["status"] = "passed" if result["passed"] else "validation_failed"
    write_json(study/"manifest.json", manifest)
    return result


def execute(directory, binary, environment, timeout):
    """Run a single owned process group; timeout always reaps that group."""
    directory, binary = Path(directory), Path(binary)
    clean = {key: value for key, value in os.environ.items() if not key.startswith("OPALX_TEST_")}
    clean.update(OMP_NUM_THREADS="1", OMP_PROC_BIND="false", OMP_DYNAMIC="false")
    clean.update(environment)
    status = dict(command=[str(binary), "--info", "2", "case.in"], environment=environment,
                  mpi_ranks=1, omp_threads=1, status="running", binary_sha256=sha256(binary), timeout_s=timeout)
    start = time.monotonic()
    with (directory/"run.log").open("w") as log:
        process = subprocess.Popen(status["command"], cwd=directory, env=clean, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        filtered = {}
        reader = threading.Thread(target=filter_step_log, args=(process.stdout, log, filtered))
        reader.start()
        status["pid"] = process.pid
        write_json(directory/"run-status.json", status)
        try:
            code = process.wait(timeout=timeout)
            status.update(returncode=code, status="completed" if code == 0 else "failed")
        except subprocess.TimeoutExpired:
            status["status"] = "timed_out"
        except BaseException:
            status["status"] = "interrupted"
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            reader.join(timeout=5)
            if reader.is_alive():
                raise RuntimeError("Owned output reader did not finish after child was reaped")
            process.stdout.close()
            status.update(filtered)
            status.update(elapsed_s=time.monotonic()-start, returncode=process.returncode, child_reaped=process.poll() is not None)
            write_json(directory/"run-status.json", status)
    if status["status"] != "completed":
        raise RuntimeError(f"Case {status['status']}: {directory}")
    return status


def prepare(binary, output=None, turns=32, dt=6.25e-11, fd=3e-5):
    """Freeze immutable executable, inputs, scripts, and dirty source evidence."""
    binary = Path(binary).resolve()
    if turns < 8 or dt <= 0 or fd <= 0:
        raise ValueError("At least eight turns and positive DT/FD required")
    study = Path(output) if output else Path(tempfile.mkdtemp(prefix="bare-ring-milestone-", dir=ROOT/"results"))
    study.mkdir(parents=True, exist_ok=True)
    if (study/"manifest.json").exists():
        raise ValueError("Refusing to overwrite an existing study")
    shutil.copy2(binary, study/"opalx")
    for source in (ROOT/"bare-cof.in", ROOT/"config.json", Path(__file__),
                   *(Path(__file__).with_name(name) for name in ("test_bare_ring_milestone.py", "bare_tune_tools.py", "kv_model.py", "run_gaussian_bare.py"))):
        shutil.copy2(source, study/source.name)
    source_hashes, deleted = snapshot_sources(REPO, study)
    base = (study/"bare-cof.in").read_text()
    sections = d1_sections(base)
    write_json(study/"sections.json", sections)
    cases = []
    for dt_index, dt_value in enumerate((dt, dt/2)):
        for fd_index, fd_value in enumerate((fd, fd/2)):
            name = f"cof-dt{dt_index}-fd{fd_index}"
            case = study/name
            case.mkdir()
            (case/"case.in").write_text(cof_deck(base, dt_value, fd_value))
            cases.append(dict(name=name, kind="cof", dt_s=dt_value, fd_m=fd_value, input_sha256=sha256(case/"case.in")))
    manifest = dict(status="prepared", binary_sha256=sha256(study/"opalx"), turns=turns, dt_s=dt, fd_m=fd,
                    mpi_ranks=1, omp_threads=1, cases=cases, source_sha256=source_hashes, deleted_source_files=deleted,
                    analysis_sha256={path.name: sha256(path) for path in study.glob("*.py")},
                    scope="Static magnetic DBA, no RF, no self fields, exact same launch for all device cases",
                    tune_tolerance=TUNE_TOLERANCE, orbit_tolerance_m=ORBIT_TOLERANCE_M,
                    momentum_tolerance_mc=MOMENTUM_TOLERANCE_MC,
                    tolerance_rationale="2e-6 tune agreement resolves anticipated ~0.1 SC shifts by 50000; 1 nm orbit gate is 1e-4 of the 10 um probe, and far below the original mm discrepancy.")
    write_json(study/"manifest.json", manifest)
    return study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=REPO/"omp-build/src/opalx")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--turns", type=int, default=32)
    parser.add_argument("--dt", type=float, default=6.25e-11)
    parser.add_argument("--fd", type=float, default=3e-5)
    parser.add_argument("--run", type=Path, help="Run an already prepared frozen study")
    parser.add_argument("--analyze", type=Path, help="Analyze an already completed study")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if args.run or args.analyze:
        result = run_study(args.run, args.timeout) if args.run else analyze(args.analyze)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["passed"] else 1
    study = prepare(args.binary, args.output, args.turns, args.dt, args.fd)
    print(study, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
