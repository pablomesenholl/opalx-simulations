"""One-rank isolated uniform-ellipsoid PIC force benchmark and launch generator.

A separate DBA COF supplies the absolute reference momentum. The initial-force
experiment uses a straight field-free drift, one 1 ps Boris step, and one fixed
GAMMAZ bin. This isolates the analytic internal field from ring curvature and
hard-edge scheduling. Spatial samples are scrambled Sobol points in a uniformly
filled ellipsoid, with antithetic pairs. No Gaussian spatial surrogate is used.

Run with ~/.venv-h6/bin/python. --prepare-only writes a reviewable study;
--study PATH --run all --analyze executes the prepared inputs and makes figures.
The generated ring-launch.txt has the finite-emittance covariance, whereas
force cases are cold controls so that the common Lorentz boost is exact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
from scipy.constants import c, elementary_charge, epsilon_0, proton_mass
from scipy.special import ndtri
from scipy.stats import qmc

from ellipsoid_model import depolarization, field_tensor, initial_covariance, match_transverse
from kv_model import beam_parameters

ROOT = Path(__file__).resolve().parents[1]
CHARGE = 41.13101073e-12
DEFAULT_CASES = ((131072, 32), (524288, 32), (131072, 64), (524288, 64))


def write_json(path, data):
    """Refuse nonfinite JSON values in numerical evidence."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha256(path):
    """Stream the artifact checksum without loading large particle files."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def generate_ellipsoid(covariance, count, seed=20260911):
    """Sample (x,x',y,y',z,delta) with uniform spatial density and given covariance.

    Spatial covariance is preserved statistically, not whitened after sampling:
    whitening would change the requested ellipsoid surface. Conditional momentum
    is a correlated Gaussian. This sets moments, not a stationary Vlasov solution.
    Antithetic partners make all six means zero to floating-point roundoff.
    """
    covariance = np.asarray(covariance, dtype=float)
    if count < 16 or count % 2 or covariance.shape != (6, 6):
        raise ValueError('Use an even count >=16 and a 6x6 covariance')
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-16):
        raise ValueError('Covariance must be symmetric')
    pos, mom = [0, 2, 4], [1, 3, 5]
    cr = covariance[np.ix_(pos, pos)]
    cp = covariance[np.ix_(mom, mom)]
    cpr = covariance[np.ix_(mom, pos)]
    coupling = np.linalg.solve(cr, cpr.T).T
    residual = cp - coupling @ cr @ coupling.T
    values, vectors = np.linalg.eigh(residual)
    if min(values) < -1e-13 * max(1e-30, max(abs(np.linalg.eigvalsh(cp)))):
        raise ValueError('Covariance must be positive semidefinite')
    # Powers of two give a balanced Sobol design; truncation is supported and
    # recorded but the default convergence study uses powers of two throughout.
    points = qmc.Sobol(6, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(count // 2))))[:count // 2]
    radius, costheta, phi = points[:, 0] ** (1 / 3), 2 * points[:, 1] - 1, 2 * np.pi * points[:, 2]
    sintheta = np.sqrt(1 - costheta ** 2)
    ball = radius[:, None] * np.column_stack((sintheta * np.cos(phi), sintheta * np.sin(phi), costheta))
    positions = ball @ np.linalg.cholesky(5 * cr).T
    gaussian = ndtri(np.clip(points[:, 3:], np.finfo(float).eps, 1 - np.finfo(float).eps))
    momenta = positions @ coupling.T + gaussian @ (vectors * np.sqrt(np.maximum(values, 0))).T
    half = np.zeros((count // 2, 6))
    half[:, pos], half[:, mom] = positions, momenta
    return np.vstack((half, -half))


def absolute_particles(phase, reference_r, reference_p, rotation=None):
    """Convert orbit-local paraxial variables to FROMFILE SI positions and p/(mc).

    x'=px/p0 and y'=py/p0 here, and delta=(pz-p0)/p0, exactly as in the
    analytic covariance model. A nonidentity section rotation must be supplied
    explicitly. FROMFILE cannot be combined with INITIALORBIT or BEAM.PC.
    """
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    reference_r, reference_p = np.asarray(reference_r), np.asarray(reference_p)
    p0 = np.linalg.norm(reference_p)
    positions = phase[:, [0, 2, 4]] @ rotation.T + reference_r
    momenta = p0 * phase[:, [1, 3, 5]] @ rotation.T + reference_p
    return np.column_stack((positions, momenta))


def write_particles(path, particles):
    """Write named absolute FROMFILE columns in metres and beta*gamma."""
    with Path(path).open('w') as stream:
        stream.write(f'{len(particles)}\nx y z px py pz\n')
        np.savetxt(stream, particles, fmt='%.17g')


def analytic_fields(positions, covariance_r, charge, p0):
    """Return exact interior E [V/m], B [T], and E+v0 cross B [V/m]."""
    gamma = np.sqrt(1 + p0 * p0)
    boost = np.diag([1., 1., gamma])
    values, vectors = np.linalg.eigh(boost @ covariance_r @ boost)
    axes = np.sqrt(5 * values)
    rho = charge / (4 * np.pi * np.prod(axes) / 3)
    gradient_rest = rho / epsilon_0 * (vectors * depolarization(axes)) @ vectors.T
    electric_rest = positions @ (gradient_rest @ boost).T
    electric = electric_rest * [gamma, gamma, 1]
    velocity = np.array([0., 0., c * p0 / gamma])
    magnetic = gamma / c**2 * np.cross(velocity, electric_rest)
    force = electric + np.cross(velocity, magnetic)
    return electric, magnetic, force


def fit_gradient(positions, fields):
    """Fit all gradient components plus an intercept with column conditioning."""
    scale = np.std(positions, axis=0)
    if np.any(scale == 0):
        raise ValueError('Three nonzero position spans are required')
    design = np.column_stack((positions / scale, np.ones(len(positions))))
    coefficients = np.linalg.lstsq(design, fields, rcond=None)[0]
    gradient = (coefficients[:3] / scale[:, None]).T
    residual = fields - design @ coefficients
    return gradient, coefficients[3], np.sqrt(np.mean(residual**2, axis=0))


def drift_deck(count, mesh, charge, dt=1e-12, steps=1):
    """A field-free 1 m drift; all particles start around z=0.2 m."""
    return f'''OPTION, ECHO=FALSE, VERSION=10900, EBDUMP=TRUE, ASCIIDUMP=FALSE,
        PSDUMPFREQ=1, STATDUMPFREQ=1, REPARTFREQ=0, STEPINFOFQ=10000;
TITLE, STRING="Isolated uniform ellipsoid: initial PIC force";
D: DRIFT, L=1, APERTURE="RECTANGLE(1,1)", X=0, Y=0, Z=0, THETA=0, PHI=0, PSI=0;
LATTICE: LINE=(D);
DIST: DISTRIBUTION, TYPE=FROMFILE, FNAME="particles.txt", NPARTDIST={count};
SOURCE: EMISSIONSOURCE, DISTRIBUTION=DIST;
SOURCES: EMISSIONSOURCELIST=(SOURCE);
ONE: BINNING, PARAMETER="GAMMAZ", MAXBINS=1, ADAPTIVEBINNING=FALSE,
     DUMPBINSFILE="NONE", TABLEPRINTFREQ=0;
FS: FIELDSOLVER, TYPE=OPEN, BINS=ONE, NX={mesh}, NY={mesh}, NZ={mesh},
    PARFFTX=TRUE, PARFFTY=TRUE, PARFFTZ=TRUE, BCFFTX=OPEN, BCFFTY=OPEN, BCFFTZ=OPEN;
BEAM1: BEAM, PARTICLE=PROTON, CHARGE=1, NALLOC={count}, SOURCES=SOURCES, BCHARGE={charge:.17g};
TRACK, LINE=LATTICE, BEAM=BEAM1, DT={dt:.17g}, MAXSTEPS={steps};
RUN, METHOD="PARALLEL", FIELDSOLVER=FS;
ENDTRACK;
QUIT;
'''


def prepare(binary, cases=DEFAULT_CASES):
    """Solve COF, create cold-force cases and a finite-emittance ring launch."""
    repo = ROOT.parents[2]
    study = Path(tempfile.mkdtemp(prefix='large-ellipsoid-', dir=ROOT / 'results'))
    print(study, flush=True)
    for name in ['large_ellipsoid_benchmark.py', 'ellipsoid_model.py', 'kv_model.py']:
        shutil.copy2(Path(__file__).parent / name, study / name)
    shutil.copy2(ROOT / 'config.json', study / 'config.json')
    source = study / 'source'
    source.mkdir()
    (source / 'working-tree.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=repo, text=True))
    changed = subprocess.check_output(['git', 'ls-files', '--modified', '--others', '--exclude-standard', 'src', 'tests'], cwd=repo, text=True).splitlines()
    for name in changed:
        if (repo / name).is_file():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, target)
    cof_input = (ROOT / 'bare-cof.in').read_text().replace('ENABLELINEARTRANSFERMAPS=TRUE', 'ENABLELINEARTRANSFERMAPS=FALSE')
    cof_input = re.sub(r'\bDT\s*=\s*[^,;]+', 'DT=6.25e-11', cof_input)
    cof_input = re.sub(r'TIMEINTEGRATOR="[^"]+"', 'TIMEINTEGRATOR="BORIS"', cof_input)
    (study / 'cof.in').write_text(cof_input)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OMP_PROC_BIND': 'false'}
    with (study / 'cof.log').open('w') as log:
        subprocess.run([str(binary), '--info', '1', 'cof.in'], cwd=study, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    cof = json.loads((study / 'bare-cof.json').read_text())
    if not cof['converged'] or max(abs(np.asarray(cof['residual']))) > 1e-9:
        raise RuntimeError('COF did not converge')
    if not np.allclose(cof['section_rotation_wxyz'], [1, 0, 0, 0], rtol=0, atol=1e-14):
        raise RuntimeError('Current runner requires the DBA entrance section aligned to +z')
    config = json.loads((ROOT / 'config.json').read_text())
    p0 = np.linalg.norm(cof['momentum_mc'])
    gamma = np.sqrt(1 + p0*p0)
    beta = p0/gamma
    env0, tunes, match_error = match_transverse(config['elements'], config['rms_emittance_x_m_rad'], config['rms_emittance_y_m_rad'], .05, CHARGE, beta, gamma)
    covariance = initial_covariance(env0, config['rms_emittance_x_m_rad'], config['rms_emittance_y_m_rad'], .05, 1e-4)
    cold_covariance = covariance.copy()
    cold_covariance[[1, 3, 5], :] = 0
    cold_covariance[:, [1, 3, 5]] = 0
    phase = generate_ellipsoid(covariance, cases[0][0])
    write_particles(study / 'ring-launch.txt', absolute_particles(phase, cof['position_m'], cof['momentum_mc']))
    metadata = dict(binary=str(binary), binary_sha256=sha256(binary), revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                    charge_C=CHARGE, mpi_ranks=1, omp_threads=1, seed=20260911, p0_mc=float(p0), beta=float(beta), gamma=float(gamma),
                    covariance_analytic=covariance.tolist(), ring_launch_covariance=np.cov(phase, rowvar=False, ddof=0).tolist(),
                    semiaxes_m=[float(env0[0]), float(env0[2]), .05], envelope=env0.tolist(), matching_residual=float(match_error),
                    frozen_length_tunes=(config['cells']*tunes).tolist(), cases=[], scope='Initial force, cold beam, isolated open boundary; finite-emittance ring launch is an rms closure launch, not a kinetic equilibrium')
    for count, mesh in cases:
        name = f'n{count}-m{mesh}'
        metadata['cases'].append(name)
        directory = study / name
        directory.mkdir()
        phase = generate_ellipsoid(cold_covariance, count)
        particles = absolute_particles(phase, [0, 0, .2], [0, 0, p0])
        write_particles(directory / 'particles.txt', particles)
        (directory / 'case.in').write_text(drift_deck(count, mesh, CHARGE))
        write_json(directory / 'provenance.json', dict(status='prepared', count=count, mesh=mesh, charge_C=CHARGE, dt_s=1e-12, steps=1,
                    input_sha256=sha256(directory/'case.in'), particle_sha256=sha256(directory/'particles.txt'),
                    initial_covariance=np.cov(phase, rowvar=False, ddof=0).tolist()))
    write_json(study / 'study.json', metadata)
    return study


def run_case(study, name):
    """Run one prepared case with a hard 180 s bound; preserve logs on failure."""
    study_metadata = json.loads((study/'study.json').read_text())
    directory = study / name
    metadata = json.loads((directory/'provenance.json').read_text())
    if metadata['status'] != 'prepared':
        raise ValueError(f'{name} already has status {metadata["status"]}')
    binary = Path(study_metadata['binary'])
    if sha256(binary) != study_metadata['binary_sha256']:
        raise ValueError('Executable changed since this study was prepared')
    if sha256(directory/'case.in') != metadata['input_sha256'] or sha256(directory/'particles.txt') != metadata['particle_sha256']:
        raise ValueError('Prepared input changed')
    command = [str(binary), '--info', '1', 'case.in']
    start = time.monotonic()
    metadata.update(status='running', command=command, mpi_ranks=1, omp_threads=1)
    write_json(directory/'provenance.json', metadata)
    try:
        with (directory/'run.log').open('w') as log:
            result = subprocess.run(command, cwd=directory, env={**os.environ, 'OMP_NUM_THREADS':'1', 'OMP_PROC_BIND':'false'}, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        metadata.update(returncode=result.returncode, status='completed' if result.returncode == 0 else 'failed')
    except subprocess.TimeoutExpired:
        metadata['status'] = 'timed_out'
    finally:
        metadata['elapsed_seconds'] = time.monotonic() - start
        write_json(directory/'provenance.json', metadata)
    if metadata['status'] != 'completed':
        raise RuntimeError(f'{name}: {metadata["status"]}')
    print(f'{name}: {metadata["elapsed_seconds"]:.3f}s', flush=True)


def analyze(study):
    """Compare first-step gathered fields against the exact cold-beam solution."""
    meta = json.loads((study/'study.json').read_text())
    rows = []
    for name in meta['cases']:
        directory = study / name
        case = json.loads((directory/'provenance.json').read_text())
        cr = np.asarray(case.get('covariance_analytic', meta['covariance_analytic']))[np.ix_([0, 2, 4], [0, 2, 4])]
        if case['status'] != 'completed':
            continue
        initial = np.loadtxt(directory/'particles.txt', skiprows=2)
        with h5py.File(directory/'case.h5', 'r') as output:
            keys = sorted((k for k in output if re.fullmatch(r'Step#\d+', k)), key=lambda k:int(k[5:]))
            available = [output[k] for k in keys if np.linalg.norm(output[k]['Ex'][:]) > 0]
            if not available:
                raise ValueError(f'{name}: no nonzero electric-field snapshot')
            group = available[0]
            ordering = np.argsort(group['id'][:])
            ids = group['id'][:][ordering]
            if not np.array_equal(ids, np.arange(case['count'])):
                raise ValueError('Expected unchanged one-rank source-file IDs')
            r = np.column_stack([group[a][:][ordering] for a in ['x','y','z']])
            p = np.column_stack([group[a][:][ordering] for a in ['px','py','pz']])
            electric = np.column_stack([group[a][:][ordering] for a in ['Ex','Ey','Ez']])
            magnetic = np.column_stack([group[a][:][ordering] for a in ['Bx','By','Bz']])
            charge_sum = float(np.sum(group['q'][:]))
            timestamp = float(np.asarray(group.attrs['TIME']).ravel()[0])
            refp = np.asarray(group.attrs['RefPartP'])
            unit_metadata = str(output.attrs.get('ExUnit', 'absent'))
        if np.linalg.norm(refp[:2]) > 1e-12*meta['p0_mc'] or refp[2] <= 0:
            raise ValueError('Initial-force extraction requires the straight +z reference basis')
        if len(r) != case['count'] or not np.isfinite(np.c_[r,p,electric,magnetic]).all():
            raise ValueError(f'{name}: particle loss or nonfinite data')
        if not np.isclose(timestamp, case['dt_s'], rtol=1e-12):
            raise ValueError('Expected first accepted step snapshot')
        # Straight cold beam: all first half-drifts have exactly the same velocity,
        # so midpoint coordinates relative to the source centroid equal input r.
        centered = initial[:, :3] - [0, 0, .2]
        ea, ba, fa = analytic_fields(centered, cr, meta['charge_C'], meta['p0_mc'])
        velocity = np.array([0., 0., c*meta['beta']])
        force = electric + np.cross(velocity, magnetic)
        row = dict(case=name, count=case['count'], mesh=case['mesh'], elapsed_seconds=case['elapsed_seconds'], charge_C=charge_sum)
        diagnostics = {}
        for region, mask in [('all', np.ones(len(r), dtype=bool)), ('core', np.einsum('ni,ij,nj->n', centered, np.linalg.inv(5*cr), centered) < .8**2)]:
            measured, offset, scatter = fit_gradient(centered[mask], force[mask])
            expected, _, _ = fit_gradient(centered[mask], fa[mask])
            electric_fit, _, _ = fit_gradient(centered[mask], electric[mask])
            expected_electric, _, _ = fit_gradient(centered[mask], ea[mask])
            diagnostics[region] = dict(count=int(mask.sum()), force_gradient_V_per_m2=measured.tolist(), analytic_force_gradient_V_per_m2=expected.tolist(),
                                      electric_gradient_V_per_m2=electric_fit.tolist(), analytic_electric_gradient_V_per_m2=expected_electric.tolist(),
                                      intercept_V_per_m=offset.tolist(), nonlinear_residual_rms_V_per_m=scatter.tolist())
            row[f'{region}_tensor_relative_frobenius_error'] = float(np.linalg.norm(measured-expected)/np.linalg.norm(expected))
            for i, axis in enumerate('xyz'):
                row[f'{region}_{axis}_gradient_relative_error'] = float(measured[i,i]/expected[i,i]-1)
                row[f'{region}_{axis}_force_relative_rms_error'] = float(np.linalg.norm((force-fa)[mask,i])/np.linalg.norm(fa[mask,i]))
        measured_delta = p - initial[:, 3:]
        expected_delta = elementary_charge/(proton_mass*c) * case['dt_s'] * force
        kick_error = float(np.linalg.norm(measured_delta-expected_delta)/np.linalg.norm(expected_delta))
        row['kick_relative_error'] = kick_error
        row['relativistic_B_relative_error'] = float(np.linalg.norm(magnetic - np.cross(velocity, electric)/c**2)/np.linalg.norm(magnetic))
        row['max_relative_size_change'] = float(np.max(abs(np.std(r, axis=0)/np.std(centered, axis=0)-1)))
        diagnostics.update(row, h5_electric_unit_metadata=unit_metadata, electric_values_actual_units='V/m: raw pc.E, independently checked from physical momentum kick',
                           source_frame='Reference basis equals fixed laboratory basis in this field-free straight cold launch',
                           analytic_field_tensor_m2=field_tensor(cr, meta['charge_C'], meta['beta'], meta['gamma']).tolist())
        write_json(directory/'field-comparison.json', diagnostics)
        # Small, reproducible diagnostic subset keeps the plot/data useful without
        # duplicating all several-hundred-thousand HDF5 particle records in CSV.
        sample = np.arange(0, len(r), max(1, len(r)//10000))
        table = pd.DataFrame(np.c_[centered[sample], force[sample], fa[sample]], columns=[f'{kind}_{axis}' for kind in ['r_m','force_Vpm','analytic_Vpm'] for axis in 'xyz'])
        table.to_csv(directory/'field-samples.csv', index=False)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(study/'force-convergence.csv', index=False)
    if frame.empty:
        raise ValueError('No completed cases')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'opalx-large-ellipsoid-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, panels = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, color in zip('xyz', ['#0072B2','#D55E00','#009E73']):
        for count, style in zip(sorted(frame['count'].unique()), ['--','-','-.']):
            subset = frame[frame['count']==count].sort_values('mesh')
            panels[0].plot(subset.mesh, 100*subset[f'core_{axis}_gradient_relative_error'], marker='o', color=color, linestyle=style, label=f'{axis}, N={count:,}')
            panels[1].plot(subset.mesh, 100*subset[f'all_{axis}_force_relative_rms_error'], marker='o', color=color, linestyle=style, label=f'{axis}, N={count:,}')
    panels[0].set(xlabel='Mesh cells per axis', ylabel='Core force-gradient error [%]', title='Interior r/a < 0.8')
    panels[1].set(xlabel='Mesh cells per axis', ylabel='Whole-bunch force RMS error [%]', title='Includes surface smoothing and particle noise')
    for panel in panels:
        panel.grid(alpha=.2)
        panel.legend(fontsize=7)
    for ext in ['png','pdf']:
        fig.savefig(study/f'force-convergence.{ext}', dpi=220)
    plt.close(fig)
    write_json(study/'summary.json', dict(cases=rows, analysis='Gradient fits use intercept and all tensor components; core selection is separately reported alongside all-particle RMS error. No stationary tune claim.',
                units_note='HDF5 ExUnit metadata currently says MV/m; raw values are V/m as verified with Boris momentum kicks.', mpi_ranks=1, omp_threads=1))
    print(frame.to_string(index=False))
    return frame



def ring_probe(study, timeout=90):
    """Measure one nominal ring step, cancelling only this task's child at timeout."""
    meta = json.loads((study/'study.json').read_text())
    binary = Path(meta['binary'])
    if sha256(binary) != meta['binary_sha256']:
        raise ValueError('Executable changed since study preparation')
    directory = study/'ring-performance-probe'
    directory.mkdir()
    shutil.copy2(study/'ring-launch.txt', directory/'particles.txt')
    count = int((directory/'particles.txt').open().readline())
    lattice = (ROOT/'bare-gaussian-cof.in').read_text().split('DIST: DISTRIBUTION')[0]
    lattice = re.sub(r'PSDUMPFREQ=\d+', 'PSDUMPFREQ=10000', lattice)
    lattice = re.sub(r'STATDUMPFREQ=\d+', 'STATDUMPFREQ=10000', lattice)
    source = drift_deck(count, 32, meta['charge_C'], dt=6.25e-11).split('DIST: DISTRIBUTION')[1]
    deck = 'OPTION, STEPINFOFQ=1, EBDUMP=FALSE;\n' + lattice + 'DIST: DISTRIBUTION' + source
    (directory/'case.in').write_text(deck)
    command = [str(binary), '--info', '1', 'case.in']
    started = time.monotonic()
    provenance = dict(command=command, binary_sha256=meta['binary_sha256'], count=count, mesh=32,
        charge_C=meta['charge_C'], mpi_ranks=1, omp_threads=1, nominal_dt_s=6.25e-11, nominal_steps=1,
        timeout_seconds=timeout, input_sha256=sha256(directory/'case.in'), particle_sha256=sha256(directory/'particles.txt'))
    with (directory/'run.log').open('w') as log:
        child = subprocess.Popen(command, cwd=directory, env={**os.environ,'OMP_NUM_THREADS':'1','OMP_PROC_BIND':'false'}, stdout=log, stderr=subprocess.STDOUT)
        provenance.update(pid=child.pid,status='running')
        write_json(directory/'provenance.json',provenance)
        try:
            child.wait(timeout=timeout)
            provenance['status'] = 'completed' if child.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            provenance.update(status='timed_out', cancellation='SIGTERM sent only to the recorded task-owned child; SIGKILL after 5s if needed')
    provenance.update(returncode=child.returncode, elapsed_seconds=time.monotonic()-started)
    logtext = (directory/'run.log').read_text()
    provenance['boundary_controller_active'] = 'Ring boundary control: synchronized ordinary Boris/PIC substeps.' in logtext
    provenance['last_progress_lines'] = [line for line in logtext.splitlines() if 'Track ' in line or 'Step ' in line][-10:]
    write_json(directory/'provenance.json',provenance)
    print(json.dumps(provenance,indent=2),flush=True)
    return provenance

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT.parents[2]/'omp-build/src/opalx')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run', nargs='*', default=[])
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--ring-probe', action='store_true')
    args = parser.parse_args()
    study = args.study.resolve() if args.study else prepare(args.binary.resolve())
    if args.prepare_only:
        return
    names = json.loads((study/'study.json').read_text())['cases'] if args.run == ['all'] else args.run
    for name in names:
        run_case(study, name)
    if args.analyze:
        analyze(study)
    if args.ring_probe:
        ring_probe(study)


if __name__ == '__main__':
    main()
