"""Independent finite-amplitude fixed-Boris RF-ring controls, one MPI rank.

Retain the10um fixed-step failure. These larger amplitudes test whether the
step-size limit is associated with unresolved hard-edge arrival-time differences.
They do not automatically validate space-charge forces or a large kinetic bunch.
"""
import argparse,gzip,json,re,shutil,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from large_ellipsoid_benchmark import ROOT,sha256,write_json,write_particles
from rf_ring_study import (execute,normal_mode_probes,read_history,rf_lattice,
                           section_observations,straight_particles,tracking_deck)
from ellipsoid_observables import individual_mode_tunes


def run(amplitudes=(.001,.005),timesteps=(4e-10,2e-10,1e-10),turns=32,study=None):
    source=ROOT/'results/rf-ring-s3a01yyz';binary=ROOT.parents[2]/'omp-build/src/opalx'
    study=Path(tempfile.mkdtemp(prefix='finite-amplitude-fixed-rf-',dir=ROOT/'results')) if study is None else Path(study)
    if (study/'finite_amplitude_fixed_probe.py').exists():
        shutil.copy2(study/'finite_amplitude_fixed_probe.py',study/'original-runner.py')
    for filename in ['finite_amplitude_fixed_probe.py','rf_ring_study.py','ellipsoid_observables.py']:
        shutil.copy2(Path(__file__).with_name(filename),study/filename)
    for filename in ['bare-cof.json','sections.json']:
        shutil.copy2(source/filename,study/filename)
    shutil.copy2(source/'synchronous-3/rf.json',study/'rf.json')
    shutil.copy2(source/'analytic/match-0pc.npz',study/'match-0pc.npz')
    cof=json.loads((source/'bare-cof.json').read_text());sections=json.loads((source/'sections.json').read_text())
    rf=json.loads((source/'synchronous-3/rf.json').read_text());model=np.load(source/'analytic/match-0pc.npz')
    expected=abs(model['signed_cell_phases'])*6/(2*np.pi)
    write_json(study/'study.json',dict(binary=str(binary),binary_sha256=sha256(binary),source=str(source),
               mpi_ranks=1,omp_threads=1,turns=turns,repartfreq=10,amplitudes_m=list(amplitudes),
               timesteps_s=list(timesteps),model_tunes_xyz=expected.tolist(),dump_interval_target_s=8e-9))
    print(study,flush=True);rows=[];diagnostics=[]
    for amplitude in amplitudes:
        phase,labels=normal_mode_probes(model['normal_mode_basis'],amplitude)
        particles=straight_particles(phase,cof['position_m'],cof['momentum_mc'],sections[0]['theta_rad'])
        for dt in timesteps:
            directory=study/f'amp{amplitude*1000:g}mm-dt{dt*1e12:g}ps';directory.mkdir(exist_ok=True)
            lattice,_=rf_lattice((source/'bare-gaussian-cof.in').read_text(),rf['frequency_hz'],rf['amplitude_mvm'],
                                  rf['phases_rad'],dump=max(1,round(8e-9/dt)))
            lattice=re.sub(r'REPARTFREQ\s*=\s*0\b','REPARTFREQ=10',lattice)
            (directory/'case.in').write_text(tracking_deck(lattice,len(phase),dt,turns))
            write_particles(directory/'particles.txt',particles)
            write_json(directory/'launch.json',dict(phase=phase.tolist(),labels=labels,rf=rf,amplitude_m=amplitude))
            if not (directory/'provenance.json').exists():
                execute(directory,binary,timeout=1200,expected_boundary_control=False)
            else:
                old=json.loads((directory/'provenance.json').read_text())
                if old['status']!='completed' or old['binary_sha256']!=sha256(binary):
                    raise RuntimeError('Cannot reuse an incomplete or differently built run')
            history=read_history(directory/'case.h5')
            observation_error=None
            try:
                observed=section_observations(history,sections,np.linalg.norm(cof['momentum_mc']),phase)
            except ValueError as error:
                observed=None;observation_error=str(error)
                print(f'{directory.name}: observation validation failed: {error}',flush=True)
            if observed is not None:
                np.savez_compressed(directory/'section-observations.npz',**observed)
                result=individual_mode_tunes(observed['phase'],observed['turn_coordinate'],model['normal_mode_inverse'],
                           model['signed_cell_phases'],observed['phase'][:,0])
                np.savez_compressed(directory/'mode-tunes.npz',ids=observed['ids'],**result)
            # Preserve all raw history in compressed form plus immediately
            # reviewable section observations and the synchronous reference.
            np.savez_compressed(directory/'reference-history.npz',t_s=history['t'],r=history['r'],p=history['p'],
                                 central_local_r=history['local_r'][:,0],central_local_p=history['local_p'][:,0])
            meta=dict(amplitude_m=amplitude,dt_s=dt,observation_valid=observed is not None,observation_error=observation_error,
                      events=len(observed['t_s']) if observed is not None else None,particles=len(history['ids']),
                      max_central_abs_phase=np.max(abs(observed['phase'][:,0]),axis=0).tolist() if observed is not None else None,
                      max_particle_reference_position_m=float(np.max(np.linalg.norm(history['local_r'][:,0],axis=1))),
                      h5_sha256=sha256(directory/'case.h5'))
            for i,label in enumerate(labels):
                if label=='centre' or observed is None:continue
                mode='xyz'.index(label[0]);row=dict(amplitude_m=amplitude,dt_s=dt,probe=label,mode=label[0],analytic_tune=expected[mode])
                for name in ['phase_tune_absolute','spectral_tune_absolute','phase_residual_rad','spectral_coherent_fraction','estimator_difference']:
                    row[name]=float(result[name][i,mode])
                row['phase_minus_model']=row['phase_tune_absolute']-expected[mode]
                rows.append(row)
            diagnostics.append(meta);write_json(directory/'observation-audit.json',meta)
            pd.DataFrame(rows).to_csv(study/'tune-comparison.csv',index=False)
            write_json(study/'observation-audits.json',diagnostics)
            with (directory/'case.h5').open('rb') as raw,gzip.open(directory/'case.h5.gz','wb',compresslevel=3) as packed:
                shutil.copyfileobj(raw,packed)
            # Verify the compressed copy by its uncompressed SHA before removal.
            import hashlib
            digest=hashlib.sha256()
            with gzip.open(directory/'case.h5.gz','rb') as packed:
                for block in iter(lambda:packed.read(1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=meta['h5_sha256']:raise RuntimeError('Compressed HDF5 verification failed')
            (directory/'case.h5').unlink()
            if observed is not None: print(pd.DataFrame(rows).query('amplitude_m==@amplitude and dt_s==@dt').groupby('mode')[
                  ['phase_tune_absolute','spectral_tune_absolute','phase_minus_model','spectral_coherent_fraction']].mean().to_string(),flush=True)
    return study

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--amplitudes',type=float,nargs='+',default=[.001,.005])
    parser.add_argument('--timesteps',type=float,nargs='+',default=[4e-10,2e-10,1e-10]);parser.add_argument('--turns',type=int,default=32)
    parser.add_argument('--study',type=Path)
    args=parser.parse_args();run(args.amplitudes,args.timesteps,args.turns,args.study)
