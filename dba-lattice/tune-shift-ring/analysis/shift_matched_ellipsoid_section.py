"""Shift a matched ring family to the centre of D1 without changing its physics."""
import argparse,json
from pathlib import Path
from shutil import copy2
import numpy as np
from match_ellipsoid_ring import cell_pass,normal_mode_basis


def shifted_layout(elements):
    """Cyclically split D1 into entrance/exit half drifts and return prefix map lattice."""
    index=next(i for i,e in enumerate(elements) if e['name']=='D1')
    drift=elements[index];half=drift['length_m']/2
    beginning=dict(drift,name='D1b',length_m=half)
    ending=dict(drift,name='D1a',length_m=half)
    prefix=[dict(e) for e in elements[:index]]+[ending]
    rotated=[beginning]+[dict(e) for e in elements[index+1:]+elements[:index]]+[ending]
    return rotated,prefix


def shift(source,output):
    summary=json.loads((source/'summary.json').read_text());cfg=json.loads((source/'config.json').read_text())
    rotated,prefix=shifted_layout(cfg['elements']);cells=cfg['cells'];cfg['elements']=rotated
    beta,gamma,g=summary['beta'],summary['gamma'],summary['rf_slope_per_m']
    output.mkdir(parents=True,exist_ok=True); records=[]
    for case in summary['cases']:
        initial=np.array(case['covariance']);charge=case['charge_pC']*1e-12
        covariance,transfer,_=cell_pass(initial,prefix,charge,beta,gamma,0.,rtol=2e-11)
        # Prefix precedes the RF gap, so g=0 avoids the legacy exit-kick fallback.
        final,matrix,rows,maps=cell_pass(covariance,rotated,charge,beta,gamma,g,rtol=2e-11,samples=12,return_maps=True)
        basis,phases=normal_mode_basis(matrix)
        old=np.load(source/f"match-{case['charge_pC']:g}pc.npz")
        conjugated=transfer@old['matrix']@np.linalg.inv(transfer)
        mode_matrix=np.linalg.solve(basis,(matrix-conjugated)@basis)
        scales=np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))
        error=float(np.max(abs((final-covariance)/scales)))
        record=dict(case,covariance=covariance.tolist(),tunes_xyz=(abs(phases)*cells/(2*np.pi)).tolist(),
                    scaled_shifted_closure=error,normal_map_conjugation_error=float(np.max(abs(mode_matrix))))
        records.append(record)
        np.savez(output/f"match-{case['charge_pC']:g}pc.npz",covariance=covariance,matrix=matrix,
                 normal_mode_basis=basis,normal_mode_inverse=np.linalg.inv(basis),signed_cell_phases=phases,
                 s=np.array([s for s,c in rows]),covariances=np.array([c for s,c in rows]),particle_maps=maps,
                 transfer_from_b1=transfer)
        print(json.dumps({k:record[k] for k in ['charge_pC','tunes_xyz','scaled_shifted_closure','normal_map_conjugation_error']}),flush=True)
    summary['cases']=records;summary['observation_section']='middle of D1, 0.5 m from either adjacent element'
    summary['source_section']='B1 entrance';summary['source_directory']=str(source.resolve())
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (output/'source').mkdir(exist_ok=True)
    for path in (source/'source').glob('*.py'):copy2(path,output/'source'/path.name)
    copy2(__file__,output/'source'/Path(__file__).name)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();shift(args.source,args.output)
