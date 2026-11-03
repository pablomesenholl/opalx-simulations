"""Independent tolerance, long-moment-trajectory and Floquet checks of saved matches."""
import argparse,json
from pathlib import Path
import numpy as np
from match_ellipsoid_ring import (cell_pass,eigenemittances,mode_covariance,
                                 envelope_multipliers)


def validate(directory, turns=32):
    summary=json.loads((directory/'summary.json').read_text())
    cfg=json.loads((directory/'config.json').read_text())
    beta,gamma,g=summary['beta'],summary['gamma'],summary['rf_slope_per_m']
    emittances=np.array(summary['ordered_eigenemittances_xyz_m'])
    rows=[]
    for case in summary['cases']:
        if case['status']!='matched_moment_closure': continue
        charge=case['charge_pC']*1e-12; initial=np.array(case['covariance'])
        scale=np.sqrt(np.outer(np.diag(initial),np.diag(initial)))
        final,matrix,_=cell_pass(initial,cfg['elements'],charge,beta,gamma,g,rtol=2e-11)
        tight_tunes=mode_covariance(matrix,emittances)[1]*cfg['cells']/(2*np.pi)
        coarse,_=envelope_multipliers(initial,cfg['elements'],charge,beta,gamma,g,step=2e-5)
        fine,_=envelope_multipliers(initial,cfg['elements'],charge,beta,gamma,g,step=1e-5)
        state=initial.copy();max_periodic=0.;max_emits=0.
        for _ in range(turns*cfg['cells']):
            state,_,_=cell_pass(state,cfg['elements'],charge,beta,gamma,g,rtol=2e-11,with_map=False)
            max_periodic=max(max_periodic,float(np.max(abs((state-initial)/scale))))
            max_emits=max(max_emits,float(np.max(abs(eigenemittances(state)/np.sort(emittances)-1))))
        row=dict(charge_pC=case['charge_pC'],tight_tunes_xyz=tight_tunes.tolist(),
                 max_tune_tolerance_difference=float(np.max(abs(tight_tunes-case['tunes_xyz']))),
                 tight_scaled_cell_closure=float(np.max(abs((final-initial)/scale))),
                 envelope_max_multiplier_coarse=float(np.max(abs(coarse))),
                 envelope_max_multiplier_fine=float(np.max(abs(fine))),
                 envelope_multipliers=[[float(v.real),float(v.imag)] for v in fine],
                 tracked_moment_turns=turns,max_scaled_periodic_covariance_error=max_periodic,
                 max_relative_eigenemittance_drift=max_emits)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='envelope_multipliers'}),flush=True)
        (directory/'validation.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--turns',type=int,default=32);args=parser.parse_args()
    validate(args.directory,args.turns)
