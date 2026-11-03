"""Extract individual normal-mode tunes from saved equivalent-cell observations."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from ellipsoid_observables import individual_mode_tunes


def analyze(observations,match,output,central_id=None):
    data=np.load(observations);model=np.load(match);ids=data['ids']
    centroid=None
    if central_id is not None:
        indices=np.flatnonzero(ids==central_id)
        if len(indices)!=1:raise ValueError('Central reference particle id must be unique')
        centroid=data['phase'][:,indices[0]]
    result=individual_mode_tunes(data['phase'],data['turn_coordinate'],model['normal_mode_inverse'],
                                 model['signed_cell_phases'],centroid)
    output.mkdir(parents=True,exist_ok=True);np.savez(output/'mode-tunes.npz',ids=ids,**result)
    rows=[]
    for i,pid in enumerate(ids):
        for mode,axis in enumerate('xyz'):
            row=dict(id=int(pid),mode=axis)
            row.update({key:(bool(value[i,mode]) if key=='active' else float(value[i,mode])) for key,value in result.items()})
            rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(output/'mode-tunes.csv',index=False)
    summary=[]
    for axis,group in table[table.active].groupby('mode'):
        summary.append(dict(mode=axis,count=len(group),phase_median=float(group.phase_tune_absolute.median()),
                            phase_min=float(group.phase_tune_absolute.min()),phase_max=float(group.phase_tune_absolute.max()),
                            spectral_median=float(group.spectral_tune_absolute.median()),
                            max_estimator_difference=float(group.estimator_difference.max()),
                            max_phase_residual_rad=float(group.phase_residual_rad.max())))
    result_summary=dict(centering='explicit central particle' if central_id is not None else 'provided reference frame',
                        central_id=central_id,modes=summary)
    (output/'summary.json').write_text(json.dumps(result_summary,indent=2)+'\n');print(json.dumps(result_summary,indent=2))
    return result_summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('observations',type=Path);parser.add_argument('match',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--central-id',type=int);args=parser.parse_args()
    analyze(args.observations,args.match,args.output,args.central_id)
