"""Compare a completed Gaussian INITIALORBIT run to the accepted cold probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
data=json.loads((args.run/'tunes.json').read_text())
f=pd.DataFrame(data['measurements'])
previous=json.loads((root/'results/bare-tunes-62.5ps-ijgjxg2t/tunes.json').read_text())
p=pd.DataFrame(previous['measurements'])
rows=[]
for axis in ['x','y']:
    s=f[f.axis==axis]
    baseline=float(p[p.axis==axis].tracking_tune.mean())
    rows.append(dict(axis=axis,particle_fits=len(s),analytic=float(s.analytic_tune.iloc[0]),
                     previous_tracking=baseline,gaussian_mean=float(s.tracking_tune.mean()),
                     gaussian_median=float(s.tracking_tune.median()),
                     gaussian_min=float(s.tracking_tune.min()),gaussian_max=float(s.tracking_tune.max()),
                     mean_minus_previous=float(s.tracking_tune.mean()-baseline)))
comparison=pd.DataFrame(rows)
checks=dict(cof_converged=json.loads((args.run/'bare-cof.json').read_text())['converged'],
            adequate_signal_count=bool((comparison.particle_fits>=16).all()),
            similar_bare_tunes=bool(comparison.mean_minus_previous.abs().max()<1e-6),
            individual_tunes_match_cof=bool(f.tracking_minus_cof.abs().max()<1e-6),
            independent_frequency_agrees=bool(f.frequency_estimator_difference.abs().max()<1e-6),
            half_record_consistency=bool(max(abs(v[0]-v[1]) for v in f.half_tunes)<2e-6),
            interpolation_consistency=bool(f.interpolation_order_difference.abs().max()<1e-6),
            energy_conservation=bool(f.relative_energy_variation.max()<1e-8))
summary=dict(checks=checks,comparison=rows,
             excluded_small_signal=data['excluded_small_signal'],
             max_tracking_minus_cof=float(f.tracking_minus_cof.abs().max()),
             max_frequency_difference=float(f.frequency_estimator_difference.abs().max()),
             max_half_difference=max(abs(v[0]-v[1]) for v in f.half_tunes),
             max_interpolation_difference=float(f.interpolation_order_difference.abs().max()),
             max_relative_energy_variation=float(f.relative_energy_variation.max()))
comparison.to_csv(args.run/'comparison.csv',index=False)
(args.run/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(comparison.to_string(index=False,float_format=lambda x:f'{x:.12g}'))
print(json.dumps(summary,indent=2))
if not all(checks.values()): raise RuntimeError('Gaussian bare-tune comparison failed; inspect results')
