"""Test whether a moment-matched uniform-spatial sample is a kinetic equilibrium.

Apply the exact frozen periodic particle maps, independently of OPALX/PIC.
The uniform ellipsoid field is retained even after profile deformation, so this
is a necessary kinetic-consistency diagnostic, not self-consistent tracking.
"""
import argparse,json
from pathlib import Path
import numpy as np
from large_ellipsoid_benchmark import generate_ellipsoid
from ellipsoid_model import field_tensor
from ellipsoid_observables import uniform_density_diagnostics,moment_diagnostics,virial_force_gradient


def check(directory,output,count=131072,charges=(0,40,80,160),cells=(0,1,6,12,48,192)):
    summary=json.loads((directory/'summary.json').read_text());output.mkdir(parents=True,exist_ok=True)
    beta,gamma=summary['beta'],summary['gamma'];rows=[]
    for charge in charges:
        data=np.load(directory/f'match-{charge:g}pc.npz');initial=data['covariance'];matrix=data['matrix']
        sample=generate_ellipsoid(initial,count,seed=20260911)
        # Charge-independent force ratios also diagnose the zero-charge density.
        comparison_charge=max(charge,1.)*1e-12
        target=field_tensor(initial[np.ix_([0,2,4],[0,2,4])],comparison_charge,beta,gamma)
        for n in cells:
            phase=sample@np.linalg.matrix_power(matrix,n).T;positions=phase[:,[0,2,4]]
            fixed=uniform_density_diagnostics(positions,initial[np.ix_([0,2,4],[0,2,4])])
            measured=uniform_density_diagnostics(positions)
            moments=moment_diagnostics(phase,initial)
            gradient,error=virial_force_gradient(positions,comparison_charge,beta,gamma)
            relative=(np.diag(gradient)/np.diag(target)-1)
            row=dict(charge_pC=charge,cells=n,turns=n/6,count=count,
                     fixed_ellipsoid_density=fixed,measured_ellipsoid_density=measured,
                     moment_diagnostics=moments,whole_gradient_relative_diag_changes=relative.tolist(),
                     whole_gradient_relative_diag_standard_errors=(np.diag(error)/np.diag(target)).tolist(),
                     whole_gradient_matrix=gradient.tolist())
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['charge_pC','cells','measured_ellipsoid_density','whole_gradient_relative_diag_changes']}),flush=True)
            (output/'kinetic-consistency.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--particles',type=int,default=131072);parser.add_argument('--charges',type=float,nargs='+',default=[0,40,80,160])
    args=parser.parse_args();check(args.directory,args.output,args.particles,args.charges)
