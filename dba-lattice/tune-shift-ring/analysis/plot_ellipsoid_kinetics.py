"""Plot intrinsic density distortion and independent Coulomb-pair diagnostics."""
import argparse,json,os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'opalx-matched-ellipsoid-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'opalx-matched-ellipsoid-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from large_ellipsoid_benchmark import generate_ellipsoid
from ellipsoid_observables import uniform_density_diagnostics


def plot(directory,model_directory):
    rows=json.loads((directory/'kinetic-consistency.json').read_text())
    selected=[r for r in rows if r['charge_pC']==160]
    model=np.load(model_directory/'match-160pc.npz');sample=generate_ellipsoid(model['covariance'],131072)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,
                         'figure.dpi':160,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.5),layout='constrained')
    for n,color in [(0,'black'),(1,'#176b9a'),(6,'#bd5d21'),(192,'#57853a')]:
        points=(sample@np.linalg.matrix_power(model['matrix'],n).T)[:,[0,2,4]]
        points-=points.mean(axis=0);covariance=points.T@points/len(points)
        radius=np.sqrt(np.maximum(0,np.einsum('ni,ni->n',points,np.linalg.solve(covariance,points.T).T)/5))
        ordered=np.sort(radius);indices=np.arange(0,len(points),64)
        axes[0].plot(ordered[indices],(indices+1)/len(points),color=color,label=f'{n} cell' if n==1 else f'{n} cells')
    axes[0].axvline(1,color='.6',lw=.7,ls='--');axes[0].set(xlim=(0,1.5),ylim=(0,1.01),xlabel='Radius in the rms ellipsoid',ylabel='Enclosed particle fraction',title='Uniform density is lost in one cell')
    axes[0].legend(frameon=False,fontsize=9)
    cell=np.array([r['cells'] for r in selected])
    axes[1].plot(cell,[r['measured_ellipsoid_density']['radial_cdf_error'] for r in selected],'o-',color='#bd5d21',label='Radial CDF error')
    axes[1].plot(cell,[r['moment_diagnostics']['max_scaled_covariance_error'] for r in selected],'o-',color='#176b9a',label='Covariance error')
    axes[1].set(xscale='symlog',xlim=(0,220),yscale='log',xlabel='Cells traversed',ylabel='Fractional error',title='Matched moments do not fix the profile')
    axes[1].legend(frameon=False,fontsize=9)
    for i,(axis,color) in enumerate(zip('xyz',['#176b9a','#bd5d21','#57853a'])):
        axes[2].errorbar(cell,[r['whole_gradient_relative_diag_changes'][i]*100 for r in selected],
                         yerr=[r['whole_gradient_relative_diag_standard_errors'][i]*100 for r in selected],
                         color=color,marker='o',ms=3,capsize=2,label=axis)
    axes[2].axhline(0,color='.6',lw=.7,ls='--');axes[2].set(xscale='symlog',xlim=(0,220),xlabel='Cells traversed',ylabel='Whole-bunch gradient change (%)',title='Coulomb-pair estimate (±1 s.e.)')
    axes[2].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('160 pC moment match · 131,072 particles · prescribed periodic linear field',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(directory/f'kinetic-consistency.{suffix}')
    plt.close(fig)
    checks=[]
    for n in [65536,131072,262144]:
        for seed in [20260911,20260912]:
            phase=generate_ellipsoid(model['covariance'],n,seed)@model['matrix'].T
            check=uniform_density_diagnostics(phase[:,[0,2,4]])
            checks.append(dict(particles=n,seed=seed,cells=1,**check))
    (directory/'density-sample-convergence.json').write_text(json.dumps(checks,indent=2)+'\n')
    pd.DataFrame(checks).to_csv(directory/'density-sample-convergence.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('model_directory',type=Path)
    args=parser.parse_args();plot(args.directory,args.model_directory)
