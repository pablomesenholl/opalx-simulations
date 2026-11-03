"""Publication-size figures for the selected ellipsoid moment-closure family."""
import argparse,json,os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"opalx-matched-ellipsoid-mpl"))
os.environ.setdefault("XDG_CACHE_HOME",str(Path(tempfile.gettempdir())/"opalx-matched-ellipsoid-cache"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(directory):
    summary=json.loads((directory/'summary.json').read_text())
    rows=[]
    bare=np.array(summary['cases'][0]['tunes_xyz'])
    for case in summary['cases']:
        q=np.array(case['tunes_xyz']);rows.append(dict(charge_pC=case['charge_pC'],Qx=q[0],Qy=q[1],Qs=q[2],
            dQx=q[0]-bare[0],dQy=q[1]-bare[1],rms_z_min_mm=1000*case['sigma_min_m'][2],rms_z_max_mm=1000*case['sigma_max_m'][2],
            rf_phase_edge_rad=case['rf_max_bunch_phase_rad']))
    table=pd.DataFrame(rows);table.to_csv(directory/'tune-prediction.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':11,
                         'figure.dpi':160,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.4),layout='constrained')
    colors={'x':'#176b9a','y':'#bd5d21','s':'#57853a'}
    for axis in ['x','y','s']:
        axes[0].plot(table.charge_pC,table['Q'+axis],'o-',ms=4,color=colors[axis],label='$Q_'+axis+'$')
    for axis in ['x','y']:
        axes[1].plot(table.charge_pC,table['dQ'+axis],'o-',ms=4,color=colors[axis],label=r'$\Delta Q_'+axis+'$')
    axes[0].set(ylabel='Oscillations per turn',title='Coupled normal-mode frequencies')
    axes[1].set(ylabel='Shift relative to the same RF lattice',title='Incoherent tune prediction')
    axes[1].axhline(-.2,color='.65',lw=.7,ls='--')
    axes[2].fill_between(table.charge_pC,table.rms_z_min_mm,table.rms_z_max_mm,color=colors['s'],alpha=.2)
    axes[2].plot(table.charge_pC,.5*(table.rms_z_min_mm+table.rms_z_max_mm),'o-',ms=4,color=colors['s'])
    axes[2].set(ylabel='Rms bunch length (mm)',title='Periodic longitudinal match')
    for ax in axes:ax.set_xlabel('Total bunch charge (pC)');ax.grid(alpha=.15)
    for ax in axes[:2]:ax.legend(frameon=False)
    fig.suptitle('DBA ring · isolated uniform-ellipsoid moment closure · 2.5 MeV protons',fontsize=12)
    for extension in ['png','pdf']:fig.savefig(directory/f'matched-ellipsoid-prediction.{extension}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.5),layout='constrained')
    for charge,line_style in [(0,'--'),(160,'-')]:
        data=np.load(directory/f'match-{charge}pc.npz')
        sizes=np.sqrt(np.diagonal(data['covariances'],axis1=1,axis2=2)[:,[0,2,4]])*1000
        for i,axis in enumerate(['x','y']): axes[0].plot(data['s'],sizes[:,i],line_style,color=colors[axis],label=f'{axis}, {charge} pC')
        axes[1].plot(data['s'],sizes[:,2],line_style,color=colors['s'],label=f'{charge} pC')
    for ax in axes:ax.set_xlabel('Path through one DBA cell (m)');ax.grid(alpha=.15);ax.legend(frameon=False)
    axes[0].set(ylabel='Transverse rms size (mm)',title='Dispersion retained in horizontal size')
    axes[1].set(ylabel='Longitudinal rms size (mm)',title='Each charge uses its own periodic match')
    for extension in ['png','pdf']:fig.savefig(directory/f'matched-ellipsoid-envelopes.{extension}')
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    plot(args.directory)
