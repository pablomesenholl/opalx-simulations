"""RF-strength continuation with particle and collective closure stability checks."""
import argparse,json
from pathlib import Path
import numpy as np
from match_ellipsoid_ring import (rf_layout,bare_cell_map,mode_covariance,match_covariance,
                                 envelope_multipliers,rf_voltage)
from kv_model import beam_parameters


def scan(output, slopes, charges, sigma_z=.05/np.sqrt(5)):
    base=Path(__file__).resolve().parents[1]
    cfg=json.loads((base/'config.json').read_text()); cfg['elements']=rf_layout(cfg['elements'])
    beta,gamma,*_=beam_parameters(cfg['kinetic_energy_MeV']); output.mkdir(parents=True,exist_ok=True)
    records=[]
    for slope in slopes:
        emits=np.array([3e-6,3e-6,1e-6])
        _,bare_phases,bases,_=mode_covariance(bare_cell_map(cfg['elements'],gamma,slope),emits)
        transverse=emits[0]*bases['x']+emits[1]*bases['y']
        emits[2]=(sigma_z**2-transverse[4,4])/bases['z'][4,4]
        initial=None
        for charge in charges:
            try:
                fit=match_covariance(cfg['elements'],charge*1e-12,beta,gamma,slope,emits,initial)
                initial=fit['covariance']
                multipliers,_=envelope_multipliers(initial,cfg['elements'],charge*1e-12,beta,gamma,slope)
                tunes=fit['phase']*cfg['cells']/(2*np.pi)
                shifts=(fit['phase']-bare_phases)*cfg['cells']/(2*np.pi)
                row=dict(rf_slope_per_m=slope,charge_pC=charge,sigma_z_zero_charge_m=sigma_z,
                         eigenemittances_xyz_m=emits.tolist(),covariance=initial.tolist(),
                         tunes_xyz=tunes.tolist(),shifts_xyz=shifts.tolist(),
                         envelope_max_multiplier=float(max(abs(multipliers))),scaled_residual=fit['residual'])
                records.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='covariance'}),flush=True)
            except (ValueError,RuntimeError) as error:
                records.append(dict(rf_slope_per_m=slope,charge_pC=charge,error=str(error)))
                print(json.dumps(records[-1]),flush=True);break
            (output/'scan.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    return records

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slopes',type=float,nargs='+',default=[.0001,.0003,.003,.006,.01])
    parser.add_argument('--charges',type=float,nargs='+',default=[0,5,10,20,30,41.13101073,60,80,100])
    parser.add_argument('--sigma-z',type=float,default=.05/np.sqrt(5));args=parser.parse_args()
    scan(args.output,args.slopes,args.charges,args.sigma_z)
