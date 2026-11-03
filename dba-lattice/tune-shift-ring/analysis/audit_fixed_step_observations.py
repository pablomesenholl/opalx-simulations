"""Independent geometry, reference and fitted-map audit of fixed-Boris probes."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from rf_ring_study import read_history,plane_crossings


def reference_audit(history,sections,p0,period):
    """Separate design-orbit error from central-particle/reference disagreement."""
    relative=np.linalg.norm(history['local_r'][:,0],axis=1)
    events=[]
    for cell,section in enumerate(sections):
        theta=section['theta_rad'];origin=np.array(section['centre_m'])
        rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        for event in plane_crossings(history,origin,theta):
            i=event['index'];f=event['fraction']
            r=((1-f)*history['r'][i]+f*history['r'][i+1]-origin)@rotation
            p=((1-f)*history['p'][i]+f*history['p'][i+1])@rotation/p0
            events.append(dict(time_s=event['time_s'],turns=event['time_s']/period,cell=cell,
                               phase=[float(r[0]),float(p[0]),float(r[1]),float(p[1]),float(r[2]),float(p[2]-1)]))
    events.sort(key=lambda item:item['time_s'])
    order=np.array([e['cell'] for e in events]);missing=int(np.count_nonzero(np.diff(order)%6!=1))
    records=[]
    for threshold in [1e-9,1e-6,1e-3]:
        indices=np.flatnonzero(relative>threshold)
        records.append(dict(threshold_m=threshold,first_turn=float(history['t'][indices[0]]/period) if len(indices) else None))
    return dict(max_particle_reference_position_m=float(relative.max()),threshold_crossings=records,
                accepted_reference_sections=len(events),unordered_section_transitions=missing,
                max_reference_abs_section_phase=np.max(abs(np.array([e['phase'] for e in events])),axis=0).tolist() if events else None,
                reference_section_events=events)


def fitted_map_audit(phase,model_inverse):
    """Fit a single linear map in conditioned normal coordinates across all probes.

    A large fit residual or unstable multipliers means a stationary linear
    frequency interpretation is poor. This is independent of the phase/spectral
    estimators and is not used to hide a failed fit.
    """
    u=(phase-phase[:,0:1])@model_inverse.T
    previous=u[:-1,1:].reshape(-1,6);following=u[1:,1:].reshape(-1,6)
    coefficient,*_=np.linalg.lstsq(previous,following,rcond=None)
    residual=following-previous@coefficient;values=np.linalg.eigvals(coefficient.T)
    return dict(relative_rms_prediction_error=float(np.linalg.norm(residual)/np.linalg.norm(following)),
                multiplier_moduli=abs(values).tolist(),positive_mode_frequencies=(np.angle(values[np.angle(values)>0])*6/(2*np.pi)).tolist(),
                normal_matrix=coefficient.T.tolist())


def audit_case(directory,source,output):
    cof=json.loads((source/'bare-cof.json').read_text());sections=json.loads((source/'sections.json').read_text())
    rf=json.loads((source/'synchronous-3/rf.json').read_text());period=rf['harmonic']/rf['frequency_hz']
    if (directory/'reference-history.npz').exists():
        h=np.load(directory/'reference-history.npz')
        history=dict(t=h['t_s'],r=h['r'],p=h['p'],local_r=h['central_local_r'][:,None,:])
    else:history=read_history(directory/'case.h5')
    result=reference_audit(history,sections,np.linalg.norm(cof['momentum_mc']),period)
    if (directory/'section-observations.npz').exists():
        observed=np.load(directory/'section-observations.npz');model=np.load(source/'analytic/match-0pc.npz')
        result['fitted_map']=fitted_map_audit(observed['phase'],model['normal_mode_inverse'])
        result['max_particle_abs_section_phase']=np.max(abs(observed['phase'][:,0]),axis=0).tolist()
    result.update(directory=str(directory.resolve()),source=str(source.resolve()))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['reference_section_events','fitted_map']},indent=2))
    if 'fitted_map' in result:print(json.dumps(result['fitted_map'],indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    audit_case(args.directory,args.source,args.output)
