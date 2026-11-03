"""Offline whole-bunch diagnostics and declared fixed-ID sampling for D1 monitors.

Run only after OPALX has closed the raw monitor files. Read one full passage at
once, compute moments and density on ALL rows, save a fixed random ID subset for
mode analysis, and retain the first/last unique full snapshot at each monitor.
This script never deletes its inputs. Verify its manifest before releasing raw
intermediate files. All times use group TIME; duplicate terminal events retain
the last record. No data is selected by fitted tune or particle amplitude.
"""
import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pandas as pd

from check_temporal_monitors import ordered_keys, read_monitor_group
from ellipsoid_observables import uniform_density_diagnostics
from large_ellipsoid_benchmark import sha256, write_json


def unique_snapshot_keys(data):
    """Deduplicate equal-time reference crossings, keeping the latest stored row."""
    result=[]; discarded=[]
    for key in ordered_keys(data):
        time=float(np.asarray(data[key].attrs['TIME']).ravel()[0])
        if result:
            previous=float(np.asarray(data[result[-1]].attrs['TIME']).ravel()[0])
            tolerance=64*np.finfo(float).eps*max(abs(previous),abs(time),1e-12)
            if abs(time-previous)<=tolerance:
                discarded.append(result.pop())
            elif time<previous:
                raise ValueError('Monitor times are not in passage order')
        result.append(key)
    return result,discarded


def fixed_ids(ids,count=1024,seed=90217):
    """Select a reproducible ID sample independently of phase-space coordinates."""
    ids=np.sort(np.asarray(ids))
    if count<1 or len(np.unique(ids))!=len(ids):
        raise ValueError('Positive requested sample count and unique particle IDs required')
    return np.sort(np.random.default_rng(seed).choice(ids,min(count,len(ids)),replace=False))


def subset_rows(ids,phase,selected):
    """Retain fixed ID identities; mark disappeared particles with explicit NaNs."""
    ids=np.asarray(ids);selected=np.asarray(selected)
    indices=np.searchsorted(ids,selected)
    exists=indices<len(ids)
    exists[exists] &= ids[indices[exists]]==selected[exists]
    subset=np.full((len(selected),6),np.nan)
    subset[exists]=phase[indices[exists]]
    return subset,exists


def full_diagnostics(phase,target_covariance=None):
    """Compute full-bunch moments and spatial-density tests, before subsampling."""
    centroid=np.mean(phase,axis=0)
    centered=phase-centroid
    covariance=centered.T@centered/len(phase)
    spatial=phase[:,[0,2,4]]
    density=uniform_density_diagnostics(spatial)
    result=dict(count=len(phase),centroid=centroid,covariance=covariance,density_measured=density)
    if target_covariance is not None:
        result['density_model']=uniform_density_diagnostics(spatial,np.asarray(target_covariance)[np.ix_([0,2,4],[0,2,4])])
    return result


def reduce(directory,sections,p0,output,count=1024,seed=90217,target_covariance=None):
    """Write compact section observations, all-particle diagnostics and raw endpoints."""
    directory=Path(directory);output=Path(output)
    output.mkdir(exist_ok=False)
    selected=None;events=[];records=[];sources=[]
    with h5py.File(output/'raw-first-last.h5','w') as retained:
        for cell,section in enumerate(sections):
            path=directory/f'monitor-{cell+1}.h5'
            source=dict(file=str(path.resolve()),sha256=sha256(path),bytes=path.stat().st_size)
            with h5py.File(path,'r') as data:
                keys,discarded=unique_snapshot_keys(data)
                source.update(unique_passages=len(keys),discarded_duplicate_keys=discarded)
                if not keys: raise ValueError('No monitor passages')
                parent=retained.create_group(f'monitor-{cell+1}')
                for key,value in data.attrs.items(): parent.attrs[key]=value
                for key in dict.fromkeys([keys[0],keys[-1]]):
                    data.copy(key,parent,name=key)
                for passage,key in enumerate(keys):
                    monitor=read_monitor_group(data[key],section,p0)
                    if selected is None: selected=fixed_ids(monitor['ids'],count,seed)
                    subset,present=subset_rows(monitor['ids'],monitor['phase'],selected)
                    diag=full_diagnostics(monitor['phase'],target_covariance)
                    row=dict(cell=cell+1,passage=passage,group=key,time_s=monitor['time_s'],particles=diag['count'],sample_present=int(present.sum()),
                        row_time_error_max_s=float(abs(monitor['row_times_s']-monitor['time_s']).max()),
                        **{'density_measured_'+key:value for key,value in diag['density_measured'].items()})
                    if 'density_model' in diag: row.update({'density_model_'+key:value for key,value in diag['density_model'].items()})
                    records.append(row)
                    events.append(dict(time_s=monitor['time_s'],cell=cell,phase=subset,present=present,centroid=diag['centroid'],covariance=diag['covariance']))
            sources.append(source)
    events.sort(key=lambda event:event['time_s'])
    cells=np.array([event['cell'] for event in events])
    if np.any(np.diff(cells)%6!=1):
        raise ValueError('Missing cell sections: cannot assign uninterrupted cell sampling')
    np.savez_compressed(output/'section-observations.npz',phase=np.array([event['phase'] for event in events]),
        t_s=np.array([event['time_s'] for event in events]),cells=cells,ids=selected,
        turn_coordinate=np.arange(len(events))/6,present=np.array([event['present'] for event in events]),
        centroid=np.array([event['centroid'] for event in events]),covariance=np.array([event['covariance'] for event in events]))
    pd.DataFrame(records).sort_values('time_s').to_csv(output/'full-bunch-diagnostics.csv',index=False)
    write_json(output/'sampling.json',dict(seed=seed,requested_count=count,actual_count=len(selected),ids=selected.tolist(),
        algorithm='numpy.default_rng(seed).choice(sorted(first_snapshot_ids), count, replace=False); same IDs in every passage',
        moments_and_density='All particles in every snapshot; no particle sampling for these observables',
        times='H5 group TIME; row time is the integration endpoint and is not used',
        duplicate_rule='Same monitor, group TIME within64 machine-epsilon relative tolerance: retain last stored group',
        raw_retention='First and last unique full snapshot for each of the6 monitors; all input files remain untouched'))
    for filename in ['provenance.json','study.json','case.in','launch.json']:
        if (directory/filename).exists(): shutil.copy2(directory/filename,output/filename)
    shutil.copy2(__file__,output/Path(__file__).name)
    outputs=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha256(p)) for p in output.iterdir() if p.is_file()]
    manifest=dict(source_directory=str(directory.resolve()),sources=sources,outputs=outputs,passages=len(events),
        raw_input_bytes=sum(v['bytes'] for v in sources),compact_output_bytes=sum(v['bytes'] for v in outputs),
        raw_inputs_deleted=False,p0_mc=p0,section_poses=sections)
    write_json(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--count',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=90217)
    args=parser.parse_args()
    sections=json.loads((args.study/'sections.json').read_text())
    p0=np.linalg.norm(json.loads((args.study/'bare-cof.json').read_text())['momentum_mc'])
    target=None if args.model is None else np.load(args.model)['covariance']
    result=reduce(args.directory,sections,p0,args.output,args.count,args.seed,target)
    print(json.dumps({key:result[key] for key in ['passages','raw_input_bytes','compact_output_bytes','raw_inputs_deleted']},indent=2))

if __name__=='__main__': main()
