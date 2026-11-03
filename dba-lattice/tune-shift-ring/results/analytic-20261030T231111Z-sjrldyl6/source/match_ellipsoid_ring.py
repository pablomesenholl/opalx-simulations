"""Periodic 6D rms closure for the isolated DBA-ring ellipsoid.

Internal analysis utility, not a new OPALX input API. SI canonical paraxial
coordinates are (x, px/p0, y, py/p0, z, delta), with z toward the bunch head.
A finite linearized cavity in each D2 centre has delta'=rf_slope*z/L. Moment matching
retains dispersion and x-z tilt, but does not establish a Vlasov equilibrium.
See Sacherer, CERN-SI-Int-DL-70-12, and Venturini/Reiser, PRL81,96(1998).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import least_squares
from ellipsoid_model import field_tensor, rf_kick, transport_generator
from kv_model import beam_parameters

J = np.kron(np.eye(3), np.array([[0.,1.],[-1.,0.]]))
POS = np.array([0,2,4]); MOM = np.array([1,3,5])
# Midplane symmetry decouples y,py from x,px,z,delta.
MATCH_ORDER = np.array([0,1,4,5,2,3])
MATCH_TRI = np.array([(i,j) for i in range(6) for j in range(i+1)
                      if (i<4 and j<4) or (i>=4 and j>=4)]).T
TRI = tuple(MATCH_ORDER[MATCH_TRI])


def normal_mode_basis(matrix):
    """Return a real symplectic normal-mode basis and signed cell phases.

    Columns are canonical (q,p) pairs ordered x,y,z. For X=W U, normal
    coordinates U=W^-1 X rotate by [[cos(mu),sin(mu)],[-sin(mu),cos(mu)]].
    Thus U_q-i U_p advances by +mu. The longitudinal phase is negative above
    transition; its quoted oscillation frequency is the absolute value.
    """
    if np.shape(matrix)!=(6,6): raise ValueError('6x6 map required')
    values,vectors=np.linalg.eig(matrix)
    if max(abs(abs(values)-1))>2e-7:
        raise ValueError(f'Unstable cell map: max |lambda|={max(abs(values)):.10g}')
    modes={}
    for j,value in enumerate(values):
        if value.imag<=1e-9: continue
        v=vectors[:,j]; symp=float(np.imag(v.conj()@J@v))
        if abs(symp)<1e-12: raise ValueError('Degenerate symplectic mode')
        key='y' if np.linalg.norm(v[2:4])>.5*np.linalg.norm(v) else ('x' if symp>0 else 'z')
        if key in modes: raise ValueError('Ambiguous mode identification')
        norm=np.sqrt(abs(symp)/2)
        modes[key]=(v.real/norm,np.sign(symp)*v.imag/norm,
                    float(np.sign(symp)*np.angle(value)))
    if set(modes)!={'x','y','z'}: raise ValueError('Expected three nondegenerate stable modes')
    basis=np.column_stack([v for axis in 'xyz' for v in modes[axis][:2]])
    return basis,np.array([modes[axis][2] for axis in 'xyz'])


def mode_covariance(matrix, emittances):
    """Invariant positive covariance and absolute x/y/z phases of a stable map.

    Modes are identified by vertical participation and symplectic signature.
    The above-transition longitudinal mode has negative Hamiltonian signature.
    Input emittances are positive geometric rms eigenemittances [m].
    """
    if np.shape(emittances)!=(3,) or min(emittances)<=0:
        raise ValueError('Three positive rms eigenemittances required')
    basis,phases=normal_mode_basis(matrix)
    bases={axis:basis[:,2*i:2*i+2]@basis[:,2*i:2*i+2].T for i,axis in enumerate('xyz')}
    covariance=sum(emittances[i]*bases[axis] for i,axis in enumerate('xyz'))
    return covariance,abs(phases),bases,{axis:float(np.sign(phases[i])) for i,axis in enumerate('xyz')}


def rf_layout(elements, gap_length=.1):
    """Replace the centre of D2 with a finite linear lens, preserving geometry.

    The existing prescribed-Ez OPALX cavity neglects transverse RF gap forces.
    This linear moment model also fixes reference energy within the zero-net-gain
    gap. The sinusoidal RF waveform/finite transit are checked separately in PIC.
    """
    result=[]; found=False
    for el in elements:
        if el['name']!='D2':
            result.append(dict(el)); continue
        if not 0<gap_length<el['length_m']: raise ValueError('RF gap must fit inside D2')
        side=(el['length_m']-gap_length)/2
        result.extend([dict(el,name='D2a',length_m=side),
                       dict(el,name='RF',length_m=gap_length,rf_weight_per_m=1/gap_length),
                       dict(el,name='D2b',length_m=side)])
        found=True
    if not found: raise ValueError('No D2 straight available for RF cavity')
    return result


def external_generator(element,gamma,rf_slope):
    """Hamiltonian external generator, including a finite ideal longitudinal lens."""
    matrix=transport_generator(element,gamma)
    matrix[5,4]+=rf_slope*element.get('rf_weight_per_m',0.)
    return matrix


def bare_cell_map(elements, gamma, rf_slope):
    """Exact piecewise-constant Hamiltonian cell map, including the RF lens.

    For backwards-compatible comparisons only, elements without an RF marker
    receive a thin cavity at the cell exit.
    """
    matrix = np.eye(6)
    for el in elements:
        matrix = expm(external_generator(el,gamma,rf_slope)*el['length_m'])@matrix
    if not any('rf_weight_per_m' in el for el in elements): matrix=rf_kick(rf_slope)@matrix
    return matrix


def cell_pass(covariance, elements, charge, beta, gamma, rf_slope,
              rtol=2e-9, samples=0, with_map=True, return_maps=False):
    """Propagate covariance plus frozen-self-field particle monodromy.

    C'=F(C)C+CF(C)^T. The monodromy M'=F(C)M gives incoherent particle
    modes of this periodic closure, not collective envelope eigenmodes.
    Internal scaling conditions both position and momentum components.
    """
    covariance=np.asarray(covariance,float)
    scales=np.sqrt(np.diag(covariance)); outer=np.outer(scales,scales)
    if not np.all(np.isfinite(scales)) or min(scales)<=0: raise ValueError('Invalid covariance')
    inverse=np.reciprocal(scales)
    state=(covariance/outer).ravel()
    if with_map: state=np.r_[state,np.eye(6).ravel()]
    rows=[(0.,covariance.copy())]; map_rows=[np.eye(6)]; offset=0.
    if return_maps and not with_map: raise ValueError('Map samples require map integration')
    def rhs(_,state,external):
        cov=state[:36].reshape(6,6)*outer; cov=(cov+cov.T)/2
        f=external.copy()
        if charge:
            f[np.ix_(MOM,POS)] += field_tensor(cov[np.ix_(POS,POS)],charge,beta,gamma)
        scaled=f*scales[None,:]*inverse[:,None]
        c=state[:36].reshape(6,6)
        result=(scaled@c+c@scaled.T).ravel()
        if with_map: result=np.r_[result,(scaled@state[36:].reshape(6,6)).ravel()]
        return result
    for el in elements:
        sol=solve_ivp(rhs,(0,el['length_m']),state,args=(external_generator(el,gamma,rf_slope),),
                      method='DOP853',rtol=rtol,atol=rtol*.01,dense_output=bool(samples))
        if not sol.success: raise RuntimeError(sol.message)
        if samples:
            grid=np.linspace(0,el['length_m'],samples+1)[1:]
            rows.extend((offset+s,sol.sol(s)[:36].reshape(6,6)*outer) for s in grid)
            if with_map: map_rows.extend(sol.sol(s)[36:].reshape(6,6)*scales[:,None]*inverse[None,:] for s in grid)
        state=sol.y[:,-1]; offset+=el['length_m']
    kick=rf_kick(0. if any('rf_weight_per_m' in el for el in elements) else rf_slope)
    final=kick@(state[:36].reshape(6,6)*outer)@kick.T
    matrix=None
    if with_map:
        matrix=kick@(state[36:].reshape(6,6)*scales[:,None]*inverse[None,:])
    rows.append((offset,final.copy()))
    if return_maps: return (final+final.T)/2,matrix,rows,np.array(map_rows+[matrix.copy()])
    return (final+final.T)/2,matrix,rows


def eigenemittances(covariance):
    """Three positive symplectic rms eigenemittances, returned sorted [m]."""
    values=np.linalg.eigvals(J@covariance)
    positive=np.sort(np.imag(values[np.imag(values)>0]))
    if len(positive)!=3 or np.max(abs(values.real))>1e-9*max(positive):
        raise ValueError('Covariance does not have positive symplectic spectrum')
    return positive


def match_covariance(elements, charge, beta, gamma, rf_slope, emittances,
                     initial=None, rtol=2e-9, tolerance=2e-7, max_evaluations=100):
    """Positive-definite Cholesky shooting at prescribed rms eigenemittances.

    Periodicity has three redundant scalar conditions; three symplectic
    eigenemittance constraints remove the family of invariant covariances.
    The overdetermined solve retains all moment residuals for honest checking.
    """
    if initial is None:
        initial=mode_covariance(bare_cell_map(elements,gamma,rf_slope),emittances)[0]
    scales=np.sqrt(np.diag(initial)); outer=np.outer(scales,scales)
    target=np.sort(emittances)
    def encode(c):
        ordered=(c/outer)[np.ix_(MATCH_ORDER,MATCH_ORDER)]
        l=np.linalg.cholesky(ordered); z=l[tuple(MATCH_TRI)].copy()
        diagonal=MATCH_TRI[0]==MATCH_TRI[1]; z[diagonal]=np.log(z[diagonal]); return z
    def decode(z):
        l=np.zeros((6,6)); l[tuple(MATCH_TRI)]=z; diagonal=np.diag_indices(6)
        l[diagonal]=np.exp(l[diagonal]); ordered=l@l.T
        inverse=np.argsort(MATCH_ORDER)
        return ordered[np.ix_(inverse,inverse)]*outer
    def residual(z):
        cov=decode(z)
        final,_,_=cell_pass(cov,elements,charge,beta,gamma,rf_slope,rtol=rtol,with_map=False)
        return np.r_[((final-cov)/outer)[TRI],np.log(np.r_[np.sqrt(np.linalg.det(cov[2:4,2:4])),
                          np.sort(np.imag(np.linalg.eigvals(J[:4,:4]@cov[np.ix_(MATCH_ORDER[:4],MATCH_ORDER[:4])])))[2:]]
                         /np.r_[emittances[1],np.sort([emittances[0],emittances[2]])])]
    def jacobian(z):
        steps=np.eye(len(z))*2e-5
        return np.column_stack([(residual(z+h)-residual(z-h))/(4e-5) for h in steps])
    fit=least_squares(residual,encode(initial),jac=jacobian,xtol=2e-10,ftol=2e-10,gtol=2e-10,
                      max_nfev=max_evaluations)
    result=decode(fit.x); resid=residual(fit.x); error=float(max(abs(resid)))
    final,matrix,rows,particle_maps=cell_pass(result,elements,charge,beta,gamma,rf_slope,rtol=rtol,samples=12,return_maps=True)
    if error>tolerance: raise RuntimeError(f'Moment matching failed: residual={error:.4g}, {fit.message}')
    _,phase,_,signs=mode_covariance(matrix,emittances)
    return dict(covariance=result,phase=phase,matrix=matrix,rows=rows,
                residual=error,evaluations=fit.nfev,mode_signatures=signs,particle_maps=particle_maps)


def envelope_multipliers(covariance,elements,charge,beta,gamma,rf_slope,
                         step=2e-5,rtol=2e-11):
    """Collective 21-moment Floquet multipliers by centred finite differences.

    Includes cross-plane perturbations omitted from midplane-symmetric matching.
    Neutral unit multipliers are expected because eigenemittances are conserved.
    Repeat with half step to distinguish physical growth from differentiation
    noise. These test the assumed ellipsoid closure, not nonlinear PIC stability.
    """
    tri=np.tril_indices(6); scale=np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))
    columns=[]
    for i,j in zip(*tri):
        perturb=np.zeros((6,6)); perturb[i,j]=perturb[j,i]=step*scale[i,j]
        plus=cell_pass(covariance+perturb,elements,charge,beta,gamma,rf_slope,rtol=rtol,with_map=False)[0]
        minus=cell_pass(covariance-perturb,elements,charge,beta,gamma,rf_slope,rtol=rtol,with_map=False)[0]
        columns.append(((plus-minus)/(2*step*scale))[tri])
    matrix=np.column_stack(columns)
    return np.linalg.eigvals(matrix),matrix


def rf_voltage(rf_slope, beta, gamma, circumference, harmonic):
    """Zero-crossing thin gap voltage magnitude [V] and frequency [Hz].

    delta=p/p0-1, z positive toward head; |g|=e V k_rf/(beta^2 gamma m c^2).
    Voltage polarity/arrival phase must realize the requested signed slope.
    """
    from scipy.constants import c, proton_mass, elementary_charge
    k=2*np.pi*harmonic/circumference
    voltage=abs(rf_slope)*beta*beta*gamma*proton_mass*c*c/(elementary_charge*k)
    return float(voltage),float(harmonic*beta*c/circumference)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parents[1]
    parser.add_argument('--config',type=Path,default=base/'config.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rf-slope',type=float,default=.001)
    parser.add_argument('--rf-location',choices=['d2-center','cell-exit'],default='d2-center')
    parser.add_argument('--rf-length',type=float,default=.1)
    parser.add_argument('--charges-pc',type=float,nargs='+',default=[0,2,5,10,20,30,41.13101073])
    parser.add_argument('--sigma-z',type=float,default=.05/np.sqrt(5))
    parser.add_argument('--harmonic',type=int,default=20)
    args=parser.parse_args(); cfg=json.loads(args.config.read_text())
    if args.rf_location=='d2-center': cfg['elements']=rf_layout(cfg['elements'],args.rf_length)
    beta,gamma,*_=beam_parameters(cfg['kinetic_energy_MeV'])
    emits=np.array([cfg['rms_emittance_x_m_rad'],cfg['rms_emittance_y_m_rad'],1e-6])
    bare=bare_cell_map(cfg['elements'],gamma,args.rf_slope)
    _,_,bases,_=mode_covariance(bare,emits)
    transverse=emits[0]*bases['x']+emits[1]*bases['y']
    emits[2]=(args.sigma_z**2-transverse[4,4])/bases['z'][4,4]
    if emits[2]<=0: raise ValueError('Requested bunch length smaller than transverse mode contribution')
    args.output.mkdir(parents=True,exist_ok=True)
    from shutil import copy2
    copy2(__file__,args.output/Path(__file__).name)
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    records=[]; initial=None
    for charge in args.charges_pc:
        print(f'Matching {charge:.8g} pC with rf_slope={args.rf_slope:.8g} /m',flush=True)
        try:
            match=match_covariance(cfg['elements'],charge*1e-12,beta,gamma,args.rf_slope,emits,initial)
        except (ValueError,RuntimeError) as e:
            records.append(dict(charge_pC=charge,status='failed',reason=str(e)))
            print(str(e),flush=True); break
        initial=match['covariance']; tunes=match['phase']*cfg['cells']/(2*np.pi)
        sizes=np.sqrt(np.array([np.diag(c)[POS] for _,c in match['rows']]))
        circumference=cfg['cells']*sum(el['length_m'] for el in cfg['elements'])
        wave_number=2*np.pi*args.harmonic/circumference
        rf_start=sum(el['length_m'] for el in cfg['elements'][:next((i for i,e in enumerate(cfg['elements']) if 'rf_weight_per_m' in e),len(cfg['elements']))])
        cavity_sizes=[np.sqrt(c[4,4]) for s,c in match['rows'] if rf_start-1e-12<=s<=rf_start+args.rf_length+1e-12]
        maximum_phase=wave_number*np.sqrt(5)*max(cavity_sizes or [sizes[:,2].max()])
        record=dict(charge_pC=charge,status='matched_moment_closure',covariance=initial.tolist(),
                    rf_max_bunch_phase_rad=float(maximum_phase),
                    rf_edge_sine_relative_deviation=float(1-np.sinc(maximum_phase/np.pi)),
                    rf_edge_cosine_relative_deviation=float(1-np.cos(maximum_phase)),
                    tunes_xyz=tunes.tolist(),scaled_residual=match['residual'],
                    eigenemittances_m=eigenemittances(initial).tolist(),
                    sigma_min_m=sizes.min(axis=0).tolist(),sigma_max_m=sizes.max(axis=0).tolist(),
                    mode_signatures=match['mode_signatures'],solver_evaluations=match['evaluations'])
        records.append(record); print(json.dumps({k:record[k] for k in ['charge_pC','tunes_xyz','scaled_residual','sigma_max_m']}),flush=True)
        basis,signed_phases=normal_mode_basis(match['matrix'])
        np.savez(args.output/f'match-{charge:g}pc.npz',covariance=initial,matrix=match['matrix'],
                 normal_mode_basis=basis,normal_mode_inverse=np.linalg.inv(basis),signed_cell_phases=signed_phases,
                 s=np.array([s for s,_ in match['rows']]),particle_maps=match['particle_maps'],covariances=np.array([c for _,c in match['rows']]))
        (args.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
    circumference=cfg['cells']*sum(el['length_m'] for el in cfg['elements'])
    voltage,frequency=rf_voltage(args.rf_slope,beta,gamma,circumference,args.harmonic)
    result=dict(model='Periodic uniform-ellipsoid rms closure; not an exact Vlasov equilibrium',
                rf_location=('finite linearized cavity at D2 centre' if args.rf_location=='d2-center' else 'thin cavity at every cell exit'),rf_length_m=args.rf_length,rf_slope_per_m=args.rf_slope,
                cavity_voltage_V=voltage,frequency_Hz=frequency,harmonic=args.harmonic,
                transit_time_factor=float(np.sinc(args.rf_length*args.harmonic/circumference)),
                cavity_peak_field_MV_per_m=float(voltage/(args.rf_length*np.sinc(args.rf_length*args.harmonic/circumference))*1e-6),
                beta=beta,gamma=gamma,circumference_m=circumference,
                ordered_eigenemittances_xyz_m=emits.tolist(),cases=records)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
