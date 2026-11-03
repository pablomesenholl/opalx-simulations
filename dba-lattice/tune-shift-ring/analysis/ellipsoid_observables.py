"""Density, moments and particle normal-mode observables for isolated-bunch studies."""
from __future__ import annotations
import numpy as np
from scipy.constants import epsilon_0, elementary_charge, proton_mass, c


def uniform_density_diagnostics(positions, covariance=None):
    """Quantify departure from a uniform spatial ellipsoid, independently of PIC.

    With covariance C, edge-normalized radius r=sqrt(x^T C^-1 x/5) has CDF r^3
    on[0,1]. Reporting both fixed-model and measured-C versions distinguishes
    profile distortion from an envelope mismatch. No universal p-value is
    assigned when C is estimated from the same sample.
    """
    positions=np.asarray(positions,float)
    if positions.ndim!=2 or positions.shape[1]!=3 or len(positions)<4:
        raise ValueError('At least four three-dimensional positions required')
    centered=positions-positions.mean(axis=0)
    if covariance is None: covariance=centered.T@centered/len(centered)
    radius2=np.einsum('ni,ni->n',centered,np.linalg.solve(covariance,centered.T).T)/5
    radii=np.sqrt(np.maximum(0,radius2));ordered=np.sort(radii);n=len(radii)
    expected=np.clip(ordered,0,1)**3
    ks=max(np.max(abs(np.arange(1,n+1)/n-expected)),np.max(abs(np.arange(n)/n-expected)))
    return dict(count=n,radial_cdf_error=float(ks),outside_ellipsoid_fraction=float(np.mean(radii>1)),
                fourth_radial_moment_relative_error=float(np.mean(radius2**2)/(3/7)-1),
                core_density_ratio_r025=float(np.mean(radii<.25)/.25**3),
                core_density_ratio_r050=float(np.mean(radii<.5)/.5**3))


def moment_diagnostics(phase, target):
    """Centroid, rms-size and covariance changes relative to a matched section."""
    from match_ellipsoid_ring import eigenemittances
    phase=np.asarray(phase,float);target=np.asarray(target,float)
    centered=phase-phase.mean(axis=0);covariance=centered.T@centered/len(phase)
    scales=np.sqrt(np.outer(np.diag(target),np.diag(target)))
    return dict(covariance=covariance.tolist(),centroid=phase.mean(axis=0).tolist(),
                max_scaled_covariance_error=float(np.max(abs((covariance-target)/scales))),
                rms_size_relative_changes=(np.sqrt(np.diag(covariance)[[0,2,4]]/np.diag(target)[[0,2,4]])-1).tolist(),
                eigenemittance_relative_changes=(eigenemittances(covariance)/eigenemittances(target)-1).tolist())


def virial_force_gradient(positions,charge,beta,gamma,pairs=1048576,seed=6171,batches=32):
    """Estimate the whole-bunch least-squares SC force gradient from Coulomb pairs.

    Independent of a PIC mesh. In the rest frame, <E_i r_j> equals
    Q/(8*pi*eps0) * <d_i d_j/|d|^3> over independent source-point pairs.
    Multiplication by C_r^-1 gives a least-squares field gradient. Transforming
    back yields the force coefficient for (px/p0,py/p0,delta)'=S(x,y,z).
    Finite pair sampling has Monte Carlo error; batch standard errors are
    reported. This is a whole-bunch linear fit, not a central force gradient.
    """
    if pairs<2*batches or len(positions)<4: raise ValueError('Insufficient points or pairs')
    boost=np.diag([1.,1.,gamma]);r=np.asarray(positions)@boost
    r=r-r.mean(axis=0);covariance=r.T@r/len(r);inverse=np.linalg.inv(covariance)
    rng=np.random.default_rng(seed);estimates=[]
    prefactor=elementary_charge/(proton_mass*c*c*beta*beta*gamma*gamma)
    for batch in range(batches):
        count=pairs//batches+(batch<pairs%batches)
        first=rng.integers(0,len(r),count);second=rng.integers(0,len(r)-1,count)
        second+=second>=first
        difference=r[first]-r[second];norm=np.linalg.norm(difference,axis=1)
        weight=np.reciprocal(norm**3)
        virial=np.einsum('ni,nj,n->ij',difference,difference,weight)/count*charge/(8*np.pi*epsilon_0)
        estimates.append(prefactor*boost@(virial@inverse)@boost)
    estimates=np.array(estimates)
    return estimates.mean(axis=0),estimates.std(axis=0,ddof=1)/np.sqrt(batches)


def normal_mode_coordinates(phase,inverse,centroid=None):
    """Transform (...,6) section observations to real symplectic mode coordinates.

    Coordinates must be in the same local straight frame and normalization as
    the saved basis. A supplied centroid may be six numbers or (events,6); no
    implicit particle-subset mean is removed.
    """
    phase=np.asarray(phase,float);inverse=np.asarray(inverse,float)
    if phase.shape[-1]!=6 or inverse.shape!=(6,6): raise ValueError('Six-dimensional phase and6x6 inverse required')
    if centroid is not None:
        centroid=np.asarray(centroid,float)
        if centroid.ndim==phase.ndim-1: centroid=centroid[...,None,:]
        phase=phase-centroid
    return phase@inverse.T


def phase_frequency(signal,turn_coordinate):
    """Amplitude-weighted unwrapped-phase frequency and residual [rad].

    This is a diagnostic estimator, not an independent-error uncertainty model;
    correlated PIC phase errors invalidate ordinary least-squares error bars.
    """
    signal=np.asarray(signal,complex);turns=np.asarray(turn_coordinate,float)
    if signal.ndim==1: signal=signal[:,None]
    if len(turns)!=len(signal) or len(turns)<4 or np.any(np.diff(turns)<=0):
        raise ValueError('At least four strictly increasing observation times required')
    weight=abs(signal)**2; total=weight.sum(axis=0)
    good=total>np.finfo(float).tiny
    safe=np.where(good,total,1.)
    centered=turns[:,None]-(weight*turns[:,None]).sum(axis=0)/safe
    angle=np.unwrap(np.angle(signal),axis=0)
    mean=(weight*angle).sum(axis=0)/safe
    denominator=(weight*centered**2).sum(axis=0)
    slope=np.divide((weight*centered*(angle-mean)).sum(axis=0),denominator,
                    out=np.full_like(total,np.nan),where=denominator>0)
    residual=np.sqrt((weight*(angle-mean-slope*centered)**2).sum(axis=0)/safe)
    return slope/(2*np.pi),residual


def spectral_frequency(signal,turn_coordinate,sign=1,oversampling=8):
    """Independent Hann-spectrum peak with parabolic log-power interpolation.

    Equivalent cell crossings must be uniformly spaced in turn coordinate.
    Searching the entire signed Nyquist interval does not constrain results
    toward the predicted tune. Returns frequency and coherent spectral fraction.
    """
    signal=np.asarray(signal,complex);turns=np.asarray(turn_coordinate,float)
    if signal.ndim==1:signal=signal[:,None]
    step=np.diff(turns)
    if len(turns)<8 or len(signal)!=len(turns) or not np.allclose(step,step[0],atol=1e-12,rtol=1e-10):
        raise ValueError('At least eight uniformly spaced section observations required')
    if sign not in [-1,1] or oversampling<2:raise ValueError('Signed mode and oversampling>=2 required')
    window=np.hanning(len(turns));weighted=signal*window[:,None]
    nfft=2**int(np.ceil(np.log2(len(turns)*oversampling)))
    frequencies=np.fft.fftfreq(nfft,step[0]);power=abs(np.fft.fft(weighted,nfft,axis=0))**2
    allowed=frequencies*sign>0
    selected=np.argmax(np.where(allowed[:,None],power,-np.inf),axis=0);columns=np.arange(signal.shape[1])
    logs=np.log(np.maximum(power,np.finfo(float).tiny))
    left=logs[(selected-1)%nfft,columns];middle=logs[selected,columns];right=logs[(selected+1)%nfft,columns]
    curvature=left-2*middle+right
    fraction=np.divide(.5*(left-right),curvature,out=np.zeros_like(curvature),where=abs(curvature)>1e-30)
    fraction=np.clip(fraction,-.5,.5)
    estimate=frequencies[selected]+fraction/(nfft*step[0])
    phasor=np.exp(-2j*np.pi*turns[:,None]*estimate[None,:])
    denominator=window.sum()*(window[:,None]*abs(signal)**2).sum(axis=0)
    coherence=np.divide(abs((weighted*phasor).sum(axis=0))**2,denominator,
                        out=np.zeros_like(denominator),where=denominator>0)
    return estimate,coherence


def individual_mode_tunes(phase,turn_coordinate,inverse,signed_cell_phases,
                          centroid=None,batch_size=512,minimum_amplitude=1e-12):
    """Individual incoherent-mode estimates, amplitudes and independent checks.

    Input shape is(events,particles,6). Qx/Qy retain their integer parts when
    observations are made at all six equivalent cell crossings per turn.
    Tiny numerical leakage into unexcited modes is marked inactive; its fitted
    frequency is NaN. Both phase and spectral estimates are retained honestly.
    """
    phase=np.asarray(phase,float);turns=np.asarray(turn_coordinate,float)
    if phase.ndim!=3 or phase.shape[2]!=6 or len(phase)!=len(turns):
        raise ValueError('Expected(events,particles,6) observations')
    if len(turns)<8 or not np.all(np.isfinite(phase)):raise ValueError('Finite observations at>=8crossings required')
    n=phase.shape[1];shape=(n,3)
    result={name:np.full(shape,np.nan) for name in ['phase_tune_signed','spectral_tune_signed','phase_residual_rad',
                                                  'spectral_coherent_fraction','rms_mode_amplitude']}
    active=np.zeros(shape,bool)
    for start in range(0,n,batch_size):
        stop=min(start+batch_size,n)
        coordinates=normal_mode_coordinates(phase[:,start:stop],inverse,centroid)
        for mode in range(3):
            signal=coordinates[:,:,2*mode]-1j*coordinates[:,:,2*mode+1]
            amplitude=np.sqrt(np.mean(abs(signal)**2,axis=0))
            # Relative leakage threshold is local to this particle batch. The
            # explicit absolute floor handles a batch with no excited modes.
            valid=amplitude>max(minimum_amplitude,float(amplitude.max())*1e-8)
            estimate,residual=phase_frequency(signal,turns)
            spectral,coherence=spectral_frequency(signal,turns,sign=int(np.sign(signed_cell_phases[mode])))
            for name,values in [('phase_tune_signed',estimate),('spectral_tune_signed',spectral),
                                ('phase_residual_rad',residual),('spectral_coherent_fraction',coherence)]:
                result[name][start:stop,mode]=np.where(valid,values,np.nan)
            result['rms_mode_amplitude'][start:stop,mode]=amplitude;active[start:stop,mode]=valid
    result['active']=active
    result['phase_tune_absolute']=abs(result['phase_tune_signed'])
    result['spectral_tune_absolute']=abs(result['spectral_tune_signed'])
    result['estimator_difference']=abs(result['phase_tune_signed']-result['spectral_tune_signed'])
    return result
