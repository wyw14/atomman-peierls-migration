"""Stationary dislocation branches and harmonic overdamped escape rates."""
import atomman as am
import numpy as np
import time
from ._energy import Energy

KB_EV=8.617333262145e-5


def _positive(value,name):
    try:
        result=float(value)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError(name) from error
    if not np.isfinite(result) or result<=0:
        raise ValueError(name)
    return result


def solve(model,settings):
    """Evaluate the continuation branches defined by the migration contract."""
    deadline=time.monotonic()+150
    if not isinstance(model,am.defect.SDVPN) or not isinstance(settings,dict):
        raise ValueError('model and settings')
    try:
        n=settings['n']
        width=settings['plateau_width']
        partial=settings['partial']
        spacing=_positive(settings['spacing_A'],'spacing_A')
        temperature=_positive(settings['temperature_K'],'temperature_K')
        length=_positive(settings['line_length_A'],'line_length_A')
        frequency=_positive(settings['frequency_s'],'frequency_s')
        mobility=np.asarray(settings['mobility'],dtype=float)
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        raise ValueError('settings') from error
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n%2!=1 or not 13<=n<=49:
        raise ValueError('n')
    if isinstance(width,bool) or not isinstance(width,(int,np.integer)) or width%2!=1 or not 3<=width<=9 or n-width<8:
        raise ValueError('plateau_width')
    if partial not in ('leading','trailing'):
        raise ValueError('partial')
    if mobility.shape!=(2,2) or not np.all(np.isfinite(mobility)) or not np.array_equal(mobility,mobility.T):
        raise ValueError('mobility')
    if np.linalg.eigvalsh(mobility)[0]<=0:
        raise ValueError('mobility')
    if not model.fullstress or model.cdiffelastic or model.cdiffstress:
        raise ValueError('SDVPN energy convention')
    if np.any(np.asarray(model.alpha)!=0) or np.any(model.beta!=0):
        raise ValueError('SDVPN correction energies')
    energy=Energy(model,int(n),spacing,dissociated=True,partial=partial,width=int(width))
    minima=[]
    minimum_hessians=[]
    minimum_energies=[]
    for which in ('left','right'):
        state,_=energy.stationary(which)
        value,_,hessian=energy.jet(state)
        minima.append(state)
        minimum_hessians.append(hessian)
        minimum_energies.append(value)
    saddle,_=energy.stationary('saddle')
    value,_,hessian=energy.jet(saddle)
    # Soft collective modes amplify binary64 force and interpolation errors.
    # This changes arithmetic precision, not the physical branch criterion.
    matrices=minimum_hessians+[hessian]
    conditions=[np.linalg.cond(matrix) for matrix in matrices]
    residuals=[np.linalg.norm(energy.jet(state)[1],np.inf)
               for state in minima+[saddle]]
    if max(conditions)>1000 or max(residuals)>1e-12:
        from ._precision import refine
        result,_=refine(model,settings,{'left':minima[0],'right':minima[1],'saddle':saddle},deadline)
        return result
    eigenvalues,eigenvectors=np.linalg.eigh(hessian)
    mode=eigenvectors[:,0]
    center=n//2+(width//2 if partial=='leading' else -(width//2))
    mode*=np.sign(mode[2*center])
    full_mobility=np.kron(np.eye(n),mobility)
    growth=float(np.linalg.eigvals(-full_mobility@hessian).real.max())
    barriers=value-np.asarray(minimum_energies)
    logs=[]
    for barrier,minimum_hessian in zip(barriers,minimum_hessians):
        logdet=np.linalg.slogdet(minimum_hessian)[1]-np.linalg.slogdet(hessian)[1]
        logs.append(np.log(frequency*growth/(2*np.pi))+.5*logdet-length*barrier/(KB_EV*temperature))
    return {'minima':np.asarray(minima),'saddle':saddle,
            'barriers_eV_per_A':barriers,'negative_eigenvalue':float(eigenvalues[0]),
            'unstable_mode':mode.reshape(n,2),'growth_rate_s':frequency*growth,
            'log_rates_s':np.asarray(logs)}
