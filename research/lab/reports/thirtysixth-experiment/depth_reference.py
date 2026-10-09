"""Fixed texture-transition proxy; no anatomical or ink-depth interpretation."""
import numpy as np
from scipy.ndimage import uniform_filter

def texture_energy(volume):
    x=np.asarray(volume,dtype=np.float64)
    if x.ndim!=3 or x.shape[0]!=63 or min(x.shape[1:])<9 or not np.isfinite(x).all():
        raise ValueError('Finite 63-plane volume with spatial halo required')
    gx=(x[:,1:-1,2:]-x[:,1:-1,:-2])/2
    gy=(x[:,2:,1:-1]-x[:,:-2,1:-1])/2
    return uniform_filter(gx*gx+gy*gy,size=(1,7,7),mode='constant')[:,3:-3,3:-3]

def boundary_from_energy(energy):
    e=np.asarray(energy,dtype=np.float64)
    if e.ndim!=3 or e.shape[0]!=63 or not np.isfinite(e).all() or e.min()<-1e-7:
        raise ValueError('Finite nonnegative 63-plane energy required')
    e=np.maximum(e,0)
    cumulative=np.concatenate([np.zeros_like(e[:1]),np.cumsum(e,axis=0)],axis=0)
    windows=(cumulative[5:]-cumulative[:-5])/5
    scale=windows.max(axis=0)
    signed=np.divide(windows[:-5]-windows[5:],scale[None],out=np.zeros_like(windows[:-5]),where=scale[None]>0)
    magnitude=abs(signed); index=magnitude.argmax(axis=0)
    peak=np.take_along_axis(magnitude,index[None],axis=0)[0]
    orientation=np.sign(np.take_along_axis(signed,index[None],axis=0)[0]).astype(np.int8)
    separated=abs(np.arange(54)[:,None,None]-index[None])>5
    competitor=np.max(np.where(separated,magnitude,0),axis=0)
    ratio=np.divide(competitor,peak,out=np.ones_like(peak),where=peak>0)
    accepted=(peak>=0.5)&(ratio<0.8)&(scale>0)
    candidate=index.astype(np.float64)+5.5
    depth=np.where(accepted,candidate,np.nan)
    return dict(depth=depth,candidate=candidate,accepted=accepted,contrast=peak,competitor_ratio=ratio,orientation=orientation,energy_scale=scale)

def infer(volume):
    energy=texture_energy(volume)
    return energy,boundary_from_energy(energy)
