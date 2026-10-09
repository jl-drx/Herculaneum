"""Prespecified known texture transitions and negative controls."""
import numpy as np

CASES=('single','sloped','double','noise','ramp')
def make_case(kind,seed):
    rng=np.random.Generator(np.random.PCG64(seed))
    z=np.arange(63)[:,None,None]; y,x=np.indices((40,40))
    texture=np.sin(2*np.pi*x/5)+0.8*np.sin(2*np.pi*y/7)+0.3*rng.normal(size=(40,40))
    truth=np.full((40,40),32.5)
    if kind=='single': amplitude=(z<32)
    elif kind=='sloped':
        boundary=25+x//4; amplitude=z<boundary;truth=boundary+0.5
    elif kind=='double': amplitude=(z>=20)&(z<42)
    elif kind=='noise': amplitude=np.zeros((63,1,1))
    elif kind=='ramp': amplitude=np.clip((50-z)/40,0,1)
    elif kind=='flat': return np.full((63,40,40),20000.),truth[4:-4,4:-4]
    else: raise ValueError(kind)
    volume=20000+1000*amplitude*texture+10*rng.normal(size=(63,40,40))
    return volume,truth[4:-4,4:-4]
