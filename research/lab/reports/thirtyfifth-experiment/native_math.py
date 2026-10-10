"""Explicit historical compatibility arithmetic; correct interpolation retained separately."""
import numpy as np
from packet_common import require
CORNERS=[(x,y,z) for z in (0,1) for y in (0,1) for x in (0,1)]
def evaluate(corners,fractions):
    require(corners.shape==(len(fractions),8) and fractions.shape[1:]==(3,),'Eight corners per XYZ fraction')
    f=fractions;standard=np.zeros(len(f),np.float64)
    for index,bits in enumerate(CORNERS):
        weight=np.ones(len(f),np.float64)
        for axis,bit in enumerate(bits):weight*=f[:,axis] if bit else 1-f[:,axis]
        standard+=corners[:,index]*weight
    historical=standard+f[:,0]*f[:,1]*(1-f[:,2])*(corners[:,1]-corners[:,3])
    require(np.isfinite(historical).all() and ((historical>=0)&(historical<=65535)).all(),'Historical intensity bounds')
    return dict(standard=standard,historical=historical,rounded_standard=np.rint(standard).astype(np.uint16),rounded_historical=np.rint(historical).astype(np.uint16))
