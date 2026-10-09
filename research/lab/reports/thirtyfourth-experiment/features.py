"""Exact integer feature numerators; no inference, fitted scaling or boundary padding."""
import numpy as np
from audit_common import require

def features(raw):
    require(raw.shape==(63,384,384) and raw.dtype==np.dtype('uint16'),'Raw contract')
    single=raw.reshape(63,128,3,128,3).sum(axis=(2,4),dtype=np.int64)
    triplet=single.reshape(21,3,128,128).sum(axis=1,dtype=np.int64)
    result={}
    for name,sums,n in [('single',single,9),('triplet',triplet,27)]:
        prefix=np.pad(sums.cumsum(1,dtype=np.int64).cumsum(2,dtype=np.int64),((0,0),(1,0),(1,0)))
        box=prefix[:,37:101,37:101]-prefix[:,28:92,37:101]-prefix[:,37:101,28:92]+prefix[:,28:92,28:92]
        center=sums[:,32:96,32:96]
        numerator=np.stack([center,81*center-box],axis=1)
        result[name]=dict(numerators=numerator,denominators=np.array([n,81*n],np.int64))
    return result

def encode(numerator,denominator,feature,polarity):
    v=numerator.astype(np.float64)/int(denominator)
    p=v/65535. if feature=='intensity' else .5+v/(2*65535.)
    require(feature in ('intensity','contrast') and polarity in (1,-1),'Fixed encoding')
    if polarity==-1:p=1-p
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Encoding range')
    return p

def distribution(values,labels,mask):
    result={}
    for cls,name in [(0,'clear'),(1,'ink')]:
        v=values[mask&(labels==cls)]
        result[name]=dict(count=int(v.size),mean=float(np.mean(v)) if v.size else None,
            q25=float(np.quantile(v,.25,method='linear')) if v.size else None,
            median=float(np.quantile(v,.5,method='linear')) if v.size else None,
            q75=float(np.quantile(v,.75,method='linear')) if v.size else None)
    return result
