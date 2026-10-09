"""Fixed raster translations and conservative label-boundary bands."""
import sys
import numpy as np
from scipy.ndimage import distance_transform_edt
from diag_common import *
sys.path.insert(0,str(REPORTS/'twentyninth-experiment'))
from reference_metrics import rank_metrics
SHIFTS=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)]
BANDS=['near_le2','middle_gt2_le4','interior_gt4']
PREDICTORS=['baseline','candidate','optical_darkness']

def shifted_labels(labels):
    require(labels.shape==(128,128) and labels.dtype.kind in 'ui' and np.isin(labels,[0,1,2]).all(),'128x128 labels0/1/2required')
    return np.stack([labels[32-dy:96-dy,32-dx:96-dx] for dy,dx in SHIFTS])

def support(labels):
    shifted=shifted_labels(labels);common=(shifted<2).all(axis=0)
    stable=common&(shifted==shifted[4]).all(axis=0)
    return shifted,common,stable

def boundary_bands(labels):
    require(labels.shape==(128,128),'Label canvas required')
    distances=np.zeros(labels.shape,np.float64)
    for cls in (0,1):
        inside=labels==cls
        d=distance_transform_edt(np.pad(inside,1,constant_values=False))[1:-1,1:-1]
        distances[inside]=d[inside]
    core=labels[32:96,32:96];dist=distances[32:96,32:96].copy();known=core<2
    bands={'near_le2':known&(dist<=2),'middle_gt2_le4':known&(dist>2)&(dist<=4),'interior_gt4':known&(dist>4)}
    require(np.array_equal(sum(v.astype(np.uint8) for v in bands.values()),known.astype(np.uint8)),'Bands must partition known support')
    return dist,bands

def measure(scores,labels,selected):
    require(selected.dtype==bool and selected.shape==labels.shape,'Explicit selected mask')
    valid=np.isfinite(scores)&(scores>=0)&(scores<=1)
    require(valid[selected&(labels<2)].all(),'Selected prediction missing')
    m=rank_metrics(scores,labels,selected=selected,prediction_valid=valid)
    n=m['counts']['scored_positive']+m['counts']['scored_negative']
    m['prevalence']=m['counts']['scored_positive']/n if n else None
    return m
