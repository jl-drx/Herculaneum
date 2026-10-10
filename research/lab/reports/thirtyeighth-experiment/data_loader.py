import numpy as np
from common import *
def batches(p,kind,torch):
    arrays=[np.load(verify(p['batch_arrays'][kind][name]),mmap_mode='r',allow_pickle=False) for name in ['inputs','mask','target']]
    require(arrays[0].shape==(32,2,1,17,128,128) and arrays[0].dtype==np.float32,'Fixed normalized batch inputs')
    require(arrays[1].shape==arrays[2].shape==(32,2,1,128,128) and arrays[1].dtype==np.bool_ and arrays[2].dtype==np.float32,'Fixed supervision shapes')
    return [tuple(torch.from_numpy(a[i].copy()) for a in arrays) for i in range(32)]
