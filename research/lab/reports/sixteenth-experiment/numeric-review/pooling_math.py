"""Independent uint16 factor-three pooling, accepting arrays without file I/O."""
import numpy as np


def require(condition,message):
    if not condition:raise ValueError(message)


def pool51(native):
    """Preserve source ordering; explicit27 residue-class additions per group."""
    require(isinstance(native,np.ndarray) and native.dtype==np.dtype(np.uint16) and
            native.ndim==3 and native.shape[0]==51 and native.shape[1]>0 and native.shape[2]>0 and
            native.shape[1]%3==native.shape[2]%3==0,'Aligned51-plane uint16 native crop required')
    output=np.empty((17,native.shape[1]//3,native.shape[2]//3),np.float32)
    for group in range(17):
        total=np.zeros(output.shape[1:],np.uint32)
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):total+=native[3*group+dz,dy::3,dx::3]
        output[group]=(total.astype(np.float64)/6939).astype(np.float32)
    return output


def scalar_reference(native):
    """Tiny synthetic oracle uses Python integer scalar iteration, not NumPy sums."""
    output=np.empty((17,native.shape[1]//3,native.shape[2]//3),np.float32)
    for z in range(output.shape[0]):
        for y in range(output.shape[1]):
            for x in range(output.shape[2]):
                total=sum(int(native[3*z+dz,3*y+dy,3*x+dx])
                    for dz in range(3) for dy in range(3) for dx in range(3))
                output[z,y,x]=np.float32(total/6939)
    return output


def fixtures():
    rng=np.random.default_rng(160)
    cases=[np.zeros((51,6,9),np.uint16),np.full((51,6,9),65535,np.uint16),
           np.ones((51,6,9),np.uint16),rng.integers(0,65536,(51,6,9),dtype=np.uint16)]
    z,y,x=np.indices((51,6,9));cases.append((z*1000+y*100+x*3).astype(np.uint16))
    sparse=np.zeros((51,6,9),np.uint16);sparse[23,4,7]=65535;cases.append(sparse)
    for native in cases:
        actual,expected=pool51(native),scalar_reference(native)
        require(np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),
                'Exact pooling bits differ from scalar integer oracle')
    require(np.array_equal(pool51(cases[0]).view(np.uint32),np.zeros((17,2,3),np.uint32)),
            'Pooled zero must be positive-zero bits')
    require(np.all(pool51(cases[1])==np.float32(255)),'Maximumuint16 scaling/overflow fixture differs')
    require(np.all(pool51(cases[2])==np.float32(1/257)),'Fractional intensity was quantized')
    invalid=[np.zeros((51,6,9),np.uint8),np.zeros((50,6,9),np.uint16),
             np.zeros((51,5,9),np.uint16),np.zeros((51,6,8),np.uint16),
             np.zeros((51,0,9),np.uint16)]
    for native in invalid:
        try:pool51(native)
        except ValueError:pass
        else:raise ValueError('Invalid native pooling input accepted')
    return dict(status='PASS',scalar_oracle_cases=6,special_bit_and_scaling_cases=3,
                rejected_input_cases=5,actual_retained_CT_arrays_read=0,network_calls=0,model_calls=0)
