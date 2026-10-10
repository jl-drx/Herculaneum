"""Independent uint16 factor-three pooling v2; declared native TIFF byte order."""
import numpy as np


def require(condition,message):
    if not condition:raise ValueError(message)


def pool51(native):
    """Preserve source ordering; explicit27 residue-class additions per group."""
    require(isinstance(native,np.ndarray) and native.dtype.kind=='u' and native.dtype.itemsize==2 and
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


def decode_declared_values(body,shape,contract):
    """A future caller must bind contract to its validated TIFF/header receipt."""
    require(contract.get('byte_order') in ('little','big') and
            contract.get('dtype')==('<u2' if contract['byte_order']=='little' else '>u2'),
            'Explicit validated TIFF byte order/dtype required')
    require(isinstance(body,bytes) and len(shape)==3 and all(type(n) is int and n>0 for n in shape),
            'Exact byte buffer/shape required')
    expected=2*int(np.prod(shape,dtype=object))
    require(len(body)==expected,'Exact uint16 raw crop extent required')
    return np.frombuffer(body,dtype=contract['dtype']).reshape(shape)


def fixtures():
    rng=np.random.default_rng(162)
    z,y,x=np.indices((51,6,9))
    cases=[np.zeros((51,6,9),np.uint16),np.full((51,6,9),65535,np.uint16),
           np.ones((51,6,9),np.uint16),np.full((51,6,9),0x0102,np.uint16),
           (z*1000+y*100+x*3).astype(np.uint16),rng.integers(0,65536,(51,6,9),dtype=np.uint16)]
    exact=0
    for logical in cases:
        expected=scalar_reference(logical)
        for order,dtype in [('little','<u2'),('big','>u2')]:
            body=logical.astype(dtype).tobytes(order='C')
            native=decode_declared_values(body,logical.shape,dict(byte_order=order,dtype=dtype))
            actual=pool51(native)
            require(np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),
                    'Declared-endian values/pooling bits differ from scalar oracle')
            require(native.tobytes(order='C')==body,'Pooling modified raw encoded input')
            exact+=1
    logical=cases[3];body=logical.astype('>u2').tobytes()
    right=pool51(decode_declared_values(body,logical.shape,dict(byte_order='big',dtype='>u2')))
    wrong=pool51(decode_declared_values(body,logical.shape,dict(byte_order='little',dtype='<u2')))
    require(np.all(right==np.float32(258/257)) and np.all(wrong==np.float32(513/257)) and
            not np.array_equal(right.view(np.uint32),wrong.view(np.uint32)),
            'Wrong-endian byte interpretation must be discriminated')
    invalid=[np.zeros((51,6,9),np.uint8),np.zeros((50,6,9),'>u2'),
             np.zeros((51,5,9),'>u2'),np.zeros((51,6,8),'<u2'),np.zeros((51,0,9),'>u2')]
    for native in invalid:
        try:pool51(native)
        except ValueError:pass
        else:raise ValueError('Invalid endian native pooling input accepted')
    for contract,payload in [(dict(byte_order='big',dtype='<u2'),body),
        (dict(byte_order='unknown',dtype='>u2'),body),(dict(byte_order='big',dtype='>u2'),body[:-1])]:
        try:decode_declared_values(payload,logical.shape,contract)
        except ValueError:pass
        else:raise ValueError('Ambiguous/truncated endian raw input accepted')
    return dict(status='PASS',scalar_oracle_endian_cases=exact,wrong_endian_discriminator_cases=1,
                rejected_input_cases=8,actual_raw_CT_objects_decoded=0,network_calls=0,model_calls=0)
