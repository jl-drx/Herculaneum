"""Independent fixed-coordinate sampler and whole-crop renderer equivalence checks."""
from pathlib import Path
import hashlib,json,itertools,sys
import numpy as np,zarr
from prepare import pin,save
P=Path(__file__).resolve().parent;W=P/'work';LAB=P.parents[1]
plan=json.loads((P/'GO.json').read_text())
assert json.loads((P/'RENDER_EXECUTION_V2.json').read_text())['status']=='PASS'
assert json.loads((P/'SOURCE_RECEIPTS.json').read_text())['all_chunks_present']
assert pin(W/'qualification-geometry.npz')==plan['geometry']
g=np.load(W/'qualification-geometry.npz');b,n=g['base'],g['normal'];assert g['valid'].all()
chunks={tuple(map(int,key.split('/')[1:])):np.frombuffer((W/'source-cache'/key).read_bytes(),np.uint8).reshape(128,128,128) for key in plan['source_chunk_keys']}

def fma(a,b,c):return (a.astype(np.float64)*b+c).astype(np.float32)
def at(zyx):
    flat=zyx.reshape(-1,3);idx=flat//128;out=np.empty(len(flat),np.float32)
    for key in np.unique(idx,axis=0):
        m=np.all(idx==key,axis=1);local=flat[m]%128
        out[m]=chunks[tuple(key)][local[:,0],local[:,1],local[:,2]]
    return out.reshape(zyx.shape[:-1])
def sample(depth):
    output=[]
    for off in np.arange(depth,dtype=np.float64)-(depth-1)/2:
        xyz=fma(n,off,b);base=np.floor(xyz).astype(int);frac=xyz-base
        v={d:at((base+np.array(d))[...,::-1]) for d in itertools.product([0,1],repeat=3)}
        # First x, then y, then z, with float32 FMA after each interpolation.
        ix={(y,z):fma(frac[...,0],v[(1,y,z)]-v[(0,y,z)],v[(0,y,z)]) for y,z in itertools.product([0,1],repeat=2)}
        iy={z:fma(frac[...,1],ix[(1,z)]-ix[(0,z)],ix[(0,z)]) for z in [0,1]}
        val=fma(frac[...,2],iy[1]-iy[0],iy[0]);output.append(np.clip(val,0,255).astype(np.uint8))
    return np.stack(output)
def stats(a,b):
    diff=np.abs(a.astype(np.int16)-b.astype(np.int16))
    return dict(shape=list(a.shape),n=int(a.size),exact=int((diff==0).sum()),exact_fraction=float((diff==0).mean()),mean_abs=float(diff.mean()),max_abs=int(diff.max()),within_one_fraction=float((diff<=1).mean()))
a31=np.asarray(zarr.open_group(str(W/'target-31.zarr'),mode='r')['0']);a66=np.asarray(zarr.open_group(str(W/'target-66.zarr'),mode='r')['0'])
assert a31.shape==(31,128,128) and a66.shape==(66,128,128)
assert a31.dtype==a66.dtype==np.uint8
old=np.load(LAB/'data/experiments/pherc0800-larger1536/surface.npy',mmap_mode='r')[:,704:832,704:832]
pred31=sample(31);pred66=sample(66)
for name,a in [('render31',a31),('render66',a66),('independent31',pred31),('independent66',pred66),('published31-crop',old)]:np.save(W/f'{name}.npy',a)
c31=stats(a31,old);c66=stats(a66,pred66)
gates=dict(published31=c31['max_abs']<=1 and c31['exact_fraction']>=.99,independent66=c66['max_abs']<=1 and c66['mean_abs']<=.1 and c66['exact_fraction']>=.95)
result=dict(status='PASS' if all(gates.values()) else 'FAIL',gates=gates,actual31_vs_published=c31,actual66_vs_independent=c66,independent31_vs_published=stats(pred31,old),wrong_order_control_actual31_vs_reversed_published=stats(a31,old[::-1]),known_source_chunks=len(chunks),unknown_source_chunks=0,geometry_all_valid=True,script=pin(__file__),limitations=['One interior128 patch does not qualify mesh boundaries, invalid normals or the whole field.','Recorded normal sign agrees with export convention; reading side is not established.','The31plane published volume and66plane volume have different half-voxel depth origins.'])
save(P/'RENDER_VERIFICATION.json',result);print(json.dumps(result))
sys.exit(0 if result['status']=='PASS' else 2)
