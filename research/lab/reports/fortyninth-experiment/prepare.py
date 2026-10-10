"""Freeze qualification inputs and a bounded raw-CT whitelist, before rendering."""
from pathlib import Path
import sys,json,hashlib,shutil,itertools,datetime
import numpy as np,tifffile
P=Path(__file__).resolve().parent;W=P/'work';LAB=P.parents[1];R=P.parent
G=LAB/'data/experiments/pherc0800-larger-geometry/20251028222030'
sys.path.insert(0,str(R/'fortyeighth-experiment'))
from support import reconstruct,corner_envelope

def pin(p):
    p=Path(p)
    with p.open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=s)

def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def main():
    assert not (P/'GO.json').exists()
    prior=json.loads((R/'fortyeighth-experiment/final-verification.json').read_text())
    lookup={r['path']:r for r in prior['records']}
    snapshot=json.loads(Path('/workspace/Herculaneum/research/workspace/MANIFEST.json').read_text())
    for row in snapshot['files']:
        lookup.setdefault(str(LAB/row['path']),dict(path=str(LAB/row['path']),sha256=row['sha256'],size_bytes=row.get('size_bytes',row.get('bytes'))))
    inputs=[G/f'mesh.{a}.tif' for a in 'xyz']+[G/'mesh.meta.json',G/'source-render-export-metadata.json',G/'surface.level0.zarray.json',LAB/'data/experiments/pherc0800-larger1536/surface.npy',R/'fortyeighth-experiment/support.py',R/'fortysixth-experiment/RENDERER_IDENTITY.json']
    bindings=[]
    for f in inputs:
        rec=pin(f);assert lookup[str(f)]==rec,str(f);bindings.append(rec)
    mesh=np.stack([tifffile.imread(G/f'mesh.{a}.tif') for a in 'xyz'],-1)
    scale=json.loads((G/'mesh.meta.json').read_text())['scale']
    b,n,valid_base,valid_normal=reconstruct(mesh,scale,(2580,2580))
    sl=np.s_[1216:1344,1216:1344]
    assert valid_base[sl].all() and valid_normal[sl].all()
    np.savez_compressed(W/'qualification-geometry.npz',base=b[sl],normal=n[sl],valid=valid_normal[sl])
    # All possibly touched trilinear integer corners, including arithmetic margin.
    lo,hi=corner_envelope(b[sl],n[sl]);lo=lo.astype(int);hi=hi.astype(int)
    keys=set()
    for low,high in zip(lo.reshape(-1,3),hi.reshape(-1,3)):
        assert (low>=0).all() and (high<np.array([9867,9867,24298])).all()
        for xyz in itertools.product(*(range(low[k]//128,high[k]//128+1) for k in range(3))):keys.add('0/'+'/'.join(map(str,xyz[::-1])))
    assert len(keys)*128**3<=32*1024**2
    # Pin the qualified executable as well as its original verified distribution.
    old=R/'fortysixth-experiment/work'
    app=old/'VC3D-41701aa-2026-09-24-linux-x86_64.AppImage'
    assert pin(app)['sha256']=='1d88a1f396751561863b0ff50b3eed35a1282455bd03d0884ba2540c34004990'
    bindings.extend([pin(app),pin(old/'squashfs-root/usr/bin/vc_render_tifxyz'),pin(old/'squashfs-root/AppRun')])
    meshdir=W/'target.tifxyz';meshdir.mkdir()
    for name in ['x.tif','y.tif','z.tif','meta.json']:shutil.copyfile(G/f'mesh.{name}',meshdir/name)
    # Only old chunks belonging to this exact PHerc0800 source are eligible.
    retained=[]
    for d in ['pherc0800-native-ct-consistency','pherc0800-larger-native-consistency','pherc0800-fifth-native-check']:
        for f in (LAB/'data/experiments'/d).glob('ct-chunk-*.bin'):
            key='0/'+f.stem.removeprefix('ct-chunk-').replace('-','/')
            if key not in keys:continue
            rec=pin(f);assert lookup.get(str(f))==rec,str(f);retained.append(dict(key=key,**rec))
    plan=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=pin(P/'PROTOCOL.md'),script=pin(__file__),geometry=pin(W/'qualification-geometry.npz'),inputs=bindings,
        source_url='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr',
        source_chunk_keys=sorted(keys),source_chunk_bytes=128**3,source_payload_upper_bound_bytes=len(keys)*128**3,source_cap_bytes=32*1024**2,retained_chunks=retained,
        crop_xywh=[1216,1216,128,128],full_shape_yx=[2580,2580],all_crop_normal_and_bilinear_valid=True,depths=[31,66],flip_normals=True,inference_authorized_only_after_render_gate=True)
    save(P/'GO.json',plan)
    print(json.dumps({k:plan[k] for k in ['created_at_utc','source_payload_upper_bound_bytes','source_chunk_keys','all_crop_normal_and_bilinear_valid']}))
if __name__=='__main__':main()
