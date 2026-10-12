"""Prepare only contributing model patches, preserving global resize coordinates."""
import argparse, json, math, shutil
from pathlib import Path
import cv2, numpy as np
import intake


def hecate_planes(volume, spacing):
    """Plane-wise equivalent of pinned preparation; avoids its multi-GB stack."""
    d,h,w=volume.shape;centre=d//2
    pos=centre+(np.arange(16)-7.5)*(9.6/spacing)
    assert pos.min()>=0 and pos.max()<=d-1
    lo=np.floor(pos).astype(int);frac=(pos-lo).astype(np.float32);hi=np.minimum(lo+1,d-1)
    hh,ww=round(h*spacing/9.6),round(w*spacing/9.6)
    method=cv2.INTER_AREA if spacing/9.6<.95 else cv2.INTER_LINEAR
    for i in range(16):
        plane=volume[lo[i]].astype(np.float32)*(1-frac[i])+volume[hi[i]].astype(np.float32)*frac[i]
        yield np.clip(cv2.resize(plane,(ww,hh),interpolation=method),0,255).astype(np.uint8)


def synthetic_check():
    rows=[]
    for depth,spacing in [(28,9.366),(31,8.64)]:
        z,y,x=np.indices((depth,385,513),dtype=np.int32)
        v=((z*11+y*19+x*23+x*y%17)%256).astype(np.uint8)
        d,h,w=v.shape;pos=d//2+(np.arange(16)-7.5)*(9.6/spacing)
        lo=np.floor(pos).astype(int);frac=(pos-lo).astype(np.float32);hi=np.minimum(lo+1,d-1)
        # Literal array expression from pinned hecate_prep_original.py.
        zs=(v[lo].astype(np.float32)*(1-frac)[:,None,None]+
            v[hi].astype(np.float32)*frac[:,None,None])
        hh,ww=round(h*spacing/9.6),round(w*spacing/9.6)
        method=cv2.INTER_AREA if spacing/9.6<.95 else cv2.INTER_LINEAR
        reference=np.stack([np.clip(cv2.resize(zs[i],(ww,hh),interpolation=method),0,255).astype(np.uint8) for i in range(16)])
        actual=np.stack(list(hecate_planes(v,spacing)))
        assert np.array_equal(reference,actual)
        rows.append(dict(depth=depth,spacing=spacing,exact_byte_parity=True,sha256=intake.ash(actual)))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();root=a.work
    intake.verify_freeze(root)
    assert intake.load(root/'INPUTS_COMPLETE.json')['status']=='PASS'
    output=root/'model-patches';output.mkdir(exist_ok=True)
    checks=synthetic_check();selection=intake.load(root/'SELECTION.json');cases=[]
    for c in selection['cases']:
        native,known=intake.source_canvas(root,c);d,h,w=native.shape
        folder=output/c['id'];folder.mkdir(exist_ok=True)
        hybrid=sorted({tuple(t) for r in c['cores'] for t in r['hybrid_tiles_xywh']},key=lambda t:(t[1],t[0]))
        hecate=sorted({tuple(t) for r in c['cores'] for t in r['hecate_tiles_xywh']},key=lambda t:(t[1],t[0]))
        records=[]
        for x,y,pw,ph in hybrid:
            assert known[y:y+ph,x:x+pw].all()
            patch=native[d//2-8:d//2+9,y:y+ph,x:x+pw].copy();assert patch.shape==(17,128,128)
            path=folder/f'hybrid-y{y:05}-x{x:05}.npy';np.save(path,patch,allow_pickle=False)
            records.append(dict(family='hybrid',origin_yx=[y,x],path=str(path.relative_to(root)),**intake.pin(path),
                                decoded_sha256=intake.ash(patch),raw_nonempty=bool(patch.any())))
        spacing=c['native']['spacing_zyx_um'][0];hh,ww=c['cores'][0]['hecate_grid_yx']
        prepared=np.empty((len(hecate),16,64,64),np.uint8)
        for iz,plane in enumerate(hecate_planes(native,spacing)):
            assert plane.shape==(hh,ww)
            for n,(x,y,pw,ph) in enumerate(hecate):prepared[n,iz]=plane[y:y+ph,x:x+pw]
        method=cv2.INTER_AREA if spacing/9.6<.95 else cv2.INTER_LINEAR
        resized_known=cv2.resize(known.astype(np.float32),(ww,hh),interpolation=method)
        for n,(x,y,pw,ph) in enumerate(hecate):
            # Conservative bounding-box assertion supplements float known-mask check.
            assert (resized_known[y:y+ph,x:x+pw]>=1-1e-6).all()
            path=folder/f'hecate-y{y:05}-x{x:05}.npy';np.save(path,prepared[n],allow_pickle=False)
            records.append(dict(family='hecate',origin_yx=[y,x],path=str(path.relative_to(root)),**intake.pin(path),
                                decoded_sha256=intake.ash(prepared[n]),raw_nonempty=bool(prepared[n].any())))
        counts={family:dict(scheduled=sum(r['family']==family for r in records),
                           nonempty=sum(r['family']==family and r['raw_nonempty'] for r in records)) for family in ['hybrid','hecate']}
        cases.append(dict(id=c['id'],centre_native_layer=d//2,hybrid_layers=list(range(d//2-8,d//2+9)),
                          hecate_z_positions=(d//2+(np.arange(16)-7.5)*9.6/spacing).tolist(),
                          hecate_grid_yx=[hh,ww],hecate_interpolation=int(method),counts=counts,patches=records))
        print(json.dumps({'case':c['id'],'counts':counts}),flush=True)
        del native,known,prepared,resized_known
    recipe=Path(__file__).parent/'hecate_prep_original.py'
    result=dict(status='PASS',utc=intake.stamp(),selection_sha256=intake.sha(root/'SELECTION.json'),
        input_receipt_sha256=intake.sha(root/'INPUTS_COMPLETE.json'),implementation=intake.pin(Path(__file__)),
        pinned_preparation=intake.pin(recipe),synthetic_vectorized_vs_planewise=checks,cv2_version=cv2.__version__,
        cases=cases,model_forwards=0,unknown_array_values_in_contributing_patches=False,
        physical_registration_verified=False)
    intake.write(root/'PATCHES_COMPLETE.json',result)
    shutil.copyfile(Path(__file__),root/'implementation'/Path(__file__).name)


if __name__=='__main__':main()
