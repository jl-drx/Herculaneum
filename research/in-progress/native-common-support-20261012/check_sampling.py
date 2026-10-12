"""No inference: source-code grid parity and full-canvas interpolation controls."""
import argparse, ast, hashlib, json
from pathlib import Path
import cv2, numpy as np
from sampling import context, starts, spread


def extract(path,name):
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env[name]


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reports=a.repo/'research/lab/reports'
    vf=reports/'fortysixth-experiment/runtime-source/vesuvius/ink_detection/inference/infer.py'
    hf=reports/'fortyfifth-experiment/source-evidence/hecate-hecate.py'
    villa=extract(vf,'_sliding_positions_1d');hecate=extract(hf,'axis_starts')
    checks=[]
    for n in [128,129,192,199,3760,4220,4640,4760,4900,5700,7040]:
        for patch,stride,original in [(128,64,villa),(64,32,hecate)]:
            assert starts(n,patch,stride)==original(n,patch,stride)
    checks.append('22 grid lengths/patch configurations equal pinned original functions')
    rows=[dict(x=x,y=y) for y in [0,128,256] for x in [0,128,256]]
    expected=[(128,128),(0,0),(256,0),(0,256),(256,256),(128,0)]
    assert [(r['x'],r['y']) for r in spread(rows)]==expected
    assert spread(list(reversed(rows)))==spread(rows)
    try:spread(rows[:5])
    except ValueError:pass
    else:raise AssertionError('Deficient segment admitted')
    checks.append('Hand-derived spread/tie result, input-order invariance and deficient cohort rejection')
    interpolation=[]
    for shape,spacing in [((4220,4760),9.366),((7040,5700),8.64)]:
        h,w=shape;hh,ww=round(h*spacing/9.6),round(w*spacing/9.6)
        # Structured high frequencies exercise interpolation phases; no real CT.
        source=((np.arange(h,dtype=np.int32)[:,None]*17+np.arange(w,dtype=np.int32)[None,:]*31)%251).astype(np.float32)
        method=cv2.INTER_AREA if spacing/9.6<.95 else cv2.INTER_LINEAR
        full=cv2.resize(source,(ww,hh),interpolation=method)
        for x,y in [(128,128),(1280,1536),((w//128-2)*128,(h//128-2)*128)]:
            c=context(shape,spacing,x,y);assert c is not None
            sparse=np.zeros_like(source)
            for z,cy,cx in c['native_chunks_zyx']:
                ys=slice(cy*128,min(h,(cy+1)*128));xs=slice(cx*128,min(w,(cx+1)*128))
                sparse[ys,xs]=source[ys,xs]
            partial=cv2.resize(sparse,(ww,hh),interpolation=method)
            grid_known=np.zeros((hh,ww),bool)
            for px,py,pw,ph in c['hecate_tiles_xywh']:
                sl=np.s_[py:py+ph,px:px+pw]
                assert np.array_equal(partial[sl],full[sl]),'Unacquired source entered an input tile'
                grid_known[sl]=True
            # Output resize also requires neighbouring Hecate output pixels.
            cropped=np.where(grid_known,full,0)
            back=cv2.resize(full,(w,h),interpolation=cv2.INTER_LINEAR)[y:y+128,x:x+128]
            partback=cv2.resize(cropped,(w,h),interpolation=cv2.INTER_LINEAR)[y:y+128,x:x+128]
            assert np.array_equal(back,partback),'Missing output interpolation neighbour'
            interpolation.append(dict(native_shape=list(shape),spacing=spacing,core_xy=[x,y],
                                      interpolation=c['hecate_interpolation'],exact_input_and_output_parity=True))
        assert context(shape,spacing,0,0) is None
        del source,full,sparse,partial,grid_known,cropped
    report={'status':'PASS','checks':checks,'interpolation':interpolation,'cv2':cv2.__version__,
            'source_sha256':{str(p.relative_to(a.repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [vf,hf]},
            'real_ct':False,'model_forwards':0,'claim':'Footprint and axis scheduling controls; not model output parity'}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
