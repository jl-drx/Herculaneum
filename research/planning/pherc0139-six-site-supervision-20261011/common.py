"""Pinned offline helpers; no target reads or network on import."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
R=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text())
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def receipt(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def verify(rows):
    for x in rows:
        p=R/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
G=module('frozen_geometry',R/'inputs/geometry/geometry.py')
T=module('frozen_transfer',R/'sources/previous_transfer.py')
CHECK=[(64,24),(24,64),(104,64),(64,104)]
PLANES=list(range(10,18))
OPS=['mean16_bilinear','point_bilinear']
def query(q,support,center,j,shift):
    dz,dy,dx=shift;x,y=center;iz=j+dz-4;x0=x+8-12+dx;y0=y+8-12+dy
    assert 0<=iz<20 and 0<=x0<x0+24<=144 and 0<=y0<y0+24<=144
    uv=np.asarray(q[iz,y0:y0+24,x0:x0+24,:2]);v=support[iz,y0:y0+24,x0:x0+24]
    assert v.all() and np.isfinite(uv).all()
    return uv
def queries(q,support,patch,j,fit):
    alt=np.stack([query(q,support,patch['centre_core_xy'],j,c['shift_zyx']) for c in patch['alternatives']])
    mean=alt.mean(0);kernel=(alt-mean).mean((1,2))
    return dict(alternatives=alt,fit=query(q,support,patch['centre_core_xy'],j,fit),
                zero=query(q,support,patch['centre_core_xy'],j,[0,0,0]),
                mean=mean,stationary=mean[None]+kernel[:,None,None,:])
