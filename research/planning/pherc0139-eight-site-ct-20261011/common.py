"""Local, pinned evidence helpers. No network on import."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
G=module('new_site_geometry',R/'inputs/geometry/geometry.py')
P=module('retained_inverse',R/'sources/pullback_source.py')
S=module('retained_ncc',R/'sources/score_source.py')
FIT=[(24,24),(104,24),(24,104),(104,104)]
CHECK=[(64,24),(24,64),(104,64),(64,104)]
S.FIT_XY=FIT;S.CHECK_XY=CHECK
PLAN=R/'inputs/geometry/NEXT_CT_PILOT_PLAN.json'
def check_files(rows):
    for row in rows:
        p=R/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
def check_inputs():check_files(read(R/'INPUTS.json')['files'])
def check_lock():check_files(read(R/'LOCK.json')['files']);check_inputs()
def receipt(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def patches():
    for x,y in FIT+CHECK:
        # In the core+8 halo, include +/-6 possible translations.
        x0=x+8-12;y0=y+8-12
        yield (slice(6,14),slice(y0,y0+24),slice(x0,x0+24)),(slice(None),slice(y0-6,y0+30),slice(x0-6,x0+30))
def pooled_field(fine,rect,matrix):
    x0,y0,x1,y1=rect;us=np.arange(x0,x1);vs=np.arange(y0,y1)
    b=np.zeros((len(vs),len(us),3));n=np.zeros_like(b);ok=np.ones(b.shape[:2],bool)
    for dy in range(4):
        for dx in range(4):
            pp,nn,vv=G.render(fine['mesh'],fine['cache'],4*us+dx,4*vs+dy,fine['canvas'],0)
            b+=pp/16;n+=nn/16;ok &= vv
    return b@matrix[:3,:3].T+matrix[:3,3],n@matrix[:3,:3].T,ok
def fine_indices(q):
    """Every group2 XY interpolation neighbor; depth smoothing changes Z only."""
    uv=np.floor(q[...,:2].reshape(-1,2)).astype(np.int64)
    return {tuple(v) for dx,dy in [(0,0),(1,0),(0,1),(1,1)] for v in np.unique(uv+[dx,dy],axis=0)}
