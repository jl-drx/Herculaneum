"""Offline admission invariants and numerical fixtures; no target pixels."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import select_regions as S

ROOT=Path(__file__).resolve().parent

def main():
    checks=[]
    def need(value,name):
        if not value:raise ValueError(name)
        checks.append(name)
    for r in S.read(ROOT/'selection/COMPUTE_BINDING.json')['files']:
        p=ROOT/r['path'];need(p.stat().st_size==r['bytes'] and S.sha(p)==r['sha256'],'bound input '+r['path'])
    result=S.read(ROOT/'selection/RESULTS.json');rows=result['selected']
    need(len(rows)==256 and len({r['core_id'] for r in rows})==256,'fixed unique256 cohort')
    refs=[r for r in rows if r['role']!='training_candidate'];train=[r for r in rows if r['role']=='training_candidate']
    need(len(refs)==64 and len(train)==192,'reference/training quotas')
    reserved=S.read(ROOT/'selection/RESERVED_CONTEXTS.json')
    need(len(reserved)==332,'265 existing +3 old sites +64 prospective references')
    need(not any(S.hits(r['bbox_conservative_xyz'],reserved) for r in train),'training clears every native reserved box')
    need(not any(S.A.intersects(a['bbox_conservative_xyz'],b['bbox_conservative_xyz']) for i,a in enumerate(refs) for b in refs[i+1:]),'all prospective references pairwise disjoint native boxes')
    need(not any(S.hits(r['bbox_conservative_xyz'],reserved[:268]) for r in refs),'new references clear declared old contexts')
    shape_xyz=np.array(S.read(ROOT/'CATALOG_EXTRACT.json')['volumes']['20250728140407']['properties']['shape'][::-1])
    need(all((np.asarray(r['bbox_conservative_xyz'][0])>=0).all() and
             (np.asarray(r['bbox_conservative_xyz'][1])<shape_xyz).all() for r in rows),'selected native ray/trilinear boxes within catalog CT extent')
    need(all(r['geometry_valid_pixels']==256*256 for r in rows),'entire native contexts pass mesh stencils')
    need(not result['paired_training_admitted'] and result['fine_teacher_context_exclusions']=='PENDING','native geometry cannot accidentally grant paired admission')
    plan=S.read(ROOT/'selection/NEXT_NATIVE_CT_PLAN.json');req=plan['requests']
    need(len(req)==1024 and len({r['url'] for r in req})==1024,'no duplicate future chunk requests')
    need(plan['max_payload_bytes']==469762048==sum(r['expected_bytes'] for r in req),'448MiB finite native ceiling')
    for w in result['counts']:
        arr=S.read(ROOT/'metadata'/f'{w}-native-array.json');sep=arr.get('dimension_separator','.')
        expected=set()
        for row in rows:
            if row['winding']==w:
                x,y,_,_=row['context_xywh'];expected.update((cy,cx) for cy in range(y//128,(y+255)//128+1) for cx in range(x//128,(x+255)//128+1))
        actual={tuple(map(int,r['key'].split(sep)))[1:] for r in req if r['winding']==w}
        need(actual==expected,w+' exact future context coverage')
    # A sloped plane has known canvas-centred coordinates and analytical normal;
    # also exercise invalid stencil/edge rejection and all28-depth enclosures.
    yy,xx=np.mgrid[:40,:40];mesh=np.stack([xx*20,yy*20,100+xx*2+yy*3],axis=-1).astype(np.float32)
    cache=S.G.normals(mesh);canvas=[800,800]
    b=S.fast_box(mesh,cache,128,128,canvas)
    exact=S.exact_box(mesh,cache,128,128,canvas)
    points,normals,valid=S.G.render(mesh,cache,np.arange(128,384),np.arange(128,384),canvas)
    need(valid.all() and np.max(np.abs(points[...,0]-(np.arange(128,384)[None,:]+.5)))<.0001,'analytic plane canvas convention')
    expected_normal=np.array([.1,.15,-1]);expected_normal/=np.linalg.norm(expected_normal)
    need(np.max(np.abs(normals-expected_normal))<1e-6,'analytic sloped-plane flipped normal')
    for depth in np.arange(28)-13.5:
        p=points+depth*normals
        need(np.all(p-1>=np.asarray(b[0])) and np.all(p+1<=np.asarray(b[1])),f'plane depth{depth}: fast hull encloses trilinear support')
    broken=mesh.copy();broken[10,10]=np.nan
    need(S.fast_box(broken,S.G.normals(broken),128,128,canvas) is None,'invalid interior geometry rejected')
    need(S.fast_box(mesh,cache,0,0,canvas) is None,'incomplete edge normal stencil rejected')
    need(S.A.intersects([[0,0,0],[1,1,1]],[[1,1,1],[2,2,2]]),'closed-touching context boxes count as overlap')
    report=dict(status='PASS',checks=checks,count=len(checks),scope='Offline native geometry/request invariants and synthetic plane; not CT registration or ink validity',
                code_sha256=S.sha(Path(__file__)),results_sha256=S.sha(ROOT/'selection/RESULTS.json'))
    S.save(ROOT/'VERIFICATION.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
