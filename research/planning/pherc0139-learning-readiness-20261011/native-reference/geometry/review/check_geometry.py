#!/usr/bin/env python3
"""Offline independent geometry/fold audit; no production/helper imports."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import tifffile
F=np.float32

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def intersect(a,b):
    # Independent separating-axis expression, with equality retained as contact.
    return not any(float(a[1][i])<float(b[0][i]) or float(b[1][i])<float(a[0][i]) for i in range(3))

def grid(values,count,canvas,scale):
    starts=(values//128)*128;local=values-starts
    c=F(count/(2*float(scale)));origin=F(-.5*F(canvas-1));ul=F(F(F(origin+starts.astype(np.float32))+c)*scale);o=F(ul.astype(float)-4*float(scale))
    q=F(o+F((local+4).astype(np.float32)*scale));q_other=F(o.astype(float)+(local+4)*float(scale))
    safe=(np.floor(q)==np.floor(q_other))&(np.floor(q.astype(float)+.5)==np.floor(q_other.astype(float)+.5))
    return q,safe

def field(mesh,xs,ys,canvas,scale):
    x,sx=grid(xs,mesh.shape[1],canvas[0],scale);y,sy=grid(ys,mesh.shape[0],canvas[1],scale)
    ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);nx=np.floor(x.astype(float)+.5).astype(int);ny=np.floor(y.astype(float)+.5).astype(int)
    assert ix.min()>=0 and ix.max()+1<mesh.shape[1] and iy.min()>=0 and iy.max()+1<mesh.shape[0]
    assert nx.min()>0 and nx.max()<mesh.shape[1]-1 and ny.min()>0 and ny.max()<mesh.shape[0]-1
    c=[mesh[iy[:,None]+dy,ix[None,:]+dx] for dy,dx in [(0,0),(0,1),(1,0),(1,1)]]
    wx=F(x-ix.astype(np.float32))[None,:,None];wy=F(y-iy.astype(np.float32))[:,None,None]
    top=F(F(c[0]*F(1-wx))+F(c[1]*wx));bottom=F(F(c[2]*F(1-wx))+F(c[3]*wx));base=F(F(top*F(1-wy))+F(bottom*wy)).astype(float)
    valid=lambda a:np.isfinite(a).all(-1)&(a[...,0]!=-1)&(a[...,2]>0)
    ok=np.logical_and.reduce([valid(a) for a in c])&sy[:,None]&sx[None,:]
    # Independent normal support and double-precision normals for ray enclosure.
    center=mesh[ny[:,None],nx[None,:]];left=mesh[ny[:,None],nx[None,:]-1];right=mesh[ny[:,None],nx[None,:]+1];above=mesh[ny[:,None]-1,nx[None,:]];below=mesh[ny[:,None]+1,nx[None,:]]
    ok&=np.logical_and.reduce([valid(a) for a in [center,left,right,above,below]])
    normal=-np.cross(right.astype(float)-left,below.astype(float)-above);norm=np.linalg.norm(normal,axis=-1);ok&=np.isfinite(norm)&(norm>0);normal/=np.where(norm>0,norm,1)[...,None]
    # Full-precision canvas alternative, evaluated with a difference form.
    qx=(xs.astype(float)-.5*(canvas[0]-1)+mesh.shape[1]/(2*float(scale)))*float(scale);qy=(ys.astype(float)-.5*(canvas[1]-1)+mesh.shape[0]/(2*float(scale)))*float(scale)
    ax=np.floor(qx).astype(int);ay=np.floor(qy).astype(int);xx=(qx-ax)[None,:,None];yy=(qy-ay)[:,None,None]
    ac=[mesh[ay[:,None]+dy,ax[None,:]+dx].astype(float) for dy,dx in [(0,0),(0,1),(1,0),(1,1)]];ok&=np.logical_and.reduce([valid(a) for a in ac]);top=ac[0]+xx*(ac[1]-ac[0]);bottom=ac[2]+xx*(ac[3]-ac[2]);alt=top+yy*(bottom-top)
    return base,alt,normal,ok

def fixtures():
    assert intersect([[0,0,0],[1,1,1]],[[1,1,1],[2,2,2]])
    assert not intersect([[0,0,0],[1,1,1]],[[1.0000001,0,0],[2,2,2]])
    assert not intersect([[0,0,0],[1,1,1]],[[0,0,1.0000001],[2,2,2]])
    # Complete rays including near-tolerance unit length and ceil/floor neighbors.
    normal=np.array([[1.00001,0,0],[-1.00001,0,0],[0,1.00001,0],[0,0,-1.00001]])
    base=np.array([.2,1000.8,-.9]);rays=base+np.arange(-13.5,14,1)[:,None,None]*normal[None,:,:]
    assert np.all(np.floor(rays)>=base-14.51) and np.all(np.ceil(rays)<=base+14.51)
    return ['closed_boundary_contact','x_axis_separation','z_axis_separation','all28_rays_and_neighbors_at_norm_tolerance']

def main(root,output):
    checks=fixtures();data=json.loads((root/'RESULTS.json').read_text());plan=json.loads((root/'SOURCE_PLAN.json').read_text());assert sha(root/'PROTOCOL.md')==plan['protocol_sha256']==data['protocol_sha256']
    for p in json.loads((root/'COMPUTE_FREEZE.json').read_text())['files']:assert sha(root/p['relative'])==p['sha256'],p['relative']
    for p in json.loads((root/'COORDINATE_LEDGER.json').read_text()):assert sha(root/'responses'/p['name'])==p['sha256']
    rows=data['cores'];assert len(rows)==265;ids={r['core_id']:r for r in rows};assert len(ids)==265
    sources=[];replays=[];max_base_diff=0.;min_ray_margin=math.inf
    for site in plan['sites']:
        w=site['winding'];attrs=site['native_attrs'];rp=site['render']['creation_info']['provenance']['parameters'];meshinfo=site['mesh'];mp=meshinfo['origins'][0]['path'];dp=site['render']['creation_info']['derived_from']['source_path'];assert dp.endswith(mp)
        assert meshinfo['parameters']['target_volume']=='20250728140407';assert attrs['source_group']==0 and attrs['num_slices']==28 and attrs['slice_step']==1
        for k,v in {'commit':'530584554c031c982ba3aa243e1ea39c4bef9818','scale':'1','group-idx':'0','num-slices':'28','rotate':'0','flip':'-1','flip-normals':'true','crop-x':'0','crop-y':'0','crop-width':'0','crop-height':'0','affine-matrix':'','affine-transform':'','affines':'','scale-segmentation':''}.items():assert rp[k]==v,(w,k)
        met=json.loads((root/'responses'/f'{w}-meta.json').read_text());assert met['scale']==[.05000000074505806]*2;scale=F(met['scale'][0]);parts=[tifffile.imread(root/'responses'/f'{w}-{a}.tif') for a in 'xyz'];assert all(x.dtype==np.float32 and x.shape==parts[0].shape for x in parts);mesh=np.stack(parts,-1)
        canvas=[int(F(mesh.shape[1])/scale),int(F(mesh.shape[0])/scale)];assert canvas==attrs['canvas_size'];assert [28,canvas[1],canvas[0]]==json.loads((root.parent/'provenance'/f'{w}-native-array.json').read_text())['shape']
        cohort=list(csv.DictReader((root.parent/f'{w}-eligible-cores.csv').open()));assert len(cohort)=={'w035':109,'w039':54,'w044':102}[w];assert [r['core_id']for r in cohort]==[r['core_id']for r in rows if r['winding']==w]
        sources.append(dict(winding=w,mesh_shape=list(mesh.shape),canvas_wh=canvas,cores=len(cohort),frame='20250728140407'))
        for r in cohort:
            row=ids[r['core_id']];x=int(r['x']);y=int(r['y']);assert row['core_xywh']==[x,y,128,128] and row['context_xywh']==[x-64,y-64,256,256];assert row['supervised_pixels']==int(r['supervised_pixels']);assert r['eligible']==r['complete_context']=='True'
            base,alt,n,valid=field(mesh,np.arange(x-64,x+192),np.arange(y-64,y+192),canvas,scale);assert row['geometry_qualified']==bool(valid.all());assert row['geometry_valid_pixels']==int(valid.sum());assert row['geometry_invalid_pixels']==int((~valid).sum())
            basebox=np.stack([np.minimum(base[valid].min(0),alt[valid].min(0)),np.maximum(base[valid].max(0),alt[valid].max(0))]);recorded=np.asarray(row['base_extrema_union_xyz']);err=float(np.max(abs(basebox-recorded)));max_base_diff=max(max_base_diff,err);assert err<3e-11,(r['core_id'],err)
            bbox=np.asarray(row['bbox_conservative_xyz']);assert np.array_equal(bbox,np.asarray([recorded[0]-14.51,recorded[1]+14.51]));raybox=np.asarray(row['bbox_endpoint_ray_plus1p01_xyz']);assert np.all(raybox[0]>=bbox[0]) and np.all(raybox[1]<=bbox[1])
            rays=np.stack([base-13.5*n,base+13.5*n]);raylow=rays[:,valid].min((0,1))-1;rayhi=rays[:,valid].max((0,1))+1;margin=float(min(np.min(raylow-bbox[0]),np.min(bbox[1]-rayhi)));assert margin>0;min_ray_margin=min(min_ray_margin,margin)
            replays.append(dict(core_id=r['core_id'],geometry_valid_pixels=int(valid.sum()),maximum_base_extrema_difference=err,independent_endpoint_plus_one_margin=margin))
    pairs=[];paircounts={};comparisons=0
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['winding']==b['winding']:continue
            comparisons+=1
            if intersect(a['bbox_conservative_xyz'],b['bbox_conservative_xyz']):
                pairs.append((a['core_id'],b['core_id']));k=a['winding']+'-'+b['winding'];paircounts[k]=paircounts.get(k,0)+1
    assert pairs==[(p['a'],p['b'])for p in data['cross_winding_intersections']]
    folds=[]
    for f in data['folds']:
        held=f['held_out'];ev=[r for r in rows if r['winding']==held];tr=[r for r in rows if r['winding']!=held];kept=[];reject=[]
        for t in tr:
            overlap=[e['core_id']for e in ev if intersect(t['bbox_conservative_xyz'],e['bbox_conservative_xyz'])];why=[]
            if not t['geometry_qualified']:why.append('unqualified_training_geometry')
            if overlap:why.append('conservative_box_intersects_heldout_context')
            if why:reject.append(dict(core_id=t['core_id'],winding=t['winding'],reasons=why,intersecting_evaluation_ids=overlap))
            else:kept.append(t)
        assert f['original_evaluation_core_ids']==[r['core_id']for r in ev];assert f['evaluation_cores']==len(ev);assert f['original_training_cores']==len(tr);assert f['retained_training_core_ids']==[r['core_id']for r in kept];assert f['retained_training_cores']==len(kept);assert f['rejected_training_contexts']==reject
        assert f['unqualified_evaluation_core_ids']==[r['core_id']for r in ev if not r['geometry_qualified']];assert f['fold_geometry_admitted']==(not f['unqualified_evaluation_core_ids'] and bool(kept));assert f['retained_training_supervised_pixels']==sum(r['supervised_pixels']for r in kept)
        for w,m in f['retained_training_by_winding'].items():assert m['cores']==sum(r['winding']==w for r in kept);assert m['supervised_pixels']==sum(r['supervised_pixels']for r in kept if r['winding']==w)
        assert (root/f'{held}-retained-training-core-ids.txt').read_text().splitlines()==f['retained_training_core_ids']
        folds.append(dict(held_out=held,evaluation=len(ev),retained_training=len(kept),rejected_training=len(reject),retained_training_supervised_pixels=f['retained_training_supervised_pixels'],all_IDs_and_reasons_match=True))
    result=dict(status='PASS',scope='Independent full-context base/validity replay and conservative box/fold comparison; double-precision normals only used for additional endpoint enclosure, not an exact C++ normal replay',fixtures=checks,sources=sources,core_count=len(rows),independent_base_extrema_max_difference=max_base_diff,minimum_independent_endpoint_plus_one_margin=min_ray_margin,cross_winding_box_comparisons=comparisons,intersections=len(pairs),intersection_counts=paircounts,folds=folds,source_result_sha256=sha(root/'RESULTS.json'),core_replay=replays)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='core_replay'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
