#!/usr/bin/env python3
"""Independent finite-control replay. Imports no production or prior analysis code."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def interpolate(source, uv, origin, operator):
    """Nested bilinear interpolation; NaN outside the complete interpolation stencil."""
    if operator == 'mean16_bilinear':
        h,w=source.shape
        src=source.reshape(h//4,4,w//4,4).mean(axis=(1,3),dtype=np.float64)
        coords=uv-np.asarray(origin,dtype=np.float64)/4
    else:
        src=source.astype(np.float64)
        coords=4*uv+1.5-np.asarray(origin,dtype=np.float64)
    good=np.isfinite(coords).all(-1)
    ij=np.floor(np.where(np.isfinite(coords),coords,0)).astype(np.int64)
    x,y=ij[...,0],ij[...,1]
    good &= (x>=0)&(y>=0)&(x+1<src.shape[1])&(y+1<src.shape[0])
    out=np.full(x.shape,np.nan)
    gx,gy=x[good],y[good]; f=coords[good]-ij[good]
    top=src[gy,gx]+f[:,0]*(src[gy,gx+1]-src[gy,gx])
    bottom=src[gy+1,gx]+f[:,0]*(src[gy+1,gx+1]-src[gy+1,gx])
    out[good]=top+f[:,1]*(bottom-top)
    return out,good

def metrics(a,b,mask):
    d=(a-b)[mask]/255
    if not len(d): return {k:None for k in ('mae','rmse','signed_difference','p95_absolute_difference','max_absolute_difference')}
    return dict(mae=float(np.mean(abs(d))),rmse=float(np.sqrt(np.mean(d*d))),signed_difference=float(np.mean(d)),p95_absolute_difference=float(np.quantile(abs(d),.95)),max_absolute_difference=float(np.max(abs(d))))

def arm_stats(a,mask):
    p=a[mask]/255
    if not len(p): return dict(mean=None,entropy_nats=None,threshold48_positive_count=0)
    ent=np.zeros_like(p); m=p>0; ent[m]-=p[m]*np.log(p[m]); m=p<1; ent[m]-=(1-p[m])*np.log1p(-p[m])
    return dict(mean=float(p.mean()),entropy_nats=float(ent.mean()),threshold48_positive_count=int(np.count_nonzero(a[mask]>=48)))

def moments(x,w,fit):
    x=np.array(x,dtype=float);w=np.array(w,dtype=float)
    mean=np.einsum('i,ij->j',w,x); c=x-mean
    return dict(mean_absolute_zyx=mean.tolist(),mean_relative_to_fit_zyx=(mean-fit).tolist(),covariance_zyx=np.einsum('i,ij,ik->jk',w,c,c).tolist())

def fixtures():
    checked=[]
    origin=np.array([100,200]); y,x=np.mgrid[:20,:24]; field=(2*x+3*y).astype(float)
    uv=np.array([[[26.25,51.5],[27.5,52.25]]])
    for op in ('mean16_bilinear','point_bilinear'):
        r,m=interpolate(field,uv,origin,op)
        expected=2*(4*uv[...,0]+1.5-origin[0])+3*(4*uv[...,1]+1.5-origin[1])
        assert m.all() and np.allclose(r,expected,atol=1e-12)
        r,m=interpolate(np.ones_like(field)*91,uv,origin,op);assert np.array_equal(r,91*np.ones(r.shape))
        checked.extend([op+':asymmetric-ramp-and-nonzero-origin',op+':constant'])
    r,m=interpolate(field,np.array([[[100.,100.]]]),origin,'point_bilinear');assert not m.any() and np.isnan(r).all();checked.append('invalid-source-is-nan-not-zero')
    kernels=[np.array([0.]),np.array([10.,10.,10.])]; equal=np.mean([v.mean() for v in kernels]); pooled=np.concatenate(kernels).mean(); assert equal==5 and pooled==7.5; checked.append('equal-kernel-not-equal-candidate')
    p=np.array([.12,.34,.56,.78]);q=np.array([[.2,.7,.3,.9],[.9,.1,.4,.2]]); bce=lambda q:-q*np.log(p)-(1-q)*np.log1p(-p)
    assert np.allclose(bce(q).mean(0),bce(q.mean(0)),atol=2e-16);checked.append('masked-bce-linear-in-target')
    mask=np.array([True,False,True,False]); assert np.allclose(bce(q)[:,mask].mean(),bce(q.mean(0))[mask].mean());checked.append('common-mask-bce')
    assert all(v is None for v in metrics(p,p,np.zeros(4,bool)).values());checked.append('empty-support-null-metrics')
    assert arm_stats(np.zeros(4),np.ones(4,bool))['entropy_nats']==0;checked.append('entropy-zero-endpoint')
    return checked

def run(root, prior, corr):
    checks=fixtures()
    assert digest(root/'PROTOCOL.md')=='dd87ce2618f75564d072fbc08ab3a1689bc37d1fbc0128620f5452b2e69e3052', 'Frozen protocol changed'
    cand=json.loads((prior/'CT_CANDIDATES.json').read_text());cm=json.loads((prior/'teacher/CROP.json').read_text());gm=json.loads((corr/'geometry-review/pullback/site1/RESULTS.json').read_text());nm=json.loads((prior/'NATIVE_ELIGIBILITY.json').read_text())
    q=np.load(corr/'geometry-review/pullback/site1/queries_uvk.npy',allow_pickle=False);geo=np.load(corr/'geometry-review/pullback/site1/support.npy',allow_pickle=False).astype(bool);native=np.load(prior/'native_eligible.npy',allow_pickle=False).astype(bool);crop=np.load(prior/'teacher/crop.npy',allow_pickle=False)
    assert digest(prior/'teacher/crop.npy')==cm['crop_sha256'], 'Teacher crop bytes changed'
    assert hashlib.sha256(crop.tobytes()).hexdigest()==cm['decoded_pixel_sha256'], 'Teacher pixel hash'
    expected_geo={r['name']:r['sha256'] for r in gm['output_files']}
    for name in ['queries_uvk.npy','support.npy']:
        assert digest(corr/'geometry-review/pullback/site1'/name)==expected_geo[name], 'Geometry source changed'
    assert digest(prior/'native_eligible.npy')==nm['mask_sha256'], 'Native mask changed'
    # Independently reconstruct eligibility from all 28 planes of the retained chunks.
    ox,oy=nm['origin_native_xy']; nh,nw=nm['shape_yx']; native_again=np.zeros((nh,nw),bool); native_coverage=np.zeros((nh,nw),bool)
    for entry in nm['chunks']:
        path=corr/'ct-study/chunks'/entry['name']; assert digest(path)==entry['sha256'], 'Native CT chunk changed'
        chunk=np.fromfile(path,dtype=np.uint8).reshape(28,128,128)
        cy,cx=map(int,path.stem.split('-')[-2:]);xx,yy=cx*128,cy*128
        lx=max(ox,xx);rx=min(ox+nw,xx+128);ty=max(oy,yy);by=min(oy+nh,yy+128)
        if lx<rx and ty<by:
            native_again[ty-oy:by-oy,lx-ox:rx-ox]=chunk[:,ty-yy:by-yy,lx-xx:rx-xx].max(0)>0
            native_coverage[ty-oy:by-oy,lx-ox:rx-ox]=True
    assert native_coverage.all() and np.array_equal(native,native_again), 'Native max28 eligibility replay mismatch'
    if (root/'INPUT_MANIFEST.json').exists():
        frozen=json.loads((root/'INPUT_MANIFEST.json').read_text())
        roots={'supervision':prior,'correspondence':corr}
        for entry in frozen['files']:
            if entry['packet'] in roots:
                assert digest(roots[entry['packet']]/entry['relative'])==entry['sha256'], 'Frozen input manifest mismatch'
    origin=np.array(gm['native_xywh'][:2]); j0=gm['native_j_half_open'][0]
    kernels=[[tuple(c['shift_zyx']) for c in s['candidates']] for s in cand['supports']]
    unique=sorted(set(v for kernel in kernels for v in kernel));fit=tuple(cand['fit_translation_zyx']); fi=unique.index(fit)
    kw=[np.array([1/len(k) if v in k else 0 for v in unique],dtype=float) for k in kernels]; cw=np.mean(kw,axis=0)
    assert np.isclose(cw.sum(),1)
    records=[]; arrs={}; threshold_near=[]
    for si,s in enumerate(cand['supports']):
        x0,y0,x1,y1=s['native_xy_half_open']; x0-=origin[0];x1-=origin[0];y0-=origin[1];y1-=origin[1]
        ng=native[y0:y1,x0:x1]
        for j in range(10,18):
            coords=[];valids=[]
            for dz,dy,dx in unique:
                coords.append(q[j+dz-j0,y0+dy:y1+dy,x0+dx:x1+dx,:2]);valids.append(geo[j+dz-j0,y0+dy:y1+dy,x0+dx:x1+dx])
            coords=np.stack(coords);valids=np.stack(valids)
            localidx=[unique.index(v) for v in kernels[si]];centroid=coords[localidx].mean(axis=0);dg=valids[localidx].all(0)
            for op in ('mean16_bilinear','point_bilinear'):
                trans,sourcevalid=interpolate(crop,coords,cm['origin_fullfine_xy'],op);D,dv=interpolate(crop,centroid,cm['origin_fullfine_xy'],op)
                geom=np.logical_and.reduce(valids)&dg&np.isfinite(coords).all(axis=(0,3))&np.isfinite(centroid).all(-1)
                source=sourcevalid.all(0)&dv
                complete=geom&source&ng
                common=complete&(trans>0).all(0)&(D>0)
                A=trans[fi];B=trans[localidx].mean(axis=0);C=np.mean([trans[[unique.index(v) for v in k]].mean(axis=0) for k in kernels],axis=0)
                arms=dict(A=A,B=B,C=C,D=D)
                positive=common&((trans>=48).any(0)|(D>=48))
                name=f's{s["support"]}_j{j}_{op}'
                record=dict(key=name,support=s['support'],j=j,operator=op,primary=(j==14 and op=='mean16_bilinear'),geometry_only_pixels=int(geom.sum()),geometry_source_native_pixels=int(complete.sum()),common_pixels=int(common.sum()),active_union_pixels=int(positive.sum()),source_excluded_pixels=int((~source).sum()),arm_stats={k:arm_stats(v,common) for k,v in arms.items()},pairs={})
                for a,b in [('A','B'),('A','C'),('B','C'),('B','D')]:
                    p=dict(common=metrics(arms[a],arms[b],common),active_union=metrics(arms[a],arms[b],positive),threshold_disagreement_common=int(np.count_nonzero(((arms[a]>=48)!=(arms[b]>=48))&common)))
                    if a=='B': p['active_union_mae_below_one_quantization_interval']=None if not positive.any() else bool(p['active_union']['mae']<1/255)
                    record['pairs'][a+'-'+b]=p
                near=(np.abs(trans-48)<1e-12)&np.broadcast_to(common,trans.shape)
                if near.any():threshold_near.append(dict(key=name,source_transfer_values_near48=int(near.sum()),values=np.unique(trans[near]).tolist()))
                records.append(record)
                for k,v in arms.items():arrs[name+'_'+k]=v
                arrs[name+'_transfers']=trans;arrs[name+'_geometry_mask']=geom;arrs[name+'_complete_mask']=complete;arrs[name+'_common_mask']=common;arrs[name+'_positive_union_mask']=positive;arrs[name+'_centroid_uv']=centroid
    result=dict(protocol_sha256=digest(root/'PROTOCOL.md'),method='Independent nested-bilinear interpolation of retained crop; normalized metric scale /255; no production imports',fixtures=checks,unique_shifts_zyx=[list(v) for v in unique],global_weights=cw.tolist(),kernel_moments=[dict(support=s['support'],**moments(k,[1/len(k)]*len(k),fit)) for s,k in zip(cand['supports'],kernels)],global_moments=moments(unique,cw,fit),cases=records,threshold_boundary_observations=threshold_near)
    review=root/'review';review.mkdir(exist_ok=True)
    (review/'INDEPENDENT_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(review/'independent_arrays.npz',**arrs)
    print(json.dumps(dict(cases=len(records),unique_shifts=len(unique),primary=[r for r in records if r['primary']],fixtures=checks),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/workspace/herculaneum-lab/reports/pherc0139-soft-controls-20261011'));ap.add_argument('--prior',type=Path,default=Path('/workspace/herculaneum-lab/reports/pherc0139-supervision-20261011'));ap.add_argument('--correspondence',type=Path,default=Path('/workspace/herculaneum-lab/reports/pherc0139-correspondence-20261011'));args=ap.parse_args();run(args.root,args.prior,args.correspondence)
