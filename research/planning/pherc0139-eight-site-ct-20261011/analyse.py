"""No network: fixed source support, remapping and local structural scores."""
from common import *
import argparse
from scipy.ndimage import gaussian_filter1d,map_coordinates,minimum_filter1d
def assemble(site,kind,ledger):
    plan=read(PLAN);reqs=[r for r in plan['requests'] if r['id'] in site['request_ids'] and r['kind']==kind]
    if kind=='native':
        x0,y0,w,h=site['native_context_xywh'];x1,y1=x0+w,y0+h;nz=28
    else:x0,y0,x1,y1=site['fine_group2_xyxy'];nz=109
    a=np.zeros((nz,y1-y0,x1-x0),dtype=np.uint8);covered=np.zeros(a.shape[1:],bool)
    for req in reqs:
        row=ledger[req['id']];p=R/'chunks'/(req['id']+'.bin');assert p.stat().st_size==req['expected_bytes']==row['bytes'] and sha(p)==row['sha256']
        assert row['status']==200 and row['etag']==req['etag']
        meta=read(R/'inputs/geometry'/req['array_metadata_path']);assert sha(R/'inputs/geometry'/req['array_metadata_path'])==req['array_metadata_sha256']
        assert meta['dtype']=='|u1' and meta['compressor'] is None and meta.get('filters') is None and meta['order']=='C'
        z,y,x=map(int,req['key'].split('/'));assert z==0
        chunk=np.fromfile(p,dtype=np.uint8).reshape(meta['chunks']);ox=x*128;oy=y*128
        xa,xb=max(x0,ox),min(x1,ox+128);ya,yb=max(y0,oy),min(y1,oy+128)
        a[:,ya-y0:yb-y0,xa-x0:xb-x0]=chunk[:,ya-oy:yb-oy,xa-ox:xb-ox];covered[ya-y0:yb-y0,xa-x0:xb-x0]=True
    assert covered.all(),'Missing CT chunks'
    return a
def remap(f,q,geometry_valid,origin):
    coords=np.stack((q[...,2],q[...,1]-origin[1],q[...,0]-origin[0]))
    ix=np.floor(coords).astype(int);shape=np.array(f.shape)[:,None,None,None]
    inside=((ix>=0)&(ix+1<shape)).all(0)&(ix[0]>=5)&(ix[0]+1<=103)
    safe=np.minimum(np.maximum(ix,0),shape-2)
    # All 11 source layers of each Gaussian contributor must be nonzero.
    positive=minimum_filter1d((f>0).astype(np.uint8),size=11,axis=0,mode='constant',cval=0).astype(bool)
    supported=inside&geometry_valid
    for dz in [0,1]:
        for dy in [0,1]:
            for dx in [0,1]:supported &= positive[safe[0]+dz,safe[1]+dy,safe[2]+dx]
    filtered=gaussian_filter1d(f.astype(float),sigma=1.5,axis=0,radius=5,mode='constant',cval=np.nan)
    mapped=map_coordinates(filtered,coords,order=1,mode='constant',cval=np.nan,prefilter=False)
    mapped[~(inside&geometry_valid)]=np.nan
    return mapped,supported
def landscapes(native,mapped):
    assert native.shape==mapped.shape==(20,144,144)
    result=[]
    for template,search in patches():
        a=native[template];b=mapped[search];assert a.shape==(8,24,24) and b.shape==(20,36,36)
        scores=S.ncc_landscape(b,a);assert scores.shape==(13,13,13);result.append(scores)
    return np.stack(result)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=R/'results');args=ap.parse_args();out=args.output
    assert not out.exists();check_lock();acq=read(R/'ACQUISITION.json');assert acq['complete'] and acq['lock_sha256']==sha(R/'LOCK.json') and acq['plan_sha256']==sha(PLAN)
    ledger={r['id']:r for r in acq['requests']};assert len(ledger)==72 and acq['bytes_received']==86114304
    geo={s['core_id']:s for s in read(R/'geometry/RESULTS.json')['sites']};out.mkdir();results=[]
    for site in read(PLAN)['fixed_selected_sites']:
        sid=site['core_id'];g=geo[sid];d=out/sid;d.mkdir();native_full=assemble(site,'native',ledger);fine_full=assemble(site,'fine78',ledger)
        nx,ny,_,_=site['native_context_xywh'];x0,y0,_,_=g['native_halo_xywh'];native=native_full[4:24,y0-ny:y0-ny+144,x0-nx:x0-nx+144]
        q=np.load(R/'geometry'/sid/'queries_uvk.npy');gv=np.load(R/'geometry'/sid/'support.npy');mapped,valid=remap(fine_full,q,gv,site['fine_group2_xyxy'][:2])
        supp=[]
        for i,(template,search) in enumerate(patches()):
            supp.append(dict(index=i,native_nonzero=bool((native[template]>0).all()),fine_all_contributors_nonzero=bool(valid[search].all()),native_zero_count=int((native[template]==0).sum()),fine_unsupported_query_count=int((~valid[search]).sum())))
        source_ok=all(s['native_nonzero'] and s['fine_all_contributors_nonzero'] for s in supp)
        record=dict(core_id=sid,geometry_ready=g['ready_for_CT_scoring'],source_support_complete=source_ok,source_supports=supp,
                    native_full28_zero_fraction=float((native_full==0).mean()),fine_full109_zero_fraction=float((fine_full==0).mean()),
                    structural_screen_pass=False,scored=False)
        if g['ready_for_CT_scoring'] and source_ok:
            try:
                scores=landscapes(native,mapped);summary=S.summarise(scores,g['bases_depth_y_x_nativeXYZ'][8],g['bases_depth_y_x_nativeXYZ'][:8])
                np.save(d/'landscapes.npy',scores);record.update(summary,scored=True)
            except ValueError as exc:record['score_failure']=str(exc)
        for name,a in [('native_full28',native_full),('fine_full109',fine_full),('fine_remapped',mapped),('remapped_support',valid)]:np.save(d/(name+'.npy'),a)
        save(d/'RESULTS.json',record);results.append(record)
        print(sid,'scored',record['scored'],'fit',record.get('fit_translation_zyx'),'checks',[round(s['at_fit_score'],4) for s in record.get('supports',[])[4:]],'pass',record['structural_screen_pass'],flush=True)
    save(out/'RESULTS.json',dict(status='COMPLETE',CT_bytes_acquired=86114304,all_eight_retained=True,sites=results,
        scored=sum(s['scored'] for s in results),strict_passes=sum(s['structural_screen_pass'] for s in results),prior_geometry_failures=read(PLAN)['failed_geometry_sites'],
        meaning='Local CT structural consistency only; not ink accuracy, known correct registration or winding-wide transfer admission.'))
    save(out/'OUTPUTS.json',[receipt(p) if p.is_relative_to(R) else dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])
if __name__=='__main__':main()
