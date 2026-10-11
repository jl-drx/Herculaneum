"""Independent direct covariance, scalar decisions and raw-source sample replay."""
from common import *
import itertools
def independent_scores(native,fine):
    answer=[]
    for cx,cy in FIT+CHECK:
        x0=cx+8-12;y0=cy+8-12
        t=native[10:18,y0:y0+24,x0:x0+24].astype(float).reshape(-1);t-=t.mean();tn=np.linalg.norm(t)
        cube=np.empty((13,13,13))
        for zi,dz in enumerate(range(-6,7)):
            a=fine[6+dz:14+dz,y0-6:y0+30,x0-6:x0+30]
            windows=np.lib.stride_tricks.sliding_window_view(a,(24,24),axis=(1,2))
            v=np.moveaxis(windows,0,2).reshape(169,-1).copy();v-=v.mean(1,keepdims=True)
            cube[zi]=((v@t)/(np.linalg.norm(v,axis=1)*tn)).reshape(13,13)
        answer.append(cube)
    return np.stack(answer)
def main():
    check_lock();res=read(R/'results/RESULTS.json');plan=read(PLAN);check_files(read(R/'results/OUTPUTS.json'))
    direct=module('independent_summary',R/'sources/direct_score_source.py');direct.XY=tuple(FIT+CHECK)
    geo={s['core_id']:s for s in read(R/'geometry/RESULTS.json')['sites']};lookup={r['id']:r for r in plan['requests']}
    ledger=read(R/'ACQUISITION.json');assert ledger['complete'] and len(ledger['requests'])==72
    for item in ledger['requests']:
        req=lookup[item['id']];p=R/'chunks'/(req['id']+'.bin')
        assert item['status']==200 and item['etag']==req['etag'] and item['bytes']==req['expected_bytes']==p.stat().st_size and sha(p)==item['sha256']
    records=[];maxdiff=0.;maxsampling=0.;samples=0
    weights=np.exp(-np.arange(-5,6,dtype=float)**2/(2*1.5**2));weights/=weights.sum()
    for site,author in zip(plan['fixed_selected_sites'],res['sites'],strict=True):
        sid=site['core_id'];assert author['core_id']==sid;g=geo[sid];d=R/'results'/sid
        n=np.load(d/'native_full28.npy');f=np.load(d/'fine_full109.npy');mapped=np.load(d/'fine_remapped.npy');valid=np.load(d/'remapped_support.npy')
        q=np.load(R/'geometry'/sid/'queries_uvk.npy');gv=np.load(R/'geometry'/sid/'support.npy')
        x0,y0,_,_=g['native_halo_xywh'];nx,ny,_,_=site['native_context_xywh'];native=n[:,y0-ny:y0-ny+144,x0-nx:x0-nx+144]
        # Verify complete assembled arrays against source chunks without using assemble().
        for kind,array in [('native',n),('fine78',f)]:
            origin=site['native_context_xywh'][:2] if kind=='native' else site['fine_group2_xyxy'][:2]
            covered=np.zeros(array.shape[1:],bool)
            for rid in site['request_ids']:
                req=lookup[rid]
                if req['kind']!=kind:continue
                _,cy,cx=map(int,req['key'].split('/'));chunk=np.fromfile(R/'chunks'/(rid+'.bin'),np.uint8).reshape(array.shape[0],128,128)
                yy=np.arange(array.shape[1])+origin[1];xx=np.arange(array.shape[2])+origin[0];ys=np.flatnonzero(yy//128==cy);xs=np.flatnonzero(xx//128==cx)
                assert np.array_equal(array[:,ys[:,None],xs],chunk[:,(yy[ys]%128)[:,None],xx[xs]%128])
                covered[np.ix_(ys,xs)]=True
            assert covered.all()
        for z,y,x in itertools.product([0,9,19],[20,72,123],[20,72,123]):
            if not gv[z,y,x]:continue
            u,v,k=q[z,y,x];u-=site['fine_group2_xyxy'][0];v-=site['fine_group2_xyxy'][1]
            ix,iy,iz=np.floor([u,v,k]).astype(int);du,dv,dk=u-ix,v-iy,k-iz
            if iz<5 or iz+1>103 or ix<0 or iy<0 or ix+1>=f.shape[2] or iy+1>=f.shape[1]:continue
            total=0.;positive=True
            for dz,dy,dx in itertools.product([0,1],repeat=3):
                column=f[iz+dz-5:iz+dz+6,iy+dy,ix+dx];positive &= bool((column>0).all())
                wt=(dk if dz else 1-dk)*(dv if dy else 1-dv)*(du if dx else 1-du)
                total+=wt*float(weights@column)
            delta=abs(total-mapped[z,y,x]);assert delta<1e-9 and positive==bool(valid[z,y,x]);maxsampling=max(maxsampling,delta);samples+=1
        source_complete=all((native[10:18,t[1],t[2]]>0).all() and valid[s].all() for t,s in patches())
        assert bool(source_complete)==author['source_support_complete']
        if not author['scored']:
            records.append(dict(core_id=sid,status='UNSCORED_AS_RECORDED'));continue
        scores=independent_scores(native,mapped);saved=np.load(d/'landscapes.npy');delta=float(np.max(abs(scores-saved)));assert delta<1e-9
        replay=direct.summarise(scores,np.array(g['bases_depth_y_x_nativeXYZ'][8]),[np.array(b) for b in g['bases_depth_y_x_nativeXYZ'][:8]])
        errors=direct.compare(replay,author);maxdiff=max(maxdiff,delta)
        records.append(dict(core_id=sid,status='PASS',scores=scores.size,maximum_score_error=delta,maximum_summary_error=max(errors),structural_screen_pass=replay['structural_screen_pass']))
        print(sid,'replay PASS','max score error',delta,flush=True)
    save(R/'RESULT_VERIFICATION.json',dict(status='PASS',all72_chunks_verified=True,assembled_arrays_rechecked=True,
        scalar_source_samples=samples,maximum_sampling_error=maxsampling,scores_recomputed=sum(s.get('scores',0) for s in records),maximum_score_error=maxdiff,sites=records,
        scope='Different numerical implementations in this runtime; no independent physical registration truth or independent research team claim.'))
if __name__=='__main__':main()
