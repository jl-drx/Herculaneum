"""Scalar source geometry and meaningful score/inverse fixtures before CT."""
from common import *
from functools import lru_cache
from scipy.ndimage import gaussian_filter
def main():
    check_inputs();scalar=module('independent_scalar',R/'sources/scalar_source.py');checks=[]
    def need(test,name):
        assert test,name
        checks.append(name)
    old=S.fixtures();need(old['status']=='PASS','retained shifted/intensity/periodic/constant fixtures')
    rng=np.random.default_rng(1390811);f=gaussian_filter(rng.normal(size=(32,80,80)),sigma=(.7,1,1))
    for shift in [(0,0,0),(2,-3,4),(-4,3,-2),(0,0,6)]:
        z,y,x=shift;t=f[12+z:20+z,24+y:48+y,24+x:48+x]*2.3+17
        cube=S.ncc_landscape(f[6:26,18:54,18:54],t);s,v=S.peak(cube)
        need(s==shift and abs(v-1)<1e-10,'24-square shift '+str(shift))
    bad=np.ones((20,36,36));bad[0,0,0]=np.nan
    try:S.ncc_landscape(bad,np.zeros((8,24,24)))
    except ValueError:checks.append('nonfinite search rejected')
    else:raise AssertionError('nonfinite accepted')
    yy,xx=np.mgrid[:16,:16];b=np.stack((xx,yy,np.full_like(xx,10)),axis=-1).astype(float);n=np.zeros_like(b);n[...,2]=-.25
    q0=np.array([[5.,6.,54.],[7.,8.,54.]]);truth=q0+np.array([.3,-.2,1.]);target=np.column_stack((truth[:,:2],10-.25*(truth[:,2]-54)))
    q,ok,_,rr,_=P.solve(b,n,np.ones((16,16),bool),q0,target,np.tile([0,0,-1.],(2,1)),np.ones(2,bool),0,0)
    need(ok.all() and np.max(np.abs(q-truth))<1e-8,'analytic inverse recovers known query')
    broken=np.zeros((16,16),bool)
    _,ok,_,_,_=P.solve(b,n,broken,q0,target,np.tile([0,0,-1.],(2,1)),np.ones(2,bool),0,0)
    need(not ok.any(),'invalid representative support rejected')
    from analyse import remap
    volume=np.full((109,20,20),80,np.uint8);query=np.array([[[[6.25,7.5,54.25]]]])
    value,valid=remap(volume,query,np.ones((1,1,1),bool),(0,0))
    need(valid.all() and np.max(abs(value-80))<1e-10,'constant CT sampling with complete source support')
    volume[49,8,7]=0
    _,valid=remap(volume,query,np.ones((1,1,1),bool),(0,0))
    need(not valid.any(),'zero in outer Gaussian/interpolation contributor rejected')
    _,valid=remap(np.full_like(volume,80),query,np.zeros((1,1,1),bool),(0,0))
    need(not valid.any(),'finite CT cannot override invalid geometry')
    records=[];geo=read(R/'geometry/RESULTS.json')
    for w in ['w040','w041','w043']:
        native=G.load(w,'native');fine=G.load(w,'fine78');matrix=G.to_native(G.read(G.I/'CATALOG_EXTRACT.json'),fine['volume'])
        @lru_cache(maxsize=20000)
        def pooled(u,v):
            rays=[scalar.ray(fine['mesh'],4*u+dx,4*v+dy,fine['canvas']) for dy in range(4) for dx in range(4)]
            bb=sum(p for p,n in rays)/16;nn=sum(n for p,n in rays)/16
            return matrix[:3,:3]@bb+matrix[:3,3],matrix[:3,:3]@nn
        for site in [s for s in geo['sites'] if s['winding']==w]:
            check_files(site['files']);sid=site['core_id'];d=R/'geometry'/sid
            qq=np.load(d/'queries_uvk.npy');valid=np.load(d/'support.npy');stored=np.load(d/'residual.npy');x0,y0,_,_=site['native_halo_xywh']
            for z in [0,9,19]:
                for y in [20,72,123]:
                    for x in [20,72,123]:
                        if not valid[z,y,x]:
                            records.append(dict(core_id=sid,array_zyx=[z,y,x],status='RETAINED_INVALID'));continue
                        u,v,k=qq[z,y,x];iu,iv=int(np.floor(u)),int(np.floor(v));du,dv=u-iu,v-iv;p=np.zeros(3)
                        for dx,dy,weight in [(0,0,(1-du)*(1-dv)),(1,0,du*(1-dv)),(0,1,(1-du)*dv),(1,1,du*dv)]:
                            bb,nn=pooled(iu+dx,iv+dy);p+=weight*(bb+nn*(k-54))
                        nb,nn=scalar.ray(native['mesh'],x0+x,y0+y,native['canvas']);error=float(np.linalg.norm(p-(nb+nn*(z+4-13.5))))
                        need(error<=.01+1e-8 and abs(error-stored[z,y,x])<1e-8,'scalar '+sid+f' {z}/{y}/{x}')
                        records.append(dict(core_id=sid,array_zyx=[z,y,x],status='PASS',residual=error,difference=abs(error-stored[z,y,x])))
            if site['ready_for_CT_scoring']:
                # Independently mark the four interpolation neighbors in a bitmap.
                grids=[];rect=next(s['fine_group2_xyxy'] for s in read(PLAN)['fixed_selected_sites'] if s['core_id']==sid)
                for cx,cy in FIT+CHECK:
                    sub=qq[:,cy-10:cy+26,cx-10:cx+26]
                    uv=np.floor(sub[...,:2]).astype(int).reshape(-1,2)-np.array(rect[:2])
                    bitmap=np.zeros((rect[3]-rect[1],rect[2]-rect[0]),bool)
                    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:bitmap[uv[:,1]+dy,uv[:,0]+dx]=True
                    grids.append(bitmap)
                need(all(not (grids[a]&grids[b]).any() for a in range(8) for b in range(a+1,8)),'independent separate-support bitmaps '+sid)
        pooled.cache_clear()
    save(R/'PREFLIGHT_VERIFICATION.json',dict(status='PASS',checks=checks,scalar_queries=records,scalar_attempts=len(records),scalar_verified=sum(r['status']=='PASS' for r in records),ready_sites=geo['ready'],CT_bytes_acquired=0))
    print('PASS',len(checks),'checks;',len(records),'scalar queries;',geo['ready'],'ready sites')
if __name__=='__main__':main()
