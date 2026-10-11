"""Frozen CT-free coordinate pullback; full resolution, one CPU thread.

Queries are representative surface-volume geometry, not a reconstruction of
the optical PSF, registration, ink truth, or exact interpolation of CT values.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import tifffile

ROOT=Path(__file__).resolve().parent
GEO=ROOT.parents[1]/'geometry'/'responses'
CAT=Path('/workspace/herculaneum-lab/reports/fortyfifth-experiment/source-evidence/PHerc0139-retained-catalog.json')
F=np.float32
S=F(.05000000074505806)
SITES=[('site1',2432,1920),('site2',3968,3200),('site3',5632,4480)]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def inputs(): return [GEO/f'{v}-{a}.tif' for v in ('20260102150214','20250728140407') for a in 'xyz']+[CAT]
def prepare():
    plan={'kind':'geometry_only_representative_ray_inverse_not_CT_registration','code_sha256':sha(Path(__file__)),
      'inputs':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs()],
      'sites':[{'id':n,'native_center_xy':[x,y],'core_xywh':[x-128,y-128,256,256], 'halo_xywh':[x-136,y-136,272,272]} for n,x,y in SITES],
      'depth_native_half_open':[4,24],'native_center_depth':13.5,'fine_center_depth':54,
      'fine_field':'mean16 group0 staged-fp32 base/normal then affine A,b; do not renormalize mean normal; 2D bilinear representative field',
      'field_halo_fine_group2':6,'query_initialization':'source-derived transformed-grid half-pixel mapping composed with renderer, mean16 footprint center; nominal physical Z scale',
      'newton':{'maximum_iterations':10,'residual_tolerance_native_voxels':.01,'max_basin_uv_fine_group2_each_axis':2.,'max_basin_k_fine_planes':8.,'max_update_uv_each_axis':.5,'max_update_k':2.,'max_condition':1000.,'minimum_singular_value':1e-6,'expected_jacobian_determinant_sign':-1,'normal_dot_native_positive':True},
      'invalid':'all16 fine base bilinear and nearest normal axial stencils; native corresponding support; no clamping, replacement crops, CT values or prediction use',
      'output':'float64 queries D,H,W,3 [u,v,k], initial queries, boolean support and failure code; failed queries retained but not admitted',
      'basis':'native-XYZ vectors columns [normal, dy, dx] at central mesh surface, native raster derivatives +/-1',
      'query_budget_samples_per_site':20*272*272}
    p=ROOT/'PLAN.json'
    if p.exists(): assert json.loads(p.read_text())==plan,'frozen plan changed'
    else: write(p,plan)
    return plan

def squared(v): return F(F(F(v[...,0]*v[...,0])+F(v[...,1]*v[...,1]))+F(v[...,2]*v[...,2]))
def normals(p):
    valid=np.isfinite(p).all(2)&(p[...,2]>0)&(p[...,0]!=-1)
    n=np.zeros_like(p);ok=np.zeros(p.shape[:2],bool)
    support=valid[1:-1,1:-1]&valid[1:-1,:-2]&valid[1:-1,2:]&valid[:-2,1:-1]&valid[2:,1:-1]
    u=F(p[1:-1,2:]-p[1:-1,:-2]);v=F(p[2:,1:-1]-p[:-2,1:-1]);z=np.empty_like(u)
    z[...,0]=F(F(u[...,1]*v[...,2])-F(u[...,2]*v[...,1]));z[...,1]=F(F(u[...,2]*v[...,0])-F(u[...,0]*v[...,2]));z[...,2]=F(F(u[...,0]*v[...,1])-F(u[...,1]*v[...,0]))
    with np.errstate(all='ignore'):
        ll=squared(z);support &= np.isfinite(z).all(2)&np.isfinite(ll)&(ll>0)
        z=F(z*F(F(1)/F(np.sqrt(np.where(support,ll,F(1)))))[...,None]);z=F(-z)
        ll=squared(z);z=F(z/F(np.sqrt(np.where(support,ll,F(1))))[...,None])
    support &= np.isfinite(z).all(2)
    n[1:-1,1:-1]=np.where(support[...,None],z,F(0));ok[1:-1,1:-1]=support
    return valid,n,ok

def axis(coords,count,canvas):
    coords=np.asarray(coords,dtype=np.int64);tile=(coords//128)*128;local=coords-tile
    center=F(float(count)/2/float(S));origin=F(F(-.5)*F(F(canvas)-F(1)))
    off=F(origin+tile.astype(np.float32));ul=F(F(off+center)*S);ox=F(ul.astype(float)-4*float(S))
    exact=F(ox+F((local+4).astype(np.float32)*S))
    alt=F(ox.astype(float)+(local+4).astype(float)*float(S))
    i0=np.floor(exact).astype(int);near=np.floor(exact.astype(float)+.5).astype(int)
    unambiguous=(i0==np.floor(alt).astype(int))&(near==np.floor(alt.astype(float)+.5).astype(int))
    return exact,i0,near,unambiguous

def render(mesh,cache,xs,ys,canvas):
    valid,normal,nok=cache;h,w=mesh.shape[:2]
    xx,ix,nx,ux=axis(xs,w,canvas[0]);yy,iy,ny,uy=axis(ys,h,canvas[1])
    assert ix.min()>=0 and ix.max()+1<w and iy.min()>=0 and iy.max()+1<h
    assert nx.min()>0 and nx.max()<w-1 and ny.min()>0 and ny.max()<h-1
    p00=mesh[iy[:,None],ix[None,:]];p01=mesh[iy[:,None],(ix+1)[None,:]];p10=mesh[(iy+1)[:,None],ix[None,:]];p11=mesh[(iy+1)[:,None],(ix+1)[None,:]]
    wx=F(xx-ix.astype(np.float32))[None,:,None];wy=F(yy-iy.astype(np.float32))[:,None,None]
    top=F(F(p00*F(F(1)-wx))+F(p01*wx));bot=F(F(p10*F(F(1)-wx))+F(p11*wx));base=F(F(top*F(F(1)-wy))+F(bot*wy))
    direction=normal[ny[:,None],nx[None,:]]
    good=valid[iy[:,None],ix[None,:]]&valid[iy[:,None],(ix+1)[None,:]]&valid[(iy+1)[:,None],ix[None,:]]&valid[(iy+1)[:,None],(ix+1)[None,:]]&nok[ny[:,None],nx[None,:]]&uy[:,None]&ux[None,:]
    good &= np.isfinite(base).all(2)&np.isfinite(direction).all(2)
    return base.astype(float),direction.astype(float),good

def initial(xs,ys):
    qx=axis(xs,399,7980)[0].astype(float);qy=axis(ys,320,6400)[0].astype(float)
    qfx=(qx+.5)*(1552/399)-.5;qfy=(qy+.5)*(1244/320)-.5
    # Representative mean of samples 4u...4u+3 has mesh index S*(4u+2).
    return (qfx/float(S)-2)/4,(qfy/float(S)-2)/4

def make_field(mesh,cache,urange,vrange,A,b):
    us=np.arange(*urange);vs=np.arange(*vrange);base=np.zeros((len(vs),len(us),3));normal=np.zeros_like(base);ok=np.ones(base.shape[:2],bool)
    for dy in range(4):
        for dx in range(4):
            pp,nn,vv=render(mesh,cache,4*us+dx,4*vs+dy,(31040,24880))
            base+=pp/16;normal+=nn/16;ok &= vv
    return base@A.T+b,normal@A.T,ok

def interp(field,u,v,ox,oy):
    x=u-ox;y=v-oy;ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
    inside=(ix>=0)&(iy>=0)&(ix+1<field.shape[1])&(iy+1<field.shape[0])
    # Clipped indices serve only safe memory access; outside queries are rejected.
    ix=np.clip(ix,0,field.shape[1]-2);iy=np.clip(iy,0,field.shape[0]-2)
    dx=(x-ix)[...,None];dy=(y-iy)[...,None]
    p00=field[iy,ix];p01=field[iy,ix+1];p10=field[iy+1,ix];p11=field[iy+1,ix+1]
    value=(p00*(1-dx)+p01*dx)*(1-dy)+(p10*(1-dx)+p11*dx)*dy
    du=(p01-p00)*(1-dy)+(p11-p10)*dy;dv=(p10-p00)*(1-dx)+(p11-p01)*dx
    return value,du,dv,inside,ix,iy

def evaluate(B,N,ok,q,ox,oy):
    bb,bu,bv,inside,ix,iy=interp(B,q[:,0],q[:,1],ox,oy)
    nn,nu,nv,_,_,_=interp(N,q[:,0],q[:,1],ox,oy)
    kz=(q[:,2]-54)[:,None];point=bb+nn*kz
    J=np.stack((bu+nu*kz,bv+nv*kz,nn),axis=-1)
    valid=inside&ok[iy,ix]&ok[iy,ix+1]&ok[iy+1,ix]&ok[iy+1,ix+1]
    valid &= np.isfinite(point).all(1)&np.isfinite(J).all((1,2))
    return point,J,valid,nn

def solve(B,N,ok,q0,target,nd,ngood,ox,oy):
    q=q0.copy();failure=np.zeros(len(q),np.uint8);converged=np.zeros(len(q),bool);steps=np.zeros(len(q),np.uint8);residual=np.full(len(q),np.inf)
    failure[~ngood]=1
    for iteration in range(11):
        ids=np.flatnonzero((failure==0)&~converged)
        if not len(ids): break
        point,J,valid,normal=evaluate(B,N,ok,q[ids],ox,oy)
        failure[ids[~valid]]=2
        ids=ids[valid];point=point[valid];J=J[valid];normal=normal[valid]
        if not len(ids):continue
        singular=np.linalg.svd(J,compute_uv=False);det=np.linalg.det(J)
        correct=(singular[:,-1]>=1e-6)&(singular[:,0]/np.maximum(singular[:,-1],1e-300)<=1000)&(det<0)&((normal*nd[ids]).sum(1)>0)
        failure[ids[~correct]]=3
        ids=ids[correct];point=point[correct];J=J[correct]
        if not len(ids):continue
        delta=target[ids]-point;rr=np.linalg.norm(delta,axis=1);residual[ids]=rr
        done=rr<=.01;converged[ids[done]]=True
        ids=ids[~done];delta=delta[~done];J=J[~done]
        if not len(ids):continue
        if iteration==10:failure[ids]=5;break
        update=np.linalg.solve(J,delta[...,None])[...,0]
        scale=np.maximum(1,np.maximum(np.max(np.abs(update[:,:2]),axis=1)/.5,np.abs(update[:,2])/2))
        proposal=q[ids]+update/scale[:,None]
        basin=(np.abs(proposal[:,:2]-q0[ids,:2])<=2).all(1)&(np.abs(proposal[:,2]-q0[ids,2])<=8)&(proposal[:,2]>=0)&(proposal[:,2]<=108)
        failure[ids[~basin]]=4;q[ids[basin]]=proposal[basin];steps[ids[basin]]+=1
    return q,converged,failure,residual,steps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args();plan=prepare()
    if args.prepare:print('PLAN frozen',sha(ROOT/'PLAN.json'));return
    for p in plan['inputs']: assert sha(Path(p['path']))==p['sha256']
    cat=json.loads(CAT.read_text());aff=np.array(cat['volumes']['20260102150214']['properties']['transforms'][0]['transformation_matrix']);A=aff[:,:3];b=aff[:,3]
    fine=np.stack([tifffile.imread(p) for p in inputs()[:3]],axis=-1);native=np.stack([tifffile.imread(p) for p in inputs()[3:6]],axis=-1)
    assert fine.shape==(1244,1552,3) and native.shape==(320,399,3)
    fc=normals(fine);nc=normals(native);summaries=[]
    for name,cx,cy in SITES:
        t0=time.monotonic();out=ROOT/name;out.mkdir(exist_ok=True);xs=np.arange(cx-136,cx+136);ys=np.arange(cy-136,cy+136)
        base,nd,ngood=render(native,nc,xs,ys,(7980,6400));uu,vv=initial(xs,ys)
        urange=[int(np.floor(uu.min()))-6,int(np.ceil(uu.max()))+8];vrange=[int(np.floor(vv.min()))-6,int(np.ceil(vv.max()))+8]
        B,N,ok=make_field(fine,fc,urange,vrange,A,b)
        u0,v0=np.meshgrid(uu,vv);queries=[];allok=[];allfail=[];allrr=[];allsteps=[];initials=[]
        for j in range(4,24):
            q0=np.stack((u0.ravel(),v0.ravel(),np.full(u0.size,54+(j-13.5)*9.362/2.399)),axis=-1)
            target=(base+nd*(j-13.5)).reshape(-1,3)
            q,success,fail,rr,steps=solve(B,N,ok,q0,target,nd.reshape(-1,3),ngood.ravel(),urange[0],vrange[0])
            queries.append(q.reshape(272,272,3));initials.append(q0.reshape(272,272,3));allok.append(success.reshape(272,272));allfail.append(fail.reshape(272,272));allrr.append(rr.reshape(272,272));allsteps.append(steps.reshape(272,272))
            print(name,'native_depth',j,'accepted',int(success.sum()),'total',len(success),flush=True)
        queries=np.stack(queries);initials=np.stack(initials);support=np.stack(allok);failure=np.stack(allfail);rr=np.stack(allrr);steps=np.stack(allsteps)
        for filename,arr in [('queries_uvk.npy',queries),('initial_uvk.npy',initials),('support.npy',support),('failure_codes.npy',failure),('residual_native_voxels.npy',rr),('iterations.npy',steps)]:np.save(out/filename,arr)
        # Surface Jacobian measured on full halo, report the interior only.
        dx=(base[1:-1,2:]-base[1:-1,:-2])/2;dy=(base[2:,1:-1]-base[:-2,1:-1])/2
        validJ=ngood[1:-1,1:-1]&ngood[1:-1,2:]&ngood[1:-1,:-2]&ngood[2:,1:-1]&ngood[:-2,1:-1]
        JJ=np.stack((nd[1:-1,1:-1],dy,dx),axis=-1);centerJ=JJ[135,135]
        goodq=queries[support];goodrr=rr[support];dd=queries-initials
        bounds=None if not len(goodq) else {'u_minmax': [float(goodq[:,0].min()),float(goodq[:,0].max())], 'v_minmax':[float(goodq[:,1].min()),float(goodq[:,1].max())], 'k_minmax':[float(goodq[:,2].min()),float(goodq[:,2].max())], 'fine_group2_interpolation_inclusive_xy': [int(np.floor(goodq[:,0].min())),int(np.floor(goodq[:,1].min())),int(np.floor(goodq[:,0].max()))+1,int(np.floor(goodq[:,1].max()))+1], 'fine_k_interpolation_inclusive':[int(np.floor(goodq[:,2].min())),int(np.floor(goodq[:,2].max()))+1]}
        summary={'site':name,'plan_sha256':sha(ROOT/'PLAN.json'),'seconds':time.monotonic()-t0,'samples':support.size,'accepted':int(support.sum()),'all_depth_xy_accepted':int(support.all(0).sum()),'failure_codes':{str(i):int((failure==i).sum()) for i in range(6)},'failure_key':{'0':'none','1':'native_geometry_invalid','2':'fine_geometry_support_or_field_escape','3':'singular_condition_fold_or_wrong_normal','4':'basin_or_depth_bound','5':'not_converged_in10'},'fine_representative_field_half_open_xy':[urange,vrange],'native_xywh':[cx-136,cy-136,272,272],'native_j_half_open':[4,24],'bounds_accepted':bounds,'residual_max_accepted':float(goodrr.max()) if len(goodrr) else None,'residual_quantiles_accepted':np.quantile(goodrr,[0,.5,.95,1]).tolist() if len(goodrr) else None,'max_abs_query_movement_accepted':np.max(np.abs(dd[support]),axis=0).tolist() if support.any() else None,'max_iterations_accepted':int(steps[support].max()) if support.any() else None,'center_native_basis_columns_depth_y_x':centerJ.tolist(),'basis_valid_points':int(validJ.sum()),'basis_column_norm_ranges':[np.quantile(np.linalg.norm(JJ[validJ,:,c],axis=1),[0,.5,1]).tolist() for c in range(3)],'basis_signed_determinant_range':np.quantile(np.linalg.det(JJ[validJ]),[0,.5,1]).tolist(),'output_files':[{'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.glob('*.npy'))]}
        write(out/'RESULTS.json',summary);summaries.append(summary);write(ROOT/'RESULTS.json',{'sites':summaries,'complete':len(summaries)==3});print(json.dumps({k:summary[k] for k in ('site','seconds','samples','accepted','failure_codes','bounds_accepted')}),flush=True)
    print('complete',flush=True)

if __name__=='__main__':main()
