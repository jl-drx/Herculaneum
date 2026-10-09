"""Fixed-mask structural registration; all outputs remain diagnostic."""
import math,time
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.signal import fftconvolve
from common import *
def ncc_grid(template,search,mask):
 t=template.astype(float);r=search.astype(float);m=mask.astype(float);n=m.sum();need(n>0,'Nonempty fixed support')
 centered=(t-(t*m).sum()/n)*m;norm=(centered**2).sum();need(norm>1e-12,'Template structural variance')
 conv=lambda a,b:fftconvolve(a,b[::-1,::-1,::-1],mode='valid')
 numerator=conv(r,centered);s=conv(r,m);ss=conv(r*r,m);denom=np.sqrt(np.maximum(ss-s*s/n,0)*norm)
 return np.divide(numerator,denom,out=np.full_like(numerator,-1),where=denom>1e-12)
def peak(scores):
 p=np.asarray(np.unravel_index(np.argmax(scores),scores.shape));v=float(scores[tuple(p)]);boundary=bool(((p==0)|(p==np.asarray(scores.shape)-1)).any());delta=np.zeros(3)
 if not boundary:
  for k in range(3):
   lo=p.copy();hi=p.copy();lo[k]-=1;hi[k]+=1;a=float(scores[tuple(lo)]);b=v;c=float(scores[tuple(hi)]);den=a-2*b+c
   if den<0:delta[k]=np.clip(.5*(a-c)/den,-.5,.5)
 grid=np.indices(scores.shape);far=(np.max(abs(grid-p[:,None,None,None]),axis=0)>3);second=float(scores[far].max())
 return dict(integer_shift_zyx=(p-8).tolist(),refined_shift_zyx=(p-8+delta).tolist(),NCC=v,far_competitor_NCC=second,margin=v-second,boundary=boundary)
def masks():
 parts=[]
 for z in range(2):
  for y in range(2):
   for x in range(2):
    m=np.zeros((32,32,32),bool);start=[5+16*k for k in [z,y,x]];m[tuple(slice(i,i+6) for i in start)]=True;parts.append(dict(corner_zyx=[z,y,x],role='fit' if (z+y+x)%2==0 else 'check',mask=m))
 return parts
def sample(array,xyz,start_zyx):
 local=xyz[...,::-1]-np.asarray(start_zyx);valid=((local>=0)&(local<=np.asarray(array.shape)-1)).all(-1)
 return map_coordinates(array.astype(np.float32),np.moveaxis(local,-1,0),order=1,mode='constant',cval=np.nan,prefilter=False),valid
def basis(old_xyz,grid_yx):
 y,x=grid_yx;u=old_xyz[y,x+2]-old_xyz[y,x-2];v=old_xyz[y+2,x]-old_xyz[y-2,x];u=u/np.linalg.norm(u);v=v-u*np.dot(v,u);v=v/np.linalg.norm(v);normal=np.cross(u,v);return np.stack([u,v,normal]).astype(float)
def main():
 started=time.monotonic();go=json.loads((HERE/'ALIGNMENT_GO.json').read_text())
 for p in go['pins']:verify(p)
 out=HERE/'alignment';out.mkdir(exist_ok=False)
 selection=json.loads((HERE/'geometry/SELECTION.json').read_text());receipt=json.loads((HERE/'ct/COMPLETE.json').read_text());meta=json.loads((PRIOR/'PAIR_FEASIBILITY.json').read_text());A=np.asarray(meta['forward']['matrix']);oldgeom=np.load(HERE/'geometry/old-xyz.npy');parts=masks();fitmask=np.logical_or.reduce([r['mask'] for r in parts if r['role']=='fit']);rows=[]
 lattice=lambda n,origin:np.moveaxis(np.indices((n,n,n),dtype=float),0,-1)[...,::-1]+np.asarray(origin)-n//2
 for c in selection['candidates']:
  bs={r['volume']:r for r in receipt['arrays'] if r['pair']==c['id']};old=np.load(verify(bs['old']['array']));new=np.load(verify(bs['new']['array']));blur=gaussian_filter(new.astype(np.float32),sigma=(7.91/2.4)/2,truncate=3)
  points=lattice(32,c['old_xyz']);mapped=points@A[:,:3].T+A[:,3];local=mapped[...,::-1]-np.asarray(bs['new']['start_zyx']);need(bool(((local>=6)&(local<=249)).all()),'Blur context and full template coverage')
  fine,valid=sample(blur,mapped,bs['new']['start_zyx']);real,v=sample(old,lattice(48,c['old_xyz']),bs['old']['start_zyx']);need(valid.all() and v.all(),'Explicit structural sampling coverage')
  fine=fine-gaussian_filter(fine,1,truncate=2);real=real-gaussian_filter(real,1,truncate=2)
  scores=ncc_grid(fine,real,fitmask);fitted=peak(scores);np.save(out/(c['id']+'-fit-ncc.npy'),scores,allow_pickle=False);checks=[];axes=basis(oldgeom,c['old_grid_yx']);normal=axes[2]
  for j,part in enumerate(r for r in parts if r['role']=='check'):
   s=ncc_grid(fine,real,part['mask']);r=peak(s);np.save(out/(c['id']+f'-check{j}-ncc.npy'),s,allow_pickle=False)
   delta=(np.asarray(r['refined_shift_zyx'])-fitted['refined_shift_zyx'])[::-1];n=float(np.dot(delta,normal));t=float(np.linalg.norm(delta-n*normal));r.update(corner_zyx=part['corner_zyx'],residual_xyz=delta.tolist(),normal_residual_coarse_voxels=n,tangential_residual_coarse_voxels=t);checks.append(r)
  qualified=not fitted['boundary'] and all(not r['boundary'] and r['NCC']>=.5 and r['margin']>=.05 for r in checks) and np.median([r['tangential_residual_coarse_voxels'] for r in checks])<=.5 and max(r['tangential_residual_coarse_voxels'] for r in checks)<=1
  public_score=float(scores[8,8,8]);row=dict(id=c['id'],nominal_role=c['nominal_role'],public_mapping_fit_NCC=public_score,fit=fitted,checks=checks,numerical_screen_passes=bool(qualified),geometry_basis_old_xyz=axes.tolist(),independent_landmarks_verified=False,wrap_identity_verified=False,physical_registration_admitted=False)
  rows.append(row);print(json.dumps(dict(id=c['id'],public_NCC=public_score,fit=fitted,heldout_NCC=[r['NCC'] for r in checks],tangential_residuals=[r['tangential_residual_coarse_voxels'] for r in checks],screen=bool(qualified))),flush=True)
 save(out/'RESULTS.json',dict(status='DIAGNOSTIC_COMPLETE',created_at_utc=now(),rows=rows,all_six_numerical_screens_pass=all(r['numerical_screen_passes'] for r in rows),physical_registration_established=False,source_reference_labels_established=False,new_model_calls=0,elapsed_seconds=time.monotonic()-started))
if __name__=='__main__':main()
