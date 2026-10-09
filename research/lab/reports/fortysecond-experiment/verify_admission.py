"""Post-run audit using raw chunk comparison and direct Pearson calculations.

Does not call production alignment, peak, mask or assembly implementations.
No new science, acquisition or threshold selection occurs here.
"""
import hashlib,itertools,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter,map_coordinates
P=Path(__file__).resolve().parent
def check(ok,message):
 if not ok:raise AssertionError(message)
def read(name):return json.loads((P/name).read_text())
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),sha256=h,size_bytes=p.stat().st_size)
def verify(item):check(pin(item['path'])==item,'Changed pin '+item['path'])
def metric(s):
 loc=np.array(np.unravel_index(s.argmax(),s.shape));value=float(s[tuple(loc)]);bound=bool(np.any((loc==0)|(loc==16)));delta=np.zeros(3)
 if not bound:
  for axis in range(3):
   left=loc.copy();right=loc.copy();left[axis]-=1;right[axis]+=1
   a,c=float(s[tuple(left)]),float(s[tuple(right)]);curvature=a+c-2*value
   if curvature<0:delta[axis]=max(-.5,min(.5,(a-c)/(2*curvature)))
 competitor=max(float(s[index]) for index in np.ndindex(s.shape) if max(abs(np.array(index)-loc))>3)
 return dict(integer_shift_zyx=(loc-8).tolist(),refined_shift_zyx=(loc-8+delta).tolist(),NCC=value,far_competitor_NCC=competitor,margin=value-competitor,boundary=bound)
def close(a,b,name):check(np.allclose(a,b,atol=1e-8,rtol=1e-8),name)
def main():
 pin_count=0
 for name in ['GEOMETRY_GO.json','CT_GO.json','ALIGNMENT_GO.json','ALIGNMENT_CONTINUATION_GO.json']:
  for item in read(name)['pins']:verify(item);pin_count+=1
 geom=read('geometry/REQUESTS.json');ct=read('ct/COMPLETE.json');selection=read('geometry/SELECTION.json');results=read('alignment/RESULTS.json')
 for r in geom['records']:
  verify(r['body']);check(r['bytes']==r['body']['size_bytes'],'Geometry delivered length')
  check(r['status']==(200 if r['start'] is None else 206),'Explicit full or ranged HTTP response')
 check(sum(r['bytes'] for r in geom['records'])==geom['delivered_bytes']==3672730,'Geometry accounting')
 old=np.stack([tifffile.imread(next(r['body']['path'] for r in geom['records'] if r['tag']=='old-'+a)) for a in 'xyz'],axis=-1)
 check(np.array_equal(old,np.load(P/'geometry/old-xyz.npy')),'Old XYZ TIFF decode')
 centers=[]
 for c in selection['candidates']:
  check(np.array_equal(old[tuple(c['old_grid_yx'])],c['old_xyz']),'Selected mesh coordinate')
  centers.append(c['old_xyz'])
 for a,b in itertools.combinations(centers,2):check(np.linalg.norm(np.array(a)-b)*7.91>=1000,'Physical center spacing')
 requests=ct['records'];check(len(requests)==len({r['url'] for r in requests})==96,'96 distinct chunks')
 check(sum(r['delivered_bytes'] for r in requests)==201326592,'CT byte accounting')
 raw={r['id']:r for r in requests}
 for r in requests:
  verify(r['raw']);check(r['status']==200 and r['raw']['size_bytes']==128**3 and r['content_encoding'] is None,'Raw native uint8 chunk')
 arrays={}
 for r in ct['arrays']:
  verify(r['array']);a=np.load(r['array']['path']);check(a.shape==(256,)*3 and a.dtype==np.uint8,'Block format');arrays[r['id']]=a
  center=np.array(r['center_zyx']);check(np.array_equal(r['start_zyx'],(np.rint(center/128)-1)*128),'Frozen block placement')
  for key in r['chunk_ids']:
   chunk=raw[key];lo=np.array(chunk['index_zyx'])*128-r['start_zyx'];check(np.all((lo==0)|(lo==128)),'Chunk placement')
   view=a[tuple(slice(v,v+128) for v in lo)]
   check(view.tobytes()==Path(chunk['raw']['path']).read_bytes(),'Exact raw-to-assembled voxel parity')
 for a,b in itertools.combinations(ct['arrays'],2):
  if a['volume']==b['volume']:
   check(np.any(np.abs(np.array(a['start_zyx'])-b['start_zyx'])>=256),'Nonoverlapping same-volume blocks')
 A=np.array(json.loads((P.parent/'fortyfirst-experiment/research/PAIR_FEASIBILITY.json').read_text())['forward']['matrix'])
 parts=[]
 for corner in itertools.product(range(2),repeat=3):
  m=np.zeros((32,)*3,bool);m[tuple(slice(5+16*v,11+16*v) for v in corner)]=True;parts.append((sum(corner)%2,m))
 masks=[np.logical_or.reduce([m for role,m in parts if role==0])]+[m for role,m in parts if role==1]
 direct_count=0;max_direct_error=0;coverage_rejected=[];verified_rows=[]
 for c,row in zip(selection['candidates'],results['rows']):
  check(c['id']==row['id'],'Region order')
  blocks={b['volume']:b for b in ct['arrays'] if b['pair']==c['id']}
  xyz=np.stack(np.meshgrid(*[np.arange(-16,16)]*3,indexing='ij'),axis=-1)[...,::-1]+c['old_xyz'];mapped=xyz@A[:,:3].T+A[:,3];local=mapped[...,::-1]-blocks['new']['start_zyx']
  covered=bool(np.all((local>=6)&(local<=249)))
  if not covered:
   check(row['status']=='NOT_ADMITTED_COVERAGE' and row['fit'] is None,'Coverage stop recorded')
   close(row['sample_bounds_zyx'],[local.min((0,1,2)),local.max((0,1,2))],'Coverage bounds');coverage_rejected.append(c['id']);continue
  blurred=gaussian_filter(arrays[blocks['new']['id']].astype(np.float32),(7.91/2.4)/2,truncate=3)
  fine=map_coordinates(blurred,np.moveaxis(local,-1,0),order=1,mode='constant',cval=np.nan,prefilter=False)
  offsets=np.stack(np.meshgrid(*[np.arange(-24,24)]*3,indexing='ij'),axis=-1)
  real_coords=offsets+np.array(c['old_xyz'])[::-1]-blocks['old']['start_zyx']
  real=map_coordinates(arrays[blocks['old']['id']].astype(np.float32),np.moveaxis(real_coords,-1,0),order=1,mode='constant',cval=np.nan,prefilter=False)
  fine-=gaussian_filter(fine,1,truncate=2);real-=gaussian_filter(real,1,truncate=2)
  peaks=[]
  for j,m in enumerate(masks):
   filename=c['id']+('-fit-ncc.npy' if j==0 else f'-check{j-1}-ncc.npy');s=np.load(P/'alignment'/filename)
   check(s.shape==(17,)*3 and np.isfinite(s).all(),'Finite score cube')
   p=metric(s);peaks.append(p);stored=row['fit'] if j==0 else row['checks'][j-1]
   for k,v in p.items():close(stored[k],v,'Independent score peak '+k)
   t=fine[m].astype(float);t-=t.mean()
   indices={tuple(np.array(p['integer_shift_zyx'])+8),(8,8,8),(0,0,0),(16,16,16),(3,12,6)}
   for index in indices:
    sample=real[tuple(slice(v,v+32) for v in index)][m].astype(float);sample-=sample.mean()
    direct=float(np.dot(t,sample)/(np.linalg.norm(t)*np.linalg.norm(sample)))
    error=abs(direct-s[index]);check(error<1e-8,'Direct Pearson vs FFT NCC');max_direct_error=max(max_direct_error,error);direct_count+=1
  y,x=c['old_grid_yx'];u=(old[y,x+2]-old[y,x-2]);v=(old[y+2,x]-old[y-2,x]);u/=np.linalg.norm(u);v-=u*np.dot(v,u);v/=np.linalg.norm(v);n=np.cross(u,v).astype(float)
  residuals=[]
  for p,r in zip(peaks[1:],row['checks']):
   d=(np.array(p['refined_shift_zyx'])-peaks[0]['refined_shift_zyx'])[::-1];normal=float(d@n);tangent=float(np.linalg.norm(d-normal*n));residuals.append(tangent)
   close(r['normal_residual_coarse_voxels'],normal,'Normal proxy residual');close(r['tangential_residual_coarse_voxels'],tangent,'Tangential proxy residual')
  passed=not peaks[0]['boundary'] and all(not p['boundary'] and p['NCC']>=.5 and p['margin']>=.05 for p in peaks[1:]) and np.median(residuals)<=.5 and max(residuals)<=1
  check(bool(passed)==row['numerical_screen_passes']==False,'Frozen screen replay')
  verified_rows.append(c['id'])
 review=read('REFERENCE_REVIEW.json');check(review['seconds_since_acquisition_GO']<=2700,'Time budget');check(not any(review[k] for k in ['native_learning_admitted','synthetic_coarse_learning_admitted','real_coarse_learning_admitted']),'No learning admitted')
 check(ct['new_model_calls']==results['new_model_calls']==review['new_model_calls']==0,'Reported model-call accounting')
 out=dict(status='PASS',created_at_utc=datetime.now(timezone.utc).isoformat(),scope='Independent implementation of audit arithmetic; same assistant and scientific libraries, not independent physical or expert validation',pinned_inputs_verified=pin_count,geometry_responses_verified=len(geom['records']),CT_chunks_byte_compared=96,assembled_voxels_compared=12*256**3,disjoint_blocks=12,score_cubes_checked=20,direct_Pearson_values_checked=direct_count,maximum_direct_Pearson_error=max_direct_error,correlation_screen_rows_reproduced=verified_rows,coverage_rejections_reproduced=coverage_rejected,reference_labels_created=0,new_model_calls=0,verifier=pin(__file__))
 with (P/'ADMISSION_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
