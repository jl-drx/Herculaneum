from pathlib import Path
import hashlib,json,csv,itertools
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
R=Path('/workspace/shared/pherc343-input'); B=R/'restored/reports'; M=Path('/workspace/shared/pherc343-independent-metadata'); O=Path('/workspace/shared/pherc343-review-output');O.mkdir(exist_ok=True)
S=M/'research/lab/reports/fortyfifth-experiment/source-evidence/pherc343'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decoded(a):return hashlib.sha256(a.tobytes()).hexdigest()
for filename,expected in [('pherc343-results-001.tar.gz','621d774c4b9e20e4ea8d3b5fa9e8d5394fbc15803a97e38d046cd9789fa9cf06'),('pherc343-reference-001.tar.gz','58eac58201435f29c5321d2793b74c4853fdceac263e6c6cc74ac7686263c232')]:assert sha(R/filename)==expected
valid=np.load(R/'valid.npy',allow_pickle=False);assert valid.shape==(2560,6720) and valid.dtype==bool
assert sha(R/'valid.npy')=='1905c0d05656b4472a533e00b1c140592c7b33ed7a8bb7780873517f747e41fc'
panelmask=valid[105:2560,380:6021]
assert decoded(np.packbits(panelmask))=='f60549218edb9871e5acb8930e176a09fe1aaafef28c10a2261014f70a58dcbc'
sourcefile=next((B/'fortyfifth-experiment/ph343-mesh-replay').glob('*npz'));source=np.load(sourcefile,allow_pickle=False)['source'];source=np.repeat(np.repeat(source,20,0),20,1)[:2560,:6720];ownership=np.isin(source,[1,4]);assert ownership.shape==valid.shape
refpath=B/'fortyfifth-experiment/source-evidence/concat_w047-w048_R5B2_z9500-11000.png';assert sha(refpath)=='59307c84ea65091de1bea14a3b75d44069bfe1fb498b3445f698ec3d1b8c1c78'
ref=np.asarray(Image.open(refpath).convert('RGB'))[182:2637,173:5814];assert ref.shape==(2455,5641,3)
def histstats(x):
 h=np.bincount(x.ravel(),minlength=256); n=h.sum(); c=h.cumsum();return {'n':int(n),'mean':float(np.dot(h,np.arange(256))/n),'median':int(np.searchsorted(c,n*.5)),'p95':int(np.searchsorted(c,n*.95)),'p99':int(np.searchsorted(c,n*.99)),'max':int(x.max())}
def compare(a,b,mask):
 x=a[mask].astype(np.float64);y=b[mask].astype(np.float64);d=np.abs(x-y).astype(np.uint8);s=histstats(d);s['pearson']=float(np.corrcoef(x,y)[0,1]);s['identical_fraction']=float((d==0).mean());return s
cropmask=np.zeros_like(valid);cropmask[105:2560,380:6021]=panelmask
edges=[];boundary=np.zeros_like(valid)
for axis in (0,1):
 a=[slice(None)]*2;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None);a=tuple(a);b=tuple(b);e=(ownership[a]!=ownership[b])&valid[a]&valid[b];edges.append((a,b,e));boundary[a]|=e;boundary[b]|=e
near=distance_transform_edt(~boundary)<=20
maps={};merged={};receipts=[];stats={}
for name,token in [('dnative','dnative'),('seed42','hybrid_3d2d-seed42'),('seed43','hybrid_3d2d-seed43')]:
 maps[name]={}
 for phase in ['plain','padded']:
  f=next((B/f'fortysixth-experiment/work/maps_{phase}').glob(f'*__{token}__*.tif'));r=json.loads(Path(str(f)+'.receipt.json').read_text());a=np.asarray(Image.open(f));assert a.dtype==np.uint8 and list(a.shape)==r['shape'];assert sha(f)==r['output']['sha256'] and f.stat().st_size==r['output']['bytes'] and decoded(a)==r['decoded_sha256'] and r['status']=='PASS';receipts.append(r)
  maps[name][phase]=a if phase=='plain' else a[40:2600,200:6920]
 a=maps[name]['plain'];b=maps[name]['padded'];assert a.shape==b.shape==valid.shape
 merged[name]=np.where(ownership,b,a)
 gradients={}
 for label,v in [('plain',a),('padded',b),('merged',merged[name])]:
  vals=np.concatenate([np.abs(v[t].astype(np.int16)-v[u].astype(np.int16))[e] for t,u,e in edges]);gradients[label]=histstats(vals)
 stats[name]={'full_valid_phase':compare(a,b,valid),'published_valid_phase':compare(a,b,cropmask),'near_ownership_boundary_phase':compare(a,b,cropmask&near),'away_ownership_boundary_phase':compare(a,b,cropmask&~near),'ownership_switch_gradients':gradients,'merged_display_on_fraction':float((merged[name][cropmask]>64).mean())}
agreement={f'{a}:{b}':compare(merged[a],merged[b],cropmask) for a,b in itertools.combinations(merged,2)}
boxes=json.loads((S/'inputs/concat_w047-w048_R5B2_z9500-11000_letter_boxes.json').read_text())['letters'];rows=[]
for box in boxes:
 x0,y0,x1,y1=box['box'];x0,y0=int(np.floor(x0)),int(np.floor(y0));x1,y1=int(np.ceil(x1)),int(np.ceil(y1));sl=(slice(y0,y1),slice(x0,x1));v=valid[sl];row={'id':box['id'],'bounds_xyxy':[x0,y0,x1,y1],'valid_pixels':int(v.sum()),'valid_fraction':float(v.mean()),'models':{},'pairwise':{}}
 for name in maps:
  row['models'][name]={'phase':compare(maps[name]['plain'][sl],maps[name]['padded'][sl],v),'merged_mean':float(merged[name][sl][v].mean()),'merged_display_on_fraction':float((merged[name][sl][v]>64).mean())}
 for a,b in itertools.combinations(merged,2):row['pairwise'][f'{a}:{b}']=compare(merged[a][sl],merged[b][sl],v)
 rows.append(row)
report={'provenance':json.loads((R/'PROVENANCE.json').read_text()),'checks':{'archive_and_payload_sha256':'PASS','all_six_file_and_decoded_receipts':'PASS','reference_png':'PASS','packed_published_valid_mask':'PASS'},'plan':json.loads((M/'ANALYSIS_PLAN.json').read_text()),'valid_pixels':int(valid.sum()),'published_valid_pixels':int(panelmask.sum()),'source_counts':{str(i):int((source==i).sum()) for i in np.unique(source)},'phase_results':stats,'model_agreement':agreement,'regions':rows,'receipts':receipts}
report['ownership_boundary_extent']={'boundary_endpoint_valid_pixels_full':int(boundary.sum()),'boundary_endpoint_valid_fraction_full':float(boundary.sum()/valid.sum()),'boundary_endpoint_valid_pixels_published':int((boundary&cropmask).sum()),'boundary_endpoint_valid_fraction_published':float((boundary&cropmask).sum()/cropmask.sum()),'within20px_published_pixels':int((near&cropmask).sum()),'within20px_published_fraction':float((near&cropmask).sum()/cropmask.sum()),'regions':[]}
for r in rows:
 x0,y0,x1,y1=r['bounds_xyxy'];sl=np.s_[y0:y1,x0:x1];report['ownership_boundary_extent']['regions'].append({'id':r['id'],'boundary_endpoint_pixels':int(boundary[sl].sum()),'within20px_pixels':int((near&valid)[sl].sum()),'within20px_valid_fraction':float((near&valid)[sl].sum()/valid[sl].sum())})
(O/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
with (O/'regions.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['region','valid_pixels','model','phase_mae_uint8','phase_p95_uint8','phase_correlation','merged_mean_uint8','fixed_display_on_fraction'])
 for r in rows:
  for n,v in r['models'].items():w.writerow([r['id'],r['valid_pixels'],n,v['phase']['mean'],v['phase']['p95'],v['phase']['pearson'],v['merged_mean'],v['merged_display_on_fraction']])
def display(a):return np.clip((a.astype(np.float32)-64)/128,0,1)
def rgb(a,v):
 b=np.repeat((a*255).astype(np.uint8)[...,None],3,axis=2);b[~v]=[35,45,65];return Image.fromarray(b)
def put(canvas,im,x,y,w,h,title):
 d=ImageDraw.Draw(canvas);d.text((x+4,y+2),title,fill='black');im.thumbnail((w,h-24));canvas.paste(im,(x+(w-im.width)//2,y+24+(h-24-im.height)//2))
overview=Image.new('RGB',(1800,990),'white');put(overview,Image.fromarray(ref),0,0,900,330,'Published four-model composite; not ground truth')
for n,pos in zip(merged,[(900,0),(0,330),(900,330)]):put(overview,rgb(display(merged[n][105:2560,380:6021][::-1,::-1]),panelmask[::-1,::-1]),*pos,900,330,n+': merged, fixed display 64..192')
maxdelta=np.maximum.reduce([np.abs(v['plain'].astype(np.int16)-v['padded'].astype(np.int16)) for v in maps.values()]);spread=np.maximum.reduce(list(merged.values())).astype(np.int16)-np.minimum.reduce(list(merged.values())).astype(np.int16)
for a,title,x in [(maxdelta,'Maximum phase difference across three models (0..128)',0),(spread,'Range across merged models (0..128)',900)]:put(overview,rgb(np.clip(a[105:2560,380:6021][::-1,::-1]/128,0,1),panelmask[::-1,::-1]),x,660,900,330,title)
overview.save(O/'overview.png')
rawref=ref[::-1,::-1]
for page in range(3):
 sheet=Image.new('RGB',(1500,1200),'white')
 for j,r in enumerate(rows[page*6:page*6+6]):
  x0,y0,x1,y1=r['bounds_xyxy'];sl=(slice(y0,y1),slice(x0,x1));v=valid[sl][::-1,::-1]
  images=[Image.fromarray(rawref[y0-105:y1-105,x0-380:x1-380][::-1,::-1])]+[rgb(display(merged[n][sl][::-1,::-1]),v) for n in merged]+[rgb(np.clip(maxdelta[sl][::-1,::-1]/128,0,1),v)]
  for col,(im,title) in enumerate(zip(images,['Reference','dnative','seed42','seed43','Max phase delta'])):put(sheet,im,col*300,j*200,300,200,f'Region {r["id"]}: {title}')
 sheet.save(O/f'regions_{page+1}.png')
print(json.dumps({k:report[k] for k in ['checks','valid_pixels','published_valid_pixels','phase_results','model_agreement']},indent=2))
