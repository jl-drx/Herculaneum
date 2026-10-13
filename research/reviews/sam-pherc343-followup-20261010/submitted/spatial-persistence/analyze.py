import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,itertools,csv
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
R=Path('/workspace/shared/pherc343-input');O=Path('/workspace/shared/pherc343-persistence-output');B=R/'restored/reports';P=Path('/workspace/shared/pherc343-review-output')
prior=json.loads((P/'metrics.json').read_text());plan=json.loads((O/'PLAN.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
valid=np.load(R/'valid.npy');assert sha(R/'valid.npy')=='1905c0d05656b4472a533e00b1c140592c7b33ed7a8bb7780873517f747e41fc'
assert hashlib.sha256(np.packbits(valid[105:2560,380:6021]).tobytes()).hexdigest()=='f60549218edb9871e5acb8930e176a09fe1aaafef28c10a2261014f70a58dcbc'
f=next((B/'fortyfifth-experiment/ph343-mesh-replay').glob('*npz'));s=np.load(f)['source'];own=np.isin(np.repeat(np.repeat(s,20,0),20,1)[:2560,:6720],[1,4]);maps={};hashes={}
for n,t in [('dnative','dnative'),('seed42','hybrid_3d2d-seed42'),('seed43','hybrid_3d2d-seed43')]:
 for phase in ['plain','padded']:
  f=next((B/f'fortysixth-experiment/work/maps_{phase}').glob(f'*__{t}__*.tif'));a=np.asarray(Image.open(f));r=json.loads(Path(str(f)+'.receipt.json').read_text());assert sha(f)==r['output']['sha256'];assert hashlib.sha256(a.tobytes()).hexdigest()==r['decoded_sha256'];assert r in prior['receipts'];hashes[n+'_'+phase]={'encoded':sha(f),'decoded':r['decoded_sha256']};maps[n+'_'+phase]=a if phase=='plain' else a[40:2600,200:6920]
 maps[n+'_merged']=np.where(own,maps[n+'_padded'],maps[n+'_plain'])
names=['dnative','seed42','seed43'];pairs=[(n+'_plain',n+'_padded',n+' phase') for n in names]+[(a+'_merged',b+'_merged',a+' / '+b) for a,b in itertools.combinations(names,2)]
supports=[{'id':'published_crop','bounds_xyxy':[380,105,6021,2560]}]+prior['regions']
def select(a,v,q):
 h=np.bincount(a[v],minlength=256);N=int(v.sum());target=q*N;k=int(np.ceil(target));t=int(np.flatnonzero(h[::-1].cumsum()>=k)[0]);t=255-t;above=int(h[t+1:].sum());ties=int(h[t]);w=(target-above)/ties;b=(a>=t)&v
 return b, {'threshold_uint8':t,'n':N,'selected':int(b.sum()),'selected_fraction':float(b.sum()/N),'strict_above_count':above,'boundary_tie_count':ties,'boundary_weight':w,'exact_mass':target},np.where(v,np.where(a>t,1,np.where(a==t,w,0)),0).astype(np.float32)
def topology(b):
 lab,n=ndi.label(b,np.ones((3,3)));sizes=np.bincount(lab.ravel())[1:];sizes.sort();assert sizes.sum()==b.sum();ys,xs=np.nonzero(b);return {'components_8':int(n),'largest_component_share':float(sizes[-1]/b.sum()),'components_covering90pct':int(np.searchsorted(sizes[::-1].cumsum(),b.sum()*.9)+1),'centroid_xy_local':[float(xs.mean()),float(ys.mean())]}
def comparison(a,b,wa,wb,ta,tb):
 na=int(a.sum());nb=int(b.sum());inter=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b));dice=2*inter/(na+nb);jac=inter/union
 # Second overlap calculation from selected flattened indices, independent of boolean intersection.
 ia=np.flatnonzero(a);ib=np.flatnonzero(b);assert np.intersect1d(ia,ib,assume_unique=True).size==inter;assert union==na+nb-inter
 da=ndi.distance_transform_edt(~b)[a];db=ndi.distance_transform_edt(~a)[b];dd=np.concatenate([da,db]);c1=np.array(ta['centroid_xy_local']);c2=np.array(tb['centroid_xy_local']);wi=float(np.minimum(wa,wb).sum(dtype=np.float64));mass=float(wa.sum(dtype=np.float64)+wb.sum(dtype=np.float64));pa=na/ta['n'];pb=nb/tb['n']
 return {'within_radius_native_px':{str(rad):{'pooled_symmetric':float((dd<=rad).mean()),'A_to_B':float((da<=rad).mean()),'B_to_A':float((db<=rad).mean())} for rad in [0,2,4,8]},'intersection':inter,'union':union,'dice':dice,'jaccard':jac,'retained_A_fraction':inter/na,'retained_B_fraction':inter/nb,'weighted_dice_exact_mass':2*wi/mass,'symmetric_nearest_mean_px':float(dd.mean()),'symmetric_nearest_p95_px':float(np.percentile(dd,95)),'centroid_displacement_px':float(np.linalg.norm(c1-c2)),'independent_uniform_occupancy_dice_scale_only':2*pa*pb/(pa+pb),'largest_component_share_absolute_change':abs(ta['largest_component_share']-tb['largest_component_share'])}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
sheets=[Image.new('RGB',(1800,1360),'white') for _ in range(3)]
for sheet in sheets:
 d=ImageDraw.Draw(sheet);d.text((15,8),'Top 10% within each box; ties included. White = both; orange = A only; blue = B only. Invalid = gray.',font=font,fill='black');d.text((15,31),'Native-pixel selections shown in published orientation (both axes reversed). Rows are pre-existing boxes; no shape/letter labels.',font=small,fill='black')
results=[]
for si,r in enumerate(supports):
 x0,y0,x1,y1=r['bounds_xyxy'];sl=np.s_[y0:y1,x0:x1];v=valid[sl];rr={'id':r['id'],'bounds_xyxy':[x0,y0,x1,y1],'valid_pixels':int(v.sum()),'fractions':[]}
 for q in plan['fractions']:
  masks={};weights={};meta={}
  for name,a in maps.items():
   b,m,w=select(a[sl],v,q);m.update(topology(b));assert abs(w.sum(dtype=np.float64)-q*v.sum())<max(.1,1e-6*q*v.sum());assert 2*np.count_nonzero(b&b)/(2*b.sum())==1;assert np.array_equal(b|b,b);m['self_control']={'dice':1,'jaccard':1,'centroid_displacement_px':0};masks[name]=b;weights[name]=w;meta[name]=m
  cm={}
  for col,(a,b,label) in enumerate(pairs):
   m=comparison(masks[a],masks[b],weights[a],weights[b],meta[a],meta[b]);cm[label]=m
   if si and q==.1:
    ar=np.zeros((*v.shape,3),dtype=np.uint8);ar[:]=[15,19,25];ar[~v]=[90,90,90];ar[masks[a]&~masks[b]]=[240,145,35];ar[masks[b]&~masks[a]]=[35,155,240];ar[masks[a]&masks[b]]=[255,255,255];im=Image.fromarray(ar[::-1,::-1]);im.thumbnail((282,155),Image.Resampling.NEAREST);page=(si-1)//6;row=(si-1)%6;ox=col*300;oy=60+row*215;d=ImageDraw.Draw(sheets[page]);d.text((ox+8,oy),f'Box {r["id"]}: {label}',font=small,fill='black');d.text((ox+8,oy+19),f'Dice {m["dice"]:.2f}; d95 {m["symmetric_nearest_p95_px"]:.1f}px',font=small,fill='black');sheets[page].paste(im,(ox+(300-im.width)//2,oy+42+(155-im.height)//2))
  rr['fractions'].append({'q':q,'selections':meta,'comparisons':cm})
 results.append(rr);print('Finished',r['id'],flush=True)
for i,im in enumerate(sheets):im.save(O/f'all_boxes_overlap_{i+1}.png')
report={'supplement_plan':json.loads((O/'SUPPLEMENT_PLAN.json').read_text()),'plan':plan,'plan_sha256':sha(O/'PLAN.json'),'input_hashes':hashes,'prior_metrics_sha256':sha(P/'metrics.json'),'ownership_source_sha256':sha(next((B/'fortyfifth-experiment/ph343-mesh-replay').glob('*npz'))),'provenance':prior['provenance'],'results':results,'verification':'All source hashes, independent index intersections, union identities, weighted-mass checks, component area sums and self overlaps passed.'}
(O/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
with (O/'comparisons.csv').open('w') as f:
 rows=[dict(support=r['id'],q=q['q'],pair=p,**{k:v for k,v in m.items() if k!='within_radius_native_px'},**{f'within_{rad}px_{direction}':value for rad,ds in m['within_radius_native_px'].items() for direction,value in ds.items()}) for r in results for q in r['fractions'] for p,m in q['comparisons'].items()];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Compact all-support overlap heatmap; rows preserve box order and columns all predeclared ranks.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,6,figsize=(17,8),sharey=True)
for ax,(_,_,p) in zip(axs,pairs):
 a=np.array([[f['comparisons'][p]['dice'] for f in r['fractions']] for r in results]);im=ax.imshow(a,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(3),['5%','10%','20%']);ax.set_title(p,fontsize=10)
 for i in range(19):
  for j in range(3):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if a[i,j]<.5 else 'black')
axs[0].set_yticks(range(19),['Valid crop']+[f'Box {r["id"]}' for r in results[1:]]);fig.suptitle('Spatial overlap of rank-selected responses: binary Dice, boundary ties included\nSame valid support within each comparison; descriptive, no truth labels or independent-pixel inference',fontsize=12);fig.tight_layout(rect=[0,0,.95,.93]);fig.colorbar(im,ax=axs,location='right',fraction=.018,pad=.02);fig.savefig(O/'all_supports_dice.png',dpi=150);plt.close(fig)
