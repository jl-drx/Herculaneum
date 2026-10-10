import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np,json,hashlib
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/workspace/shared/pherc343-grid-output');R=Path('/workspace/shared/pherc343-input'); B=R/'restored/reports/fortysixth-experiment/work'
assert hashlib.sha256((O/'source/infer.py').read_bytes()).hexdigest()=='979aed8e1bf047bea8e30bd0bbe55538556289c72b5914bbbbecf355e5940389'
prior=json.loads(Path('/workspace/shared/pherc343-review-output/metrics.json').read_text())
v=np.load(R/'valid.npy');crop=np.zeros_like(v);crop[105:2560,380:6021]=v[105:2560,380:6021];inside=crop.copy();inside[:128]=False;inside[2432:]=False;inside[:,:128]=False;inside[:,6592:]=False
boxes={};union=np.zeros_like(v)
for r in prior['regions']:
 x0,y0,x1,y1=r['bounds_xyxy'];m=np.zeros_like(v);m[y0:y1,x0:x1]=v[y0:y1,x0:x1];boxes['box'+str(r['id'])]=m;union|=m
Y,X=np.indices(v.shape,dtype=np.int16);phase=(Y%64//8)*8+X%64//8
# Exact finite-axis starts, including appended boundary aligned positions.
def starts(n):
 a=list(range(0,n-128+1,64));return np.array(a+([] if a[-1]==n-128 else [n-128]))
def distance(n,pad):
 q=np.arange(n);return np.min(np.abs(q[:,None]-(starts(n+pad)-pad)[None,:]),axis=1)
dists={p:np.minimum(distance(2560,dy)[:,None],distance(6720,dx)[None,:]) for p,dy,dx in [('plain',0,0),('padded',40,200)]}
def summary(d,s,amp,m):
 z=d[m];out={'n':int(m.sum()),'mae':float(z.mean()),'mean_signed_padded_minus_plain':float(s[m].mean()),'grid':{}}
 for p,dist in dists.items():
  near=m&(dist<=4);far=m&(dist>=13)
  def stat(mask):return {'n':int(mask.sum()),'mae':float(d[mask].mean()) if mask.any() else None,'error_share':float(d[mask].sum()/z.sum())}
  out['grid'][p]={'near':stat(near),'far':stat(far),'bands':[{'distance_min':lo,'distance_max':hi,**stat(m&(dist>=lo)&(dist<=hi))} for lo,hi in [(0,4),(5,12),(13,20),(21,32),(33,128)]], 'amplitude':[]}
  for lo,hi in [(0,64),(64,128),(128,192),(192,256)]:
   a=(amp>=lo)&(amp<hi)
   out['grid'][p]['amplitude'].append({'min':lo,'max_exclusive':hi,'near':stat(near&a),'far':stat(far&a)})
 return out
out={'plan':json.loads((O/'PLAN.json').read_text()),'models':{}}
fig,axs=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
for col,(name,token) in enumerate([('dnative','dnative'),('seed42','hybrid_3d2d-seed42'),('seed43','hybrid_3d2d-seed43')]):
 arr=[]
 for p in ['plain','padded']:
  f=next((B/f'maps_{p}').glob(f'*__{token}__*.tif'));a=np.array(Image.open(f));r=json.loads(Path(str(f)+'.receipt.json').read_text());assert hashlib.sha256(a.tobytes()).hexdigest()==r['decoded_sha256'];arr.append(a if p=='plain' else a[40:2600,200:6920])
 a,b=arr;s=b.astype(np.int16)-a.astype(np.int16);d=np.abs(s);amp=(a.astype(np.float32)+b)/2
 pop={'crop':crop,'interior_crop':inside,'all18_union':union,**boxes};rec={k:summary(d,s,amp,m) for k,m in pop.items()}
 # binned phase, interior only: positions are verified coordinates, no search
 counts=np.bincount(phase[inside],minlength=64);means=np.bincount(phase[inside],weights=d[inside],minlength=64)/counts
 rec['phase8x8']={'count':counts.tolist(),'mae':means.tolist(),'signed_mean':(np.bincount(phase[inside],weights=s[inside],minlength=64)/counts).tolist()};out['models'][name]=rec
 assert abs(rec['crop']['mae']-prior['phase_results'][name]['published_valid_phase']['mean'])<1e-10
 ax=axs[0,col];im=ax.imshow(means.reshape(8,8)/rec['interior_crop']['mae'],origin='lower',extent=[0,64,0,64],vmin=.75,vmax=1.25,cmap='coolwarm');ax.set_title(name+' | error / interior mean');ax.set_xlabel('Native column modulo 64');ax.set_ylabel('Native row modulo 64')
 ax=axs[1,col]
 for pname,style in [('crop','-'),('all18_union','--')]:
  for grid,color in [('plain','#3366aa'),('padded','#d17c17')]:
   rr=rec[pname]['grid'][grid];ax.plot([2,8.5,16.5,26.5],[x['mae'] for x in rr['bands'][:4]],style+'o',color=color,label=pname.replace('all18_union','18-box union')+' / '+grid)
 ax.set(xlabel='Distance to potential tile-start line (px)',ylabel='Mean absolute phase difference (uint8)',ylim=(0,None));ax.grid(alpha=.2)
 if col==0:ax.legend(fontsize=7)
fig.colorbar(im,ax=list(axs[0,:]),label='Relative phase error',shrink=.8)
fig.suptitle('Padding sensitivity is not confined to tile-start lines\n128px patches, 64px stride; descriptive spatial association, not an accuracy or causal test',fontsize=13)
fig.savefig(O/'grid_phase_summary.png',dpi=160)
(O/'metrics.json').write_text(json.dumps(out,indent=2)+'\n')
for name,rec in out['models'].items():
 print(name,'phase range',min(rec['phase8x8']['mae']),max(rec['phase8x8']['mae']))
 for pop in ['crop','interior_crop','all18_union']:
  print(pop,'mae',rec[pop]['mae'],{g:{t:rec[pop]['grid'][g][t] for t in ['near','far']} for g in dists})
