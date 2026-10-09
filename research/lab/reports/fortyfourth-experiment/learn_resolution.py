"""Prespecified CPU-only, physical-parent-held-out paired observation comparison."""
import os,time,copy
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
from scipy.ndimage import distance_transform_edt,uniform_filter
import torch
from torch import nn
from torch.nn import functional as F
from pixel_http import P,pin,save,need,now,json
SEED=20261009
STEPS=2048
QUALITIES=['native','coarse4']
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def degrade(a):
 v=torch.from_numpy(np.ascontiguousarray(a))[None,None].float()
 return F.interpolate(F.avg_pool2d(v,4),size=a.shape,mode='bilinear',align_corners=False)[0,0].numpy()
def normalize(a):
 lo,med,hi=np.quantile(a,[.01,.5,.99]);need(hi>lo,'Nonconstant height')
 return np.clip((a-med)/(hi-lo),-4,4).astype(np.float32),dict(lower=float(lo),median=float(med),upper=float(hi))
def prepare():
 data=[]
 for row in json.loads((P/'CROP_SELECTION.json').read_text()):
  cid=row['case'];base=P/'crops';z=np.load(base/f'{cid}-height.npy');finite=np.isfinite(z);indices=distance_transform_edt(~finite,return_distances=False,return_indices=True);filled=z[tuple(indices)];need(np.isfinite(filled).all(),'Filled context finite')
  support=np.load(base/f'{cid}-common-support.npy');v=finite.reshape(384,4,384,4).all((1,3)).astype(np.float32);interpolated=F.interpolate(torch.from_numpy(v)[None,None],size=z.shape,mode='bilinear',align_corners=False)[0,0].numpy();support&=interpolated==1
  support[:32]=False;support[-32:]=False;support[:,:32]=False;support[:,-32:]=False
  images={};stats={}
  for quality,a in [('native',filled),('coarse4',degrade(filled))]:images[quality],stats[quality]=normalize(a)
  label=(np.load(base/f'{cid}-label.npy')!=0).astype(np.float32);need((support&(label>0)).sum()>=10000 and (support&(label==0)).sum()>=10000,'Final support classes')
  eligible=support.copy();eligible[:96]=False;eligible[-96:]=False;eligible[:,:96]=False;eligible[:,-96:]=False
  pools=[np.argwhere(eligible),np.argwhere(eligible&(label>0))];need(all(len(v)>0 for v in pools),'Sample pools')
  data.append(dict(case=cid,parent=row['parent'],images=images,stats=stats,label=label,support=support,pools=pools,hole_density=uniform_filter((~finite).astype(np.float32),32,mode='reflect')))
 return data
class Block(nn.Sequential):
 def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.a=Block(1,8);self.b=Block(8,16);self.c=Block(16,32);self.d=Block(48,16);self.e=Block(24,8);self.out=nn.Conv2d(8,1,1)
 def forward(self,x):
  a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));e=self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));return self.out(e)
def ranking_metrics(truth,score):
 order=np.argsort(-score,kind='stable');t=truth[order].astype(np.float64);v=score[order];ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1];tp=np.cumsum(t)[ends];fp=(ends+1)-tp;positives=float(t.sum());negatives=len(t)-positives;need(positives>0 and negatives>0,'Both metric classes')
 ap=float(np.sum(np.diff(np.r_[0.,tp])/positives * tp/(ends+1)))
 auc=float(np.trapezoid(np.r_[0.,tp]/positives,np.r_[0.,fp]/negatives))
 return ap,auc

def metrics(y,s,m):
 truth=y[m].astype(np.uint8);score=s[m].astype(np.float64);need(np.isfinite(score).all(),'Finite predictions');pred=score>=.5;tp=int((pred&(truth>0)).sum());den=int(pred.sum()+truth.sum())
 ap,auc=ranking_metrics(truth,score)
 return dict(ap=ap,auroc=auc,dice_at_half=float(2*tp/den) if den else 1.,brier=float(np.mean((score-truth)**2)),prevalence=float(truth.mean()),n=int(len(truth)),positive=int(truth.sum()))
@torch.inference_mode()
def predict(model,a):
 model.eval();pad=np.pad(a,32,mode='reflect');out=np.empty_like(a);coords=[(y,x) for y in range(0,a.shape[0],128) for x in range(0,a.shape[1],128)]
 for start in range(0,len(coords),8):
  group=coords[start:start+8];v=np.stack([pad[y:y+192,x:x+192] for y,x in group]);scores=model(torch.from_numpy(v[:,None])).sigmoid()[:,0,32:160,32:160].numpy()
  for (y,x),s in zip(group,scores):out[y:y+128,x:x+128]=s
 return out

def draw_batch(data,train_ids,quality,rng):
 xs=[];ys=[];ms=[]
 for j in range(8):
  d=data[train_ids[int(rng.integers(len(train_ids)))]];pool=d['pools'][int(rng.integers(2))];cy,cx=pool[int(rng.integers(len(pool)))];yslice=slice(cy-96,cy+96);xslice=slice(cx-96,cx+96)
  a=d['images'][quality][yslice,xslice];y=d['label'][yslice,xslice];m=d['support'][yslice,xslice]
  for axis in [0,1]:
   if rng.integers(2):a=np.flip(a,axis);y=np.flip(y,axis);m=np.flip(m,axis)
  if rng.integers(2):a=a.T;y=y.T;m=m.T
  xs.append(a);ys.append(y[32:160,32:160]);ms.append(m[32:160,32:160])
 return tuple(torch.from_numpy(np.stack(v).copy())[:,None] for v in [xs,ys,ms])
def posterior(train_features,train_y,test):
 f=np.concatenate(train_features);y=np.concatenate(train_y);edges=np.unique(np.quantile(f,np.linspace(0,1,21))[1:-1]);idx=np.searchsorted(edges,f,side='right');counts=np.bincount(idx,minlength=len(edges)+1);pos=np.bincount(idx,weights=y,minlength=len(edges)+1);p=(pos+1)/(counts+2);return p[np.searchsorted(edges,test,side='right')].astype(np.float32),dict(edges=edges.tolist(),positive=pos.tolist(),count=counts.tolist())
def controls(data,train_ids,held):
 results=[];d=data[held];train_y=[data[i]['label'][data[i]['support']] for i in train_ids];prev=float(np.concatenate(train_y).mean())
 for q in QUALITIES:
  feat=[uniform_filter(v['images'][q],32,mode='reflect') for v in data]
  for name,fs in [('mean_height',feat),('hole_density',[v['hole_density'] for v in data])]:
   score,fit=posterior([fs[i][data[i]['support']] for i in train_ids],train_y,fs[held]);results.append(dict(quality=q,name=name,metrics=metrics(d['label'],score,d['support']),fit=fit))
  results.append(dict(quality=q,name='constant',metrics=metrics(d['label'],np.full(d['label'].shape,prev,np.float32),d['support']),training_prevalence=prev))
 return results

def main():
 go=json.loads((P/'TRAINING_GO.json').read_text())
 for p in go['pins']:need(pin(p['path'])==p,'Frozen training pin')
 run=P/'training';run.mkdir(exist_ok=False);data=prepare();stats=[]
 for d in data:
  np.save(run/f"{d['case']}-support.npy",d['support']);stats.append(dict(case=d['case'],parent=d['parent'],normalization=d['stats'],support=int(d['support'].sum()),positive=int(d['label'][d['support']].sum())))
 save(run/'input-stats.json',stats);all_results=[];start_time=time.monotonic()
 for held in range(3):
  train_ids=[i for i in range(3) if i!=held];need(data[held]['parent'] not in [data[i]['parent'] for i in train_ids],'Parent holdout');ctrl=controls(data,train_ids,held);save(run/f'fold{held}-controls.json',ctrl)
  paired_initial=None
  for quality in QUALITIES:
   torch.manual_seed(SEED+held);rng=np.random.default_rng(SEED+held);model=Net();initial={k:v.detach().clone() for k,v in model.state_dict().items()}
   if paired_initial is None:paired_initial=initial
   else:need(all(torch.equal(v,paired_initial[k]) for k,v in initial.items()),'Paired initialization equality')
   optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);truth=np.concatenate([data[i]['label'][data[i]['support']] for i in train_ids]);prevalence=float(truth.mean());pos_weight=torch.tensor((1-prevalence)/prevalence);history=[];fit_history=[];arm_start=time.monotonic()
   arm=f'fold{held}-{quality}';log=(run/f'{arm}-progress.jsonl').open('x');losses=[]
   for step in range(1,STEPS+1):
    model.train();x,y,m=draw_batch(data,train_ids,quality,rng);optimizer.zero_grad(set_to_none=True);logits=model(x)[:,:,32:160,32:160];pixel_loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight,reduction='none');loss=pixel_loss[m].mean();need(torch.isfinite(loss).item(),'Finite optimization');loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
    if step%128==0:
     row=dict(step=step,mean_loss=float(np.mean(losses[-128:])),elapsed_seconds=time.monotonic()-arm_start);history.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(dict(arm=arm,**row)),flush=True)
    if step in [512,1024,2048]:
     fit=[]
     for i in train_ids:
      score=predict(model,data[i]['images'][quality]);fit.append(dict(case=data[i]['case'],metrics=metrics(data[i]['label'],score,data[i]['support'])))
     fit_history.append(dict(step=step,cases=fit));log.write(json.dumps(dict(training_fit=fit_history[-1]))+'\n');log.flush();print(json.dumps(dict(arm=arm,training_fit=fit_history[-1])),flush=True)
   log.close();torch.save(model.state_dict(),run/f'{arm}.pt');evaluation=[]
   for test_q in QUALITIES:
    scores=predict(model,data[held]['images'][test_q]);np.save(run/f'{arm}-test-{test_q}.npy',scores);evaluation.append(dict(quality=test_q,metrics=metrics(data[held]['label'],scores,data[held]['support'])))
   result=dict(arm=arm,held_out=data[held]['case'],train_cases=[data[i]['case'] for i in train_ids],train_quality=quality,parameters=sum(v.numel() for v in model.parameters()),seed=SEED+held,updates=STEPS,prevalence=prevalence,pos_weight=float(pos_weight),history=history,training_fit=fit_history,test=evaluation,elapsed_seconds=time.monotonic()-arm_start,controls=ctrl);save(run/f'{arm}-result.json',result);all_results.append(result);print(json.dumps(dict(completed_arm=arm,test=evaluation)),flush=True)
 save(P/'TRAINING_RESULTS.json',dict(created_at_utc=now(),elapsed_seconds=time.monotonic()-start_time,results=all_results,total_optimizer_updates=STEPS*6));print('ALL_SIX_ARMS_COMPLETE',flush=True)
if __name__=='__main__':main()
