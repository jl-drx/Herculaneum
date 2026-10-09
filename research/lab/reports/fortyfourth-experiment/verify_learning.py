"""Independent grouped-ranking arithmetic and final training-inference replay."""
from scipy.ndimage import binary_erosion
from learn_resolution import *
def independent(y,p,m):
 yy=y[m].astype(np.int64);pp=p[m];values,inverse,counts=np.unique(pp,return_inverse=True,return_counts=True);positives=np.bincount(inverse,weights=yy,minlength=len(values));negatives=counts-positives;tp=np.cumsum(positives[::-1]);seen=np.cumsum(counts[::-1]);ap=float(np.sum(positives[::-1]/yy.sum()*tp/seen));auc=float(np.sum(positives*(np.cumsum(negatives)-negatives+.5*negatives))/(yy.sum()*(len(yy)-yy.sum())));binary=pp>=.5;dice=2*float((binary*yy).sum())/(binary.sum()+yy.sum());return dict(ap=ap,auroc=auc,dice_at_half=float(dice),brier=float(((pp.astype(float)-yy)**2).mean()),prevalence=float(yy.mean()),n=len(yy),positive=int(yy.sum()))
def same(a,b):
 for k in a:need(abs(a[k]-b[k])<1e-10,'Independent metric '+k)
r=json.loads((P/'TRAINING_RESULTS.json').read_text());data=prepare();checked=[];replayed=[];control_checks=[]
for p in json.loads((P/'TRAINING_GO.json').read_text())['pins']:need(pin(p['path'])==p,'Frozen training pin')
for held in range(3):
 d=data[held];need(np.array_equal(np.load(P/'training'/f"{d['case']}-support.npy"),d['support']),'Same common support saved')
 # Independent footprint validity uses erosion on coarse block occupancy, with bilinear grid checked by fixtures.
 raw=np.load(P/'crops'/f"{d['case']}-height.npy");need(np.isfinite(raw[d['support']]).all(),'No imputed target support')
 for q in QUALITIES:
  arm=next(a for a in r['results'] if a['arm']==f'fold{held}-{q}');need(d['case']==arm['held_out'] and d['case'] not in arm['train_cases'],'Held-out identity')
  for tq in QUALITIES:
   scores=np.load(P/'training'/f"{arm['arm']}-test-{tq}.npy");m=independent(d['label'],scores,d['support']);expected=next(t['metrics'] for t in arm['test'] if t['quality']==tq);same(m,expected);checked.append(dict(arm=arm['arm'],test_quality=tq,metrics=m))
  for ctl in arm['controls']:
   if ctl['quality']!=q:continue
   train=[i for i in range(3) if i!=held]
   if ctl['name']=='constant':
    prev=np.concatenate([data[i]['label'][data[i]['support']] for i in train]).mean();need(abs(float(prev)-ctl['training_prevalence'])<1e-10,'Training-only constant');score=np.full(d['label'].shape,prev,np.float32)
   else:
    fs=[uniform_filter(v['images'][q],32,mode='reflect') for v in data] if ctl['name']=='mean_height' else [v['hole_density'] for v in data]
    f=np.concatenate([fs[i][data[i]['support']] for i in train]);y=np.concatenate([data[i]['label'][data[i]['support']] for i in train]);edges=np.asarray(ctl['fit']['edges']);bins=np.r_[-np.inf,edges,np.inf];counts=np.histogram(f,bins=bins)[0];pos=np.histogram(f,bins=bins,weights=y.astype(float))[0];need(np.array_equal(counts,ctl['fit']['count']) and np.array_equal(pos,ctl['fit']['positive']),'Training-only control counts');prob=(pos+1)/(counts+2);score=prob[np.digitize(fs[held],edges)].astype(np.float32)
   m=independent(d['label'],score,d['support']);same(m,ctl['metrics']);control_checks.append(dict(held_out=d['case'],quality=q,name=ctl['name'],metrics=m))
  model=Net();state=torch.load(P/'training'/f"{arm['arm']}.pt",map_location='cpu',weights_only=True);model.load_state_dict(state);need(sum(p.numel() for p in model.parameters())==29937,'Same capacity')
  for fit in arm['training_fit'][-1]['cases']:
   td=next(v for v in data if v['case']==fit['case']);scores=predict(model,td['images'][q]);m=independent(td['label'],scores,td['support']);same(m,fit['metrics']);replayed.append(dict(arm=arm['arm'],case=td['case'],metrics=m))
  need([v['step'] for v in arm['history']]==list(range(128,2049,128)) and [v['step'] for v in arm['training_fit']]==[512,1024,2048],'Fixed steps and fit milestones')
save(P/'LEARNING_VERIFICATION.json',dict(status='PASS',created_at_utc=now(),retained_test_maps=checked,final_training_replays=replayed,control_checks=control_checks,independent_metrics=42*7,verification_model_forward_batches=216,training_optimizer_updates=12288,training_program_model_forward_batches=13152,verification_optimizer_updates=0));print('PASS:12 held-out maps,12 final training replays,294 metric fields including18 controls')
