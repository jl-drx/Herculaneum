"""Whole-core descriptive comparisons; no fitted display, labels or ink claim."""
from pathlib import Path
import json,itertools
import numpy as np
P=Path(__file__).resolve().parent;W=P/'work';O=W/'inference'
r=json.loads((P/'INFERENCE_RESULTS.json').read_text());assert r['status']=='PASS' and len(r['maps'])==16
maps={};summary=[]
for rec in r['maps']:
 key=(rec['model'],rec['phase'],rec['direction']);a=np.load(rec['outputs']['aligned']['path']);assert a.shape==(128,128) and np.isfinite(a).all();maps[key]=a
 summary.append(dict(model=key[0],phase=key[1],direction=key[2],mean=float(a.mean()),standard_deviation=float(a.std()),min=float(a.min()),max=float(a.max()),q95=float(np.quantile(a,.95))))
def compare(a,b):
 d=np.abs(a.astype(np.float64)-b.astype(np.float64));c=float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if a.std()>0 and b.std()>0 else None
 return dict(pearson_correlation=c,mean_abs_uint8_levels=float(d.mean()*255),max_abs_uint8_levels=float(d.max()*255))
names=[r['name'] for r in json.loads((P/'INFERENCE_GO.json').read_text())['models']];rows=[]
for name in names:
 for direction in ['forward','reverse']:rows.append(dict(comparison='padding',model=name,direction=direction,**compare(maps[name,'plain',direction],maps[name,'padded',direction])))
 for phase in ['plain','padded']:rows.append(dict(comparison='depth-order',model=name,phase=phase,**compare(maps[name,phase,'forward'],maps[name,phase,'reverse'])))
for phase in ['plain','padded']:
 for direction in ['forward','reverse']:
  for a,b in itertools.combinations(names,2):rows.append(dict(comparison='models',phase=phase,direction=direction,model_a=a,model_b=b,**compare(maps[a,phase,direction],maps[b,phase,direction])))
result=dict(status='PASS',core_xywh=[1024,280,128,128],maps=summary,comparisons=rows,interpretation='Descriptive response stability only; no ink accuracy, AP or authenticated reading. Tiny1.1mm core; one already studied specimen. No ensemble or fitted contrast.')
(P/'MAP_COMPARISONS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status='PASS',map_summaries=len(summary),pair_comparisons=len(rows),padding=[r for r in rows if r['comparison']=='padding'])))
