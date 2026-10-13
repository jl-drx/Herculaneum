"""All fixed-region descriptive comparisons on common observed context; no labels."""
import json,itertools
import numpy as np
from common import *
assert json.loads((P/'COMPLETION_VERIFICATION.json').read_text())['status']=='PASS'
r=json.loads((P/'INFERENCE_RESULTS.json').read_text());mask=np.load(W/'output-support.npz')['common'];maps={};summaries=[]
for row in r['maps']:
 key=(row['model'],row['phase'],row['direction']);a=np.load(row['outputs']['aligned']['path'])[mask].astype(np.float64);maps[key]=a
 summaries.append(dict(model=key[0],phase=key[1],direction=key[2],pixels=len(a),mean=float(a.mean()),standard_deviation=float(a.std()),min=float(a.min()),max=float(a.max()),q95=float(np.quantile(a,.95)),q99=float(np.quantile(a,.99)),fraction_at_least_128_uint8=float((a>=128/255).mean())))
def compare(a,b):
 d=np.abs(a-b);return dict(pearson_correlation=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None,mean_abs_uint8_levels=float(d.mean()*255),max_abs_uint8_levels=float(d.max()*255))
models=[m['name'] for m in json.loads((P/'INFERENCE_GO.json').read_text())['models']];rows=[]
for m in models:
 for direction in ['forward','reverse']:rows.append(dict(comparison='padding',model=m,direction=direction,**compare(maps[m,'plain',direction],maps[m,'padded',direction])))
 for phase in ['plain','padded']:rows.append(dict(comparison='depth-order',model=m,phase=phase,**compare(maps[m,phase,'forward'],maps[m,phase,'reverse'])))
for phase in ['plain','padded']:
 for direction in ['forward','reverse']:
  for a,b in itertools.combinations(models,2):rows.append(dict(comparison='models',phase=phase,direction=direction,model_a=a,model_b=b,**compare(maps[a,phase,direction],maps[b,phase,direction])))
save(P/'MAP_COMPARISONS.json',dict(status='PASS',common_support_pixels=int(mask.sum()),maps=summaries,comparisons=rows,scope='Descriptive response stability on reused development material; no ink accuracy or reading claim.'))
print(json.dumps(dict(status='PASS',maps=summaries,padding=[r for r in rows if r['comparison']=='padding'],depth_order=[r for r in rows if r['comparison']=='depth-order'])))
