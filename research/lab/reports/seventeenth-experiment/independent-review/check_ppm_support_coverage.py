"""Cross-tabulate all already-fixed PPM centers against literal support; no resampling."""
import json
from pathlib import Path
from reconstruct_ct import PINS,read_array,digest,json_dump
P=Path('/workspace/herculaneum-lab/reports/seventeenth-experiment')
descpath=next(p for p in PINS if p.name=='pooled-descriptor.json');assert digest(descpath.read_bytes())==PINS[descpath]
desc=json.loads(descpath.read_text());cases={c['id']:c for c in desc['cases']}
ppmpath=P/'independent-review/ppm-reconstruction/independent-ppm-diagnostics.json';ppm=json.loads(ppmpath.read_text())
rows=[];groups={};zeros=[]
for obj in ppm['objects']:
 for point in obj['points']:
  case=cases[point['case_id']];key=(point['case_id'],point['variant'])
  if key not in groups:
   groups[key]=dict(case_id=key[0],variant=key[1],centers=0,zero_centers=0,zero_inside_primary=0,zero_inside_surface=0,zero_inside_known=0)
  arrays={k:read_array(case['arrays'][k])for k in ['primary_mask','surface_mask','known_mask']}
  y,x=point['coarse_yx'];r=dict(case_id=key[0],variant=key[1],coarse_yx=[y,x],native_center_yx=point['native_center_yx'],zero_sentinel=point['geometry']['zero_sentinel'],primary=bool(arrays['primary_mask'][y,x]),surface=bool(arrays['surface_mask'][y,x]),known=bool(arrays['known_mask'][y,x]))
  rows.append(r);g=groups[key];g['centers']+=1
  if r['zero_sentinel']:
   zeros.append(r);g['zero_centers']+=1
   for name in ['primary','surface','known']:g['zero_inside_'+name]+=int(r[name])
assert len(rows)==972 and all(g['centers']==81 for g in groups.values())
result=dict(status='PASS',scope='All fixed81centers percase/map; no new samples or support changes',ppm_input_sha256=digest(ppmpath.read_bytes()),all_centers=rows,groups=list(groups.values()),zero_centers=zeros,total_centers=len(rows),total_zero_centers=len(zeros),zeros_inside_primary=sum(r['primary']for r in zeros),zeros_inside_surface=sum(r['surface']for r in zeros),zeros_inside_known=sum(r['known']for r in zeros))
json_dump(P/'independent-review/ppm-support-coverage.json',result)
print(json.dumps({k:v for k,v in result.items()if k not in ['all_centers','groups']},indent=2))
