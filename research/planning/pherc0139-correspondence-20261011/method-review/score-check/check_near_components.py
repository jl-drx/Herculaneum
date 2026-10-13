#!/usr/bin/env python3
"""Descriptive post-screen topology; preserves frozen admission decision."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent
scores=np.load(P/'site1-independent-landscapes.npy');r=json.loads((P/'REAL_DATA_REPLAY.json').read_text())['sites'][0]
fit=np.array(r['fit_translation_zyx']);fi=tuple(fit+6);results=[]
for i,e in enumerate(r['supports'][4:],4):
 mask=scores[i]>=e['at_fit_score']-.05;unseen=set(map(tuple,np.argwhere(mask)));comps=[]
 while unseen:
  start=min(unseen);unseen.remove(start);todo=[start];visited={start}
  while todo:
   v=todo.pop()
   for ax in range(3):
    for inc in (-1,1):
     w=list(v);w[ax]+=inc;w=tuple(w)
     if w in unseen:unseen.remove(w);visited.add(w);todo.append(w)
  comps.append(visited)
 component=next(c for c in comps if fi in c);idx=np.array(sorted(component));shifts=idx-6
 results.append({'support':i,'comparison_level':'score>=NCC_at_fit-0.05; descriptive, not calibrated confidence or new gate','score_threshold':e['at_fit_score']-.05,'total_candidates':int(mask.sum()),'components':len(comps),'fit_component_candidates':len(component),'fit_component_min_zyx':shifts.min(0).tolist(),'fit_component_max_zyx':shifts.max(0).tolist(),'fit_component_touches_search_boundary':bool(np.any((idx==0)|(idx==12))),'strongest_distinct_competitor_same_component':tuple(np.array(e['distinct_competitor_zyx'])+6) in component})
(P/'SITE1_NEAR_SCORE_COMPONENTS.json').write_text(json.dumps({'status':'descriptive post-screen analysis, no criteria changed','supports':results},indent=2)+'\n')
