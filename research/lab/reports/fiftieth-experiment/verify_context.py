"""Integer-rational source-footprint union check for the admitted128-square core."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent;W=P/'work';R=P.parent
plan=json.loads((P/'RENDER_GO.json').read_text());needed=np.zeros((2300,2260),bool);rows=[]
for grid in plan['grids']:
 py,px=(0,0) if grid['phase']=='plain' else (40,200);h,w=2300+py,2260+px;hh,ww=grid['grid_shape_yx'];mask=np.zeros_like(needed)
 for x,y,pw,ph in grid['patches_xywh']:
  if grid['family']=='villa':x0,x1,y0,y1=x,x+pw,y,y+ph
  else:
   x0=x*w//ww;x1=((x+pw)*w+ww-1)//ww;y0=y*h//hh;y1=((y+ph)*h+hh-1)//hh
  mask[y0-py:y1-py,x0-px:x1-px]=True
 needed|=mask;rows.append(dict(family=grid['family'],phase=grid['phase'],source_pixels=int(mask.sum())))
old=np.load(R/'fortyeighth-experiment/support-results/20251028225813.npz');ind=np.load(P/'independent-audit/output-locked/20251028225813-masks.npz');cur=np.load(W/'candidate-admission-masks.npz')
assert old['point_support'][needed].all() and ind['point_support'][needed].all()
assert not cur['new_chunk_intersection_archived'][needed].any() and not cur['new_chunk_intersection_independent'][needed].any()
observed=np.zeros_like(needed);x,y,w,h=plan['candidate_render_xywh'];observed[y:y+h,x:x+w]=np.load(W/'render-observed-mask.npy');assert observed[needed].all()
assert not (old['old_CT_box_bits'][needed]>0).any()
np.savez_compressed(W/'exact-required-source.npz',needed=needed)
result=dict(status='PASS',source_pixels=int(needed.sum()),rendered_context_pixels=int(observed.sum()),extra_renderer_pixels=int((observed&~needed).sum()),all_required_context_observed=True,both_arithmetic_exclusions_pass=True,current_exposure_inventory_pass=True,minimum_archived_nearest_old_centre_distance=float(old['nearest_old_distance'][needed].min()),rows=rows)
(P/'CONTEXT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
