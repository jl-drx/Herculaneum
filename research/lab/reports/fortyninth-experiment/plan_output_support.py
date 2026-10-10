"""Conservative output support under every positive-weight scheduled contributor.

This plans exclusion of invalid geometry; it does not validate unrendered CT.
"""
from pathlib import Path
import json
import numpy as np,cv2
from prepare import save,pin
P=Path(__file__).resolve().parent;W=P/'work'
plan=json.loads((P/'TRANSFER_PLAN.json').read_text());valid=np.load(W/'target-planning-masks.npz')['valid_geometry']
common=np.ones((1536,1536),bool);out={};rows=[]
for row in plan['rows']:
    py,px=(0,0) if row['phase']=='plain' else (40,200)
    # Recipe padding is known synthetic zero; other invalid geometry is unknown.
    padded=np.pad(valid,((py,0),(px,0)),constant_values=True)
    h,w=padded.shape;hh,ww=row['grid_yx'];p=row['patch']
    known=padded if row['family']=='villa' else cv2.resize(padded.astype(np.float32),(ww,hh),interpolation=cv2.INTER_AREA)>=1.0
    s=np.pad((~known).astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    bad=np.zeros((hh,ww),bool);coverage=np.zeros_like(bad);bad_tiles=0
    for y,x in row['patches_yx']:
        coverage[y:y+p,x:x+p]=True
        count=int(s[y+p,x+p]-s[y,x+p]-s[y+p,x]+s[y,x])
        if count:bad[y:y+p,x:x+p]=True;bad_tiles+=1
    if row['family']=='hecate':
        # INTER_LINEAR has nonnegative weights: positive bad weight marks unknown.
        bad=cv2.resize(bad.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>0
        coverage=cv2.resize(coverage.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>=1.0
    v=(~bad&coverage)[512+py:2048+py,512+px:2048+px]
    out[row['family']+'_'+row['phase']]=v;common&=v
    rows.append(dict(family=row['family'],phase=row['phase'],contributing_grid_tiles=len(row['patches_yx']),tiles_with_unknown_geometry=bad_tiles,fully_known_tiles=len(row['patches_yx'])-bad_tiles,output_supported_pixels=int(v.sum())))
np.savez_compressed(W/'planned-output-support.npz',common=common,**out)
save(P/'OUTPUT_SUPPORT_PLAN.json',dict(status='PASS',scope='Conservative geometry-only planning; no full source/render qualification yet',rows=rows,common_support_pixels=int(common.sum()),common_support_fraction=float(common.mean()),scoring_pixels=common.size,mask=pin(W/'planned-output-support.npz'),unknown_not_observed_zero=True,all_original_contributor_contexts_required=True,script=pin(__file__)))
print(json.dumps(dict(rows=rows,common_support_pixels=int(common.sum()),common_support_fraction=float(common.mean()))))
