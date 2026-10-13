"""Enumerate global-grid ROI tiles and conservative source work, without fetching it."""
from pathlib import Path
import sys,json,itertools,hashlib
import numpy as np,tifffile
from prepare import pin,save
P=Path(__file__).resolve().parent;W=P/'work';R=P.parent;LAB=P.parents[1]
sys.path.insert(0,str(R/'fortyeighth-experiment'));from support import reconstruct,corner_envelope
G=LAB/'data/experiments/pherc0800-larger-geometry/20251028222030'
mesh=np.stack([tifffile.imread(G/f'mesh.{a}.tif') for a in 'xyz'],-1)
b,n,validbase,valid=reconstruct(mesh,json.loads((G/'mesh.meta.json').read_text())['scale'],(2580,2580))
def starts(length,patch,stride):
    return sorted(set(list(range(0,max(1,length-patch+1),stride))+[max(0,length-patch)]))
def overlap(a,p,l,h):return a<h and a+p>l
def source_envelope(lo,hi,src,dst):
    # Area downsampling contributors, extended by one pixel for float endpoint
    # arithmetic. This is deliberately a bound, not a minimal exact fetch set.
    return max(0,int(np.floor(lo*src/dst))-1),min(src,int(np.ceil(hi*src/dst))+1)
rows=[];required=np.zeros((2580,2580),bool)
for phase,py,px in [('plain',0,0),('padded',40,200)]:
    h,w=2580+py,2580+px
    for family,patch,stride in [('villa',128,64),('hecate',64,32)]:
        hh,ww=(h,w) if family=='villa' else (round(h*.9),round(w*.9))
        ys,xs=starts(hh,patch,stride),starts(ww,patch,stride)
        roi=(512+py,2048+py,512+px,2048+px)
        if family=='hecate':
            # Both neighbors from the exact cv2 half-pixel upsampling mapping.
            roi=(int(np.floor((roi[0]+.5)*hh/h-.5)),int(np.floor((roi[1]-.5)*hh/h-.5))+2,int(np.floor((roi[2]+.5)*ww/w-.5)),int(np.floor((roi[3]-.5)*ww/w-.5))+2)
        chosen=[(y,x) for y in ys for x in xs if overlap(y,patch,roi[0],roi[1]) and overlap(x,patch,roi[2],roi[3])]
        y0=min(y for y,x in chosen);y1=max(y for y,x in chosen)+patch;x0=min(x for y,x in chosen);x1=max(x for y,x in chosen)+patch
        if family=='hecate':y0,y1=source_envelope(y0,y1,h,hh);x0,x1=source_envelope(x0,x1,w,ww)
        y0,y1=max(0,y0-py),min(2580,y1-py);x0,x1=max(0,x0-px),min(2580,x1-px)
        required[y0:y1,x0:x1]=True
        rows.append(dict(phase=phase,family=family,grid_yx=[hh,ww],patch=patch,stride=stride,full_grid_patches=len(ys)*len(xs),roi_contributing_grid_patches=len(chosen),native_context_bbox_xyxy=[x0,y0,x1,y1],count_role='unfiltered grid count; Villa nonzero occupancy can lower it',patches_yx=chosen))
inside=required&valid
lo,hi=corner_envelope(b[inside],n[inside]);lo=lo.astype(np.int64);hi=hi.astype(np.int64)
assert (lo>=0).all() and (hi<np.array([9867,9867,24298])).all()
low=lo//128;high=hi//128;assert (high-low<=1).all()
keys=set()
for delta in itertools.product([0,1],repeat=3):
    ix=low+delta;keep=(ix<=high).all(-1)
    keys.update(tuple(map(int,v[::-1])) for v in np.unique(ix[keep],axis=0))
np.savez_compressed(W/'target-planning-masks.npz',required_context=required,valid_geometry=valid,required_valid=inside)
# Compute estimates from actual target timing if available, otherwise explicitly
# label the earlier PHerc343 calibration (the current caller waits for timing).
qual=json.loads((P/'INFERENCE_QUALIFICATION.json').read_text())
timing={r['model']:r['batch4_seconds']/4 for r in qual['models']}
estimates=[]
for name,seconds in timing.items():
    fam='hecate' if name=='hecate' else 'villa';count=sum(r['roi_contributing_grid_patches'] for r in rows if r['family']==fam)
    full=sum(r['full_grid_patches'] for r in rows if r['family']==fam)
    estimates.append(dict(model=name,roi_tiles_both_phases=count,full_canvas_tiles_both_phases=full,measured_seconds_per_tile_batch4=seconds,roi_estimated_seconds=count*seconds,full_canvas_estimated_seconds=full*seconds))
result=dict(status='PASS',scope='Compute/source planning only; no larger source fetch or map scheduled',scoring_xywh=[512,512,1536,1536],rows=rows,source_chunk_indices_zyx=sorted(keys),source_chunk_count=len(keys),source_payload_upper_bound_bytes=len(keys)*128**3,required_context_pixels=int(required.sum()),required_context_valid_geometry_pixels=int(inside.sum()),required_context_unknown_geometry_pixels=int((required&~valid).sum()),scoring_valid_geometry_pixels=int(valid[512:2048,512:2048].sum()),scoring_pixels=1536**2,compute=estimates,roi_model_seconds=sum(r['roi_estimated_seconds'] for r in estimates),full_canvas_model_seconds=sum(r['full_canvas_estimated_seconds'] for r in estimates),dense_array_bytes=dict(plain66=66*2580**2,padded66=66*2620*2780,hecate_plain16=16*2322**2,hecate_padded16=16*2358*2502,two_float_accumulators_largest_hecate=2*4*2358*2502),timing_source=pin(P/'INFERENCE_QUALIFICATION.json'),script=pin(__file__),limits=['Forecast uses four repeated copies of one fixed tile; full workloads, I/O and CPU contention vary.','No occupancy mask has been inferred from unobserved CT; these grid counts bound a later occupancy schedule.','All-grid ROI tile selection does not freeze original batch cohorts; later scheduler must qualify its batch grouping.','Source union includes all normal offsets through66layers on valid context; it is conservative, not an exact minimal fetch plan.','Invalid geometry must remain unknown, not observed zero. A later field run must freeze admissible output support including every patch contributor.','A31depth sweep multiplies inference cost roughly31x; testing both directions multiplies2x. Not scheduled.'])
save(P/'TRANSFER_PLAN.json',result)
print(json.dumps({k:result[k] for k in ['status','source_chunk_count','source_payload_upper_bound_bytes','required_context_pixels','required_context_valid_geometry_pixels','scoring_valid_geometry_pixels','roi_model_seconds','full_canvas_model_seconds','compute']}))
