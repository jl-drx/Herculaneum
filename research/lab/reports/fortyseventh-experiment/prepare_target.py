"""Read-only geometric overlap screen on retained PHerc0800 coordinate maps."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent;LAB=P.parents[1];D=LAB/'data/experiments'
out=P/'target-preparation';out.mkdir(exist_ok=False)
seal=LAB/'reports/fortysixth-experiment/final-verification.json'
assert hashlib.sha256(seal.read_bytes()).hexdigest()=='c91572b8763e74331638468208a7d15a5d9f01e2c4e38bbc0b4891352cf0e271'
retained={r['path']:r for r in json.loads(seal.read_text())['records']}
pins=[]
def verified(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    r=retained[str(p)];assert h==r['sha256'] and p.stat().st_size==r['size_bytes']
    pins.append(r);return p

provenance=[];ct_views=[]
for p in sorted(D.glob('pherc0800*/provenance.json')):
    d=json.loads(verified(p).read_text())
    if d.get('bbox_source_xywh') is not None:
        provenance.append({'path':str(p),'segment':d['segment_id'],'bbox_source_xywh':d['bbox_source_xywh'],'shape_zyx':d['shape_zyx']})
    elif 'array_start_zyx' in d:
        assert '/3/' in d['chunk_source']['url']
        lo=np.array(d['array_start_zyx'])*8;hi=lo+np.array(d['valid_shape_zyx'])*8
        ct_views.append({'path':str(p),'native_bbox_zyx':[lo.tolist(),hi.tolist()], 'role':d['role']})
    else:raise AssertionError('Unclassified target provenance: '+str(p))

maximal=[('20251029010146',[0,0,980,980]),('20251028222030',[512,512,1536,1536]),('20251028220955',[768,256,768,768])]
for r in provenance:
    x,y,w,h=r['bbox_source_xywh']
    assert any(r['segment']==seg and X<=x and Y<=y and x+w<=X+W and y+h<=Y+H for seg,(X,Y,W,H) in maximal)

def geometry(seg):
    root=D/'pherc0800-surface-geometry' if seg=='20251029010146' else D/'pherc0800-larger-geometry'/seg
    coords=np.load(verified(root/'full-surface-ct-coordinates.xyz.npy'),mmap_mode='r')
    mask=np.load(verified(root/'full-surface-valid-mask.npy'),mmap_mode='r')
    assert coords.shape==(*mask.shape,3)
    return coords,mask

centres=[];source_ids=[]
for idx,(seg,(x,y,w,h)) in enumerate(maximal):
    xyz,v=geometry(seg);sel=v[y:y+h,x:x+w]
    points=np.asarray(xyz[y:y+h,x:x+w][sel],dtype=np.float64)
    assert np.isfinite(points).all()
    centres.append(points);source_ids.append(np.full(len(points),idx,np.uint8))
centres=np.concatenate(centres);source_ids=np.concatenate(source_ids)
tree=cKDTree(centres)
# Exact self lookup on deterministic input positions must recover distance zero.
assert np.all(tree.query(centres[::1000],workers=1)[0]==0)
segments=sorted(p.name for p in (D/'pherc0800-larger-geometry').iterdir() if p.is_dir() and (p/'full-surface-ct-coordinates.xyz.npy').exists())
assert len(segments)==5
fig,axes=plt.subplots(1,5,figsize=(17,4),layout='constrained')
records=[]
for ax,seg in zip(axes,segments):
    xyz,valid=geometry(seg);q=np.asarray(xyz[::20,::20]);ok=np.asarray(valid[::20,::20])
    dist,nearest=tree.query(q[ok],workers=2)
    field=np.full(ok.shape,np.nan);field[ok]=dist
    far=ok&(field>64)
    labs,n=label(far)
    counts=np.bincount(labs.ravel());counts[0]=0
    if counts.max():
        yy,xx=np.where(labs==counts.argmax());box=[int(xx.min()*20),int(yy.min()*20),int(xx.max()*20+1),int(yy.max()*20+1)]
    else:box=None
    ct=[]
    for c in ct_views:
        lo,hi=np.array(c['native_bbox_zyx'])[:,::-1]
        ct.append({'source':c['path'],'sample_centres_inside':int(((q[ok]>=lo)&(q[ok]<hi)).all(1).sum())})
    np.savez_compressed(out/(seg+'.npz'),native_source_yx=np.indices(ok.shape).transpose(1,2,0)*20,
                        valid=ok,nearest_distance=field,nearest_source=source_ids[nearest],far_component_labels=labs)
    records.append({'segment':seg,'full_native_shape':list(valid.shape),'sampled_valid':int(ok.sum()),
        'nearest_prior_distance_quantiles':dict(zip(['min','p25','median','p75','p95','max'],map(float,np.quantile(dist,[0,.25,.5,.75,.95,1])))),
        'within_voxels':{str(r):{'count':int((dist<=r).sum()),'fraction':float((dist<=r).mean())} for r in [2,16,64]},
        'farther_than64':int(far.sum()),'largest_far_component_samples':int(counts.max()),'largest_far_component_native_bounds_xyxy':box,
        'prior_lowres_ct_views':ct,'prospective_field_admitted':False})
    ax.imshow(np.ma.array(field,mask=~ok),vmin=0,vmax=128,cmap='viridis');ax.contour(far,levels=[.5],colors='red',linewidths=.5)
    ax.set_title(seg+'\n'+str(int(far.sum()))+' samples >64 voxels',fontsize=8);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Geometry-only screen: nearest old surface centre, fixed0–128 native-voxel display\nRed outlines candidate separation; sampled centres do not certify full context/slab independence')
fig.savefig(out/'geometry_screen.png',dpi=130);plt.close(fig)
report={'status':'PASS','provenance_files_checked':len(provenance)+len(ct_views),'surface_acquisitions':provenance,
        'maximal_surface_extents':maximal,'prior_valid_centres':len(centres),'prior_ct_views':ct_views,'candidates':records,
        'pins':pins,'no_new_target_inference':True,'fresh_assessment_field_certified':False,
        'limitations':'Every20px centre screen only. Complete CT depth/context, normals/interpolation, old raw-CT spot checks and physical identity still require admission audit.'}
with (out/'RESULTS.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':'PASS','prior_centres':len(centres),'candidates':[{k:r[k] for k in ['segment','sampled_valid','farther_than64','largest_far_component_samples']} for r in records]},indent=2))
