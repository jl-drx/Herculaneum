"""Recount every geometry screen and brute-force fixed nearest-point checks."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent;D=P.parents[1]/'data/experiments';O=P/'target-preparation-v2'
r=json.loads((O/'RESULTS.json').read_text())
points=[]
for seg,(x,y,w,h) in r['maximal_surface_extents']:
    root=D/'pherc0800-surface-geometry' if seg=='20251029010146' else D/'pherc0800-larger-geometry'/seg
    xyz=np.load(root/'full-surface-ct-coordinates.xyz.npy',mmap_mode='r');v=np.load(root/'full-surface-valid-mask.npy',mmap_mode='r')
    points.append(np.asarray(xyz[y:y+h,x:x+w][v[y:y+h,x:x+w]],dtype=np.float64))
points=np.concatenate(points);assert len(points)==r['prior_valid_centres']
checks=[]
for row in r['candidates']:
    z=np.load(O/(row['segment']+'.npz'));v=z['valid'];dist=z['nearest_distance'][v]
    assert v.sum()==row['sampled_valid']
    for radius in [2,16,64]:assert int((dist<=radius).sum())==row['within_voxels'][str(radius)]['count']
    assert int((dist>64).sum())==row['farther_than64']
    xyz=np.load(D/'pherc0800-larger-geometry'/row['segment']/'full-surface-ct-coordinates.xyz.npy',mmap_mode='r')[::20,::20][v].astype(np.float64)
    for idx in np.linspace(0,len(xyz)-1,6,dtype=int):
        best=float('inf')
        for start in range(0,len(points),262144):
            delta=points[start:start+262144]-xyz[idx]
            best=min(best,float(np.sum(delta*delta,axis=1).min()))
        error=abs(best**.5-float(dist[idx]));assert error<1e-9
        checks.append({'segment':row['segment'],'selected_valid_index':int(idx),'absolute_error':error})
with (O/'VERIFICATION.json').open('x') as f:json.dump({'status':'PASS','summary_counts_recomputed_for_all_five_meshes':True,'fixed_bruteforce_checks':checks,'scope':'Numerical geometry screen only; no physical-overlap or fresh-field admission.'},f,indent=2);f.write('\n')
print(json.dumps({'status':'PASS','bruteforce_queries':len(checks),'max_error':max(x['absolute_error'] for x in checks)}))
