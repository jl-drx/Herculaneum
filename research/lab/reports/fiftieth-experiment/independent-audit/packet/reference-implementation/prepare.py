"""Freeze inputs and conservative exposure inventory, without loading intensities."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;LAB=P.parents[1];D=LAB/'data/experiments'
def pin(p):
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':h}
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
seal=LAB/'reports/fortyseventh-experiment/final-verification.json';assert pin(seal)['sha256']=='0aef8d525d1140a04796268abb349740860f0416d7fa2ca185ce012786986532'
records={x['path']:x for x in json.loads(seal.read_text())['records']}
index=Path('/workspace/Herculaneum/research/workspace/MANIFEST.json');snapshot={str(LAB/r['path']):r for r in json.loads(index.read_text())['files']}
pins=[]
def verified(p):
 r=pin(p);expected=records.get(str(p),snapshot.get(str(p)));assert expected is not None,str(p)
 assert r['sha256']==expected['sha256'] and r['size_bytes']==expected.get('size_bytes',expected.get('bytes'))
 pins.append(r);return p
prior=json.loads(verified(LAB/'reports/fortyseventh-experiment/target-preparation-v2/RESULTS.json').read_text())
meshes=[]
segments=['20251029010146']+[r['segment'] for r in prior['candidates']]
for seg in segments:
 root=D/'pherc0800-surface-geometry' if seg==segments[0] else D/'pherc0800-larger-geometry'/seg
 files={n:str(verified(root/n)) for n in ['mesh.meta.json','mesh.x.tif','mesh.y.tif','mesh.z.tif','surface.level0.zarray.json','full-surface-valid-mask.npy','full-surface-ct-coordinates.xyz.npy']}
 if seg!=segments[0]:files['source-render-export-metadata.json']=str(verified(root/'source-render-export-metadata.json'))
 meshes.append({'segment':seg,'files':files})
 if seg!=segments[0]:verified(LAB/'reports/fortyseventh-experiment/target-preparation-v2'/f'{seg}.npz')
boxes=[]
for row in prior['prior_ct_views']:
 lo,hi=row['native_bbox_zyx'];boxes.append({'source':row['path'],'kind':'level3_cube_conservative','lo_xyz':[v-1 for v in lo[::-1]],'hi_xyz_inclusive':hi[::-1]})
chunks=set()
for name in ['reports/first-experiment/native-ct-consistency.json','reports/third-experiment/native-ct-consistency.json','reports/fifth-experiment/native-check/frozen-plan.json']:
 p=verified(LAB/name);d=json.loads(p.read_text())
 for row in d.get('native_chunk_sources',d.get('chunks',[])):
  if row.get('missing'):continue
  index3=tuple(row['chunk_index_zyx']);chunks.add(index3)
  boxes.append({'source':str(p),'kind':'native_chunk_conservative','lo_xyz':[v*128 for v in index3[::-1]],'hi_xyz_inclusive':[(v+1)*128-1 for v in index3[::-1]]})
# Completeness check: known raw native CT files match the declared chunk union.
actual=set()
for p in D.glob('pherc0800*/ct-chunk-*.bin'):actual.add(tuple(map(int,p.stem.removeprefix('ct-chunk-').split('-'))))
assert actual==chunks,(actual,chunks)
inputs={'created_at_utc':datetime.now(timezone.utc).isoformat(),'phase47_seal':pin(seal),'workspace_fallback':pin(index),'pins':pins,
 'current_sources':[pin(P/n) for n in ['PROTOCOL.md','support.py','prepare.py','run_admission.py']],
 'meshes':meshes,'previous_surface_extents':prior['maximal_surface_extents'],'previous_CT_boxes':boxes,
 'candidate_segments':segments[1:],'surface_bound_voxels':51.1,'context_radius_native_pixels':128,'minimum_core_square':128,
 'target_shape_zyx':[24298,9867,9867],'new_slab_offsets':{'start':-32.5,'end':32.5,'step':1,'count':66},'raw_native_chunk_indices':sorted(chunks),
 'scope':'CPU-only support screen; conservative prior retained-CT exclusion, no ink evidence or admission of a model study.'}
save(P/'GO.json',inputs);print(json.dumps({'status':'FROZEN','pins':len(pins),'old_CT_boxes':len(boxes),'meshes':len(meshes)}))
