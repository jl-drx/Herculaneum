"""Validate public pair metadata algebra, without claiming physical registration."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def pin(p):return dict(path=str(p),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
catalog=ROOT/'pair-registration-metadata/PHercParis4-catalog.json'
x=json.loads(catalog.read_text());old='20230205180739';new='20260411134726'
transforms=x['sample']['properties']['volume_transforms']
def matrix(a,b):
 row=next(r for r in transforms if r['from_volume_id']==a)
 item=next(t for t in row['transforms'] if t['to_volume_id']==b)
 m=np.eye(4);m[:3,:]=item['matrix'];return m,item
forward,fi=matrix(old,new);inverse,iv=matrix(new,old)
read=lambda name:json.loads((ROOT/'pair-final-metadata'/name).read_text())
om=read('old-mesh-meta.txt');nm=read('new-mesh-meta.txt')
oa=read('old-native-array.txt');na=read('new-native-array.txt')
def inside(bbox,shape):
 b=np.asarray(bbox);return bool((b[0]>=0).all() and (b[1]<np.array(shape)[::-1]).all())
records=[]
for path in ROOT.glob('pair-*/RECEIPT.json'):records.extend(json.loads(path.read_text())['records'])
result=dict(status='PASS_METADATA_ALGEBRA_ONLY',source=pin(catalog),schema=pin(ROOT/'pair-final-metadata/catalog-schema.txt'),old_volume_id=old,new_volume_id=new,nominal_voxel_um=[x['volumes'][old]['properties']['pixel_size_um'],x['volumes'][new]['properties']['pixel_size_um']],volume_metadata_public=True,coordinates='XYZ level0 voxel units; array shapes ZYX',forward=fi,inverse=iv,max_inverse_roundtrip_error=float(abs(inverse@forward-np.eye(4)).max()),max_forward_roundtrip_error=float(abs(forward@inverse-np.eye(4)).max()),linear_determinant=float(np.linalg.det(forward[:3,:3])),linear_singular_values=np.linalg.svd(forward[:3,:3],compute_uv=False).tolist(),old_shape_ZYX=oa['shape'],new_shape_ZYX=na['shape'],native_chunks_ZYX=oa['chunks'],bytes_per_uncompressed_uint8_chunk=int(np.prod(oa['chunks'])),segment_id='20231016151002',paired_mesh_files_listed=True,mesh_metadata={'old':om,'new':nm},old_bbox_in_volume=inside(om['bbox'],oa['shape']),new_bbox_in_volume=inside(nm['bbox'],na['shape']),metadata_GET_requests=len(records),metadata_delivered_bytes=sum(r['bytes'] for r in records),additional_HEAD_requests=1,scientific_image_or_mesh_payloads_read=0,physical_registration_validated=False,held_out_landmark_error_measured=False,independent_reference_labels_established=False,legacy_seed42_parent_exposure='Paris4 appears in the named seed42 training representations; do not call it upstream-unseen',scope='Transforms, paired meshes and both public volume metadata are available. Algebra and in-bounds boxes are not landmark accuracy, surface equivalence or ink evidence.')
assert result['max_inverse_roundtrip_error']<1e-8 and result['old_bbox_in_volume'] and result['new_bbox_in_volume']
(ROOT/'PAIR_FEASIBILITY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['forward','inverse','mesh_metadata']},indent=2))
