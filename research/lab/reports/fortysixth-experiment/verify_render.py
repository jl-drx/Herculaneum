"""Check decoded public reference hashes before any scientific ink inference."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import zarr
P=Path(__file__).resolve().parent
OLD=P.parent/'fortyfifth-experiment'
NAME='concat_w047-w048_R5B2_z9500-11000'
reference=json.loads((OLD/'source-evidence/pherc343/outputs'/f'{NAME}.json').read_text())
root=P/'work'/f'{NAME}_66.zarr'
a=zarr.open_group(str(root),mode='r')['0']
assert tuple(a.shape)==(66,2560,6720) and a.dtype==np.uint8
area=reference['area']['raw_render_px'];sl=(slice(area['row0'],area['row1']),slice(area['col0'],area['col1']))
layer=np.asarray(a[34,*sl]);sha=hashlib.sha256(layer.tobytes()).hexdigest()
valid=np.ones((2560,6720),bool)
for y in range(0,2560,128):valid[y:y+128]=np.all(np.asarray(a[:,y:y+128,:])>0,axis=0)
masksha=hashlib.sha256(np.packbits(valid[sl]).tobytes()).hexdigest()
np.save(P/'work/valid.npy',valid)
checks={'area_layer34_sha256':sha,'area_valid_mask_sha256':masksha}
expected={'area_layer34_sha256':reference['ct']['area_layers_sha256']['34'],'area_valid_mask_sha256':reference['ct']['area_valid_mask_sha256']}
tree=hashlib.sha256();files=[]
for f in sorted(x for x in root.rglob('*') if x.is_file()):
    digest=hashlib.sha256(f.read_bytes()).hexdigest();rel=f.relative_to(root).as_posix();tree.update(f'{rel}\t{digest}\n'.encode());files.append(dict(path=rel,sha256=digest,bytes=f.stat().st_size))
record={'status':'PASS' if checks==expected else 'FAIL','shape':list(a.shape),'dtype':str(a.dtype),'actual':checks,'expected':expected,'valid_pixels':int(valid.sum()),'tree_sha256':tree.hexdigest(),'published_tree_sha256':reference['ct']['tree_sha256'],'tree_equal':tree.hexdigest()==reference['ct']['tree_sha256'],'files':files,'bytes':sum(f['bytes'] for f in files),'interpretation':'Decoded source fidelity; no inference or physical ink claim.'}
(P/'RENDER_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='files'}),flush=True)
assert record['status']=='PASS','Stop: published decoded layer/mask mismatch'
