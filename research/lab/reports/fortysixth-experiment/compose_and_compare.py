"""Author's fixed composition plus descriptive RGB comparison, never ink AP."""
from pathlib import Path
import hashlib,json,subprocess,sys
import cv2,numpy as np
P=Path(__file__).resolve().parent;W=P/'work';OLD=P.parent/'fortyfifth-experiment';SRC=OLD/'source-evidence/pherc343';S=SRC/'ink/scripts';NAME='concat_w047-w048_R5B2_z9500-11000'
assert json.loads((P/'FIXED_DEPTH_INFERENCE.json').read_text())['status']=='PASS'
assert json.loads((P/'RENDER_VERIFICATION.json').read_text())['status']=='PASS'
maps=W/'maps';maps.mkdir(exist_ok=False)
subprocess.run([sys.executable,'-B',str(S/'merge_tile_phase.py'),'--plain',str(W/'maps_plain'),'--plain-name',NAME+'_66','--padded',str(W/'maps_padded'),'--padded-name',NAME+'_66_padded','--node-source',str(OLD/'ph343-mesh-replay'/f'{NAME}_node_source.npz'),'--out',str(maps),'--out-name',NAME],check=True)
inputs=[maps/f'{NAME}__{suffix}' for suffix in ['dnative__L66s26_reverse.tif','hecate__L66zc33.50_reverse.png','hybrid_3d2d-seed42__L66s26_reverse.tif','hybrid_3d2d-seed43__L66s26_reverse.tif']]
assert all(p.is_file() for p in inputs)
image_dir=W/'image'
cmd=[sys.executable,'-B',str(S/'make_submission_image.py'),'--mesh',str(OLD/'ph343-mesh-replay'/f'{NAME}.tifxyz'),'--umbilicus',str(SRC/'inputs/umbilicus_PHerc0343.json'),'--ct',str(W/f'{NAME}_66.zarr'),'--overlay-style','dimmed-ct','--caption','compact','--geometry-sums',str(SRC/'outputs/SHA256SUMS'),'--area','380,105,6021,2560','--max-cm2','12','--ct-layer','d+1','--ink',*map(str,inputs),'--ensemble','--ink-stats',str(P/'FIXED_DISPLAY_STATS.json'),'--rows',str(SRC/'inputs'/f'{NAME}_rows.json'),'--letters',str(SRC/'inputs'/f'{NAME}_letter_boxes.json'),'--letter-markers','off','--sizes-csv',str(image_dir/f'{NAME}_letter_sizes.csv'),'--out-dir',str(image_dir),'--write-components']
subprocess.run(cmd,check=True)
reference=json.loads((SRC/'outputs'/f'{NAME}.json').read_text());actual=json.loads((image_dir/f'{NAME}.json').read_text())
ref_file=OLD/'source-evidence'/f'{NAME}.png';ours=image_dir/f'{NAME}.png'
assert hashlib.sha256(ref_file.read_bytes()).hexdigest()==reference['outputs']['image']['sha256']
def panel(file,side):
    im=cv2.imread(str(file));box=side['outputs']['image']['data_panel'];return im[box['y']:box['y']+box['height'],box['x']:box['x']+box['width']]
a=panel(ref_file,reference);b=panel(ours,actual);assert a.shape==b.shape
delta=np.abs(a.astype(np.int16)-b.astype(np.int16));valid=np.load(W/'valid.npy')[105:2560,380:6021][::-1,::-1]
def visible(im):
    B,G,R=np.moveaxis(im.astype(np.int16),-1,0)
    return valid&(R-B>12)&(G-B>8)&(np.abs(R-G)<50)
ra,rb=visible(a),visible(b)
report=dict(shape=list(a.shape),exact_png=ours.read_bytes()==ref_file.read_bytes(),valid_rgb_mae=float(delta[valid].mean()),valid_rgb_max_difference=int(delta[valid].max()),reference_display_visible_pixels=int(ra.sum()),reproduction_display_visible_pixels=int(rb.sum()),visible_display_intersection=int((ra&rb).sum()),visible_display_union=int((ra|rb).sum()),reference_visible_rgb_mae=float(delta[ra].mean()),definition='Display-color comparison on fixed published panel; yellow-visible rule R-B>12,G-B>8,abs(R-G)<50 excludes cyan annotations. Not pixel ink truth, AP, or an untouched evaluation.',raw_reference_maps_available=False,command=cmd)
(P/'IMAGE_COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
