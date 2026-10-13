"""Exercise original padding/resampling with analytically known depth and tile phases."""
from pathlib import Path
import json,subprocess,sys
import numpy as np,zarr
P=Path(__file__).resolve().parent;S=P.parent/'fortyfifth-experiment/source-evidence/pherc343/ink/scripts';W=P/'work/preprocessing-fixture-v2';W.mkdir(exist_ok=False)
raw=np.broadcast_to((30+2*np.arange(66,dtype=np.uint8))[:,None,None],(66,160,320)).copy()
g=zarr.open_group(str(W/'plain.zarr'),mode='w',zarr_format=2);g.create_array('0',data=raw,chunks=(66,128,128))
subprocess.run([sys.executable,'-B',str(S/'pad_render.py'),str(W/'plain.zarr'),str(W/'padded.zarr')],check=True)
padded=np.asarray(zarr.open_group(str(W/'padded.zarr'),mode='r')['0'][:])
assert np.array_equal(padded[:,40:,200:],raw) and not padded[:,:40,:].any() and not padded[:,:,:200].any()
for phase in ['plain','padded']:
    subprocess.run([sys.executable,'-B',str(S/'hecate_prep.py'),'--src',str(W/f'{phase}.zarr'),'--out',str(W/f'hecate-{phase}.zarr'),'--native-um','8.64','--native-um-z','8.64','--target-um','9.6','--z-center','33.5','--nz','16'],check=True)
a=np.asarray(zarr.open_group(str(W/'hecate-plain.zarr'),mode='r')['0'][:]);b=np.asarray(zarr.open_group(str(W/'hecate-padded.zarr'),mode='r')['0'][:])
assert a.shape==(16,144,288) and b.shape==(16,180,468)
assert np.array_equal(a,b[:,36:,180:]) and not b[:,:36,:].any() and not b[:,:,:180].any()
expected=np.array([80,82,84,87,89,91,93,95,98,100,102,104,107,109,111,113],np.uint8)
# CV2's float32 area weights can put an integral ideal value just below the
# integer before the author's uint8 truncation. Require the mathematical value
# or its immediately lower integer, rather than silently altering that recipe.
delta=a.astype(np.int16)-expected[:,None,None].astype(np.int16)
assert delta.min()>=-1 and delta.max()<=0,(delta.min(),delta.max())
sys.path.insert(0,str(P/'runtime-source'))
from vesuvius.ink_detection.inference.infer import select_layer_indices
assert select_layer_indices(66,layer_start=26,layer_end=43,output_depth=17,direction='reverse').tolist()==list(range(42,25,-1))
(P/'PREPROCESSING_VERIFICATION.json').write_text(json.dumps(dict(status='PASS',fixture='spatially constant affine depth ramp, analytic 16-plane values within one downward truncation unit',native_shape=list(raw.shape),hecate_plain_shape=list(a.shape),hecate_padded_shape=list(b.shape),ideal_depth_values=expected.tolist(),actual_center_values=a[:,72,144].tolist(),error_range=[int(delta.min()),int(delta.max())],native_padding_yx=[40,200],hecate_padding_yx=[36,180],padded_phase_realignment_exact=True,villa_reverse_indices=list(range(42,25,-1)),limitations='Checks mechanics on synthetic input; no full CT qualification or ink reference comparison.'),indent=2)+'\n')
print('PASS: analytic depth values, all-axis dimensions, both tile phases and Villa reversal',flush=True)
