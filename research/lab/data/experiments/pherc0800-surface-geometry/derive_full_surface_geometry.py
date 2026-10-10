#!/usr/bin/env python3
"""Derive the full geometry mask/map offline from already verified local TIFXYZ."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parent
INPUT_NAMES = [
    'mesh.meta.json', 'mesh.x.tif', 'mesh.y.tif', 'mesh.z.tif',
    'surface.level0.zarray.json', 'surface.zattrs.json',
    'native-roi-valid-mask.npy', 'native-roi-ct-coordinates.xyz.npy',
    'geometry-validation.json', 'manifest.json',
]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_new_or_identical(path: Path, array: np.ndarray) -> None:
    if path.exists():
        assert np.array_equal(np.load(path), array, equal_nan=True), f'Existing {path.name} differs'
        return
    with path.open('xb') as stream:
        np.save(stream, array)

input_hashes = {name: digest(ROOT/name) for name in INPUT_NAMES}
meta = json.loads((ROOT/'mesh.meta.json').read_text())
array_meta = json.loads((ROOT/'surface.level0.zarray.json').read_text())
prior = json.loads((ROOT/'geometry-validation.json').read_text())
coords = np.stack([tifffile.imread(ROOT/f'mesh.{axis}.tif') for axis in 'xyz'], axis=-1)
assert coords.shape == (49, 49, 3) and coords.dtype == np.float32
finite_non_sentinel = np.isfinite(coords).all(axis=-1) & (coords != -1).all(axis=-1)
vertex_valid = np.isfinite(coords).all(axis=-1) & (coords > 0).all(axis=-1)
# The stricter positive test is equivalent for this retrieved source, not an assumption.
assert np.array_equal(vertex_valid, finite_non_sentinel)
scale_x, scale_y = map(float, meta['scale'])
height, width = map(int, array_meta['shape'][1:])
assert (height, width) == (980, 980)
assert tuple(np.rint(np.array(coords.shape[:2])/np.array([scale_y, scale_x])).astype(int)) == (height,width)
full_mask = np.zeros((height,width), dtype=bool)
full_xyz = np.full((height,width,3), np.nan, dtype=np.float32)

# Bound temporary arrays by mapping 128 rows at a time. No fresh network accesses.
for first_row in range(0,height,128):
    ys, xs = np.meshgrid(np.arange(first_row,min(first_row+128,height),dtype=np.float64),
                         np.arange(width,dtype=np.float64), indexing='ij')
    gy, gx = (ys+0.5)*scale_y, (xs+0.5)*scale_x
    iy, ix = np.floor(gy).astype(int), np.floor(gx).astype(int)
    inside = (iy>=0)&(ix>=0)&(iy+1<coords.shape[0])&(ix+1<coords.shape[1])
    iyc, ixc = np.clip(iy,0,coords.shape[0]-2), np.clip(ix,0,coords.shape[1]-2)
    fy, fx = gy-iy, gx-ix
    ok = inside & vertex_valid[iyc,ixc] & vertex_valid[iyc+1,ixc] & vertex_valid[iyc,ixc+1] & vertex_valid[iyc+1,ixc+1]
    interpolated = (((1-fy)*(1-fx))[...,None]*coords[iyc,ixc]
                    + ((1-fy)*fx)[...,None]*coords[iyc,ixc+1]
                    + (fy*(1-fx))[...,None]*coords[iyc+1,ixc]
                    + (fy*fx)[...,None]*coords[iyc+1,ixc+1])
    full_mask[first_row:first_row+ok.shape[0]] = ok
    full_xyz[first_row:first_row+ok.shape[0]] = np.where(ok[...,None],interpolated,np.nan).astype(np.float32)

existing_roi_mask = np.load(ROOT/'native-roi-valid-mask.npy')
existing_roi_xyz = np.load(ROOT/'native-roi-ct-coordinates.xyz.npy')
assert full_mask.shape == (980,980)
assert np.array_equal(full_mask[256:512,256:512],existing_roi_mask)
assert np.array_equal(full_xyz[256:512,256:512],existing_roi_xyz,equal_nan=True)
assert np.isfinite(full_xyz[full_mask]).all()
assert np.isnan(full_xyz[~full_mask]).all()
assert (full_xyz[full_mask] > 0).all()
source_bbox = np.array(meta['bbox'],dtype=np.float64)
point_bbox = np.stack([full_xyz[full_mask].min(axis=0),full_xyz[full_mask].max(axis=0)])
assert (point_bbox[0] >= source_bbox[0]-0.002).all()
assert (point_bbox[1] <= source_bbox[1]+0.002).all()

components = {}
for name, structure in [('four_neighbor',np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)),
                         ('eight_neighbor',np.ones((3,3),dtype=np.uint8))]:
    labels,count = label(full_mask,structure=structure)
    areas = np.bincount(labels.ravel())[1:]
    components[name] = {'count':int(count),'areas_pixels_descending':sorted(map(int,areas),reverse=True)}
background_labels,background_count = label(~full_mask,structure=np.ones((3,3),dtype=np.uint8))
boundary_ids = np.unique(np.concatenate([background_labels[0],background_labels[-1],background_labels[:,0],background_labels[:,-1]]))
background_areas = np.bincount(background_labels.ravel())
interior_holes = [int(background_areas[k]) for k in range(1,background_count+1) if k not in boundary_ids]
mask_name = 'full-surface-valid-mask.npy'
map_name = 'full-surface-ct-coordinates.xyz.npy'
save_new_or_identical(ROOT/mask_name,full_mask)
save_new_or_identical(ROOT/map_name,full_xyz)
valid_y, valid_x = np.where(full_mask)
# A complete bilinear cell requires floor(g)+1 <=48. With g=(p+0.5)*0.05,
# p=960..979 lies past the final complete source-grid cell on either axis.
grid_limit_y = coords.shape[0]-1
grid_limit_x = coords.shape[1]-1
edge_invalid_rows = np.where((np.arange(height)+0.5)*scale_y >=grid_limit_y)[0]
edge_invalid_cols = np.where((np.arange(width)+0.5)*scale_x >=grid_limit_x)[0]
summary = {
    'sample_id':'PHerc0800','volume_id':'20250521135224','segment_id':'20251029010146',
    'derivation':'Offline bilinear geometry interpolation of existing verified local source coordinate TIFFs; no model or ink inference.',
    'surface_shape_yx':[height,width], 'mesh_shape_yx':list(coords.shape[:2]), 'mesh_scale_xy':[scale_x,scale_y],
    'native_planar_pixel_size_um':8.64,
    'mapping_convention':'grid_x=(native_x+0.5)*scale_x; grid_y=(native_y+0.5)*scale_y, from C++ renderer centered-canvas convention.',
    'mapping_sources':prior['mapping_sources'], 'source_commit':prior['mapping_source_commit'],
    'validity_convention':'All four bilinear-corner XYZ vertices must be positive and finite, within source grid. For this source this equals the finite/non-minus-one criterion used previously.',
    'valid_pixels':int(full_mask.sum()), 'total_pixels':int(full_mask.size), 'valid_fraction':float(full_mask.mean()),
    'valid_pixel_bbox_yx_half_open':[[int(valid_y.min()),int(valid_y.max()+1)],[int(valid_x.min()),int(valid_x.max()+1)]],
    'valid_bilinear_point_bbox_xyz':point_bbox.tolist(), 'source_vertex_bbox_xyz':source_bbox.tolist(),
    'connected_valid_components':components,
    'interior_invalid_components_eight_neighbor':{'count':len(interior_holes),'areas_pixels_descending':sorted(interior_holes,reverse=True)},
    'existing_roi_yx_half_open':[[256,512],[256,512]],
    'existing_roi_mask_identical':True, 'existing_roi_xyz_map_identical':True,
    'edge_support':{
        'source_grid_index_extent_yx':[[0,48],[0,48]],
        'mapped_render_pixel_center_grid_extent_yx':[[0.025,48.975],[0.025,48.975]],
        'rows_without_complete_bilinear_cell':edge_invalid_rows.tolist(),
        'columns_without_complete_bilinear_cell':edge_invalid_cols.tolist(),
        'explanation':'The declared980px canvas divides49 stored vertices by0.05. Complete bilinear support extends only through render pixel959; the last20 rows/columns need out-of-grid neighbors and are invalid in this conservative mask. First render pixel maps to grid0.025, not0.0. This is support geometry, not evidence of absent ink.'
    },
    'coordinate_map_storage':'float32(H,W,3),XYZ CT voxel indices; NaN on invalid pixels. Mask bool(H,W).',
    'source_inputs_sha256':input_hashes,
    'derived_outputs':[{ 'file':name,'byte_count':(ROOT/name).stat().st_size,'sha256':digest(ROOT/name)} for name in [mask_name,map_name]],
    'offline_helper_sha256':digest(Path(__file__)),
    'limitations':['Original export renderer commit/offsets not recorded; maintained pinned C++ convention must be checked against raw CT for exact image correspondence.',
                   'This mask excludes missing or unsupported geometry, not non-ink pixels; connected footprint does not establish biological papyrus-layer topology.',
                   'Bilinear surface XYZ does not incorporate through-plane31slice offsets or normals; source CT mapping remains the base geometry.']
}
summary_path = ROOT/'full-surface-geometry-summary.json'
new_summary = json.dumps(summary,indent=2)+'\n'
if summary_path.exists():
    assert summary_path.read_text() == new_summary, 'Existing full-surface summary differs'
else:
    summary_path.write_text(new_summary)
assert input_hashes == {name:digest(ROOT/name) for name in INPUT_NAMES}, 'Existing inputs changed'
print(json.dumps({k:summary[k] for k in ['surface_shape_yx','valid_pixels','total_pixels','valid_fraction','valid_pixel_bbox_yx_half_open','connected_valid_components','interior_invalid_components_eight_neighbor','existing_roi_mask_identical','existing_roi_xyz_map_identical','derived_outputs']},indent=2))
