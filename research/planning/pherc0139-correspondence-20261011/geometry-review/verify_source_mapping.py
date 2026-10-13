"""Coordinate-only fixed 3x3 check; no CT, labels, predictions, fit or network."""
from pathlib import Path
import hashlib
import json
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
GEOM = ROOT.parent / "geometry" / "responses"
CATALOG = ROOT.parent / "metadata" / "CATALOG_EXTRACT.json"
# Use the retained full catalog because its schema is fixed and its hash is pinned.
FULL_CATALOG = Path("/workspace/herculaneum-lab/reports/fortyfifth-experiment/source-evidence/PHerc0139-retained-catalog.json")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def bilinear(arr, x, y):
    ix, iy = int(np.floor(x)), int(np.floor(y))
    pp = arr[iy:iy+2, ix:ix+2].astype(np.float64)
    if pp.shape != (2, 2, 3) or not np.isfinite(pp).all() or (pp == -1).any():
        return None
    dx, dy = x-ix, y-iy
    return ((1-dx)*pp[0, 0]+dx*pp[0, 1])*(1-dy) + ((1-dx)*pp[1, 0]+dx*pp[1, 1])*dy

def main():
    assert sha(FULL_CATALOG) == "2a460d1f57ca0d3535ce2a8a8ab8b80fe4141f143e8de486394a150bc93f094d"
    catalog = json.loads(FULL_CATALOG.read_text())
    affine = np.array(catalog['volumes']['20260102150214']['properties']['transforms'][0]['transformation_matrix'])
    inputs = [GEOM / f'{volume}-{axis}.tif' for volume in ('20260102150214', '20250728140407') for axis in 'xyz']
    fine = np.stack([tifffile.imread(p) for p in inputs[:3]], axis=-1)
    native = np.stack([tifffile.imread(p) for p in inputs[3:]], axis=-1)
    assert fine.shape == (1244, 1552, 3) and native.shape == (320, 399, 3)
    records = []
    for fy in (.25, .5, .75):
        for fx in (.25, .5, .75):
            xn, yn = (native.shape[1]-1)*fx, (native.shape[0]-1)*fy
            ratio_x, ratio_y = fine.shape[1]/native.shape[1], fine.shape[0]/native.shape[0]
            correct_xy = [(xn+.5)*ratio_x-.5, (yn+.5)*ratio_y-.5]
            naive_xy = [xn*ratio_x, yn*ratio_y]
            pn, pc, pb = bilinear(native, xn, yn), bilinear(fine, *correct_xy), bilinear(fine, *naive_xy)
            record = {'fraction_xy': [fx, fy], 'native_mesh_xy': [xn, yn], 'fine_mesh_source_mapping_xy': correct_xy, 'fine_mesh_naive_ratio_xy': naive_xy}
            if any(p is None for p in (pn, pc, pb)):
                record['status'] = 'invalid_coordinate_footprint'
            else:
                dc = affine[:, :3] @ pc + affine[:, 3] - pn
                db = affine[:, :3] @ pb + affine[:, 3] - pn
                record.update(status='measured', source_mapping_residual_xyz_native_voxels=dc.tolist(), source_mapping_residual_native_voxels=float(np.linalg.norm(dc)), naive_ratio_residual_xyz_native_voxels=db.tolist(), naive_ratio_residual_native_voxels=float(np.linalg.norm(db)))
            records.append(record)
    result = {'kind': 'coordinate_only_pipeline_mapping_check_not_CT_registration', 'site_selection': 'fixed Cartesian product (.25,.5,.75) times native mesh dimension minus1; no fitting or selection by response', 'interpolation': 'float64 bilinear representative geometry, not exact staged FP32 renderer replay', 'source_mapping': 'qF=(qN+0.5)*NF/NN-0.5', 'shape_fine_yxc': list(fine.shape), 'shape_native_yxc': list(native.shape), 'inputs': [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in inputs]+[{'path':str(FULL_CATALOG), 'sha256':sha(FULL_CATALOG)}], 'records': records}
    measured = [r for r in records if r['status']=='measured']
    result['summary'] = {'measured': len(measured), 'total': len(records), 'source_mapping_residual_range_native_voxels': [min(r['source_mapping_residual_native_voxels'] for r in measured), max(r['source_mapping_residual_native_voxels'] for r in measured)], 'naive_ratio_residual_range_native_voxels': [min(r['naive_ratio_residual_native_voxels'] for r in measured), max(r['naive_ratio_residual_native_voxels'] for r in measured)]}
    (ROOT / 'SOURCE_MAPPING_COORDINATE_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))

if __name__ == '__main__':
    main()
