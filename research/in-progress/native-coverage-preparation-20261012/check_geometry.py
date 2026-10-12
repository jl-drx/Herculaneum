"""Training-source geometry coverage audit; no ink-dependent selection."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import tifffile


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def records(a):
    a=np.ascontiguousarray(a)
    return a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).ravel()


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();r=a.work
    if (r/'GEOMETRY_COVERAGE.json').exists():raise FileExistsError('Already audited')
    sources=json.loads((r/'MESH_SOURCES.json').read_text())['rows'];rows=[];clouds={};voxels={};all_points=[]
    for source in sources:
        ident=source['id'];axes=[];pins=[]
        for axis in 'xyz':
            path=r/'mesh-arrays'/f'{ident}-mesh-{axis}.raw';rec=json.loads(path.with_suffix('.receipt.json').read_text())
            assert sha(path)==rec['sha256'] and path.stat().st_size==rec['bytes'];pins.append(dict(path=str(path.relative_to(r)),sha256=sha(path)))
            axes.append(tifffile.imread(path))
        xyz=np.stack(axes,-1);assert xyz.dtype==np.float32 and np.isfinite(xyz).all()
        invalid=(xyz==-1).all(-1);valid=(xyz>0).all(-1);assert np.all(valid|invalid),'Unexpected mixed sentinel or nonpositive coordinates'
        points=xyz[valid];assert len(points)>0
        meta=json.loads((r/'mesh-metadata'/f'{ident}-mesh-meta.raw').read_text());assert meta['scale']==[.05,.05]
        native=json.loads((r/'metadata'/f'{ident}-array.raw').read_text());assert list(xyz.shape[:2])==[n//20 for n in native['shape'][1:]]
        bbox=np.asarray([points.min(0),points.max(0)]);assert np.array_equal(bbox,np.asarray(meta['bbox'],np.float32))
        exact=np.unique(records(points));rounded=np.unique(records(np.rint(points).astype(np.int32)))
        clouds[ident]=exact;voxels[ident]=rounded;all_points.append(points)
        # At each fixed smoke-crop center, sample its four surrounding stored
        # vertices. This is only a geometry-support screen, not CT/normal parity.
        center=next(c for c in json.loads((r/'SOURCE_CROP_GO.json').read_text())['cases'] if c['id']==ident)
        x,y,_,_=center['native_crop_xy'];gy=(y+63.5)*.05;gx=(x+63.5)*.05;iy=int(np.floor(gy));ix=int(np.floor(gx))
        corners=valid[iy:iy+2,ix:ix+2];assert corners.shape==(2,2)
        rows.append(dict(id=ident,shape_yx=list(xyz.shape[:2]),valid_vertices=int(valid.sum()),
            invalid_vertices=int(invalid.sum()),valid_vertex_fraction=float(valid.mean()),
            bbox_xyz=bbox.tolist(),unique_exact_vertices=len(exact),unique_rounded_voxel_centres=len(rounded),
            fixed_smoke_centre_valid_corners=int(corners.sum()),source_pins=pins))
    pairs=[]
    for left,right in itertools.combinations(rows,2):
        l,u=np.maximum(left['bbox_xyz'][0],right['bbox_xyz'][0]),np.minimum(left['bbox_xyz'][1],right['bbox_xyz'][1])
        intersect=np.intersect1d(clouds[left['id']],clouds[right['id']],assume_unique=True).size
        rounded=np.intersect1d(voxels[left['id']],voxels[right['id']],assume_unique=True).size
        pairs.append(dict(left=left['id'],right=right['id'],bounding_boxes_overlap=bool(np.all(u>l)),
            coincident_exact_xyz_vertices=int(intersect),coincident_rounded_native_voxel_centres=int(rounded)))
    all_xyz=np.concatenate(all_points);total=len(all_xyz);unique=len(np.unique(records(all_xyz)))
    result=dict(status='GEOMETRY_SOURCE_AUDIT_COMPLETE',cases=rows,pairs=pairs,total_valid_stored_vertices=total,
        unique_exact_xyz_vertices=unique,duplicate_vertex_records=total-unique,
        bbox_overlap_pairs=sum(p['bounding_boxes_overlap'] for p in pairs),
        pairs_with_exact_coincidences=sum(p['coincident_exact_xyz_vertices']>0 for p in pairs),
        pairs_with_rounded_voxel_coincidences=sum(p['coincident_rounded_native_voxel_centres']>0 for p in pairs),
        physical_independent_regions= 'not established; eight representations of one parent',
        no_ink_values_used=True,interpretation='Exact point/voxel coincidences screen repeated source locations. They are not a surface-area estimate, same-sheet certificate or complete input-footprint deduplication. Overlapping bounding boxes alone do not prove overlap of thin surfaces.',
        pending='Freeze patch footprints/aliases and count coverage on the actual valid training support before a physical-coverage improvement claim',
        implementation_sha256=sha(Path(__file__)))
    (r/'GEOMETRY_COVERAGE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','pairs']},indent=2),flush=True)


if __name__=='__main__':main()
