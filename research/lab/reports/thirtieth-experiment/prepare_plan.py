"""Derive immutable sampling support from retained geometry; no data/network/model access."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPORTS=HERE.parent
sys.path.insert(0,str(REPORTS/'twentyeighth-experiment'))
sys.path.insert(0,str(REPORTS/'twentyseventh-experiment'))
from chunk_accumulator import build_plan
from local_zarr_reader import schema


def pin(path):
    raw=path.read_bytes()
    return dict(path=str(path.resolve()),size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def write(path, obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    geometry=REPORTS/'twentyfirst-experiment/frag6-stage-b/run-v1/acquisition/SPARSE_GEOMETRY.json'
    structural=REPORTS/'twentyfirst-experiment/frag6-stage-a/run-v1/acquisition/STRUCTURAL_CONTRACT.json'
    g=json.loads(geometry.read_text());s=json.loads(structural.read_text())
    potential=[row for row in g['geometry']['samples'] if row['potential_native_site']]
    sites=[row for row in potential if row['complete_mapped_neighborhood']]
    assert len(potential)==9 and len(sites)==7
    xyz=np.array([[r['xyz'] for r in row['records_row_major']] for row in sites])
    normals=np.array([[r['normal'] for r in row['records_row_major']] for row in sites])
    assert np.max(abs(np.linalg.norm(normals,axis=2)-1))<1e-5
    # site, layer, neighborhood pixel, XYZ; sign -1 is just layer-axis reversal.
    base=xyz[:,None,:,:]+np.array([-32.,0.,32.])[None,:,None,None]*normals[:,None,:,:]
    blocks=[]
    for displacement in (np.zeros(3),np.array([17.,13.,11.])):
        q=base+displacement
        blocks.extend((q,np.floor(q),np.floor(q+.5)))
    points=np.concatenate([q.reshape(-1,3) for q in blocks])
    meta_dir=REPORTS/'twentyseventh-experiment/metadata-run-v1'
    ledger=json.loads((meta_dir/'request-ledger.json').read_text())
    meta=ledger['requests'][:3]
    bodies={r['relative_path']:json.loads(Path(r['body']['path']).read_text()) for r in meta}
    spec=schema(bodies['.zattrs'],bodies['.zgroup'],bodies['0/.zarray'])
    plan=build_plan(points,spec,max_total_chunks=32,max_total_bytes=128*1024**2)
    assert plan.geometric_valid.all()
    np.save(HERE/'POINTS.npy',points,allow_pickle=False)
    ranges=[]
    for row in sites:
        cy,cx=row['center_yx']
        for y in range(cy-1,cy+2):
            start=8+2*(y*6205+cx-1)
            ranges.append([start,start+5])
    assert len(ranges)==21 and len(set(map(tuple,ranges)))==21
    source_meta=[]
    for r in meta:
        source_meta.append(dict(name=r['relative_path'],url=r['url'],etag=r['headers']['ETag'][0],body=r['body']))
    layers=[]
    for t in s['tiffs']:
        assert t['parsed']['width']==6205 and t['parsed']['height']==8853
        assert t['parsed']['dtype']=='uint16' and t['parsed']['byte_order']=='little' and t['parsed']['strip_offset']==8
        layers.append(dict(layer=t['layer_index'],object=t['object'],ranges=ranges))
    assert [t['layer'] for t in layers]==[0,32,64]
    payload=dict(geometry=pin(geometry),structural=pin(structural),points=pin(HERE/'POINTS.npy'),
        metadata=source_meta,native_base_url=ledger['base_url'],source_shape_zyx=list(spec.shape_zyx),
        native_plan=plan.summary(),sites=[dict(center_yx=r['center_yx'],role='calibration' if i<3 else 'check') for i,r in enumerate(sites)],
        excluded=[dict(center_yx=r['center_yx'],reason='previously unmapped; not replaced') for r in potential if not r['complete_mapped_neighborhood']],
        layers=layers,point_layout='6blocks: direct trilinear/floor/round, wrong-coordinate trilinear/floor/round; each7sites×3layers×9pixels',
        hypothesis_order=['trilinear:+1','trilinear:-1','floor:+1','floor:-1','round:+1','round:-1'],
        limits=dict(max_requests=100,max_reserved_bytes=160*1024**2,max_phase_bytes=192*1024**2,max_rss_bytes=512*1024**2,request_seconds=30,total_seconds=1200),
        acceptance=dict(calibration_mae_max=2.,calibration_max_error_max=8.,runner_up_mae_gap_min=10.,
            each_check_site_mae_max=2.,each_check_site_max_error_max=8.,informative_check_range_min=512.,informative_check_sites_min=3,
            wrong_coordinate_check_mae_min=100.,wrong_coordinate_mae_ratio_min=20.),
        scientific_model_forwards=0,labels_allowed=False,optical_pixels_allowed=False)
    write(HERE/'SAMPLING_PLAN.json',payload)
    print(json.dumps(dict(status='PLANNED',points=len(points),sites=payload['sites'],excluded=payload['excluded'],native=plan.summary()),indent=2))


if __name__=='__main__':main()
