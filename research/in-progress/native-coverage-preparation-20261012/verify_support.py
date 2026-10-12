"""Separate integral-mask and dense-occupancy check of the reserved support."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import tifffile


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=a.work;folder=root/'support-reservation-v2'
    result=json.loads((folder/'RESULTS.json').read_text());rule=json.loads((folder/'RULE.json').read_text())
    data=(folder/'CANDIDATES.csv').read_bytes();assert hashlib.sha256(data).hexdigest()==result['csv_sha256']
    rows=list(csv.DictReader(data.decode().splitlines()));sources=json.loads((root/'MESH_SOURCES.json').read_text())['rows'];expected_ids=set();count_a=0
    lookup={r['id']:r for r in rows};assert len(lookup)==len(rows)
    for source in sources:
        ident=source['id'];xyz=np.stack([tifffile.imread(root/'mesh-arrays'/f'{ident}-mesh-{axis}.raw') for axis in 'xyz'],-1)
        bad=~(np.isfinite(xyz).all(-1)&(xyz>0).all(-1));integral=np.pad(bad.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
        h,w=(np.asarray(xyz.shape[:2])*20).tolist()
        for y in range(0,h-127,96):
            for x in range(0,w-127,96):
                if x//512!=(x+127)//512 or y//512!=(y+127)//512:continue
                # Integer arithmetic independent of the producer's floating
                # floor/ceil conversion at 0.05 stored nodes per pixel.
                x0=x//20-2;y0=y//20-2;x1=(x+127+19)//20+3;y1=(y+127+19)//20+3
                if min(x0,y0)<0 or x1>xyz.shape[1] or y1>xyz.shape[0]:continue
                invalid=integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0]
                if invalid:continue
                key=f'{ident}-x{x:05}-y{y:05}';expected_ids.add(key);r=lookup[key]
                selected=(hashlib.sha256(f'plinius-native-coverage-v1|{source["segment"]}|{x//512}|{y//512}'.encode()).digest()[0]&3)==0
                assert int(r['in_restricted_A'])==selected and int(r['in_broad_B'])==1
                count_a+=selected
                vertices=xyz[y0:y1,x0:x1].reshape(-1,3).astype(np.float64)
                assert np.array_equal(vertices.min(0)-11,[float(r[axis+'_lo']) for axis in 'xyz'])
                assert np.array_equal(vertices.max(0)+11,[float(r[axis+'_hi']) for axis in 'xyz'])
    assert expected_ids==set(lookup) and len(rows)==result['broad_candidates'] and count_a==result['restricted_candidates']
    occupancy={}
    maximum=np.max([[float(r[axis+'_hi']) for axis in 'xyz'] for r in rows],axis=0)
    for spacing in [64,128]:
        shape=(maximum//spacing+1).astype(int);dense={arm:np.zeros(shape,bool) for arm in ['A','B']}
        for r in rows:
            low=(np.array([float(r[axis+'_lo']) for axis in 'xyz'])//spacing).astype(int)
            high=(np.array([float(r[axis+'_hi']) for axis in 'xyz'])//spacing).astype(int)
            sl=tuple(slice(l,h+1) for l,h in zip(low,high));dense['B'][sl]=True
            if int(r['in_restricted_A']):dense['A'][sl]=True
        counts={k:int(v.sum()) for k,v in dense.items()};assert np.all(~dense['A']|dense['B'])
        for arm in counts:assert counts[arm]==result['pooled_context_grid_proxy'][str(spacing)][arm]
        occupancy[str(spacing)]=counts
    a.output.write_text(json.dumps(dict(status='PASS',candidate_records=len(rows),restricted_records=int(count_a),
        eligibility='Independent integer coordinate conversion and summed-area invalid-mask checks',
        occupancy='Dense 3D boolean grids reproduce producer set-union counts',context_grid_counts=occupancy,
        new_model_forwards=0,scope='Separate arithmetic in this workspace; not independent physical annotation or exact sheet-area validation',
        rule_sha256=hashlib.sha256((folder/'RULE.json').read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(a.output.read_text(),flush=True)


if __name__=='__main__':main()
