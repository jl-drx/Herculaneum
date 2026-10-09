"""Independent phase-17 retained-row reconstruction and profile calculation."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time
import numpy as np
from independent_math import exact_pool, profile

ROOT = Path('/workspace/herculaneum-lab')
P17 = ROOT/'reports/seventeenth-experiment'
P16 = ROOT/'reports/sixteenth-experiment'
P15 = ROOT/'reports/fifteenth-experiment'
PINS = {
    P17/'RESEARCH_DESIGN.md': '08bc48f203d6722216c70d54b961d0d29ce59204359da49d2b6110661a9a9cc1',
    P17/'PREEXECUTION_CLARIFICATIONS.md': '22ed3e0749d56bd61113b6a2cd74f6ae7373082a34a1c090352ccba99eb2c643',
    P16/'production-pooling-v2/pooling-run/pooled-descriptor.json': 'fef036010df97b465830da24a69a6ee583f61d0ac7831898f30b4e6e351d6e07',
    P16/'production-acquisition/pixel-run/raw-provenance.json': 'c9cc64e4c39ee063bbc67bcddf547fabd51c71af36143a1527c4130867e0185c',
    P15/'BENCHMARK_CONTRACT.json': 'a4026cce7ca55d5aaeeaa0736061ab6b7c8aa13875155682d24e74c139fe5735',
}


def digest(data): return hashlib.sha256(data).hexdigest()


def read_pin(pin):
    path = Path(pin['path'])
    data = path.read_bytes()
    assert len(data) == pin.get('size_bytes',pin.get('bytes')) and digest(data)==pin['sha256'], str(path)
    return data


def read_array(pin): return np.load(io.BytesIO(read_pin(pin)), allow_pickle=False)


def json_dump(path,value):
    with Path(path).open('x') as out: json.dump(value,out,indent=2,allow_nan=False); out.write('\n')


def reconstruct_row(body,row,contract,case):
    x0,y0,width,height = case['bbox_source_xywh']
    y = row['native_y']
    assert y0 <= y < y0+height
    offset = contract['strip_offset']+2*(y*contract['width']+x0)
    assert row['range_inclusive'] == [offset,offset+2*width-1]
    assert row['payload_bytes']==2*width
    start = row['payload_offset']; data=body[start:start+2*width]
    assert len(data)==2*width and digest(data)==row['payload_sha256']
    endian = {'<u2':'<','>u2':'>'}[contract['dtype']]
    # struct unpack checks native interpretation independently of ndarray buffer dtype.
    return np.asarray(struct.unpack(endian+str(width)+'H',data), dtype=np.uint16)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); args=parser.parse_args()
    output=Path(args.out); output.mkdir(parents=True,exist_ok=False)
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
    resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    started=time.monotonic(); cpu_started=time.process_time()
    metadata={}
    for path, expected in PINS.items():
        data=path.read_bytes(); assert digest(data)==expected,str(path)
        if path.suffix=='.json': metadata[path.name]=json.loads(data)
    descriptor=metadata['pooled-descriptor.json']; raw=metadata['raw-provenance.json']; benchmark=metadata['BENCHMARK_CONTRACT.json']
    cases=descriptor['cases']; assert [c['id'] for c in cases]==[f'frag{f}-context{c}' for f in (1,2,3) for c in (1,2)]
    bench={c['case_id']:c for c in benchmark['selected_contexts']}
    contracts={(c['fragment'],c['plane_index']):c for c in raw['tiff_contracts']}
    schedule={}
    for b in raw['batches']: schedule.setdefault((b['case_id'],b['plane_index']),[]).append(b)
    assert len(contracts)==153 and len(schedule)==306
    json_dump(output/'start-receipt.json',dict(input_pins={str(k):v for k,v in PINS.items()},
       source_sha256={p.name:digest(p.read_bytes()) for p in [Path(__file__),Path(__file__).with_name('independent_math.py')]},
       python=sys.version,numpy=np.__version__,started_epoch=time.time(),argv=sys.argv,author_helper_imports=0,torch_imports=0))
    results=[]; batches=0; body_bytes=0; numeric_values=0
    for case in cases:
        caseid=case['id']; bc=bench[caseid]
        x0,y0,w,h=case['bbox_source_xywh']
        assert [y0,x0,h,w] == case['native_bbox_yxhw'] == bc['native_bbox_yxhw'] and case['bbox_source_xywh']==bc['bbox_source_xywh']
        assert bc['source_pixel_center_mapping']==dict(x=f'{x0}+1+3*x_coarse',y=f'{y0}+1+3*y_coarse')
        for key, name in [('labels','labels.npy'),('surface_mask','surface-mask.npy'),('known_mask','known-mask.npy'),('primary_mask','primary-mask.npy')]:
            assert case['arrays'][key]['sha256']==bc['files'][name]['sha256']
        arrays={k:read_array(v) for k,v in case['arrays'].items()}
        assert arrays['surface'].shape==(17,384,384) and arrays['surface'].dtype==np.float32
        assert np.isfinite(arrays['surface']).all() and np.min(arrays['surface'])>=0 and np.max(arrays['surface'])<=255
        label=arrays['labels']; primary=arrays['primary_mask']; surface=arrays['surface_mask']; known=arrays['known_mask']
        assert label.dtype==np.uint8 and np.isin(label,[0,1]).all()
        assert label.shape==primary.shape==surface.shape==known.shape==(384,384)
        assert primary.dtype==surface.dtype==known.dtype==np.bool_
        border=np.zeros((384,384),bool); border[32:352,32:352]=True
        assert np.array_equal(primary,known&border) and not np.any(known&~surface)
        support=dict(all=np.ones((384,384),bool),surface=surface,primary_positive=primary&(label==1),primary_negative=primary&(label==0))
        native_support={k:np.kron(v,np.ones((3,3),bool)) for k,v in support.items()}
        assert int(primary.sum())==bc['primary_count'] and int(support['primary_positive'].sum())==bc['primary_positive'] and int(support['primary_negative'].sum())==bc['primary_negative']
        outcase=dict(id=caseid,support_counts={k:int(v.sum()) for k,v in support.items()},native_profiles=[],pooled_profiles=[],bitwise_pooled_planes=0)
        triplet=[]; native_panels=[]; rows=[]; columns=[]
        for z in range(7,58):
            assert time.monotonic()-started<600,'wall cap'
            contract=contracts[(caseid.split('-')[0],z)]
            assert contract['orientation']==1 and contract['photometric']=='min-is-black'
            plane=np.empty((1152,1152),np.uint16); seen=set()
            for index in sorted(schedule[(caseid,z)],key=lambda i:i['native_row_start']):
                batch=json.loads(read_pin(index['batch']))
                for k in ['case_id','fragment','plane_index','native_row_start','row_count','request_id']: assert batch[k]==index[k]
                assert batch['etag']==contract['etag'] and batch['source_url']==contract['url'] and batch['object_total_bytes']==contract['object_total_bytes']
                body=read_pin(batch['multipart']); batches+=1; body_bytes+=len(body)
                assert len(batch['rows'])==len(batch['parts'])==index['row_count']
                parts={tuple(p['range_inclusive']):p for p in batch['parts']}
                for row in batch['rows']:
                    part=parts[tuple(row['range_inclusive'])]
                    for k in ['payload_offset','payload_bytes','payload_sha256']: assert part[k]==row[k]
                    local=row['native_y']-y0; assert local not in seen; seen.add(local)
                    plane[local]=reconstruct_row(body,row,contract,case)
            assert seen==set(range(1152))
            numeric_values+=plane.size
            outcase['native_profiles'].append(dict(plane_index=z,nominal_depth_um=3.24*(z-32),supports={k:profile(plane,v,65535) for k,v in native_support.items()}))
            if z in [8,20,32,44,56]: native_panels.append(plane.copy())
            rows.append(plane[577].copy()); columns.append(plane[:,577].copy())
            triplet.append(plane)
            if len(triplet)==3:
                j=(z-9)//3; pooled=exact_pool(np.stack(triplet)); triplet=[]
                assert pooled.tobytes()==arrays['surface'][j].tobytes(),(caseid,j)
                outcase['bitwise_pooled_planes']+=1
                outcase['pooled_profiles'].append(dict(plane_index=8+3*j,pooled_index=j,nominal_depth_um=3.24*((8+3*j)-32),supports={k:profile(pooled,v,255) for k,v in support.items()}))
        assert not triplet and outcase['bitwise_pooled_planes']==17
        np.savez(output/(caseid+'-display-source.npz'),native_display=np.stack(native_panels),row577=np.stack(rows),column577=np.stack(columns))
        results.append(outcase); print(caseid,'PASS',flush=True)
    assert batches==1530 and body_bytes==848908566 and numeric_values==406093824
    stats=dict(status='PASS',cases=results,raw_batches=batches,raw_body_bytes=body_bytes,native_values=numeric_values,
               native_profiles=306,pooled_profiles=102,profile_support_records=1632,bitwise_pooled_arrays=6,
               wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu_started,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    assert stats['peak_rss_bytes']<=3*1024**3 and stats['wall_seconds']<=600 and stats['cpu_seconds']<=1200
    json_dump(output/'independent-ct-diagnostics.json',stats)
    print(json.dumps({k:v for k,v in stats.items() if k!='cases'},indent=2))

if __name__=='__main__': main()
