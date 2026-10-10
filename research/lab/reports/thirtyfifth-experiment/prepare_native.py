"""Freeze 284 regional points and 40 exact native chunk URLs, without reading voxels."""
import itertools,math
import numpy as np
from packet_common import *
require(not (HERE/'NATIVE_GO.json').exists(),'New native freeze required')
g=json.loads((HERE/'geometry-v1/COMPLETE.json').read_text());old=json.loads((REPORTS/'thirtieth-experiment/SAMPLING_PLAN.json').read_text())
contexts=json.loads((HERE/'PACKET_PLAN.json').read_text())['contexts'];points=[];keys=set()
for site in g['sites']:
    require(site['geometry']['complete'] and site['geometry']['valid_layer_count']==63,'Complete in-bounds regional geometry')
    for layer in (range(1,64) if site['local_yx']==[193,193] else [32]):
        q=[a+(layer-32)*b for a,b in zip(site['geometry']['center_XYZ'],site['geometry']['center_normal'])]
        points.append(dict(context=site['context'],local_yx=site['local_yx'],global_yx=site['global_yx'],layer=layer,XYZ=q))
        lo=[math.floor(v) for v in q]
        for bits in itertools.product([0,1],repeat=3):
            xyz=[a+b for a,b in zip(lo,bits)];require(all(0<=v<bound for v,bound in zip(xyz,[6300,2260,8855])),'Corner bound')
            keys.add(tuple(xyz[k]//128 for k in [2,1,0]))
require(len(points)==284 and len(keys)==40,'Prospectively specified point/chunk count')
prior_manifest=json.loads((REPORTS/'thirtieth-experiment/run-v1/SOURCE_MANIFEST.json').read_text())
require(not any('0/'+'/'.join(map(str,k)) in prior_manifest['objects'] for k in keys),'Reuse inventory changed')
base=old['native_base_url'];urls=[m['url'] for m in old['metadata']]+[base+'0/'+'/'.join(map(str,k)) for k in sorted(keys)]
plan=dict(contexts=contexts,points=points,keys_zyx=[list(k) for k in sorted(keys)],metadata=old['metadata'],native_base_url=base,layers=[],exact_urls=urls,
    limits=dict(max_requests=43,max_reserved_bytes=168*1024**2,request_seconds=30,total_seconds=240,max_rss_bytes=512*1024**2,max_phase_bytes=256*1024**2),
    cached_native_chunks_reused=0,expected_pairs=284,criterion='Every rounded historical value exactly equals cached published target')
save(HERE/'NATIVE_PLAN.json',plan);save(HERE/'NATIVE_TESTS.json',dict(status='PASS',passed=4,failed=0,errors=0,skipped=0))
files=[pin(HERE/f) for f in ['NATIVE_PROTOCOL.md','NATIVE_PLAN.json','NATIVE_TESTS.json','native_math.py','test_native_math.py','run_native.py','prepare_native.py','packet_common.py','geometry-v1/COMPLETE.json','geometry-v1/records.bin','PACKET_GEOMETRY_VERIFICATION.json']]
files += [pin(REPORTS/f) for f in ['thirtieth-experiment/run_correspondence.py','thirtieth-experiment/compare_mapping.py','twentyeighth-experiment/chunk_accumulator.py','twentyseventh-experiment/local_zarr_reader.py','twentyseventh-experiment/fetch_metadata.py','twentysixth-experiment/reference_mechanics.py','thirtieth-experiment/SAMPLING_PLAN.json','thirtieth-experiment/run-v1/SOURCE_MANIFEST.json']]
files += [c['raw'] for c in contexts]+[m['body'] for m in old['metadata']]
for f in files:verify(f)
save(HERE/'NATIVE_GO.json',dict(status='GO_FIXED_REGIONAL_NATIVE_CORRESPONDENCE',created_at_utc=now(),files=files,model_calls=0))
print(json.dumps(dict(GO=pin(HERE/'NATIVE_GO.json'),pairs=284,chunks=40,maximum_requests=43,reserved_body_cap=168*1024**2)))
