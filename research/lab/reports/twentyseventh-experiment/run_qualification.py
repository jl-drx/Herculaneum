"""Frozen offline qualification and all-record geometry planning."""
import argparse
import io
import json
import math
from pathlib import Path
import resource
import time
import unittest

import numpy as np

from fetch_metadata import pin,save,require
from local_zarr_reader import LocalReader,schema,sample_checked
from reference_mechanics import ray_points,plan_samples
import test_local_zarr_reader as test

HERE=Path(__file__).resolve().parent
GEOMETRY=HERE.parent/'twentyfirst-experiment/frag6-stage-b/run-v1/acquisition/raw/records.bin'


def dry_plan(spec):
    data=np.fromfile(GEOMETRY,dtype='<f8').reshape(729,6)
    mapped=~np.all(data==0,axis=1)
    assert int(mapped.sum())==558
    points,known=ray_points(data[:,:3],data[:,3:],[-3.24,0,3.24],[3.24]*3,
                           direction_space='physical_unit',mapped=mapped)
    points=points.reshape(-1,3);known=known.ravel()
    physical_chunk=math.prod(spec.chunks_zyx)*2
    try:
        whole=plan_samples(points,spec,mapped=known)
        require(len(whole.groups)*physical_chunk<=64*1024**2,'Physical chunk byte budget')
        whole_status='PASS';whole_reason=None
    except ValueError as error:whole_status='REJECTED';whole_reason=str(error)
    # Each point fits the unchanged limits. Union is diagnostic accounting, not a budget override.
    all_keys=set();current=set();ranges=[];start=0;corner_count=0;valid_count=0
    for i,p in enumerate(points):
        one=plan_samples(p[None,:],spec,mapped=known[i:i+1])
        keys=set(one.groups);all_keys|=keys;corner_count+=one.positive_weight_corners;valid_count+=int(one.geometric_valid.sum())
        assert len(keys)*physical_chunk<=64*1024**2
        proposed=current|keys
        if len(proposed)>128 or len(proposed)*physical_chunk>64*1024**2:
            ranges.append((start,i));start=i;current=keys
        else:current=proposed
    ranges.append((start,len(points)))
    batches=[];covered=[]
    for start,stop in ranges:
        plan=plan_samples(points[start:stop],spec,mapped=known[start:stop]);s=plan.summary()
        assert len(plan.groups)*physical_chunk<=64*1024**2
        batches.append(dict(start_inclusive=start,stop_exclusive=stop,physical_storage_bytes=len(plan.groups)*physical_chunk,**s))
        covered.extend(range(start,stop))
    assert covered==list(range(len(points)))
    assert sum(b['geometric_valid_count'] for b in batches)==valid_count
    assert sum(b['positive_weight_corners'] for b in batches)==corner_count
    union_bytes=sum(math.prod(spec.chunk_shape(k))*2 for k in all_keys)
    total_chunks=sum(b['unique_chunks'] for b in batches)
    return dict(status='PASS_DETERMINISTIC_SPLIT',geometry=pin(GEOMETRY),
        whole_batch_status=whole_status,whole_batch_reason=whole_reason,shape_zyx=spec.shape_zyx,chunks_zyx=spec.chunks_zyx,
        retained_records=729,mapped_records=558,invalid_records=171,point_count=len(points),
        geometric_valid_count=valid_count,positive_weight_corners=corner_count,
        source_frame_confirmed=False,physical_spacing_confirmed=False,hypothetical_spacing_um=3.24,
        probe_offsets_um=[-3.24,0,3.24],point_order='depth-major then original retained-record order',
        unique_chunks=len(all_keys),unique_logical_bytes=union_bytes,unique_physical_bytes=len(all_keys)*physical_chunk,
        unique_keys_zyx=[list(k) for k in sorted(all_keys)],batch_count=len(batches),batches=batches,
        total_chunk_decodes_without_cross_batch_cache=total_chunks,
        repeated_chunk_decodes_without_cross_batch_cache=total_chunks-len(all_keys),
        total_physical_bytes_without_cross_batch_cache=total_chunks*physical_chunk,
        encoded_network_bytes_unknown=True,max_batch_bytes=64*1024**2,max_batch_chunks=128,
        source_objects_read=0,original_stage_C_open=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    freeze=json.loads(Path(args.freeze).read_text())
    for item in freeze['files']:assert pin(item['path'])==item
    out=Path(args.output);out.mkdir(exist_ok=False);started=time.monotonic()
    buffer=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromModule(test)
    result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    (out/'test-log.txt').write_text(buffer.getvalue())
    save(out/'TESTS.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),success=result.wasSuccessful()))
    save(out/'SYNTHETIC_TEST_IDENTITIES.json',test.FIXTURE_IDENTITIES)
    assert result.wasSuccessful(),buffer.getvalue()
    known,manifest=test.fixture(out/'native-sized-synthetic-store',(129,129,129),(128,128,128))
    save(out/'SYNTHETIC_MANIFEST.json',manifest)
    reader=LocalReader(out/'native-sized-synthetic-store',manifest,cache=out/'encoded-cache')
    max_error=0
    for key in np.ndindex(2,2,2):
        chunk=reader(key);slices=tuple(slice(k*128,min(k*128+128,s)) for k,s in zip(key,known.shape))
        np.testing.assert_array_equal(chunk,known[slices])
    # Include both sides of every native chunk boundary, exact last voxel and valid zero.
    points=np.array([[0,0,0],[1.2,2.3,3.4],[127.5,127.5,127.5],[128,128,128],[127.1,128,126.8],[128.1,128,128]])
    values,valid,stats=sample_checked(points,reader)
    np.testing.assert_array_equal(valid,[True,True,True,True,True,False])
    error=np.abs(values[:5]-points[:5]@np.array([7,5,3]));assert error.max()<1e-10
    save(out/'NATIVE_SIZE_SYNTHETIC_RESULT.json',dict(status='PASS',shape=[129]*3,chunks=[128]*3,
        source_voxels=known.size,reference_writer='zarr3.4.0 writing format2 + numcodecs0.17.0',
        all_voxels_exact=True,full_and_edge_chunks_checked=8,sample_points_xyz=points.tolist(),
        sample_values=[float(v) if ok else None for v,ok in zip(values,valid)],valid=valid.tolist(),
        maximum_sample_absolute_error=float(error.max()),reader_stats=reader.stats,sampling_stats=stats,
        native_candidate_voxels_read=0))
    attrs=json.loads((HERE/'metadata-run-v1/01-metadata.body').read_text())
    group=json.loads((HERE/'metadata-run-v1/02-metadata.body').read_text())
    array=json.loads((HERE/'metadata-run-v1/03-metadata.body').read_text())
    spec=schema(attrs,group,array);plan=dry_plan(spec);save(out/'ACTUAL_LAYOUT_DRY_PLAN.json',plan)
    for item in freeze['files']:assert pin(item['path'])==item
    phase_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());assert phase_bytes<32*1024**2
    save(out/'RUN_RECEIPT.json',dict(status='PASS_OFFLINE_QUALIFICATION',source_freeze=pin(args.freeze),
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        tests=result.testsRun,failures=0,errors=0,skips=0,network_requests=0,new_candidate_pixels=0,model_calls=0,
        phase_bytes_before_receipt=phase_bytes,outputs=[pin(out/n) for n in ['TESTS.json','NATIVE_SIZE_SYNTHETIC_RESULT.json','ACTUAL_LAYOUT_DRY_PLAN.json','SYNTHETIC_MANIFEST.json']]))
    print(json.dumps(dict(receipt=json.loads((out/'RUN_RECEIPT.json').read_text()),plan_summary={k:v for k,v in plan.items() if k not in ('batches','unique_keys_zyx')}),indent=2))


if __name__=='__main__':main()
