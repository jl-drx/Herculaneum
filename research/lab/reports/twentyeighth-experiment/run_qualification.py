"""Offline frozen scheduler qualification using sparse synthetic encoded objects."""
import argparse
import io
import json
from pathlib import Path
import resource
import sys
import time
import unittest
import weakref

import numpy as np
import numcodecs
import zarr

from chunk_accumulator import build_plan,execute,PRIOR
from reference_mechanics import ray_points
sys.path.insert(0,str(PRIOR/'twentyseventh-experiment'))
from fetch_metadata import pin,save
from local_zarr_reader import schema,LocalReader,sample_checked,COMPRESSOR
import test_chunk_accumulator as test

HERE=Path(__file__).resolve().parent
PREVIOUS=PRIOR/'twentyseventh-experiment'


def synthetic_store(root,plan):
    attrs=json.loads((PREVIOUS/'metadata-run-v1/01-metadata.body').read_text())
    group=zarr.open_group(str(root),mode='w',zarr_format=2,attributes=attrs)
    array=group.create_array('0',shape=plan.spec.shape_zyx,chunks=plan.spec.chunks_zyx,dtype='<u2',
        compressors=numcodecs.Blosc(**{k:v for k,v in COMPRESSOR.items() if k!='id'}),filters=None,
        order='C',fill_value=0,chunk_key_encoding={'name':'v2','separator':'/'},config={'write_empty_chunks':True})
    checked_voxels=0
    for g in plan.groups:
        start=[k*c for k,c in zip(g.key,plan.spec.chunks_zyx)];shape=plan.spec.chunk_shape(g.key)
        z=np.arange(start[0],start[0]+shape[0],dtype=np.uint16)[:,None,None]
        y=np.arange(start[1],start[1]+shape[1],dtype=np.uint16)[None,:,None]
        x=np.arange(start[2],start[2]+shape[2],dtype=np.uint16)[None,None,:]
        known=z+2*y+3*x;slices=tuple(slice(a,a+s) for a,s in zip(start,shape))
        array[slices]=known
        np.testing.assert_array_equal(array[slices],known)
        checked_voxels+=known.size
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<24*1024**2
        del known,z,y,x
    def identity(p):
        r=pin(p);r.pop('path');return r
    manifest=dict(metadata={n:identity(root/n) for n in ('.zattrs','.zgroup','0/.zarray')},
        objects={'0/'+'/'.join(map(str,g.key)):identity(root/'0'/('/'.join(map(str,g.key)))) for g in plan.groups})
    return manifest,checked_voxels


def main():
    a=argparse.ArgumentParser();a.add_argument('--freeze',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    freeze=json.loads(Path(args.freeze).read_text())
    for item in freeze['files']:assert pin(item['path'])==item
    out=Path(args.output);out.mkdir(exist_ok=False);begin=time.monotonic()
    log=io.StringIO();r=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test))
    (out/'test-log.txt').write_text(log.getvalue())
    save(out/'TESTS.json',dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skips=len(r.skipped),success=r.wasSuccessful()))
    assert r.wasSuccessful(),log.getvalue()
    old=json.loads((PREVIOUS/'run-v1/ACTUAL_LAYOUT_DRY_PLAN.json').read_text())
    geometry=Path(old['geometry']['path']);data=np.fromfile(geometry,dtype='<f8').reshape(729,6)
    mapped=~np.all(data==0,axis=1)
    points,known=ray_points(data[:,:3],data[:,3:],[-3.24,0,3.24],[3.24]*3,direction_space='physical_unit',mapped=mapped)
    points=points.reshape(-1,3);known=known.ravel()
    meta=[json.loads((PREVIOUS/f'metadata-run-v1/{i:02d}-metadata.body').read_text()) for i in (1,2,3)]
    spec=schema(*meta);plan_started=time.monotonic();plan=build_plan(points,spec,mapped=known);planning_seconds=time.monotonic()-plan_started
    summary=plan.summary();assert summary['keys_zyx']==old['unique_keys_zyx']
    assert summary['batch_chunks']==[16,16,16,16,6] and summary['positive_contributions']==13392
    save(out/'CHUNK_PLAN.json',dict(**summary,source_frame_authenticated=False,physical_spacing_confirmed=False,
        point_order='same depth-major2187slots as phase27',probe_offsets_um=[-3.24,0,3.24],hypothetical_spacing_um=3.24,
        geometry=pin(geometry),physical_slot_count=729,mapped_slots=558,invalid_slots=171))
    manifest,checked_voxels=synthetic_store(out/'synthetic-store',plan)
    save(out/'SYNTHETIC_MANIFEST.json',manifest)
    baseline=LocalReader(out/'synthetic-store',manifest);baseline_values=np.full(len(points),np.nan);baseline_valid=np.zeros(len(points),dtype=bool)
    then=time.monotonic()
    for b in old['batches']:
        start,stop=b['start_inclusive'],b['stop_exclusive']
        values,valid,stats=sample_checked(points[start:stop],baseline,mapped=known[start:stop])
        baseline_values[start:stop]=values;baseline_valid[start:stop]=valid
    baseline_seconds=time.monotonic()-then
    optimized=LocalReader(out/'synthetic-store',manifest);refs=[]
    def tracked(key):
        assert all(ref() is None for ref in refs),'Previous returned source chunk retained'
        chunk=optimized(key)
        if chunk is not None:refs.append(weakref.ref(chunk))
        return chunk
    then=time.monotonic();result=execute(plan,tracked);optimized_seconds=time.monotonic()-then
    assert all(ref() is None for ref in refs)
    np.testing.assert_array_equal(result['valid'],baseline_valid)
    np.testing.assert_allclose(result['values'],baseline_values,atol=1e-10,rtol=0,equal_nan=True)
    expected=points[:,2]+2*points[:,1]+3*points[:,0]
    error=np.abs(result['values'][result['valid']]-expected[result['valid']]);assert error.max()<=1e-10
    assert not result['missing_source'].any() and not result['nonfinite_source'].any()
    np.testing.assert_array_equal(result['processed'],result['expected'])
    assert baseline.stats['source_reads']==baseline.stats['decoded']==200
    assert optimized.stats['source_reads']==optimized.stats['decoded']==70
    np.savez(out/'COMPARISON_ARRAYS.npz',points_xyz=points,mapped=known,baseline_values=baseline_values,
        baseline_valid=baseline_valid,**{k:result[k] for k in ['values','valid','geometric_valid','missing_source','nonfinite_source','expected','processed']})
    delta=np.abs(result['values'][result['valid']]-baseline_values[result['valid']])
    save(out/'COMPARISON.json',dict(status='PASS',source_kind='synthetic_z_plus_2y_plus_3x_only',
        writer_roundtrip_checked_voxels=checked_voxels,positive_contributions=13392,point_slots=2187,valid_points=1674,
        invalid_points=513,validity_exact=True,coverage_exact=True,maximum_value_difference=float(delta.max()),
        maximum_analytic_error=float(error.max()),tolerance_absolute=1e-10,tolerance_relative=0,
        baseline_stats=baseline.stats,optimized_stats=optimized.stats,baseline_batches=13,optimized_batches=5,
        baseline_decoded_work_bytes=200*4194304,optimized_decoded_work_bytes=70*4194304,
        source_read_reduction_fraction=0.65,planning_seconds=planning_seconds,baseline_seconds=baseline_seconds,
        optimized_seconds=optimized_seconds,one_returned_buffer_verified=True,scheduler_stats=result['stats'],
        native_scan_wire_bytes_measured=False,model_throughput_measured=False,accumulator_resume_supported=False))
    for item in freeze['files']:assert pin(item['path'])==item
    size=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());assert size<26*1024**2
    save(out/'RUN_RECEIPT.json',dict(status='PASS',tests=r.testsRun,failures=0,errors=0,skips=0,
        elapsed_seconds=time.monotonic()-begin,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_freeze=pin(args.freeze),outputs=[pin(out/n) for n in ['TESTS.json','CHUNK_PLAN.json','SYNTHETIC_MANIFEST.json','COMPARISON.json','COMPARISON_ARRAYS.npz']],
        phase_bytes_before_receipt=size,network_requests=0,candidate_pixels=0,model_calls=0,GPU_spend_GBP=0))
    print(json.dumps(dict(receipt=json.loads((out/'RUN_RECEIPT.json').read_text()),comparison=json.loads((out/'COMPARISON.json').read_text())),indent=2))


if __name__=='__main__':main()
