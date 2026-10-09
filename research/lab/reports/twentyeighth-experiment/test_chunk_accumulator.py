from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
import weakref

import numpy as np

from chunk_accumulator import build_plan,execute,VolumeSpec,PRIOR
from reference_mechanics import sample_trilinear
from test_reference_mechanics import polynomial,source_volume,loader_for,scalar_oracle
sys.path.insert(0,str(PRIOR/'twentyseventh-experiment'))
from local_zarr_reader import LocalReader
from test_local_zarr_reader import fixture


class Tests(unittest.TestCase):
    def setUp(self):
        self.volume=source_volume();self.spec=VolumeSpec(self.volume.shape,(4,3,5),'<f8')
        self.points=np.random.default_rng(2801).uniform([0,0,0],[12,8,10],size=(200,3))

    def run_points(self,points,*,volume=None,spec=None,missing=(),mapped=None,**limits):
        v=self.volume if volume is None else volume;s=self.spec if spec is None else spec
        load,calls=loader_for(v,s,missing);plan=build_plan(points,s,mapped=mapped,**limits)
        result=execute(plan,load);return result,plan,calls

    def test_multilinear_analytic_and_scalar(self):
        result,plan,calls=self.run_points(self.points,max_batch_chunks=1)
        np.testing.assert_allclose(result['values'],polynomial(*self.points.T),atol=1e-10,rtol=0)
        np.testing.assert_allclose(result['values'],[scalar_oracle(self.volume,p) for p in self.points],atol=1e-10,rtol=0)
        self.assertTrue(result['valid'].all());self.assertEqual(len(calls),len(set(calls)))
        np.testing.assert_array_equal(result['expected'],result['processed'])

    def test_phase26_sampler_equivalence(self):
        load,_=loader_for(self.volume,self.spec);values,valid,_=sample_trilinear(self.points,self.spec,load)
        result,_,_=self.run_points(self.points)
        np.testing.assert_array_equal(result['valid'],valid)
        np.testing.assert_allclose(result['values'],values,atol=1e-10,rtol=0)

    def test_batch_size_does_not_change_values(self):
        a,_,_=self.run_points(self.points,max_batch_chunks=1)
        b,_,_=self.run_points(self.points,max_batch_chunks=3)
        np.testing.assert_array_equal(a['values'],b['values']);np.testing.assert_array_equal(a['valid'],b['valid'])

    def test_boundary_corner_and_edge_shapes(self):
        points=np.array([[4.5,2.5,3.5],[12,8,10],[0,0,0],[11.99,7.99,9.99]])
        result,_,_=self.run_points(points,max_batch_chunks=1)
        np.testing.assert_allclose(result['values'],polynomial(*points.T),atol=1e-10,rtol=0)
        self.assertEqual(result['expected'].tolist(),[8,1,1,8])

    def test_invalid_geometry_preserved(self):
        points=np.array([[0,0,0],[-1e-12,0,0],[13,0,0],[float('nan'),1,1],[1,1,1]])
        r,_,calls=self.run_points(points,mapped=np.array([True,True,True,True,False]))
        self.assertEqual(r['valid'].tolist(),[True,False,False,False,False])
        self.assertEqual(r['processed'].tolist(),[1,0,0,0,0]);self.assertEqual(len(calls),1)
        self.assertFalse(r['missing_source'].any());self.assertTrue(np.isnan(r['values'][1:]).all())

    def test_zero_valid_and_missing_separate(self):
        r,_,_=self.run_points([[1,1,1],[5,3,4]],volume=np.zeros_like(self.volume),missing={(1,1,1)})
        self.assertEqual(r['valid'].tolist(),[True,False]);self.assertEqual(r['values'][0],0)
        self.assertEqual(r['missing_source'].tolist(),[False,True]);self.assertTrue(np.isnan(r['values'][1]))

    def test_missing_contribution_spans_batches(self):
        points=np.array([[4.5,2.5,3.5],[1,1,1]])
        r,plan,calls=self.run_points(points,missing={(1,1,1)},max_batch_chunks=1)
        self.assertEqual(r['valid'].tolist(),[False,True]);self.assertEqual(r['processed'].tolist(),[8,1])
        load,_=loader_for(self.volume,self.spec,missing={(1,1,1)})
        v,ok,_=sample_trilinear(points,self.spec,load)
        np.testing.assert_array_equal(r['valid'],ok);np.testing.assert_allclose(r['values'],v,atol=1e-10,rtol=0,equal_nan=True)

    def test_zero_weight_missing_not_requested(self):
        r,_,calls=self.run_points([[4,2,3]],missing={(1,1,1)})
        self.assertTrue(r['valid'][0]);self.assertEqual(calls,[(0,0,0)])

    def test_nonfinite_source(self):
        v=self.volume.copy();v[1,1,1]=np.nan;v[2,2,2]=np.inf
        r,_,_=self.run_points([[1.5,1.5,1.5],[0,0,0]],volume=v)
        self.assertEqual(r['valid'].tolist(),[False,True]);self.assertEqual(r['nonfinite_source'].tolist(),[True,False])
        self.assertFalse(r['missing_source'].any());self.assertEqual(r['processed'].tolist(),[8,1])

    def test_multiple_unknown_reasons_remain_separate(self):
        v=self.volume.copy();v[3,2,4]=np.nan
        r,_,_=self.run_points([[4.5,2.5,3.5]],volume=v,missing={(1,1,1)},max_batch_chunks=1)
        self.assertTrue(r['missing_source'][0]);self.assertTrue(r['nonfinite_source'][0]);self.assertFalse(r['valid'][0])

    def test_noncontributing_nonfinite_ignored(self):
        v=self.volume.copy();v[1,1,2]=np.nan
        r,_,_=self.run_points([[1,1,1]],volume=v)
        self.assertTrue(r['valid'][0]);self.assertFalse(r['nonfinite_source'][0])

    def test_duplicates_not_duplicate_chunk_loads(self):
        r,plan,calls=self.run_points([[1.3,1.4,1.5]]*100)
        self.assertEqual(len(calls),1);self.assertTrue(r['valid'].all());self.assertEqual(len(set(r['values'])),1)
        self.assertTrue((r['processed']==8).all())

    def test_budget_rejections(self):
        limits=[dict(max_points=0),dict(max_total_chunks=0),dict(max_total_bytes=0),dict(max_plan_bytes=10),
                dict(max_batch_bytes=479),dict(max_batch_chunks=0),dict(max_plan_bytes=30)]
        for limit in limits:
            with self.subTest(limit=limit),self.assertRaises(ValueError):build_plan([[1.5,1.5,1.5]],self.spec,**limit)

    def test_physical_padded_edge_budget(self):
        spec=VolumeSpec((1,1,1),(128,128,128),'<u2')
        with self.assertRaises(ValueError):build_plan([[0,0,0]],spec,max_batch_bytes=1024**2)

    def test_mutated_plan_fails_before_loader(self):
        p=build_plan(self.points,self.spec);calls=[]
        p.groups[0].weights.flags.writeable=True;p.groups[0].weights[0]+=0.1
        with self.assertRaisesRegex(ValueError,'Changed plan'):execute(p,lambda k:calls.append(k))
        self.assertEqual(calls,[])

    def test_changed_schedule_fails_before_loader(self):
        p=build_plan(self.points,self.spec);calls=[]
        bad=replace(p,batches=p.batches+p.batches[:1])
        with self.assertRaises(ValueError):execute(bad,lambda k:calls.append(k))
        self.assertEqual(calls,[])

    def test_caller_mutation_does_not_change_plan(self):
        points=self.points.copy();p=build_plan(points,self.spec);points[:]=0
        load,_=loader_for(self.volume,self.spec);r=execute(p,load)
        np.testing.assert_allclose(r['values'],polynomial(*self.points.T),atol=1e-10,rtol=0)

    def test_wrong_shape_dtype_endian(self):
        p=build_plan([[1,1,1]],self.spec)
        for a in [np.zeros((1,1,1)),np.zeros((4,3,5),dtype='<f4'),np.zeros((4,3,5),dtype='>f8')]:
            with self.assertRaises(ValueError):execute(p,lambda key:a)

    def test_single_live_returned_chunk(self):
        p=build_plan(self.points,self.spec);refs=[];load,_=loader_for(self.volume,self.spec)
        def tracked(k):
            self.assertTrue(all(r() is None for r in refs))
            a=load(k);refs.append(weakref.ref(a));return a
        r=execute(p,tracked)
        self.assertTrue(all(ref() is None for ref in refs));self.assertEqual(r['stats']['max_scheduler_returned_chunks'],1)

    def test_interruption_no_partial_result(self):
        p=build_plan(self.points,self.spec)
        for count in [0,1,len(p.groups)]:
            load,calls=loader_for(self.volume,self.spec)
            with self.subTest(count=count),self.assertRaises(InterruptedError):execute(p,load,interrupt_after_chunks=count)
            self.assertEqual(len(calls),count)
        load,calls=loader_for(self.volume,self.spec);r=execute(p,load)
        np.testing.assert_allclose(r['values'],polynomial(*self.points.T),atol=1e-10,rtol=0)
        self.assertEqual(len(calls),len(p.groups))

    def test_loader_failure_propagates(self):
        p=build_plan(self.points,self.spec);load,calls=loader_for(self.volume,self.spec)
        def fail(k):
            if calls:raise RuntimeError('Injected object error')
            return load(k)
        with self.assertRaises(RuntimeError):execute(p,fail)
        self.assertEqual(len(calls),1)

    def test_composed_encoded_corruption(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'store';known,manifest=fixture(root)
            path=root/'0/0/0/0';b=bytearray(path.read_bytes());b[-1]^=1;path.write_bytes(b)
            reader=LocalReader(root,manifest);p=build_plan([[1,1,1]],reader.spec)
            with self.assertRaises(ValueError):execute(p,reader)

    def test_empty_and_singleton(self):
        r,_,calls=self.run_points(np.empty((0,3)));self.assertEqual(len(r['values']),0);self.assertEqual(calls,[])
        spec=VolumeSpec((1,1,3),(1,1,2),'<u2');v=np.array([[[0,2,4]]],dtype='<u2')
        r,_,_=self.run_points([[0.5,0,0],[2,0,0]],volume=v,spec=spec,max_batch_chunks=1)
        np.testing.assert_array_equal(r['values'],[1,4])

    def test_axes_negative_control(self):
        r,_,_=self.run_points([[7,2,1]])
        self.assertGreater(abs(r['values'][0]-polynomial(1,2,7)),100)


if __name__=='__main__':unittest.main(verbosity=2)
