import itertools
import math
import unittest
import numpy as np
from reference_mechanics import VolumeSpec,ray_points,plan_samples,sample_trilinear,affine_landmark_check


def polynomial(x,y,z):
    return 7+2*x+11*y+101*z+0.1*x*y+0.2*y*z+0.3*x*z+0.01*x*y*z


def source_volume(shape=(11,9,13)):
    z,y,x=np.indices(shape,dtype=np.float64)
    return polynomial(x,y,z)


def loader_for(volume,spec,missing=()):
    calls=[]
    def load(key):
        calls.append(key)
        if key in missing:return None
        slices=tuple(slice(k*c,min((k+1)*c,s)) for k,c,s in zip(key,spec.chunks_zyx,spec.shape_zyx))
        return volume[slices].copy()
    return load,calls


def scalar_oracle(volume,point):
    # Independent direct whole-array interpolation, no chunk planning imports.
    x,y,z=map(float,point)
    if not all(math.isfinite(v) for v in (x,y,z)):return math.nan
    if not (0<=x<=volume.shape[2]-1 and 0<=y<=volume.shape[1]-1 and 0<=z<=volume.shape[0]-1):return math.nan
    total=[]
    for k in (math.floor(z),math.floor(z)+1):
        wz=max(0,1-abs(z-k))
        for j in (math.floor(y),math.floor(y)+1):
            wy=max(0,1-abs(y-j))
            for i in (math.floor(x),math.floor(x)+1):
                wx=max(0,1-abs(x-i));w=wx*wy*wz
                if w==0:continue
                value=float(volume[k,j,i])
                if not math.isfinite(value):return math.nan
                total.append(w*value)
    return math.fsum(total)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.volume=source_volume();self.spec=VolumeSpec(self.volume.shape,(4,3,5),'<f8')

    def sample(self,points,volume=None,spec=None,missing=(),**kwargs):
        load,calls=loader_for(self.volume if volume is None else volume,self.spec if spec is None else spec,missing)
        answer=sample_trilinear(points,self.spec if spec is None else spec,load,**kwargs)
        return *answer,calls

    def test_multilinear_analytic_truth(self):
        rng=np.random.default_rng(826)
        points=rng.uniform([0,0,0],[12,8,10],size=(400,3))
        values,valid,_,_=self.sample(points)
        self.assertTrue(valid.all());np.testing.assert_allclose(values,polynomial(*points.T),rtol=0,atol=1e-10)

    def test_independent_scalar_oracle(self):
        rng=np.random.default_rng(827)
        points=rng.uniform([0,0,0],[12,8,10],size=(200,3))
        values,valid,_,_=self.sample(points)
        expected=[scalar_oracle(self.volume,p) for p in points]
        self.assertTrue(valid.all());np.testing.assert_allclose(values,expected,rtol=0,atol=1e-10)

    def test_chunk_faces_edges_corners(self):
        points=np.array(list(itertools.product([4.,4.5,5.],[2.,2.5,3.],[3.,3.5,4.])))
        values,valid,_,_=self.sample(points)
        self.assertTrue(valid.all());np.testing.assert_allclose(values,polynomial(*points.T),rtol=0,atol=1e-10)

    def test_outer_integer_boundaries(self):
        points=np.array(list(itertools.product([0.,12.],[0.,8.],[0.,10.])))
        values,valid,_,_=self.sample(points)
        self.assertTrue(valid.all());np.testing.assert_array_equal(values,polynomial(*points.T))

    def test_no_clamping_or_nonfinite_fetch(self):
        points=[[-1e-10,0,0],[12+1e-10,0,0],[0,float('nan'),0],[0,0,float('inf')]]
        values,valid,stats,calls=self.sample(points)
        self.assertFalse(valid.any());self.assertTrue(np.isnan(values).all());self.assertEqual(calls,[])

    def test_valid_zero_is_not_missing(self):
        volume=np.zeros(self.volume.shape,dtype='<f8')
        values,valid,_,_=self.sample([[1.3,2.2,3.1]],volume=volume)
        self.assertTrue(valid[0]);self.assertEqual(values[0],0)

    def test_missing_positive_weight_chunk_invalidates(self):
        values,valid,stats,calls=self.sample([[4.5,2.5,3.5]],missing={(1,1,1)})
        self.assertFalse(valid[0]);self.assertTrue(np.isnan(values[0]));self.assertEqual(stats['missing_chunks'],1)

    def test_zero_weight_missing_chunk_not_loaded(self):
        values,valid,_,calls=self.sample([[4.,2.,3.]],missing={(1,1,1)})
        self.assertTrue(valid[0]);self.assertEqual(calls,[(0,0,0)])

    def test_nonfinite_positive_corner_invalidates(self):
        volume=self.volume.copy();volume[1,1,1]=np.nan
        values,valid,_,_=self.sample([[1.5,1.5,1.5]],volume=volume)
        self.assertFalse(valid[0]);self.assertTrue(np.isnan(values[0]))

    def test_nonfinite_zero_weight_or_unused_voxel_ignored(self):
        volume=self.volume.copy();volume[1,1,2]=np.nan;volume[0,0,0]=np.inf
        values,valid,_,_=self.sample([[1.,1.,1.]],volume=volume)
        self.assertTrue(valid[0]);self.assertEqual(values[0],self.volume[1,1,1])

    def test_duplicate_points_deduplicate_chunk_reads(self):
        values,valid,stats,calls=self.sample([[1.2,1.3,1.4]]*20)
        self.assertEqual(len(calls),1);self.assertTrue(valid.all());self.assertEqual(len(set(values)),1)

    def test_missing_chunk_cached_once(self):
        values,valid,stats,calls=self.sample([[1.2,1.3,1.4]]*20,missing={(0,0,0)})
        self.assertEqual(len(calls),1);self.assertFalse(valid.any());self.assertEqual(stats['missing_chunks'],1)

    def test_partition_invariance(self):
        points=np.random.default_rng(828).uniform([0,0,0],[12,8,10],size=(200,3))
        a,av,_,_=self.sample(points)
        alternate=VolumeSpec(self.volume.shape,(6,5,7),'<f8')
        b,bv,_,_=self.sample(points,spec=alternate)
        np.testing.assert_array_equal(av,bv);np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)

    def test_explicit_invalid_geometry_not_sampled(self):
        values,valid,_,calls=self.sample([[1,1,1]],mapped=np.array([False]))
        self.assertFalse(valid[0]);self.assertEqual(calls,[])

    def test_single_voxel_dimensions(self):
        volume=np.array([[[0.,2.,4.]]]);spec=VolumeSpec(volume.shape,(1,1,2),'<f8')
        values,valid,_,_=self.sample([[.5,0,0],[2,0,0]],volume=volume,spec=spec)
        self.assertTrue(valid.all());np.testing.assert_array_equal(values,[1,4])

    def test_uint16_source(self):
        volume=np.arange(27,dtype='<u2').reshape(3,3,3);spec=VolumeSpec(volume.shape,(2,2,2),'<u2')
        values,valid,_,_=self.sample([[.5,.5,.5]],volume=volume,spec=spec)
        self.assertTrue(valid[0]);self.assertEqual(values[0],6.5)

    def test_budgets_fail_before_loader(self):
        for limit in ({'max_chunks':0},{'max_bytes':1},{'max_points':0}):
            calls=[]
            with self.assertRaises(ValueError):sample_trilinear([[1.5,1.5,1.5]],self.spec,lambda key:calls.append(key),**limit)
            self.assertEqual(calls,[])

    def test_wrong_chunk_shape_rejected(self):
        with self.assertRaisesRegex(ValueError,'shape'):
            sample_trilinear([[1,1,1]],self.spec,lambda key:np.zeros((1,1,1)))

    def test_wrong_dtype_rejected(self):
        with self.assertRaisesRegex(ValueError,'dtype'):
            sample_trilinear([[1,1,1]],self.spec,lambda key:np.zeros((4,3,5),dtype=np.float32))

    def test_swapped_axes_negative_control(self):
        point=np.array([[1.,2.,7.]])
        wrong,_,_,_=self.sample(point[:,::-1])
        self.assertGreater(abs(wrong[0]-polynomial(*point[0])),100)

    def test_half_voxel_negative_control(self):
        point=np.array([[2.,2.,2.]])
        wrong,_,_,_=self.sample(point+.5)
        self.assertGreater(abs(wrong[0]-polynomial(*point[0])),10)

    def test_empty_point_batch(self):
        values,valid,stats,calls=self.sample(np.empty((0,3)))
        self.assertEqual(len(values),0);self.assertEqual(calls,[])


class RayTests(unittest.TestCase):
    def test_anisotropic_physical_unit(self):
        vector=np.array([[1.,2.,2.]])/3
        points,valid=ray_points([[10,10,10]],vector,[-6,0,6],[2,3,4],direction_space='physical_unit',mapped=np.array([True]))
        distance=(points[:,0]-[10,10,10])*[2,3,4]
        np.testing.assert_allclose(distance,np.array([-6,0,6])[:,None]*vector,rtol=0,atol=1e-12)

    def test_anisotropic_voxel_direction_vector(self):
        vector=np.array([[1.,2.,2.]])/3
        points,valid=ray_points([[10,10,10]],vector,[6],[2,3,4],direction_space='voxel_unit_vector',mapped=np.array([True]))
        displacement=(points[0,0]-10)*[2,3,4]
        self.assertAlmostEqual(np.linalg.norm(displacement),6,places=12)
        expected=vector[0]*6/np.linalg.norm(vector[0]*[2,3,4])
        np.testing.assert_allclose(points[0,0]-10,expected,rtol=0,atol=1e-12)

    def test_reverse_offsets_reorder_exactly(self):
        args=([[4,4,4]],[[0,0,1]])
        a,_=ray_points(*args,[-2,0,2],[2,2,2],direction_space='physical_unit',mapped=np.array([True]))
        b,_=ray_points(*args,[2,0,-2],[2,2,2],direction_space='physical_unit',mapped=np.array([True]))
        np.testing.assert_array_equal(a[::-1],b)

    def test_sentinel_preserved_not_replaced(self):
        points,valid=ray_points([[0,0,0],[1,1,1]],[[0,0,0],[0,0,1]],[0],[1,1,1],direction_space='physical_unit',mapped=np.array([False,True]))
        self.assertTrue(np.isnan(points[0,0]).all());self.assertFalse(valid[0,0]);self.assertTrue(valid[0,1])

    def test_nonunit_mapped_vector_rejected(self):
        with self.assertRaisesRegex(ValueError,'unit'):
            ray_points([[1,1,1]],[[0,0,2]],[0],[1,1,1],direction_space='physical_unit',mapped=np.array([True]))

    def test_ambiguous_convention_rejected(self):
        with self.assertRaisesRegex(ValueError,'convention'):
            ray_points([[1,1,1]],[[0,0,1]],[0],[1,1,1],direction_space='unknown',mapped=np.array([True]))

    def test_nonpositive_spacing_rejected(self):
        with self.assertRaisesRegex(ValueError,'spacing'):
            ray_points([[1,1,1]],[[0,0,1]],[0],[1,0,1],direction_space='physical_unit',mapped=np.array([True]))


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.source=np.array([[0,0],[10,0],[0,10],[10,10],[2,3],[7,8]],dtype=float)
        self.matrix=np.array([[1.1,.2,3],[-.1,.9,-2.]])
        self.target=self.source@self.matrix[:,:2].T+self.matrix[:,2]
        self.ids=['a','b','c','d','e','f']

    def check(self,target=None,**kw):
        return affine_landmark_check(self.source,self.target if target is None else target,self.ids,
            fit_ids=kw.get('fit_ids',self.ids[:4]),check_ids=kw.get('check_ids',self.ids[4:]))

    def test_known_affine_fit_and_heldout(self):
        r=self.check();np.testing.assert_allclose(r['affine_source_xy_to_target_xy'],self.matrix,rtol=0,atol=1e-12)
        self.assertLess(r['check']['maximum_pixels'],1e-12)

    def test_check_error_does_not_change_fit(self):
        target=self.target.copy();target[4]+=[3,4]
        r=self.check(target);self.assertLess(r['fit']['maximum_pixels'],1e-12)
        self.assertAlmostEqual(r['check']['distances_pixels'][0],5,places=12)
        np.testing.assert_allclose(r['affine_source_xy_to_target_xy'],self.matrix,rtol=0,atol=1e-12)

    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):self.check(check_ids=['a'])

    def test_duplicate_names_rejected(self):
        self.ids[-1]='a'
        with self.assertRaisesRegex(ValueError,'Unique'):self.check()

    def test_unknown_name_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown'):self.check(check_ids=['z'])

    def test_collinear_fit_rejected(self):
        self.source[:4]=[[0,0],[1,1],[2,2],[3,3]]
        with self.assertRaisesRegex(ValueError,'Collinear'):self.check()

    def test_no_check_set_rejected(self):
        with self.assertRaisesRegex(ValueError,'check'):self.check(check_ids=[])

    def test_nonfinite_landmark_rejected(self):
        self.target[4,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Finite'):self.check()
