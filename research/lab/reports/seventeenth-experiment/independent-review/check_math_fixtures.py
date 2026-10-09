import hashlib
import json
import math
from pathlib import Path
import numpy as np
from independent_math import profile, exact_pool, block_average, gaussian_reflect, ranking, geometry


def close(actual, expected, tol=1e-12):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), (actual, expected)


def main():
    checks = []
    a = np.array([[0, 1, 9], [3, 7, 65535]], dtype=np.uint16)
    m = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    s = profile(a, m, 65535)
    vals = [0, 1, 7, 65535]
    avg = math.fsum(vals)/4
    assert s['count'] == 4 and s['horizontal_edge_count'] == 2 and s['vertical_edge_count'] == 1
    close(s['mean'], avg)
    close(s['std'], math.sqrt(math.fsum((x-avg)**2 for x in vals)/4))
    close(s['horizontal_mean_absolute_difference'], (1+65528)/2)
    close(s['vertical_mean_absolute_difference'], 6)
    close(s['q10'], .3); close(s['q50'], 4); close(s['q90'], 45876.600000000006)
    assert s['endpoint_min_count'] == s['endpoint_max_count'] == 1
    checks.append('asymmetric_mask_population_std_linear_quantiles_endpoints_unsigned_safe_edges')
    empty = profile(a, np.zeros(a.shape, bool), 65535)
    assert empty['count'] == 0 and empty['mean'] is None and empty['std'] is None and empty['q10'] is None
    assert empty['endpoint_min_fraction'] is None and empty['horizontal_mean_absolute_difference'] is None
    singleton = np.zeros(a.shape, bool); singleton[1, 1] = 1
    s = profile(a, singleton, 65535)
    assert s['std'] == 0 and s['q10'] == s['q90'] == 7 and s['horizontal_mean_absolute_difference'] is None
    checks.append('empty_singleton_and_no_edge_nulls')
    for bad in [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-1.]]), np.array([[256.]])]:
        try: profile(bad, np.ones((1, 1), bool), 255)
        except AssertionError: pass
        else: raise AssertionError('invalid profile accepted')
    checks.append('nonfinite_and_range_rejection')
    ramps = np.fromfunction(lambda z,y,x: 10000*z+100*y+x, (3,6,9), dtype=int).astype(np.uint16)
    got = exact_pool(ramps)
    expected = np.empty((2,3), np.float32)
    for y in range(2):
        for x in range(3):
            total = sum(int(ramps[z,3*y+dy,3*x+dx]) for z in range(3) for dy in range(3) for dx in range(3))
            expected[y,x] = total/6939
    assert np.array_equal(got,expected)
    assert exact_pool(np.full((3,3,3),65535,np.uint16))[0,0] == 255
    assert exact_pool(np.zeros((3,3,3),np.uint16))[0,0] == 0
    checks.append('non_square_axis_ramp_and_uint16_overflow_pooling')
    assert ranking([.5,.5,.5,.5],[0,1,0,1])['roc_auc'] == .5
    assert ranking([.5,.5,.5,.5],[0,1,0,1])['average_precision'] == .5
    r = ranking([0, 1, 1, 2], [0, 1, 0, 1]); close(r['roc_auc'], .875); close(r['average_precision'], 5/6)
    assert ranking([1,2],[0,0])['average_precision'] is None
    assert ranking([1,2],[1,1])['roc_auc'] is None and ranking([1,2],[1,1])['average_precision'] == 1
    assert ranking([],[])['roc_auc'] is None
    checks.append('rank_ties_pairwise_auc_step_ap_single_class_empty')
    # Hand-calculated radius-one convolution on [0,10,0], symmetric reflection.
    radius = 1; sigma = 1
    w = math.exp(-.5)/(1+2*math.exp(-.5)); center = 1-2*w
    one = gaussian_reflect(np.array([[0.,10.,0.]]),sigma=sigma,radius=radius)
    np.testing.assert_allclose(one,[[10*w,10*center,10*w]],rtol=1e-14,atol=1e-14)
    assert np.max(np.abs(gaussian_reflect(np.full((4,7),17.0))-17)) < 1e-12
    im = np.arange(54,dtype=np.float64).reshape(6,9)
    np.testing.assert_array_equal(block_average(im),[[10,13,16],[37,40,43]])
    checks.append('explicit_gaussian_reflect_constant_and_hand_boundary_block_mean')
    n = np.zeros((3,3,6));
    for y in range(3):
        for x in range(3): n[y,x] = [2*x,3*y,0,0,0,-2]
    g = geometry(n)
    assert g['du_length']==2 and g['dv_length']==3 and g['cross_area']==6
    assert g['signed_normal_agreement']==-1 and g['tangent_agreement']==0 and g['normal_norm']==2
    n[:,:,0:3]=0; g=geometry(n)
    assert g['cross_area']==0 and g['signed_normal_agreement'] is None and g['tangent_agreement'] is None
    checks.append('anisotropic_ppm_tangent_normal_sign_degenerate')
    here = Path(__file__).resolve().parent
    receipt = dict(status='PASS',checks=checks,check_count=len(checks),scientific_array_reads=0,
                   python_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),here/'independent_math.py']},numpy_version=np.__version__)
    target = here/'math-fixture-receipt.json'
    with target.open('x') as f: json.dump(receipt,f,indent=2); f.write('\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
