"""Meaningful fabricated evidence checks; no native checkpoint/factory/output reads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import verify_parity as v

class ArithmeticTests(unittest.TestCase):
    def test_formula_uint8(self):
        a=v.volume((7,9,11)); self.assertEqual(a.dtype,np.uint8)
        for z,y,x in [(0,0,0),(6,8,10),(3,4,5)]:
            self.assertEqual(int(a[z,y,x]),1+(17*z+3*y+5*x+y*x%29+z*y%23)%251)
    def test_nine_selection_contracts(self):
        bounds=[(6,22),(2,18),(5,21),(7,23),(9,25),(2,18),(22,6),(6,22),(6,22)]
        for case,(first,last) in zip(v.CASES,bounds,strict=True):
            ids=v.indices(case['interval'],case['reverse']); self.assertEqual((int(ids[0]),int(ids[-1])),(first,last)); self.assertEqual(len(ids),17)
    def test_even_center_not_lower_center(self):
        self.assertFalse(np.array_equal(v.indices([0,28]),np.arange(5,22)))
    def test_odd_center(self): self.assertEqual(v.indices([8,27]).tolist(),list(range(9,26)))
    def test_reverse_after_crop(self): self.assertEqual(v.indices([0,28],True).tolist(),list(range(22,5,-1)))
    def test_raw_zero_skip(self): self.assertEqual(np.count_nonzero(v.volume((28,128,128),True)),0)
    def test_constant_normalization(self):
        self.assertTrue(np.array_equal(v.normalize(np.full((17,8,8),67,dtype=np.uint8)),np.zeros((17,8,8),dtype=np.float32)))
    def test_normalization_ownership(self):
        a=v.volume((17,8,8)); before=a.copy(); output=v.normalize(a)
        self.assertTrue(np.array_equal(a,before)); self.assertFalse(np.shares_memory(a,output))
    def test_normalization_fallback(self):
        a=np.zeros((17,16,16),dtype=np.uint8); a.flat[0]=255
        out=v.normalize(a); self.assertTrue(np.isfinite(out).all()); self.assertTrue((out==0).all())
    def test_clipped_quantization(self):
        a=np.asarray([-1,0,64/255,192/255,1,2],dtype=np.float32)
        self.assertEqual(v.encode(a).tolist(),[0,0,64,192,255,255])
        self.assertEqual(v.decode(v.encode(a)).tolist(),[0,0,0,1,1,1])
    def test_truncation_not_rounding(self): self.assertEqual(int(v.encode(np.array([.5],dtype=np.float32))[0]),127)
    def test_hann_floor(self):
        h=v.hann(); self.assertEqual(float(h.max()),1); self.assertEqual(float(h[0,0]),float(np.float32(.001)))
        self.assertTrue(np.array_equal(h,h.T))
    def test_overlap_schedule(self): self.assertEqual(v.blocks([28,192,192]).tolist(),[[0,0,128,128],[0,64,128,128],[64,0,128,128],[64,64,128,128]])
    def test_edge_aligned_schedule(self): self.assertEqual(v.positions(200),[0,64,72])
    def test_padding_schedule(self): self.assertEqual(v.blocks([17,60,80]).tolist(),[[0,0,60,80]])
    def test_overlap_normalization(self):
        p=np.full((4,1,1,128,128),.375,dtype=np.float32)
        ps,ws,raw=v.accumulate(p,v.hann(),v.blocks([28,192,192]),(192,192))
        self.assertTrue((ws>0).all()); self.assertLess(float(np.max(np.abs(raw-.375))),1e-6)
        self.assertGreater(float(ws[70,70]),float(v.hann()[70,70]))
    def test_sigmoid_stability(self):
        got=v.sigmoid(np.array([-1000,-1,0,1,1000],dtype=np.float32))
        self.assertEqual(got[0],0); self.assertEqual(got[2],.5); self.assertEqual(got[-1],1)
        self.assertTrue(np.isfinite(got).all())
    def test_wrong_shape_rejected(self):
        with self.assertRaises(v.Reject): v.compare(np.zeros(3),np.zeros(4),1e-5,1e-5,'shape')
    def test_nonfinite_rejected(self):
        with self.assertRaises(v.Reject): v.compare(np.zeros(1),np.array([np.nan]),1e-5,1e-5,'nan')
    def test_tolerance_failure(self):
        with self.assertRaises(v.Reject): v.compare(np.zeros(1),np.array([1e-4]),1e-5,1e-5,'difference')
    def test_bitwise_signed_zero(self):
        with self.assertRaises(v.Reject): v.exact(np.array([0.],dtype=np.float32),np.array([-0.],dtype=np.float32),'signedzero')

def state_fixture():
    names=[f'key{i:03d}' for i in range(508)]; hashes={n:'0'*64 for n in names}
    aliases=sorted([sorted(names[i:i+2]) for i in range(0,488,2)])
    schema=dict(expected_factory_alias_groups=[dict(names=x) for x in aliases])
    receipt=dict(status='PASS',named_tensors=508,named_elements=68175426,unique_groups=264,unique_parameters=34546498,
        all_loaded_bytes_exact=True,all_finite=True,named_value_sha256=hashes.copy(),alias_pairs=aliases,
        semantic_value_sha256=v.hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return receipt,hashes,schema
class StateTests(unittest.TestCase):
    def test_valid_fabricated_state(self): v.verify_state(*state_fixture())
    def test_wrong_key_rejected(self):
        r,h,s=state_fixture(); del r['named_value_sha256']['key000']
        with self.assertRaises(v.Reject): v.verify_state(r,h,s)
    def test_wrong_value_hash_rejected(self):
        r,h,s=state_fixture(); r['named_value_sha256']['key000']='1'*64
        with self.assertRaises(v.Reject): v.verify_state(r,h,s)
    def test_wrong_alias_rejected(self):
        r,h,s=state_fixture(); r['alias_pairs']=copy.deepcopy(r['alias_pairs']); r['alias_pairs'][0][0]='key500'
        with self.assertRaises(v.Reject): v.verify_state(r,h,s)
    def test_wrong_count_rejected(self):
        r,h,s=state_fixture(); r['unique_groups']=265
        with self.assertRaises(v.Reject): v.verify_state(r,h,s)
    def test_wrong_semantic_hash_rejected(self):
        r,h,s=state_fixture(); r['semantic_value_sha256']='x'
        with self.assertRaises(v.Reject): v.verify_state(r,h,s)

def case_fixture(root,case):
    vol=v.volume(case['shape'],case.get('zero',False)); ids=v.indices(case['interval'],case['reverse'])
    schedule=v.blocks(case['shape']); count=case['forwards']; weights=v.hann()
    inputs=np.stack([v.normalize(vol[ids,y:y+h,x:x+w])[None,None] for y,x,h,w in schedule]) if count else np.empty((0,1,1,17,128,128),dtype=np.float32)
    logits=np.broadcast_to(np.linspace(-4,4,128,dtype=np.float32),(count,1,1,128,128)).copy()
    probs=v.sigmoid(logits); ps,ws,raw=v.accumulate(probs,weights,schedule if count else schedule[:0],tuple(case['shape'][-2:]))
    arrays=dict(selected_indices=ids,blocks=schedule,hann=weights,normalized=inputs,logits=logits,probabilities=probs,
        probability_sum=ps,weight_sum=ws,raw_score=raw,uint8=v.encode(raw),author_clipped_score=v.decode(v.encode(raw)),evaluated=ws>0)
    meta=dict(case=case,forwards=count,arrays={k:v.array_record(a) for k,a in arrays.items()},input=v.array_record(vol),evaluated_pixels=int((ws>0).sum()))
    for key,array in arrays.items(): np.save(root/(key+'.npy'),array,allow_pickle=False)
    return meta,arrays
class EvidenceTests(unittest.TestCase):
    def test_complete_tiled_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); case=v.CASES[7]; meta,_=case_fixture(path,case)
            result,_=v.verify_case(path,meta,case); self.assertEqual(result['forwards'],4)
    def test_complete_zero_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); case=v.CASES[8]; meta,_=case_fixture(path,case); v.verify_case(path,meta,case)
    def test_wrong_normalized_tensor_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); case=v.CASES[0]; meta,a=case_fixture(path,case)
            a['normalized'][0,0,0,0,0,0]+=1
            np.save(path/'normalized.npy',a['normalized'],allow_pickle=False); meta['arrays']['normalized']=v.array_record(a['normalized'])
            with self.assertRaises(v.Reject): v.verify_case(path,meta,case)
    def test_wrong_encoded_output_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); case=v.CASES[0]; meta,a=case_fixture(path,case)
            a['uint8'][0,0]+=1; np.save(path/'uint8.npy',a['uint8'],allow_pickle=False); meta['arrays']['uint8']=v.array_record(a['uint8'])
            with self.assertRaises(v.Reject): v.verify_case(path,meta,case)
    def test_wrong_call_count_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); case=v.CASES[0]; meta,_=case_fixture(path,case); meta['forwards']=2
            with self.assertRaises(v.Reject): v.verify_case(path,meta,case)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'STOPPED',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),native_weight_reads=0,real_factory_constructions=0,model_forwards=0)))
    raise SystemExit(0 if result.wasSuccessful() else 2)
