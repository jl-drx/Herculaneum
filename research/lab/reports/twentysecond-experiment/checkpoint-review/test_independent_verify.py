"""Synthetic files only. No production checkpoint file or tensor value reads."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file
import independent_verify as checker

class VerificationFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'toy.safetensors'
        self.values={'a':np.array([1.,0.],dtype=np.float32),
            'b':np.array([1.,0.],dtype=np.float32),'c':np.array(3.,dtype=np.float32)}
        self.reference={'tensor_schema':[
            dict(name=n,dtype='F32',shape=list(v.shape),numel=v.size,data_bytes=v.nbytes)
            for n,v in self.values.items()],
            'expected_factory_alias_groups':[dict(names=['a','b'],shape=[2])]}
        self.contract=None
        self.write()
    def tearDown(self): self.tmp.cleanup()
    def write(self):
        save_file(self.values,str(self.path))
        self.refresh_contract()
    def refresh_contract(self):
        raw=self.path.read_bytes(); h=struct.unpack('<Q',raw[:8])[0]
        self.contract=dict(size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            prefix_sha256=hashlib.sha256(raw[:8]).hexdigest(),
            header_sha256=hashlib.sha256(raw[8:8+h]).hexdigest(),header_bytes=h,
            named_count=3,named_numel=5,data_bytes=20,alias_pair_count=1,
            group_count=2,unique_numel=3,unique_bytes=12)
    def change_header(self,edit):
        raw=self.path.read_bytes(); h=struct.unpack('<Q',raw[:8])[0]
        head=json.loads(raw[8:8+h]); edit(head)
        new=json.dumps(head,separators=(',',':')).encode()
        new+=b' '*((-len(new))%8)
        self.path.write_bytes(struct.pack('<Q',len(new))+new+raw[8+h:]); self.refresh_contract()
    def verify(self): return checker.verify_file(self.path,self.reference,self.contract)
    def reject(self,code):
        with self.assertRaisesRegex(checker.Reject,code): self.verify()
    def test_valid_allfinite_and_exact_alias(self):
        r=self.verify(); self.assertEqual(r['status'],'PASS'); self.assertEqual(r['alias_pair_count'],1)
        self.assertEqual(r['conditional_factory_unique_parameters'],3)
    def test_library_validates_synthetic_file(self):
        r=self.verify(); self.assertEqual(r['safetensors_version'],'0.8.0')
    def test_sha_rejection_precedes_library_open(self):
        self.contract['sha256']='0'*64
        def forbidden(*args,**kwargs): self.fail('tensor library opened before integrity')
        with self.assertRaisesRegex(checker.Reject,'whole_sha256'):
            checker.verify_file(self.path,self.reference,self.contract,forbidden)
    def test_wrong_length(self):
        self.contract['size_bytes']+=1; self.reject('whole_size')
    def test_prefix_pin(self):
        self.contract['prefix_sha256']='0'*64; self.reject('prefix_sha256')
    def test_header_pin(self):
        self.contract['header_sha256']='0'*64; self.reject('header_sha256')
    def test_trailing_bytes(self):
        with self.path.open('ab') as f:f.write(b'BAD!')
        self.refresh_contract(); self.reject('file_extent')
    def test_nonfinite_nan(self):
        self.values['a'][0]=np.nan; self.write(); self.reject('nonfinite_tensor')
    def test_nonfinite_positive_inf(self):
        self.values['a'][0]=np.inf; self.write(); self.reject('nonfinite_tensor')
    def test_nonfinite_negative_inf(self):
        self.values['c'][...]=-np.inf; self.write(); self.reject('nonfinite_tensor')
    def test_signed_zero_alias_mismatch(self):
        self.values['b'][1]=-0.; self.write(); self.reject('alias_bit_mismatch')
    def test_finite_alias_mismatch(self):
        self.values['b'][0]=np.nextafter(np.float32(1),np.float32(2));self.write();self.reject('alias_bit_mismatch')
    def test_duplicate_top_key(self):
        with self.assertRaisesRegex(checker.Reject,'duplicate_json_key'):
            checker.parse_header(b'{"a":1,"a":2}')
    def test_duplicate_nested_key(self):
        with self.assertRaisesRegex(checker.Reject,'duplicate_json_key'):
            checker.parse_header(b'{"a":{"shape":[],"shape":[]}}')
    def test_nonstandard_json_constant(self):
        with self.assertRaisesRegex(checker.Reject,'nonstandard_json_number'):
            checker.parse_header(b'{"a":NaN}')
    def test_metadata_rejected(self):
        self.change_header(lambda h:h.update(__metadata__={'author':'toy'}));self.reject('unexpected_metadata')
    def test_unexpected_name(self):
        self.change_header(lambda h:h.update(extra=h.pop('c')));self.reject('tensor_names')
    def test_unexpected_field(self):
        self.change_header(lambda h:h['a'].update(extra=1));self.reject('tensor_fields')
    def test_wrong_dtype(self):
        self.change_header(lambda h:h['a'].update(dtype='I32'));self.reject('tensor_dtype')
    def test_boolean_shape(self):
        self.change_header(lambda h:h['a'].update(shape=[True,2]));self.reject('shape_integer')
    def test_negative_shape(self):
        self.change_header(lambda h:h['a'].update(shape=[-2]));self.reject('shape_integer')
    def test_wrong_shape(self):
        self.change_header(lambda h:h['a'].update(shape=[1,2]));self.reject('tensor_shape')
    def test_bool_offset(self):
        self.change_header(lambda h:h['a'].update(data_offsets=[False,8]));self.reject('offset_integer')
    def test_outside_offset(self):
        self.change_header(lambda h:h['a'].update(data_offsets=[20,28]));self.reject('tensor_extent')
    def test_overlap_offsets(self):
        self.change_header(lambda h:h['b'].update(data_offsets=h['a']['data_offsets']));self.reject('noncontiguous_or_overlapping')
    def test_hole_offsets(self):
        self.change_header(lambda h:h['a'].update(data_offsets=[1,9]));self.reject('noncontiguous_or_overlapping')
    def test_alias_duplicate_name(self):
        self.reference['expected_factory_alias_groups'][0]['names']=['a','a'];self.reject('alias_pair_definition')
    def test_alias_nonexistent(self):
        self.reference['expected_factory_alias_groups'][0]['names']=['a','absent'];self.reject('alias_partition')
    def test_alias_shape_wrong(self):
        self.reference['expected_factory_alias_groups'][0]['shape']=[1,2];self.reject('alias_shape')
    def test_group_count_wrong(self):
        self.contract['group_count']=3;self.reject('group_count')
    def test_unique_count_wrong(self):
        self.contract['unique_numel']=4;self.reject('unique_totals')
    def test_reference_duplicate_name(self):
        self.reference['tensor_schema'][2]['name']='a';self.reject('reference_duplicate_name')
    def test_socket_audit_hook_blocks_network(self):
        # Directly exercise hook without installing a process-global hook in fixture runner.
        with self.assertRaisesRegex(checker.Reject,'network_forbidden'):
            checker.deny_network('socket.connect',('unused',443))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(VerificationFixtures)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'STOPPED',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),
        actual_checkpoint_reads=0,network_calls=0,model_instantiations=0,
        tests=[name for name in unittest.defaultTestLoader.getTestCaseNames(VerificationFixtures)])
    out=Path(__file__).with_name('synthetic-fixtures.json')
    with out.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 2)
