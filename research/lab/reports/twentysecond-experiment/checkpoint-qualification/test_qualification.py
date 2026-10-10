"""Synthetic format and fake-transport contracts; no live network or native tensors."""
import contextlib
import copy
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
from unittest.mock import patch
import qualify_checkpoint as q

CASES=[]
def check(name, fn):
    fn(); CASES.append({'name':name,'status':'PASS'})
def stops(code, fn):
    try: fn()
    except q.Stop as e:
        assert e.code==code,(e.code,code)
    else: raise AssertionError('expected '+code)
def fixture(payload=None):
    values={'a':struct.pack('<2f',1,-0.),'b':struct.pack('<2f',1,-0.),'c':struct.pack('<f',2),'d':struct.pack('<f',2)}
    if payload: values.update(payload)
    header={}; schema=[]; data=b''
    for name,value in values.items():
        shape=[len(value)//4]; start=len(data);data+=value
        header[name]={'dtype':'F32','shape':shape,'data_offsets':[start,len(data)]}
        schema.append({'name':name,'dtype':'F32','shape':shape,'numel':shape[0],'data_bytes':len(value)})
    raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    prefix=struct.pack('<Q',len(raw)); blob=prefix+raw+data
    reference={'expected_named_tensor_count':4,'expected_named_data_bytes':24,'expected_named_numel':6,
      'tensor_schema':schema,'expected_factory_alias_groups':[{'names':['a','b'],'shape':[2]},{'names':['c','d'],'shape':[1]}],
      'retained_factory_unique_parameters':3,'retained_factory_unique_storage_groups':2}
    source={'expected_file_extent_bytes':len(blob),'expected_full_file_sha256':q.sha(blob)}
    return blob,prefix,raw,reference,source

def inspect_mutation(mutator,code):
    blob,prefix,raw,reference,source=fixture(); blob,prefix,raw,reference,source=mutator(blob,prefix,raw,reference,source)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'test.safetensors';p.write_bytes(blob)
        stops(code,lambda:q.inspect_file(p,source,reference,prefix,raw))

def header_mutation(change,code):
    blob,prefix,raw,reference,source=fixture();h=json.loads(raw);change(h)
    encoded=json.dumps(h).encode()
    stops(code,lambda:q.validate_header(encoded,len(encoded)+8+24,reference))

class Response:
    def __init__(self,status=206,body=b'x'*128,headers=None,fail_after=None):
        self.status=status;self.headers=Message();self.reader=io.BytesIO(body);self.closed=False;self.reads=[];self.fail_after=fail_after
        defaults={'Content-Length':str(len(body)),'Content-Range':f'bytes 0-{len(body)-1}/{len(body)}','ETag':'"stable"'}
        defaults.update(headers or {})
        for key,value in defaults.items():
            if value is not None:
                for v in value if isinstance(value,list) else [value]: self.headers[key]=v
    def read(self,n):
        self.reads.append(n)
        if self.fail_after is not None and self.reader.tell()>=self.fail_after: raise RuntimeError('https://us.aws.cdn.hf.co/signed?SYNTHETIC_SECRET')
        if self.fail_after is not None: n=min(n,self.fail_after-self.reader.tell())
        return self.reader.read(n)
    def close(self): self.closed=True
class Opener:
    def __init__(self,responses,out): self.responses=list(responses);self.requests=[];self.out=out
    def open(self,request,timeout):
        ledger=json.loads((self.out/'ledger.json').read_text()); ev=ledger['attempts'][-1]
        assert ev['state']=='DURABLY_RECORDED_BEFORE_OPEN'
        assert ev['pre_open_reserved_body_bytes']==128
        assert request.get_header('Range')=='bytes=0-127'
        assert request.get_header('Accept-encoding')=='identity'
        assert request.get_header('If-match')=='"stable"'
        assert timeout<=30
        self.requests.append(request)
        return self.responses.pop(0)
@contextlib.contextmanager
def no_deadline(_):yield

def transport(responses,code=None,plan_change=None,extra=None):
    body=b'x'*128
    plan={'source':{'canonical_url':'https://huggingface.co/repo/resolve/pin/file','expected_file_extent_bytes':128,'expected_full_file_sha256':q.sha(body),'strong_etag':'"stable"'},
      'transport':{'allowed_exact_hosts':['huggingface.co','us.aws.cdn.hf.co']},
      'limits':{'attempts':4,'redirects':2,'body_bytes':280*1024**2,'absolute_request_seconds':180}}
    if plan_change:plan_change(plan)
    with tempfile.TemporaryDirectory() as d:
        out=Path(d);opener=Opener(responses,out);client=q.Client(plan,out,opener,no_deadline)
        if code:stops(code,client.download)
        else:
            final=client.download();assert final.read_bytes()==body
            assert client.ledger['status']=='DOWNLOAD_HASH_VERIFIED'
        serialized=(out/'ledger.json').read_text()
        assert 'SYNTHETIC_SECRET' not in serialized
        if extra:extra(out,opener,client)
        return len(opener.requests)

def main():
    blob,prefix,raw,reference,source=fixture()
    def valid():
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f';p.write_bytes(blob);result=q.inspect_file(p,source,reference,prefix,raw)
            assert result['finite_named_elements']==6 and result['alias_pair_count']==2
            assert result['conditional_factory_unique_parameters']==3
            assert not result['strict_model_load_verified'] and not result['shared_serialized_storage_claimed']
    check('finite_and_alias_equal_full_inspection',valid)
    check('whole_file_hash_before_value_parse',lambda: inspect_mutation(lambda b,p,h,r,s:(b[:-1]+b'1',p,h,r,s),'FULL_FILE_SHA256_MISMATCH'))
    check('file_extent_before_hash',lambda: inspect_mutation(lambda b,p,h,r,s:(b[:-1],p,h,r,s),'FULL_FILE_EXTENT_MISMATCH'))
    check('retained_prefix_identity',lambda: inspect_mutation(lambda b,p,h,r,s:(b,b'0'*8,h,r,s),'RETAINED_PREFIX_MISMATCH'))
    check('retained_header_identity',lambda: inspect_mutation(lambda b,p,h,r,s:(b,p,h.replace(b'a',b'z',1),r,s),'RETAINED_HEADER_MISMATCH'))
    def finite_or_alias(changes,code):
        b,p,h,r,s=fixture(changes)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'f';path.write_bytes(b);stops(code,lambda:q.inspect_file(path,s,r,p,h))
    for label,v in [('NaN',float('nan')),('positive_infinity',float('inf')),('negative_infinity',-float('inf'))]:
        check('reject_'+label,lambda v=v:finite_or_alias({'c':struct.pack('<f',v)},'NONFINITE_F32_VALUE'))
    check('alias_value_disagreement',lambda:finite_or_alias({'b':struct.pack('<2f',1,1)},'FACTORY_ALIAS_VALUES_NOT_BYTE_EQUAL'))
    check('signed_zero_is_byte_disagreement',lambda:finite_or_alias({'b':struct.pack('<2f',1,0.)},'FACTORY_ALIAS_VALUES_NOT_BYTE_EQUAL'))
    mutations=[
      ('missing_name',lambda h:h.pop('a'),'NAME_CARDINALITY_MISMATCH'),
      ('extra_name',lambda h:h.update(extra=h['a']),'NAME_CARDINALITY_MISMATCH'),
      ('metadata',lambda h:h.update(__metadata__={'x':'y'}),'UNEXPECTED_METADATA'),
      ('dtype',lambda h:h['a'].update(dtype='F16'),'DTYPE_SHAPE_MISMATCH'),
      ('bool_dimension',lambda h:h['a'].update(shape=[True]),'DTYPE_SHAPE_MISMATCH'),
      ('negative_dimension',lambda h:h['a'].update(shape=[-2]),'DTYPE_SHAPE_MISMATCH'),
      ('shape',lambda h:h['a'].update(shape=[1,2]),'DTYPE_SHAPE_MISMATCH'),
      ('unknown_field',lambda h:h['a'].update(unexpected=1),'DESCRIPTOR_FIELDS'),
      ('bool_offset',lambda h:h['a'].update(data_offsets=[False,8]),'INVALID_OFFSETS'),
      ('negative_offset',lambda h:h['a'].update(data_offsets=[-1,7]),'INVALID_OFFSETS'),
      ('reversed_offset',lambda h:h['a'].update(data_offsets=[8,0]),'OFFSET_LENGTH_MISMATCH'),
      ('overlap',lambda h:h['b'].update(data_offsets=[0,8]),'INTERVAL_OVERLAP_OR_GAP'),
      ('gap',lambda h:h['a'].update(data_offsets=[4,12]),'INTERVAL_OVERLAP_OR_GAP'),
      ('trailing_bytes',lambda h:h['d'].update(data_offsets=[20,28]),'OFFSET_LENGTH_MISMATCH')]
    for name,change,code in mutations:check('header_'+name,lambda change=change,code=code:header_mutation(change,code))
    check('duplicate_JSON_keys',lambda:stops('DUPLICATE_JSON_KEY',lambda:q.safe_json(b'{"a":1,"a":2}')))
    check('nonfinite_JSON',lambda:stops('NONFINITE_JSON',lambda:q.safe_json(b'{"a":NaN}')))
    check('malformed_UTF8',lambda:stops('MALFORMED_JSON',lambda:q.safe_json(b'\xff')))
    check('root_not_object',lambda:stops('HEADER_ROOT_NOT_OBJECT',lambda:q.validate_header(b'[]',32,reference)))
    check('transport_exact206',lambda:transport([Response()]))
    redirect=lambda target='https://us.aws.cdn.hf.co/file?SYNTHETIC_SECRET':Response(302,b'SYNTHETIC_SECRET',{'Location':target})
    check('manual_redirect_sanitized',lambda:transport([redirect(),Response()]))
    check('maximum_two_redirects',lambda:transport([redirect(),redirect(),Response()]))
    check('third_redirect_stops',lambda:transport([redirect(),redirect(),redirect()],'REDIRECT_BUDGET'))
    for name,url in [('http','http://us.aws.cdn.hf.co/f'),('other_host','https://evil.example/f'),('userinfo','https://u@us.aws.cdn.hf.co/f'),('port','https://us.aws.cdn.hf.co:444/f'),('fragment','https://us.aws.cdn.hf.co/f#x')]:
        check('redirect_reject_'+name,lambda url=url:transport([redirect(url)],'DESTINATION_OUTSIDE_ALLOWLIST'))
    ignored=Response(200)
    check('HTTP200_closes_before_read',lambda:transport([ignored],'HTTP200_RANGE_IGNORED',extra=lambda o,p,c: (_ for _ in ()).throw(AssertionError()) if ignored.reads else None))
    for name,headers,code in [
      ('range',{'Content-Range':'bytes 1-128/129'},'CONTENT_RANGE_MISMATCH'),
      ('length',{'Content-Length':'127'},'CONTENT_LENGTH_MISMATCH'),
      ('length_missing',{'Content-Length':None},'MISSING_OR_DUPLICATE_CONTENT_LENGTH'),
      ('length_duplicate',{'Content-Length':['128','128']},'MISSING_OR_DUPLICATE_CONTENT_LENGTH'),
      ('length_nonnumeric',{'Content-Length':'12.8'},'INVALID_CONTENT_LENGTH'),
      ('gzip',{'Content-Encoding':'gzip'},'NON_IDENTITY_ENCODING'),
      ('transfer_encoding',{'Transfer-Encoding':'chunked'},'TRANSFER_ENCODING_DISALLOWED'),
      ('weak_etag',{'ETag':'W/"stable"'},'STRONG_ETAG_MISMATCH'),
      ('changed_etag',{'ETag':'"changed"'},'STRONG_ETAG_MISMATCH'),
      ('missing_etag',{'ETag':None},'MISSING_OR_DUPLICATE_ETAG'),
      ('duplicate_range',{'Content-Range':['bytes 0-127/128','bytes 0-127/128']},'MISSING_OR_DUPLICATE_CONTENT_RANGE')]:
        check('transport_reject_'+name,lambda headers=headers,code=code:transport([Response(headers=headers)],code))
    check('error_body_hash_only',lambda:transport([Response(403,b'SYNTHETIC_SECRET')],'HTTP_STATUS_REJECTED'))
    check('oversized_error_closed',lambda:transport([Response(403,b'x'*4097)],'ERROR_BODY_LIMIT'))
    check('oversized_redirect_closed',lambda:transport([Response(302,b'x'*4097,{'Location':'https://us.aws.cdn.hf.co/f'})],'REDIRECT_BODY_LIMIT'))
    check('truncated_retained',lambda:transport([Response(body=b'x'*64,headers={'Content-Length':'128','Content-Range':'bytes 0-127/128'})],'TRUNCATED_RESPONSE',extra=lambda o,p,c:assert_partial(o,64)))
    check('read_exception_partial_sanitized',lambda:transport([Response(fail_after=32)],'TRANSPORT_EXCEPTION',extra=lambda o,p,c:assert_partial(o,32)))
    check('full_file_hash_mismatch_retains',lambda:transport([Response(body=b'z'*128)],'FULL_FILE_SHA256_MISMATCH',extra=lambda o,p,c:assert_partial(o,128)))
    check('body_reservation_before_open',lambda:transport([],'INSUFFICIENT_DURABLE_BODY_RESERVATION',lambda p:p['limits'].update(body_bytes=127)))
    check('attempt_budget_before_open',lambda:transport([],'ATTEMPT_BUDGET',lambda p:p['limits'].update(attempts=0)))
    check('aggregate_redirect_body_counted',lambda:transport([redirect(),Response()],'INSUFFICIENT_DURABLE_BODY_RESERVATION',lambda p:p['limits'].update(body_bytes=128)))
    def file_no_overwrite():
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'evidence'; q.durable_json(path,{'a':1}); q.durable_json(path,{'a':2})
            assert json.loads(path.read_text())=={'a':2} and not (Path(d)/'evidence.tmp').exists()
    check('atomic_evidence_replace_no_temp',file_no_overwrite)
    result={'status':'PASS','fixture_count':len(CASES),'cases':CASES,'live_network_attempts':0,'native_tensor_reads':0,'model_calls':0,
        'scope':'Small authored synthetic F32 files and fake HTTP responses only.'}
    print(json.dumps(result,sort_keys=True,indent=2))
def assert_partial(out,n):
    assert (out/'checkpoint.partial').stat().st_size==n
    assert not (out/'dense_native_016000.safetensors').exists()
if __name__=='__main__':main()
