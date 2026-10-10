"""Independent boundary probes of the frozen author's downloader; fake transport only."""
import contextlib
from email.message import Message
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import ssl
import tempfile

BASE=Path('/workspace/herculaneum-lab/reports/twentysecond-experiment/checkpoint-qualification')
source=BASE/'qualify_checkpoint.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='99c9cc655ac0d41fea4e44969193fdd93bd4c32ff3b110f850eff8a3838bc966'
spec=importlib.util.spec_from_file_location('reviewed_downloader',source)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

BODY=b'toy-synthetic-body'*19
SECRET='SIGNED_URL_SHOULD_NOT_SURVIVE'
@contextlib.contextmanager
def deadline(seconds):
    assert seconds==180
    yield
class Response:
    def __init__(self,status,body=BODY,headers=None,throw=False):
        self.status=status;self.body=io.BytesIO(body);self.reads=0;self.closed=False;self.throw=throw
        self.headers=Message()
        fields={'Content-Length':str(len(body)),'Content-Range':f'bytes 0-{len(BODY)-1}/{len(BODY)}','ETag':'"pinned"'}
        fields.update(headers or {})
        for key,value in fields.items():
            for item in (value if isinstance(value,list) else [value]):
                if item is not None:self.headers[key]=item
    def read(self,n):
        self.reads+=1
        if self.throw:raise RuntimeError('https://us.aws.cdn.hf.co/path?'+SECRET)
        return self.body.read(n)
    def close(self):self.closed=True
class FakeOpen:
    def __init__(self,root,responses):self.root=root;self.responses=iter(responses);self.calls=0
    def open(self,request,timeout):
        self.calls+=1
        # Evidence exists on disk before the transport is entered.
        ledger=json.loads((self.root/'ledger.json').read_text());event=ledger['attempts'][-1]
        assert event['attempt']==self.calls and event['state']=='DURABLY_RECORDED_BEFORE_OPEN'
        assert event['pre_open_reserved_body_bytes']==len(BODY)
        assert ledger['observed_body_bytes']+event['pre_open_reserved_body_bytes']<=293601280
        assert request.get_header('Range')==f'bytes=0-{len(BODY)-1}'
        assert request.get_header('If-match')=='"pinned"'
        assert request.get_header('Accept-encoding')=='identity' and timeout==30
        return next(self.responses)

def execute(responses,expected=None,body_budget=293601280):
    plan={'source':{'canonical_url':'https://huggingface.co/toy/resolve/immutable/toy',
        'expected_file_extent_bytes':len(BODY),'expected_full_file_sha256':hashlib.sha256(BODY).hexdigest(),'strong_etag':'"pinned"'},
        'transport':{'allowed_exact_hosts':['huggingface.co','us.aws.cdn.hf.co']},
        'limits':{'attempts':4,'redirects':2,'body_bytes':body_budget,'absolute_request_seconds':180}}
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);fake=FakeOpen(root,responses);client=q.Client(plan,root,fake,deadline)
        if expected is None:
            result=client.download();assert result.read_bytes()==BODY
        else:
            try:client.download()
            except q.Stop as exc:assert exc.code==expected,(exc.code,expected)
            else:raise AssertionError('accepted malformed response')
        files=list(root.iterdir())
        assert all(SECRET.encode() not in f.read_bytes() for f in files)
        ledger=json.loads((root/'ledger.json').read_text())
        return ledger,fake.calls

def main():
    checks=[]
    def record(name):checks.append(name)
    context=ssl.create_default_context()
    assert context.verify_mode==ssl.CERT_REQUIRED and context.check_hostname;record('default_TLS_context_verifies_cert_and_hostname')
    assert q.NoRedirect().redirect_request(None,None,302,'',{},'https://us.aws.cdn.hf.co/?'+SECRET) is None;record('automatic_redirect_handler_disabled')
    l,c=execute([Response(206)]);assert c==1 and l['observed_body_bytes']==len(BODY);record('exact206_preopen_durable_reservation_and_count')
    bad=Response(302,b'SECRET',{'Location':'https://forbidden.example/?'+SECRET})
    execute([bad],'DESTINATION_OUTSIDE_ALLOWLIST');assert bad.reads==0 and bad.closed;record('offhost_redirect_closed_before_body')
    bad=Response(200);execute([bad],'HTTP200_RANGE_IGNORED');assert bad.reads==0 and bad.closed;record('HTTP200_closed_without_body')
    for fields,code in [({'ETag':'"changed"'},'STRONG_ETAG_MISMATCH'),
        ({'Content-Length':[str(len(BODY)),str(len(BODY))]},'MISSING_OR_DUPLICATE_CONTENT_LENGTH'),
        ({'Transfer-Encoding':'chunked'},'TRANSFER_ENCODING_DISALLOWED'),
        ({'Content-Encoding':['identity','identity']},'NON_IDENTITY_ENCODING')]:
        bad=Response(206,headers=fields);execute([bad],code);assert bad.reads==0 and bad.closed;record(code+'_before_body')
    bad=Response(403,SECRET.encode())
    l,c=execute([bad],'HTTP_STATUS_REJECTED')
    assert l['observed_body_bytes']==len(SECRET) and l['attempts'][0]['observed_body_sha256']==hashlib.sha256(SECRET.encode()).hexdigest();record('error_body_count_hash_only')
    bad=Response(206,throw=True);l,c=execute([bad],'TRANSPORT_EXCEPTION');assert l['attempts'][0]['exception_type']=='RuntimeError';record('raw_exception_sanitized')
    bad=Response(206,BODY[:11],{'Content-Length':str(len(BODY))})
    l,c=execute([bad],'TRUNCATED_RESPONSE');assert l['observed_body_bytes']==11 and l['attempts'][0]['observed_body_bytes']==11;record('truncated_partial_accounted')
    l,c=execute([],'INSUFFICIENT_DURABLE_BODY_RESERVATION',body_budget=len(BODY)-1);assert c==0;record('insufficient_budget_prevents_open')
    out=Path(__file__).with_name('transport-review-fixtures.json')
    report={'status':'PASS','count':len(checks),'checks':checks,'network_calls':0,'actual_checkpoint_reads':0,'model_calls':0,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    with out.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
