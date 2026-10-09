"""Targeted independent fsync/framing review. Fake responses, no live sockets."""
import sys
sys.dont_write_bytecode=True
import email.message
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import stat
import time

OUT=Path(__file__).resolve().parent
IMPL=OUT.parent/'pherc0139-geometry/public-geometry-implementation-v2'
sys.path.insert(0,str(IMPL))
import bounded_transport as B
import geometry_kernels as K
import run_public_geometry as R
def denied(*args,**kwargs): raise AssertionError('No live sockets authorized')
socket.socket.connect=denied
socket.socket.connect_ex=denied
PLAN=IMPL.parent/'public-geometry-plan/COORDINATE_QUERY_PLAN.json'
plan=json.loads(PLAN.read_text())
freeze=json.loads((IMPL/'CODE_FREEZE.json').read_text())
cases=[]
def pin(p):
    b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def check(name,value,**evidence):
    assert value,name
    cases.append(dict(name=name,status='PASS',**evidence))
for p in freeze['sources']+freeze['inputs']+[freeze['fixture'],freeze['readme']]:
    check('Frozen pin '+Path(p['path']).name,pin(Path(p['path']))==p)
check('Unchanged geometry source',pin(IMPL/'geometry_kernels.py')['sha256']=='62cf2a9912a6e53c04594f45f5b37fb6d858a36e8713640ce1c6912c2d33c2ae')
check('Pinned runtime',R.runtime()==freeze['runtime']=={'python':'3.12.14','numpy':'2.5.3','platform':'linux'})
R.verify_plan(plan)
TESTS=OUT/'transport-v2-fixtures'
TESTS.mkdir(exist_ok=False)

class Response:
    def __init__(self,q,headers=None,duplicate=None,status=206,url=None,size_delta=0,partial=False,slow=False):
        self.q=q;self.status=status;self.url=url;self.closed=False;self.reads=0;self.partial=partial;self.slow=slow
        self.remaining=q['expected_body_bytes']+size_delta
        lo,hi=q['range_inclusive']
        h={'Content-Range':f'bytes {lo}-{hi}/{q["source"]["object_size_bytes"]}',
           'Content-Length':str(q['expected_body_bytes']),'ETag':q['source']['strong_etag']}
        for k,v in (headers or {}).items():
            if v is None: h.pop(k,None)
            else: h[k]=v
        self.headers=email.message.Message()
        for k,v in h.items():self.headers[k]=v
        for k,v in (duplicate or {}).items():self.headers[k]=v
    def geturl(self):return self.url or self.q['source']['url']
    def read(self,n):
        self.reads+=1
        assert 0<n<=65536
        if self.slow:time.sleep(.10)
        if self.partial:
            self.partial=False
            raise http.client.IncompleteRead(b'partial',100)
        length=min(n,self.remaining);self.remaining-=length
        return b'\0'*length
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True

events=[]
real_fsync,real_replace=B.os.fsync,B.os.replace
def observed_fsync(fd):
    kind='fsync_directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'fsync_file'
    real_fsync(fd);events.append(kind)
def observed_replace(src,dst):
    real_replace(src,dst);events.append('replace')
class Opener:
    def __init__(self,q,out,response=None,slow=False):
        self.q=q;self.out=out;self.response=response or Response(q);self.opens=0;self.slow=slow;self.before_open=[]
    def open(self,request,timeout):
        self.opens+=1;self.before_open=list(events)
        assert self.before_open[-3:]==['fsync_file','replace','fsync_directory']
        ledger=json.loads((self.out/'transport-ledger.json').read_text())
        assert ledger['requests'][-1]['status']=='RESERVED'
        assert ledger['reserved_bytes_all_attempts']==456625
        assert ledger['charged_bytes']==456625
        assert request.get_method()=='GET' and timeout==30
        assert request.headers['If-match']==self.q['source']['strong_etag']
        assert request.headers['Range']=='bytes 5615708-6072331'
        assert request.headers['Accept-encoding']=='identity'
        assert 0<signal.getitimer(signal.ITIMER_REAL)[0]<=30
        events.append('open')
        if self.slow:time.sleep(.10)
        return self.response

def one(name,success=False,slow_open=False,slow_read=False,**kwargs):
    out=TESTS/name;out.mkdir()
    q=plan['requests'][0];response=Response(q,slow=slow_read,**kwargs)
    op=Opener(q,out,response,slow=slow_open)
    client=B.Client(plan,out,lambda:None,op)
    events.clear()
    B.os.fsync=observed_fsync;B.os.replace=observed_replace
    real_timer=B.signal.setitimer
    if slow_open or slow_read:
        B.signal.setitimer=lambda which,seconds,*args:real_timer(which,min(seconds,.02) if seconds else 0,*args)
    caught=None
    try:
        try:client.fetch(q)
        except BaseException as exc:caught=exc
    finally:
        B.os.fsync=real_fsync;B.os.replace=real_replace;B.signal.setitimer=real_timer
    row=client.ledger['requests'][0]
    if success:
        check(name,caught is None and row['status']=='PASS' and row['retained_body_bytes']==456624 and client.ledger['charged_bytes']==456624)
    else:
        check(name,isinstance(caught,K.ContractError) and row['status']=='STOP' and client.ledger['reserved_bytes_all_attempts']==client.ledger['charged_bytes']==456625,
              stop_reason=row.get('stop_reason'),retained_bytes=row['retained_body_bytes'])
        if not(kwargs.get('size_delta') or kwargs.get('partial') or slow_open or slow_read):
            check(name+' rejected before body',response.reads==0)
    check(name+' durable reservation before open',op.opens==1 and op.before_open[-3:]==['fsync_file','replace','fsync_directory'])
    if not slow_open:check(name+' response closed',response.closed)
    check(name+' absolute timer cleared',signal.getitimer(signal.ITIMER_REAL)[0]==0)
    check(name+' saved ledger matches memory',json.loads((out/'transport-ledger.json').read_text())==client.ledger)

one('exact_single_range',success=True)
one('explicit_identity',success=True,headers={'Content-Encoding':'identity'})
one('missing_content_length',headers={'Content-Length':None})
one('transfer_encoding_with_length',headers={'Transfer-Encoding':'chunked'})
one('transfer_encoding_without_length',headers={'Transfer-Encoding':'chunked','Content-Length':None})
one('duplicate_content_length',duplicate={'Content-Length':'456624'})
one('duplicate_content_range',duplicate={'Content-Range':'bytes 5615708-6072331/8109638'})
one('duplicate_etag',duplicate={'ETag':plan['requests'][0]['source']['strong_etag']})
one('changed_total',headers={'Content-Range':'bytes 5615708-6072331/8109639'})
one('weak_etag',headers={'ETag':'W/'+plan['requests'][0]['source']['strong_etag']})
one('ignored_range',status=200)
one('redirect_status',status=302)
one('changed_url',url='https://example.invalid/elsewhere')
one('content_encoding',headers={'Content-Encoding':'gzip'})
one('truncated',size_delta=-1)
one('extra_body_byte',size_delta=1)
one('partial_exception',partial=True)
one('absolute_deadline_open',slow_open=True)
one('absolute_deadline_read',slow_read=True)

# fsync failure must prevent any HTTP open, including failure after replacement.
for fail_kind in ['file','directory']:
    out=TESTS/('fsync_failure_'+fail_kind);out.mkdir()
    op=Opener(plan['requests'][0],out)
    client=B.Client(plan,out,lambda:None,op)
    def failed_fsync(fd):
        kind='directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file'
        if kind==fail_kind:raise OSError('synthetic fsync failure')
        real_fsync(fd)
    B.os.fsync=failed_fsync
    caught=None
    try:
        try:client.fetch(plan['requests'][0])
        except BaseException as e:caught=e
    finally:B.os.fsync=real_fsync
    check('fsync failure '+fail_kind+' prevents HTTP',isinstance(caught,OSError) and op.opens==0)

check('No redirect handler follow',B.NoRedirect().redirect_request(None,None,None,None,None,None) is None)
receipt={'status':'PASS','scope':'Targeted additive transport correction; fake HTTP and synthetic bytes only. No live sockets or real coordinate payloads.','checks':len(cases),'cases':cases,'code_freeze':pin(IMPL/'CODE_FREEZE.json'),'query_plan':pin(PLAN),'review_source':pin(Path(__file__)),'network_calls':0,'actual_coordinate_payloads_read':0}
with (OUT/'transport-v2-fixtures.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'status':'PASS','checks':len(cases),'receipt':pin(OUT/'transport-v2-fixtures.json')}))
