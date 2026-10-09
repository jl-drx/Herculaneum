"""Independent fake-HTTP controls. No opener can reach a network."""
import argparse
import email.message
import hashlib
import http.client
import json
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
AUTHOR=HERE.parent/'frag6-stage-b'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-pins',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    pins=json.loads(args.source_pins.read_text())
    for p in pins['author_sources']:
        b=Path(p['path']).read_bytes();assert len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256']
    sys.path.insert(0,str(AUTHOR))
    from common import ContractError,LIMITS
    from transport import MultipartClient,URL,NoRedirect
    fixture_root=HERE/'transport-fixtures-v1';fixture_root.mkdir()
    total=2636777593;etag='"656fa2a5-9d2a0879"'
    ranges=[]
    for i in range(1,10):
        cy=(i*8852)//10
        for j in range(1,10):
            cx=(j*6204)//10
            for y in range(cy-1,cy+2):
                a=73+48*(y*6205+cx-1);ranges.append([a,a+143])
    boundary=b'ReviewerTransport21'
    body=b''.join(b'--'+boundary+b'\r\nContent-Type: application/octet-stream\r\n'+
        f'Content-Range: bytes {a}-{b}/{total}\r\n\r\n'.encode()+bytes(range(144))+b'\r\n' for a,b in ranges)
    body+=b'--'+boundary+b'--\r\n'
    base_headers=[('Content-Type','multipart/byteranges; boundary='+boundary.decode()),
                  ('Content-Length',str(len(body))),('ETag',etag)]
    object_record=dict(url=URL,total_bytes=total,etag=etag)
    checks=[]
    def check(name,condition,**detail):
        assert condition,name
        checks.append(dict(name=name,passed=True,**detail))

    class Response:
        def __init__(self,data,headers,status=206,mode=None):
            self.data=data;self.cursor=0;self.status=status;self.mode=mode
            self.headers=email.message.Message();self.calls=0;self.closed=False
            for k,v in headers:self.headers[k]=v
        def read(self,n):
            self.calls+=1
            if self.mode=='slow_read':time.sleep(.08)
            if self.mode=='incomplete':raise http.client.IncompleteRead(b'partial',100)
            block=self.data[self.cursor:self.cursor+n];self.cursor+=len(block);return block
        def close(self):self.closed=True

    class Opener:
        def __init__(self,response,root,mode=None):self.response=response;self.root=root;self.mode=mode;self.calls=0
        def open(self,request,timeout):
            self.calls+=1
            durable=json.loads((self.root/'request-ledger.json').read_text())
            assert len(durable['requests'])==1 and durable['reserved_response_body_bytes']==222640
            assert durable['requests'][0]['reserved_response_body_bytes']==222640
            assert request.full_url==URL and request.get_method()=='GET' and timeout==20
            assert request.get_header('Range')=='bytes='+','.join(f'{a}-{b}' for a,b in ranges)
            assert request.get_header('If-match')==etag and request.get_header('Accept-encoding')=='identity'
            if self.mode=='slow_open':time.sleep(.08)
            if self.mode=='open_failure':raise OSError('synthetic connection failure')
            return self.response

    def trial(name,headers=None,data=body,status=206,response_mode=None,open_mode=None,
              expected=True,read_expected=None,limits=None):
        root=fixture_root/name
        response=Response(data,base_headers if headers is None else headers,status,response_mode)
        opener=Opener(response,root,open_mode)
        selected_limits=dict(LIMITS)
        if limits:selected_limits.update(limits)
        client=MultipartClient(root,object_record,ranges,dict(path='synthetic',sha256='0'*64,bytes=0),
                               opener=opener,limits=selected_limits)
        failure=None;result=None
        try:result=client.get()
        except (ContractError,OSError) as error:failure=error
        check(name+'_status',(failure is None)==expected,reason=None if failure is None else str(failure))
        ledger=json.loads((root/'request-ledger.json').read_text())
        check(name+'_one_durable_reservation',opener.calls==1 and len(ledger['requests'])==1
              and ledger['reserved_response_body_bytes']==222640 and ledger['retained_response_body_bytes']<=222640
              and ledger['automatic_retries']==0 and ledger['redirects_followed']==0)
        entry=ledger['requests'][0]
        check(name+'_retained_accounting',entry['retained_body_bytes']==ledger['retained_response_body_bytes'])
        if read_expected=='none':check(name+'_no_body_read',response.calls==0 and entry['retained_body_bytes']==0)
        if expected:
            returned,parts,receipt=result
            check(name+'_exact_body_and_parts',returned==body and len(parts)==243 and receipt['success']
                  and (root/'raw/multipart.bin').read_bytes()==body and entry['retained_body_bytes']==len(body))
        else:
            check(name+'_failure_not_published',not entry['success'] and not (root/'raw/multipart.bin').exists())
            if 'partial_file' in entry:
                partial=Path(entry['partial_file']['path']).read_bytes()
                check(name+'_partial_bytes_preserved',len(partial)==entry['retained_body_bytes']
                      and hashlib.sha256(partial).hexdigest()==entry['body_sha256'])
            try:client.get()
            except ContractError:pass
            else:raise AssertionError(name+'_retry_not_rejected')
            check(name+'_retry_rejected_before_open',opener.calls==1
                  and json.loads((root/'request-ledger.json').read_text())['reserved_response_body_bytes']==222640)
        if open_mode not in ('slow_open','open_failure'):check(name+'_response_closed',response.closed)
        return ledger

    trial('valid')
    for status in (200,301,302,307,308,403,412,416,500):
        trial('status_'+str(status),status=status,expected=False,read_expected='none')
    def replace_header(key,value):return [(k,v) for k,v in base_headers if k!=key]+[(key,value)]
    for name,header in [('weak_etag',('ETag','W/'+etag)),('changed_etag',('ETag','"different"')),
        ('length_at_cap',('Content-Length','222640')),('length_above_cap',('Content-Length','222641')),
        ('length_below_useful',('Content-Length','34992')),('length_negative',('Content-Length','-1')),
        ('length_plus',('Content-Length','+'+str(len(body)))),('length_junk',('Content-Length','3.5')),
        ('wrong_mime',('Content-Type','application/octet-stream')),
        ('mime_missing_boundary',('Content-Type','multipart/byteranges')),
        ('mime_second_boundary',('Content-Type',base_headers[0][1]+'; boundary=other')),
        ('encoded_body',('Content-Encoding','gzip')),('transfer_encoding',('Transfer-Encoding','chunked')),
        ('top_range',('Content-Range',f'bytes {ranges[0][0]}-{ranges[0][1]}/{total}'))]:
        trial(name,headers=replace_header(*header),expected=False,read_expected='none')
    for key in ('Content-Type','Content-Length','ETag'):
        trial('missing_'+key.lower(),headers=[(k,v) for k,v in base_headers if k!=key],expected=False,read_expected='none')
    for key in ('Content-Type','Content-Length','ETag','Content-Encoding','Content-Range','Transfer-Encoding','Last-Modified'):
        initial=dict(base_headers).get(key,'identity' if key=='Content-Encoding' else 'dummy')
        clean=[(k,v) for k,v in base_headers if k!=key]
        trial('duplicate_'+key.lower(),headers=clean+[(key,initial),(key,initial)],expected=False,read_expected='none')
    trial('oversized_http_headers',headers=base_headers+[('X-Large','x'*17000)],expected=False,read_expected='none')
    trial('truncated_body',data=body[:-7],expected=False)
    trial('extra_byte',data=body+b'X',expected=False)
    partial=trial('incomplete_read',response_mode='incomplete',expected=False)
    check('incomplete_read_retained_exact_partial',partial['retained_response_body_bytes']==7)
    bad=body.replace(b'application/octet-stream',b'application/octet-evilxx',1)
    trial('postread_mime_rejection',data=bad,headers=replace_header('Content-Length',str(len(bad))),expected=False)
    trial('connection_failure',open_mode='open_failure',expected=False,read_expected='none')
    trial('absolute_open_deadline',open_mode='slow_open',expected=False,read_expected='none',limits={'per_request_wall_seconds':.02})
    trial('absolute_read_deadline',response_mode='slow_read',expected=False,limits={'per_request_wall_seconds':.02})
    check('redirect_handler_never_creates_redirect_request',NoRedirect().redirect_request(None,None,302,'',{},'https://other.invalid') is None)
    for name,overrides in [('attempt_budget',{'maximum_attempts':0}),('byte_budget',{'maximum_reserved_response_body_bytes':222639}),
                           ('rss_budget',{'maximum_rss_bytes':0}),('artifact_budget',{'maximum_artifact_bytes':0})]:
        root=fixture_root/name;response=Response(body,base_headers);opener=Opener(response,root)
        limits=dict(LIMITS);limits.update(overrides)
        client=MultipartClient(root,object_record,ranges,{},opener=opener,limits=limits)
        try:client.get()
        except ContractError:pass
        else:raise AssertionError(name)
        ledger=json.loads((root/'request-ledger.json').read_text())
        check(name+'_prevents_connection',opener.calls==0 and not ledger['requests'] and ledger['reserved_response_body_bytes']==0)
    check('resource_bounds',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<256*1024**2
          and sum(p.stat().st_size for p in fixture_root.rglob('*') if p.is_file())<32*1024**2)
    result=dict(kind='phase21-independent-frag6-b-transport-v1',status='PASS',check_count=len(checks),checks=checks,
        source_pins_sha256=hashlib.sha256(args.source_pins.read_bytes()).hexdigest(),
        resources=dict(wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        execution=dict(real_http_attempts=0,real_ppm_records_decoded=0,model_calls=0,annotation_reads=0),
        limitation='Fake opener/response exercises source logic and accounting, not external server or TLS behavior.')
    with (HERE/'transport-fixtures-v1.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','check_count','resources','execution')}))


if __name__=='__main__':main()
