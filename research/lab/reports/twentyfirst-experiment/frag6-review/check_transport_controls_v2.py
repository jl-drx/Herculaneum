"""Independent fake HTTP controls; live sockets are disabled for this process."""
from email.message import Message
import hashlib
import http.client
import io
import json
from pathlib import Path
import socket
import sys
import time
import urllib.error

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'frag6-stage-a'
sys.path.insert(0,str(SOURCE))
from common import ContractError, LIMITS
from transport import RangeClient, NoRedirect


def deny_network(*args,**kwargs):
    raise AssertionError('Live network forbidden in review fixture')
socket.socket.connect=deny_network
socket.socket.connect_ex=deny_network
socket.create_connection=deny_network

URL='https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/surface_volume/00.tif'
OBJECT=dict(url=URL,etag='"synthetic-only"',total_bytes=512)
HEADERS=[('Content-Length','8'),('Content-Range','bytes 0-7/512'),
         ('ETag','"synthetic-only"'),('Content-Type','image/tiff')]


class Response:
    def __init__(self,body=b'12345678',status=206,headers=HEADERS,mode='normal'):
        self.body=body;self.status=status;self.mode=mode
        self.headers=Message()
        for k,v in headers:self.headers[k]=v
        self.read_calls=[];self.cursor=0;self.closed=False
    def read(self,n):
        self.read_calls.append(n)
        if self.mode=='incomplete':
            raise http.client.IncompleteRead(b'123',5)
        if self.mode=='slow':time.sleep(.015)
        take=1 if self.mode=='slow' else n
        value=self.body[self.cursor:self.cursor+take]
        self.cursor+=len(value)
        return value
    def close(self):self.closed=True


class Opener:
    def __init__(self,response,root,mode='normal'):
        self.response=response;self.root=root;self.mode=mode;self.calls=[]
    def open(self,request,timeout):
        saved=json.loads((self.root/'request-ledger.json').read_text())
        assert len(saved['requests'])==1 and saved['reserved_response_body_bytes']==9
        assert saved['requests'][0]['success'] is False and saved['requests'][0]['retained_body_bytes']==0
        assert request.get_header('If-match')==OBJECT['etag']
        assert request.get_header('Range')=='bytes=0-7' and request.get_header('Accept-encoding')=='identity'
        assert request.get_method()=='GET' and request.full_url==URL
        assert timeout==20
        self.calls.append(dict(timeout=timeout,headers=dict(request.header_items())))
        if self.mode=='connect-failure':raise urllib.error.URLError('synthetic connection failure')
        if self.mode=='slow-open':time.sleep(.015)
        return self.response


def pin(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    root=HERE/'transport-fixtures-v2'
    root.mkdir(exist_ok=False)
    checks=[];case_records=[]
    def check(name,result):
        assert result,name
        checks.append(name)
    def run_case(name,response,expect_success=False,limits=None,opener_mode='normal'):
        target=root/name
        opener=Opener(response,target,opener_mode)
        client=RangeClient(target,[OBJECT],{'synthetic':True},opener=opener,limits=limits)
        failure=None
        try: result=client.get(URL,0,7,'header.bin','tiff_header')
        except Exception as error:failure=error
        check(name+': expected outcome',(failure is None)==expect_success)
        saved=json.loads((target/'request-ledger.json').read_text())
        check(name+': permanent attempt and reservation',len(saved['requests'])==1 and saved['reserved_response_body_bytes']==9)
        check(name+': exactly one open attempt',len(opener.calls)==1)
        check(name+': no retry or redirect',saved['automatic_retries']==0 and saved['redirects_followed']==0)
        if opener_mode not in ('connect-failure','slow-open'):check(name+': response closed',response.closed)
        entry=saved['requests'][0]
        check(name+': retained count within reservation',entry['retained_body_bytes']<=9)
        if expect_success:
            check(name+': exact success payload',result[0]==b'12345678' and entry['success'] is True)
            check(name+': excess-byte probe',response.read_calls==[8,1])
        else:
            check(name+': failure retained',entry['success'] is False and 'stop_reason' in entry)
        case_records.append(dict(name=name,ledger=pin(target/'request-ledger.json'),
                                 read_calls=response.read_calls,opener_calls=len(opener.calls),
                                 failure=None if failure is None else type(failure).__name__))
        return client,response,entry

    client,response,entry=run_case('valid',Response(),True)
    check('valid: exact readback length',entry['retained_body_bytes']==8)
    before=len(client.ledger['requests'])
    try:client.get(URL,0,7,'different-name.bin','tiff_header')
    except ContractError:pass
    else:raise AssertionError('duplicate category was retried')
    check('duplicate category rejected before reservation',len(client.ledger['requests'])==before)
    check('redirect handler refuses followup',NoRedirect().redirect_request(None,None,302,'x',Message(),'https://elsewhere.invalid/') is None)
    for status in (200,301,302,307,412,416,500):
        _,response,entry=run_case('status-'+str(status),Response(status=status))
        check('status '+str(status)+': no body read',not response.read_calls and entry['retained_body_bytes']==0)

    header_cases=[]
    for key in ('Content-Length','Content-Range','ETag','Content-Type'):
        header_cases.append(('missing-'+key.lower(),[(k,v) for k,v in HEADERS if k!=key]))
    for key,bad in [('Content-Length','7'),('Content-Length','9'),('Content-Length','-8'),
                    ('Content-Range','bytes 1-8/512'),('Content-Range','bytes 0-7/513'),
                    ('Content-Range','bytes 0-7/*'),('ETag','W/"synthetic-only"'),
                    ('ETag','"changed"'),('ETag','synthetic-only'),
                    ('Content-Type','multipart/byteranges; boundary=x')]:
        label='bad-'+key.lower()+'-'+str(len(header_cases))
        header_cases.append((label,[(k,bad if k==key else v) for k,v in HEADERS]))
    header_cases.extend([('gzip',HEADERS+[('Content-Encoding','gzip')]),
                         ('chunked',HEADERS+[('Transfer-Encoding','chunked')]),
                         ('huge-header',HEADERS+[('X-Synthetic','x'*16385)])])
    for key,value in [('Content-Length','8'),('Content-Range','bytes 0-7/512'),
                      ('ETag','"synthetic-only"'),('Content-Type','image/tiff'),
                      ('Content-Encoding','identity'),('Transfer-Encoding',''),
                      ('Last-Modified','Thu, 01 Jan 1970 00:00:00 GMT')]:
        base=list(HEADERS)
        if key not in dict(HEADERS):base.append((key,value))
        header_cases.append(('duplicate-'+key.lower(),base+[(key.lower(),value)]))
    for name,headers in header_cases:
        _,response,entry=run_case(name,Response(headers=headers))
        check(name+': no body read',not response.read_calls and entry['retained_body_bytes']==0)

    _,response,entry=run_case('truncated',Response(body=b'123'))
    check('truncated bytes preserved',entry['retained_body_bytes']==3 and Path(entry['partial_file']['path']).read_bytes()==b'123')
    _,response,entry=run_case('incomplete-read',Response(mode='incomplete'))
    check('IncompleteRead partial preserved',entry['retained_body_bytes']==3 and Path(entry['partial_file']['path']).read_bytes()==b'123')
    _,response,entry=run_case('excess-byte',Response(body=b'123456789'))
    check('excess byte charged and retained',entry['retained_body_bytes']==9 and Path(entry['partial_file']['path']).read_bytes()==b'123456789')
    _,response,entry=run_case('connection-failure',Response(),opener_mode='connect-failure')
    check('connection failure keeps full reservation without body',entry['retained_body_bytes']==0 and not response.read_calls)
    limits=dict(LIMITS,per_request_wall_seconds=.005)
    _,response,entry=run_case('absolute-open-timeout',Response(),limits=limits,opener_mode='slow-open')
    check('absolute open deadline stopped before body',not response.read_calls and 'deadline' in entry['stop_reason'])
    _,response,entry=run_case('absolute-read-timeout',Response(mode='slow'),limits=limits)
    check('absolute read deadline enforced','deadline' in entry['stop_reason'])

    for name,altered in [('attempt-budget',dict(LIMITS,maximum_attempts=0)),
                          ('reservation-budget',dict(LIMITS,maximum_reserved_response_body_bytes=8))]:
        target=root/name;opener=Opener(Response(),target)
        client=RangeClient(target,[OBJECT],{'synthetic':True},opener=opener,limits=altered)
        try:client.get(URL,0,7,'header.bin','tiff_header')
        except ContractError:pass
        else:raise AssertionError(name)
        check(name+': no opening or charge',not opener.calls and not client.ledger['requests'] and client.ledger['reserved_response_body_bytes']==0)
        case_records.append(dict(name=name,ledger=pin(target/'request-ledger.json'),opener_calls=0))

    result=dict(status='PASS',checks=checks,check_count=len(checks),cases=case_records,
                source_pins=[pin(SOURCE/name) for name in ('common.py','transport.py')],
                fixture_source=pin(Path(__file__).resolve()),live_requests=0,
                scientific_arrays_read=0,real_response_bodies_read=0,
                limitations='Fake HTTP calls exercise source contracts; not a live network or process-crash durability test.')
    with (HERE/'transport-controls-v2.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='PASS',check_count=len(checks),scenario_count=len(case_records))))


if __name__=='__main__':main()
