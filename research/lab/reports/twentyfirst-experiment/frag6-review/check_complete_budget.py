"""Independent maximum Stage A traversal with only synthetic metadata spans."""
from email.message import Message
import hashlib
import json
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'frag6-stage-a'
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(HERE))
from acquire import run
from check_parser_controls_v2 import ppm,tiff


def deny(*args,**kwargs): raise AssertionError('No live requests in fixture')
socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny


def pin(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    plan_path=SOURCE/'execution-plan.json'
    plan=json.loads(plan_path.read_text())
    objects=json.loads(json.dumps(plan['objects']))
    payloads={};description=b'X'*65535+b'\0'
    prefix,total=ppm();objects[0]['total_bytes']=total
    payloads[(objects[0]['url'],0,1023)]=prefix
    for index,obj in enumerate(objects[1:]):
        header,table,total,info=tiff(order='>' if index==1 else '<',description=description,
                                   extra=tuple((500+i,3,1,i) for i in range(118)))
        assert len(table)==1542
        obj['total_bytes']=total
        payloads[(obj['url'],0,7)]=header
        payloads[(obj['url'],8,9)]=table[:2]
        payloads[(obj['url'],8,8+len(table)-1)]=table
        a=info['description_offset']
        payloads[(obj['url'],a,a+65535)]=description
    output=HERE/'complete-budget-fixture-v1'
    calls=[]
    by_url={obj['url']:obj for obj in objects}
    class Response:
        def __init__(self,body,url,a,b):
            self.body=body;self.position=0;self.status=206;self.headers=Message()
            for key,value in [('Content-Length',str(len(body))),('Content-Range',f'bytes {a}-{b}/{by_url[url]["total_bytes"]}'),
                              ('ETag',by_url[url]['etag']),('Content-Type','application/octet-stream')]:
                self.headers[key]=value
        def read(self,n):
            block=self.body[self.position:self.position+n];self.position+=len(block);return block
        def close(self):pass
    class Opener:
        def open(self,request,timeout):
            assert request.get_method()=='GET' and request.get_header('Accept-encoding')=='identity'
            assert request.get_header('If-match')==by_url[request.full_url]['etag']
            a,b=map(int,request.get_header('Range')[6:].split('-'))
            key=(request.full_url,a,b)
            assert key in payloads and key not in calls
            saved=json.loads((output/'request-ledger.json').read_text())
            assert len(saved['requests'])==len(calls)+1
            assert saved['reserved_response_body_bytes']==sum(e['reserved_response_body_bytes'] for e in saved['requests'])
            assert saved['requests'][-1]['range_inclusive']==[a,b]
            calls.append(key)
            return Response(payloads[key],request.full_url,a,b)
    result=run(output,objects,{'synthetic_only':True},opener=Opener())
    assert result['status']=='PASS' and len(calls)==result['http_attempts']==13
    assert result['reserved_response_body_bytes']==202301
    assert result['retained_response_body_bytes']==202301-13
    assert [x[0] for x in calls[:4]]==[o['url'] for o in objects]
    assert result['numerical_ppm_records_decoded']==result['image_pixels_decoded']==result['model_calls']==0
    assert result['canvas_shape_yx']==[13,11]
    assert result['unknown_render_fields'] and all(v is None for v in result['unknown_render_fields'].values())
    assert result['automatic_stage_advance'] is False
    receipt=dict(status='PASS',live_requests=0,maximum_attempts=13,maximum_reserved_response_body_bytes=202301,
                 retained_synthetic_body_bytes=202288,plan=pin(plan_path),
                 source_pins=[pin(SOURCE/name) for name in ('common.py','parsers.py','transport.py','acquire.py')],
                 fixture_source=pin(Path(__file__).resolve()),fixture_generator=pin(HERE/'check_parser_controls_v2.py'),
                 structural_result=pin(output/'STRUCTURAL_CONTRACT.json'),ledger=pin(output/'request-ledger.json'),
                 initial_order_exact=True,three_independent_byte_orders=True,no_scientific_decoding=True,
                 render_fields_unresolved=True,no_stage_advance=True)
    with (HERE/'complete-budget-controls-v1.json').open('x') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='PASS',requests=13,reserved_bytes=202301,live_requests=0)))


if __name__=='__main__':main()
