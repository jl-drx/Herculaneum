"""Tiny synthetic receipt fixtures; no real CT, network or model operations."""
import argparse
import copy
import json
from pathlib import Path
import time

from pooling_common import HERE,record,identity,write_json,require,row_range
from pooling_provenance import (PRODUCER_ROOT,shared_elapsed,required_root_coverage,
    validate_global_receipts,validate_terminal_chain,validate_ready_pixel_plan)


def run():
    passed=[]
    def accept(name,fn):fn();passed.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError):passed.append(name);return
        raise AssertionError('Bad fixture accepted: '+name)
    pin=dict(path=str(HERE/'synthetic-only.bin'),size_bytes=8,sha256='a'*64)
    other=dict(pin,sha256='b'*64);body=dict(pin,size_bytes=5000)
    contract=dict(url='https://example.invalid/07.tif',etag='"same"',object_total_bytes=100000,width=20,height=20,strip_offset=8,strip_bytes=800)
    spans=[row_range(contract,3,y,2) for y in (4,5)]
    request=dict(request_id='synthetic',url=contract['url'],mode='multipart',ranges=spans,
        pin=dict(etag=contract['etag'],object_total_bytes=contract['object_total_bytes']),reserved_cap_bytes=7168,
        case_id='fixture',fragment='frag1',plane_index=7,native_row_start=4,row_count=2,native_x_start=3,native_crop_width=2)
    batch=dict(request,source_url=request['url'],multipart=body,content_type='multipart/byteranges; boundary=x',parts=[])
    entry=dict(request=pin,transport_ledger=pin,charged_response_bytes=5000,retained_body_bytes=5000,reserved_cap_bytes=7168,
        url=request['url'],mode=request['mode'])
    terminal=dict(status='PASS',request_id='synthetic',request=pin,body=body,transport_ledger=pin,status_code=206,
        content_type=batch['content_type'],etag=contract['etag'],object_total_bytes=100000,
        charged_response_bytes=5000,retained_body_bytes=5000)
    config=dict(stage='pixels',request=request,plan=pin)
    http=dict(success=True,url=request['url'],mode='multipart',ranges_inclusive=spans,http_status=206,file=body,parts=[],
        etag=contract['etag'],object_total_bytes=100000,body_sha256=body['sha256'],charged_response_bytes=5000,
        retained_body_bytes=5000,reserved_cap_bytes=7168,request_headers={'Range':'bytes='+','.join(f'{a}-{b}' for a,b in spans),
        'If-Match':contract['etag'],'Accept-Encoding':'identity'},http_identity={'etag':contract['etag'],
        'content-type':batch['content_type'],'content-length':'5000'})
    transport=dict(plan=pin,requests=[http],automatic_retries=0,redirects_followed=0,verified_tls=True,
        charged_response_bytes=5000,retained_body_bytes=5000)
    chain=lambda t=terminal,c=config,l=transport:validate_terminal_chain(batch,request,entry,t,c,l,pin,contract)
    accept('valid synthetic terminal chain after JSON roundtrip',lambda:chain(json.loads(json.dumps(terminal))))
    for key,value in [('status','STOP'),('body',other),('etag','"changed"'),('charged_response_bytes',4999),('status_code',200)]:
        reject('terminal '+key,lambda key=key,value=value:chain(dict(terminal,**{key:value})))
    reject('foreign request plan',lambda:chain(c=dict(config,plan=other)))
    altered=copy.deepcopy(transport);altered['requests'][0]['request_headers']['If-Match']='"changed"'
    reject('child If-Match differs',lambda:chain(l=altered))
    reject('child retry scope differs',lambda:chain(l=dict(transport,automatic_retries=1)))
    altered=copy.deepcopy(transport);altered['requests'][0]['success']=False
    reject('child unsuccessful',lambda:chain(l=altered))
    accept('current shared window',lambda:shared_elapsed(100,200))
    reject('expired shared window',lambda:shared_elapsed(100,1600.0001))
    reject('future shared epoch',lambda:shared_elapsed(200,100))
    reject('nonfinite shared epoch',lambda:shared_elapsed(float('nan'),200))
    accept('mandatory producer root covers plans',lambda:required_root_coverage([PRODUCER_ROOT],[PRODUCER_ROOT/'plans/x']))
    reject('omitted producer root',lambda:required_root_coverage([HERE],[PRODUCER_ROOT/'plans/x']))
    reject('foreign required production path',lambda:required_root_coverage([PRODUCER_ROOT],['/tmp/unaccounted']))
    reject('foreign pixel stage',lambda:validate_ready_pixel_plan({},dict(schema_version=1,stage='metadata',ready=True),{}, {},None))
    reject('unready pixel plan',lambda:validate_ready_pixel_plan({},dict(schema_version=1,stage='pixels',ready=False),{}, {},None))
    requests=[dict(request_id=str(i)) for i in range(1530)]
    entries=[dict(request_id=str(i),status='PASS',returncode=0,terminal=pin,charged_response_bytes=1,retained_body_bytes=1) for i in range(1530)]
    index=dict(schema_version=1,terminals=[dict(request_id=str(i),terminal=pin) for i in range(1530)])
    raw=dict(plan=pin,http_attempts=1530,charged_response_bytes=1530,retained_body_bytes=1530)
    ledger=dict(stage='pixels',plan=pin,automatic_retries=0,redirects_followed=0,verified_tls=True,requests=entries,
        charged_response_bytes=1530,retained_body_bytes=1530)
    global_check=lambda l=ledger,t=index:validate_global_receipts(raw,dict(requests=requests),l,t)
    accept('complete1530global tiny receipts JSON roundtrip',lambda:global_check(json.loads(json.dumps(ledger)),json.loads(json.dumps(index))))
    reject('global foreign plan',lambda:global_check(dict(ledger,plan=other)))
    reject('missing global request',lambda:global_check(dict(ledger,requests=entries[:-1])))
    reject('duplicate terminal ID',lambda:global_check(t=dict(index,terminals=index['terminals'][:-1]+[index['terminals'][0]])))
    reject('failed global request',lambda:global_check(dict(ledger,requests=[dict(entries[0],status='STOP')]+entries[1:])))
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    before=[record(p) for p in sorted(HERE.glob('*.py'))];passed=run();require(before==[record(p) for p in sorted(HERE.glob('*.py'))],'Fixture source changed')
    args.output_dir.mkdir(parents=True,exist_ok=False);path=args.output_dir/'fixture-receipt.json'
    write_json(path,dict(status='PASS',checks=passed,count=len(passed),code=before,actual_CT_reads=0,model_calls=0,network_calls=0))
    print(json.dumps(dict(status='PASS',receipt=record(path))))


if __name__=='__main__':main()
