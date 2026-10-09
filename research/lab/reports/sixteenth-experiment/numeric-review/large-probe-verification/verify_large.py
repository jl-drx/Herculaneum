"""Single offline256-row large-probe byte verification; no numeric CT decode."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
NUMERIC=HERE.parent
spec=importlib.util.spec_from_file_location('sealed_independent_small',NUMERIC/'probe-verification/verify_probe.py')
BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)
require,record,pin,read=BASE.require,BASE.record,BASE.pin,BASE.read
MIME=BASE.helper(NUMERIC/'production-preparation/multipart_256.py','3a901cb45dab6e680ac8d25c4892fce5f277b25889e9eb1bb50ef9dbcadceccc','sealed_independent_large_mime')

def transport(entry,span,total,etag,if_match):
    body=pin(entry['file']).read_bytes();headers=entry['http_identity']
    require(entry['success']is True and entry['http_status']==206 and entry['ranges_inclusive']==span,
            'Exact successful206 range required')
    require(entry['body_sha256']==hashlib.sha256(body).hexdigest() and
            len(body)==entry['retained_body_bytes']==entry['charged_response_bytes'],'Transport bytes/accounting differ')
    cap=sum(b-a+1 for a,b in span)+(1 if entry['mode']=='single' else 768*len(span)+1024)
    require(entry['reserved_cap_bytes']==cap and len(body)<=cap,'Transport reserved cap differs')
    require(entry['etag']==headers['etag']==etag and entry['object_total_bytes']==total and
            re.fullmatch(r'"[\x21\x23-\x7e]+"',etag) is not None,'Strong source identity differs')
    req=entry['request_headers']
    require(req['Range']=='bytes='+','.join(f'{a}-{b}'for a,b in span) and req['Accept-Encoding']=='identity' and
            req.get('If-Match')==if_match,'Request header/If-Match differs')
    require(headers.get('content-encoding','identity').lower()in ('','identity'),'Transport encoding differs')
    if 'content-length'in headers:require(headers['content-length'].isdigit()and int(headers['content-length'])==len(body),'Declared body length differs')
    if entry['mode']=='single':
        require(headers['content-range']==f'bytes {span[0][0]}-{span[0][1]}/{total}'and len(body)==span[0][1]-span[0][0]+1,
                'Exact single-range framing differs')
    else:require('content-range'not in headers,'Multipart top-level Content-Range unsupported')
    return body


def synthetic(output):
    values=[(tuple([i*10,i*10+3]),b'abcd',3000)for i in range(256)]
    body=MIME.fixture_body(values);p=output/'synthetic-multipart.bin';p.write_bytes(body)
    spans=[list(span)for span,_,_ in values]
    e=dict(file=record(p),http_identity={'etag':'"fixture"','content-type':'multipart/byteranges; boundary=fixed'},
        success=True,http_status=206,ranges_inclusive=spans,body_sha256=hashlib.sha256(body).hexdigest(),
        retained_body_bytes=len(body),charged_response_bytes=len(body),reserved_cap_bytes=1024+768*256+1024,
        mode='multipart',etag='"fixture"',object_total_bytes=3000,
        request_headers={'Range':'bytes='+','.join(f'{a}-{b}'for a,b in spans),'Accept-Encoding':'identity','If-Match':'"fixture"'})
    require(transport(e,spans,3000,'"fixture"','"fixture"')==body,'Large synthetic transport differs')
    parsed=MIME.parse(body,e['http_identity']['content-type'],spans,3000)
    require(len(parsed['parts'])==256,'Synthetic256part coverage differs')
    for field,value in [('reserved_cap_bytes',len(body)),('http_status',200),('object_total_bytes',3001)]:
        bad=dict(e);bad[field]=value
        try:transport(bad,spans,3000,'"fixture"','"fixture"')
        except ValueError:pass
        else:raise ValueError('Bad large synthetic transport accepted')
    return dict(status='PASS',transport_cases=4,parts=256,actual_probe_reads=0,numeric_CT_decodes=0)


def verify(args):
    started=time.perf_counter();resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
    entry=record(args.result);require(entry['sha256']==args.result_sha256,'Root large entry differs')
    x=read(entry);plan=read(x['plan'])
    small_path=NUMERIC/'probe-verification/completed-review/verification.json'
    small_pin=record(small_path);require(small_pin['sha256']=='f1215845e4e91e653b94c42dec0e03b1d61ca9b6023755a4475b3c9adc9891df','Prior independent proof differs')
    proof=read(small_pin);header=read(proof['header_result'])
    require(x['status']=='PASS'and x['stage']=='large-serving'and plan['ready']is True and
            x['design']==plan['design']and x['design']['sha256']==record(NUMERIC.parent/'LARGE_RANGE_DESIGN.md')['sha256'],
            'Frozen completed large design differs')
    prereq=plan['prerequisites']
    require(x['prerequisites']==prereq and prereq['header_result']==proof['header_result']and
            prereq['serving_result']==proof['serving_result']and x['sources']==header['sources']==prereq['sources']and
            x['tiff_contracts']==header['tiff_contracts']==prereq['tiff_contracts'],'Verified prerequisite/source contracts differ')
    requests=BASE.ledger(x,6,12644692,plan['limits']['http'])
    rows=[];useful=0;reserves=0
    for index,(source,contract)in enumerate(zip(x['sources'],x['tiff_contracts'],strict=True)):
        x0,y,w,h=source['bbox_source_xywh'];width=contract['width'];base=contract['strip_offset']
        require(w==h==1152 and y==1152,'Frozen first context differs')
        spans=[[base+2*((y+i)*width+x0),base+2*((y+i)*width+x0+w)-1]for i in range(256)]
        cmp=[base+2*y*width,base+2*(y+256)*width-1]
        planned=plan['requests'][index]
        require(planned['crop_row_ranges']==spans and planned['comparator_range']==cmp and
                planned['row_count']==256 and planned['native_row_start']==y and planned['url']==source['url'],'Large planned geometry differs')
        me,ce=requests[2*index:2*index+2]
        require(me['url']==ce['url']==source['url']and me['mode']=='multipart'and ce['mode']=='single','Large transport sequence differs')
        body=transport(me,spans,contract['object_total_bytes'],contract['etag'],contract['etag'])
        comparator=transport(ce,[cmp],contract['object_total_bytes'],contract['etag'],contract['etag'])
        require(planned['multipart_cap_bytes']==me['reserved_cap_bytes']and planned['comparator_cap_bytes']==ce['reserved_cap_bytes'],
                'Large planned reservation differs')
        parsed=MIME.parse(body,me['http_identity']['content-type'],spans,contract['object_total_bytes'])
        expected_parts=[dict(**m,payload_sha256=hashlib.sha256(parsed['parts'][tuple(m['range_inclusive'])]).hexdigest())for m in parsed['part_metadata']]
        report=x['serving_comparisons'][index]
        require(me['parts']==report['parts']==expected_parts,'Independent exact part offsets/headers differ')
        require(report['fragment']==source['fragment']and report['case_id']==source['case_id']and report['url']==source['url']and
                report['etag']==contract['etag']and report['object_total_bytes']==contract['object_total_bytes']and
                report['dtype']==contract['dtype']and report['byte_order']==contract['byte_order']and
                report['bbox_source_xywh']==source['bbox_source_xywh'],'Large comparison identity differs')
        require(pin(report['multipart']).read_bytes()==body and pin(report['comparator']).read_bytes()==comparator,'Large raw-body pins differ')
        by_range={tuple(m['range_inclusive']):m for m in parsed['part_metadata']}
        expected_rows=[]
        for i,span in enumerate(spans):
            payload=parsed['parts'][tuple(span)];offset=2*(i*width+x0)
            require(len(payload)==2304 and comparator[offset:offset+2304]==payload,'Large comparator row bytes differ')
            row=dict(row_native_y=y+i,range_inclusive=span,multipart_body_offset=by_range[tuple(span)]['payload_offset'],
                     comparator_body_offset=offset,payload_bytes=2304,bitwise_equal=True,sha256=hashlib.sha256(payload).hexdigest())
            expected_rows.append(row);rows.append(dict(fragment=source['fragment'],**row));useful+=2304
        require(report['rows']==expected_rows and report['numeric_pixel_decodes']==0,'Reported256rows differ')
        reserves+=me['reserved_cap_bytes']+ce['reserved_cap_bytes']
    require(len(rows)==768 and useful==1769472 and reserves==plan['maximum_reserved_body_bytes']==13157891,'Large coverage/reservation totals differ')
    require(x['numeric_pixel_decodes']==x['model_calls']==0 and x['production_acquisition']is False,'Large scope differs')
    for item in BASE.PINS.values():require(record(item['path'])==item,'Input changed during large verification')
    require('torch'not in sys.modules,'Torch forbidden')
    return dict(status='PASS',large_result=entry,prior_independent_small_probe=small_pin,rows=rows,
        all768_rows_byte_exact=True,all_exact_part_offsets_and_headers_verified=True,request_count=6,
        retained_response_bytes=12644692,reserved_response_bytes=reserves,useful_crop_bytes=useful,
        wall_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        byte_pins=list(BASE.PINS.values()),inputs_unchanged=True,numeric_CT_decodes=0,arrays_read=0,
        Torch_imports=0,checkpoint_reads=0,model_calls=0,network_calls=0,
        limitation='256raw rows on three observed plane07 objects; other150metadata contracts and production inputs remain unverified.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--result',type=Path)
    p.add_argument('--result-sha256');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    require(HERE in a.output_dir.resolve().parents and not a.output_dir.exists(),'Fresh owned large review required');a.output_dir.mkdir()
    try:x=synthetic(a.output_dir)if a.self_test else verify(a)
    except Exception as e:
        (a.output_dir/'verification.json').write_text(json.dumps(dict(status='STOPPED',reason=str(e),error_type=type(e).__name__),indent=2)+'\n');raise
    x['driver']=record(__file__);x['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    q=a.output_dir/'verification.json';q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps(record(q)))


if __name__=='__main__':main()
