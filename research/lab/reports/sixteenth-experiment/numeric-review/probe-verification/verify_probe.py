"""One offline byte-only check of completed V2 metadata and eight-row probes."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import resource
import socket
import sys
import time

HERE=Path(__file__).resolve().parent
NUMERIC=HERE.parent
PINS={}


def require(value,message):
    if not value:raise ValueError(message)


def offline(*args,**kwargs):raise RuntimeError('Independent probe network forbidden')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline


def record(path):
    path=Path(path).resolve(strict=True)
    require(path.suffix not in ('.npy','.npz','.pt','.pth','.ckpt','.safetensors'),'Arrays/checkpoints forbidden')
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha)


def pin(item):
    wanted=dict(path=str(Path(item['path']).resolve()),bytes=item.get('bytes',item.get('size_bytes')),sha256=item['sha256'])
    require(record(wanted['path'])==wanted,'Byte pin changed: '+Path(wanted['path']).name)
    require(wanted['path'] not in PINS or PINS[wanted['path']]==wanted,'Conflicting byte pin')
    PINS[wanted['path']]=wanted
    return Path(wanted['path'])


def collect(value):
    if isinstance(value,dict):
        if {'path','sha256'}<=value.keys() and ('bytes'in value or 'size_bytes'in value):
            if Path(value['path']).suffix not in ('.npy','.npz','.pt','.pth','.ckpt','.safetensors'):pin(value)
        for child in value.values():collect(child)
    elif isinstance(value,list):
        for child in value:collect(child)


def read(item):
    x=json.loads(pin(item).read_text());collect(x);return x


def helper(path,sha,name):
    require(record(path)['sha256']==sha,'Frozen helper differs')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


TIFF=helper(NUMERIC/'endian-v2/tiff_math_v2.py','0157c1b5bde34395e0af958d72d0892ac97823b87ab476bc6841c2da6f2a613a','independent_probe_tiff')
MIME=helper(NUMERIC/'multipart_math.py','c2ac19ad7964eeecabb180a20fc5ca9136418613ca6d3f5c5ad0aaadac4a5439','independent_probe_mime')


def transport(entry,span,total,etag,if_match):
    body=pin(entry['file']).read_bytes();headers=entry['http_identity']
    require(entry['success']is True and entry['http_status']==206 and entry['ranges_inclusive']==span,
            'Exact successful206 range required')
    require(entry['body_sha256']==hashlib.sha256(body).hexdigest() and
            len(body)==entry['retained_body_bytes']==entry['charged_response_bytes'],'Transport bytes/accounting differ')
    cap=sum(b-a+1 for a,b in span)+(1 if entry['mode']=='single' else 8192)
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


def ledger(result,expected_count,expected_bytes,limits):
    x=read(result['request_ledger']);require(x['plan']==result['plan'],'Ledger plan differs')
    require(x['verified_tls']is True and x['redirects_followed']==x['automatic_retries']==0 and
            len(x['requests'])==result['http_attempts']==expected_count and x['limits']==limits,'Ledger scope differs')
    require(sum(r['retained_body_bytes']for r in x['requests'])==x['retained_body_bytes']==result['retained_body_bytes']==expected_bytes and
            sum(r['charged_response_bytes']for r in x['requests'])==x['charged_response_bytes']==result['charged_response_bytes']==expected_bytes,
            'Stage byte totals differ')
    require(expected_count<=limits['attempts']and expected_bytes<=limits['body_bytes'],'Stage cap exceeded')
    return x['requests']


def synthetic_driver(output):
    body=b'abcd';path=output/'synthetic-body.bin';path.write_bytes(body)
    e=dict(file=record(path),http_identity={'etag':'"fixture"','content-range':'bytes 10-13/30','content-length':'4'},
        success=True,http_status=206,ranges_inclusive=[[10,13]],body_sha256=hashlib.sha256(body).hexdigest(),
        retained_body_bytes=4,charged_response_bytes=4,reserved_cap_bytes=5,mode='single',etag='"fixture"',object_total_bytes=30,
        request_headers={'Range':'bytes=10-13','Accept-Encoding':'identity','If-Match':'"fixture"'})
    require(transport(e,[[10,13]],30,'"fixture"','"fixture"')==body,'Synthetic transport differs')
    rejected=0
    for key,value in [('http_status',200),('body_sha256','0'*64),('retained_body_bytes',3),('object_total_bytes',31)]:
        bad=dict(e);bad[key]=value
        try:transport(bad,[[10,13]],30,'"fixture"','"fixture"')
        except ValueError:rejected+=1
        else:raise ValueError('Bad synthetic transport accepted')
    for pin_value in [None,'"wrong"']:
        try:transport(e,[[10,13]],30,'"fixture"',pin_value)
        except ValueError:rejected+=1
        else:raise ValueError('Bad synthetic If-Match accepted')
    return dict(status='PASS',TIFF=TIFF.fixtures(),multipart=MIME.fixtures(),driver_transport_cases=7,
                actual_inputs_read=0,numeric_CT_decodes=0,network_calls=0,model_calls=0)


def verify(args):
    started=time.perf_counter();resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
    hp=record(args.header);sp=record(args.serving)
    require(hp['sha256']==args.header_sha256 and sp['sha256']==args.serving_sha256,'Root entry pins differ')
    h,s=read(hp),read(sp);require(h['status']==s['status']=='PASS'and h['stage']=='headers'and s['stage']=='serving','Completed probes required')
    ph,ps=read(h['plan']),read(s['plan'])
    require(ph['ready']is True and ps['ready']is True and h['design']==s['design']==ph['design']==ps['design'] and
            h['design']['sha256']=='b19be5cc4de3622a38cea5c586f55559c7211333438408bfcdcaebd54f7d22bd','FrozenV2 design differs')
    require(ps['header_result']==hp and h['sources']==s['sources']==ph['sources']==ps['sources'] and
            h['tiff_contracts']==s['tiff_contracts']==ps['tiff_contracts'],'Header/source binding differs')
    expected=[('frag1','PHercParis2Fr47','Paris2','evaluation',[8181,6330],[2304,1152,1152,1152]),
              ('frag2','PHercParis2Fr143','Paris2','evaluation',[14830,9506],[1152,1152,1152,1152]),
              ('frag3','PHercParis1Fr34','Paris1','calibration',[7606,5249],[2304,1152,1152,1152])]
    require([(a['fragment'],a['identity'],a['parent_papyrus'],a['role'],a['shape_yx'],a['bbox_source_xywh'])for a in h['sources']]==expected,
            'Literal source/role/box identities differ')
    source=read(ph['source_contract']);benchmark=read(ph['benchmark_contract'])
    require(ph['source_contract']['sha256']=='53ff9fd5cccb2604ebcd8b116b69b1f3c4ba25ed79fa624aaedf21bdaa709890'and
            ph['benchmark_contract']['sha256']=='a4026cce7ca55d5aaeeaa0736061ab6b7c8aa13875155682d24e74c139fe5735','Source/benchmark pins differ')
    for a,c in zip(h['sources'],source['eligible_candidates'],strict=True):
        require(a['url']==c['surface_volume_url']+'07.tif','Literal TIFF URL differs')
        b=next(k for k in benchmark['selected_contexts']if k['case_id']==a['case_id'])
        require(b['bbox_source_xywh']==a['bbox_source_xywh']and b['fragment']==a['identity']and b['role']==a['role'],'First selected box/role differs')
    hr=ledger(h,8,484,ph['limits']['headers']);sr=ledger(s,6,395228,ps['limits']['serving'])
    reuse=h['prefix_reuse'];prior=read(reuse['result']);pr=read(reuse['ledger'])['requests'][0]
    require(prior['status']=='STOP'and prior['http_attempts']==1 and reuse==ph['prefix_reuse']==ps['prefix_reuse'] and
            h['reused_metadata_bytes']==8 and reuse['reused_bytes']==8 and reuse['new_GETs_for_prefix']==0,'Carried history differs')
    old=transport(pr,[[0,7]],reuse['object_total_bytes'],reuse['etag'],None)
    require(old==bytes.fromhex('4d4d002a00000008')and hashlib.sha256(old).hexdigest()==
            '39740c796de25c70b056b87da657c7fdf56b5b1eb841a25ed0f3e504ff7013d8','Original prefix differs')
    cursor=0;contracts={};row_receipts=[]
    for index,(a,c)in enumerate(zip(h['sources'],h['tiff_contracts'],strict=True)):
        require(c['url']==a['url']and c['fragment']==a['fragment'],'TIFF identity differs')
        raw=c['raw_metadata'];prefix=pin(raw['header']).read_bytes();count_bytes=pin(raw['ifd_count']).read_bytes();ifd=pin(raw['ifd']).read_bytes()
        rebuilt=TIFF.inspect(prefix,ifd,a['shape_yx'],c['object_total_bytes']);origin=rebuilt['ifd_offset']
        require(len(count_bytes)==2 and int.from_bytes(count_bytes,rebuilt['byte_order'])==(len(ifd)-6)//12,'IFD count bytes differ')
        for key in ['dtype','byte_order','strip_offset','strip_bytes','ifd_offset','orientation','photometric']:
            require(c[key]==rebuilt[key],'TIFF metadata scalar differs: '+key)
        require(c['width']==a['shape_yx'][1]and c['height']==a['shape_yx'][0]and c['ifd_bytes']==len(ifd),'TIFF dimensions/table extent differ')
        require(c['specification_defaults_used']==rebuilt['specification_defaults_applied'],'TIFF defaults differ')
        require(c['auxiliary_metadata_extents']==[dict(tag=e['tag'],extent_half_open=[e['start'],e['stop_exclusive']],dereferenced=False)
                for e in rebuilt['external_metadata_extents']],'TIFF external extents differ')
        require(set(c['tags'])==set(rebuilt['tags']),'TIFF tag set differs')
        for tag,t in rebuilt['tags'].items():
            require(c['tags'][tag]==dict(**t,storage_bytes=TIFF.TYPE_BYTES[t['type']]*t['count']),'TIFF raw tag bytes differ')
        if index==0:
            require(prefix==old and c['prefix_reused']is True and c['prefix_source']==reuse['header'],'Prefix carry differs')
        else:
            require(transport(hr[cursor],[[0,7]],c['object_total_bytes'],c['etag'],None)==prefix,'Header request body differs');cursor+=1
        for span,body in [([origin,origin+1],count_bytes),([origin,origin+len(ifd)-1],ifd)]:
            require(hr[cursor]['url']==a['url']and transport(hr[cursor],[span],c['object_total_bytes'],c['etag'],c['etag'])==body,
                    'Count/table transport differs');cursor+=1
        x,y,w,_=a['bbox_source_xywh'];width=c['width'];base=c['strip_offset']
        ranges=[[base+2*((y+i)*width+x),base+2*((y+i)*width+x+w)-1]for i in range(8)]
        comparator=[base+2*y*width,base+2*(y+8)*width-1]
        me,ce=sr[2*index:2*index+2]
        require(me['url']==ce['url']==a['url']and me['mode']=='multipart'and ce['mode']=='single','Serving request order differs')
        multipart=transport(me,ranges,c['object_total_bytes'],c['etag'],c['etag'])
        band=transport(ce,[comparator],c['object_total_bytes'],c['etag'],c['etag'])
        parsed=MIME.parse(multipart,me['http_identity']['content-type'],ranges,c['object_total_bytes'])
        report=s['serving_comparisons'][index]
        require(report['fragment']==a['fragment']and report['url']==a['url']and report['etag']==c['etag']and
                report['object_total_bytes']==c['object_total_bytes']and report['parts']==me['parts'],'Serving receipt identity differs')
        require(pin(report['multipart']).read_bytes()==multipart and pin(report['comparator']).read_bytes()==band,'Serving body pins differ')
        require(len(report['rows'])==len(report['parts'])==8,'Exactly8 parts/rows required')
        for part in report['parts']:
            key=tuple(part['range_inclusive']);payload=parsed['parts'][key]
            require(part['payload_bytes']==len(payload)==2304 and hashlib.sha256(payload).hexdigest()==part['payload_sha256']and
                    multipart[part['payload_offset']:part['payload_offset']+2304]==payload,'Recorded MIME extraction differs')
        for i,span in enumerate(ranges):
            payload=parsed['parts'][tuple(span)];offset=2*(i*width+x)
            require(band[offset:offset+2304]==payload,'Independent comparator row bytes differ')
            wanted=dict(row_native_y=y+i,range_inclusive=span,comparator_body_offset=offset,payload_bytes=2304,
                        bitwise_equal=True,sha256=hashlib.sha256(payload).hexdigest())
            require(report['rows'][i]==wanted,'Reported row comparison differs');row_receipts.append(dict(fragment=a['fragment'],**wanted))
        contracts[a['fragment']]=rebuilt
    require(cursor==8 and len(row_receipts)==24 and [contracts[k]['dtype']for k in contracts]==['>u2','<u2','<u2'],'Expected observed byte orders differ')
    for result in [h,s]:require(result['numeric_pixel_decodes']==result['model_calls']==0 and result['production_acquisition']is False,'Probe scope differs')
    for item in PINS.values():require(record(item['path'])==item,'Input changed during verification')
    require('torch'not in sys.modules,'Torch import forbidden')
    return dict(status='PASS',header_result=hp,serving_result=sp,TIFF_contracts=contracts,rows=row_receipts,
        all24_rows_byte_exact=True,transport_requests_verified=15,new_metadata_requests=8,serving_requests=6,
        prior_prefix_requests=1,new_response_bytes=395712,prior_prefix_bytes=8,known_byte_order_verified=True,
        inputs_unchanged=True,verified_byte_pins=list(PINS.values()),wall_seconds=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        numeric_CT_decodes=0,arrays_read=0,Torch_imports=0,checkpoint_reads=0,model_calls=0,network_calls=0,
        limitations=['Eight-row transport/layout evidence only; no production part-count/153-layer/CT/model readiness claim.',
                     'HTTP/TLS and execution claims remain pinned source/recorded receipts; no network replay.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true')
    for name in ['header','serving']:
        p.add_argument('--'+name,type=Path);p.add_argument('--'+name+'-sha256')
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    require(HERE in args.output_dir.resolve().parents and not args.output_dir.exists(),'Fresh owned output required')
    args.output_dir.mkdir()
    try:
        x=synthetic_driver(args.output_dir)if args.self_test else verify(args)
    except Exception as e:
        (args.output_dir/'verification.json').write_text(json.dumps(dict(status='STOPPED',reason=str(e),error_type=type(e).__name__),indent=2)+'\n')
        raise
    x['driver']=record(__file__);x['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    q=args.output_dir/'verification.json';q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps(record(q)))


if __name__=='__main__':main()
