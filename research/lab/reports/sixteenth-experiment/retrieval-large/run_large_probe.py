"""Root-only six-request 256-row serving test; byte comparisons without pixel decode."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
from large_common import *
from prepare_plan import load_plan
from range_http import RangeClient


def inspect_serving(client,source,contract,request):
    name,url=source['fragment'],source['url']
    path,mrec=client.get(url,request['crop_row_ranges'],name+'-multipart.bin',mode='multipart')
    # Exact 256-part acceptance is required before the ordinary comparator GET.
    compare_path,crec=client.get(url,[request['comparator_range']],name+'-comparator.bin')
    body,band=path.read_bytes(),compare_path.read_bytes()
    parts={tuple(p['range_inclusive']):p for p in mrec['parts']}
    rows=[]
    for i,span in enumerate(request['crop_row_ranges']):
        part=parts[tuple(span)]
        crop=body[part['payload_offset']:part['payload_offset']+part['payload_bytes']]
        offset=i*contract['width']*2+request['comparator_crop_byte_offset_per_row']
        expected=band[offset:offset+request['crop_bytes_per_row']]
        require(len(expected) == len(crop) == 2304 and expected == crop,
                'Multipart/comparator raw bytes differ')
        rows.append(dict(row_native_y=request['native_row_start']+i,range_inclusive=span,
                         multipart_body_offset=part['payload_offset'],comparator_body_offset=offset,
                         payload_bytes=len(crop),bitwise_equal=True,sha256=hashlib.sha256(crop).hexdigest()))
    return dict(fragment=name,case_id=source['case_id'],url=url,etag=contract['etag'],
                object_total_bytes=contract['object_total_bytes'],dtype=contract['dtype'],
                byte_order=contract['byte_order'],bbox_source_xywh=source['bbox_source_xywh'],
                multipart=record(path),comparator=record(compare_path),parts=mrec['parts'],rows=rows,
                numeric_pixel_decodes=0,
                limitation='Only declared256rows on this plane07 object; other150planes remain unaudited')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    started=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Large-probe wall cap')))
    signal.alarm(LIMITS['wall_seconds'])
    plan=load_plan(args.plan)
    root=Path(args.output_dir).resolve()
    proof=plan['prerequisites']
    pins={c['url']:{'etag':c['etag'],'object_total_bytes':c['object_total_bytes']}for c in proof['tiff_contracts']}
    client=RangeClient(root,[s['url']for s in proof['sources']],LIMITS['http'],record(args.plan),pins=pins)
    result=dict(schema_version=1,status='STARTED',stage='large-serving',plan=record(args.plan),
                design=plan['design'],prerequisites=proof,sources=proof['sources'],
                tiff_contracts=proof['tiff_contracts'],serving_comparisons=[],
                numeric_pixel_decodes=0,model_calls=0,production_acquisition=False)
    try:
        for source,contract,request in zip(proof['sources'],proof['tiff_contracts'],plan['requests'],strict=True):
            result['serving_comparisons'].append(inspect_serving(client,source,contract,request))
        require(len(client.ledger['requests']) == 6,'All six requests required for PASS')
        load_plan(args.plan)
        result['status']='PASS'
    except Exception as error:
        result.update(status='STOP',failure_type=type(error).__name__)
        if isinstance(error,ContractError):
            result['stop_reason']=str(error)
        raise
    finally:
        signal.alarm(0)
        result.update(wall_seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      request_ledger=record(root/'request-ledger.json'),http_attempts=len(client.ledger['requests']),
                      charged_response_bytes=client.ledger['charged_response_bytes'],
                      retained_body_bytes=client.ledger['retained_body_bytes'])
        save(root/'result.json',result)
        if sum(p.stat().st_size for p in root.rglob('*')if p.is_file()) > LIMITS['output_bytes']:
            result.update(status='STOP',failure_type='ContractError',stop_reason='Probe output exceeds cap')
            save(root/'result.json',result)
            raise ContractError('Probe output exceeds cap')
    print(json.dumps({'status':'PASS','result':record(root/'result.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'STOP','failure_type':type(error).__name__,
                          **({'stop_reason':str(error)}if isinstance(error,ContractError)else {})}))
        raise SystemExit(1)
