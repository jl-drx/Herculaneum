"""Root-only bounded header or serving probe; no numeric pixel decode/models/fallback."""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import signal
import struct
import time
from probe_common import *
from prepare_probe_plan import load_plan
from range_http import RangeClient


def inspect_header(client, source, reuse=None):
    name, url = source['fragment'],source['url']
    if reuse is not None:
        require(reuse['url'] == url,'Cached header URL differs')
        hpath = client.root/'raw'/(name+'-header-reused.bin')
        require(not hpath.exists(),'Reused header output already exists')
        hpath.write_bytes(verify(reuse['header']).read_bytes())
        hrec = {'etag':reuse['etag'],'object_total_bytes':reuse['object_total_bytes']}
    else:
        hpath,hrec = client.get(url,[[0,7]],name+'-header.bin')
    header = hpath.read_bytes()
    layout=tiff_header(header)
    order,offset=layout['struct_endian'],layout['ifd_offset']
    total = hrec['object_total_bytes']
    require(8 <= offset and offset+2 <= total,'IFD offset out of object bounds')
    cpath,crec = client.get(url,[[offset,offset+1]],name+'-ifd-count.bin')
    count = struct.unpack(order+'H',cpath.read_bytes())[0]
    require(0 < count <= 128,'IFD count outside pinned adaptive rule')
    length = 2+12*count+4
    require(offset+length <= total,'IFD table exceeds object length')
    ipath,irec = client.get(url,[[offset,offset+length-1]],name+'-ifd.bin')
    require(struct.unpack_from(order+'H',ipath.read_bytes())[0] == count,'IFD count changed')
    contract = parse_tiff(header,ipath.read_bytes(),total,source['shape_yx'])
    return dict(**contract,fragment=name,url=url,etag=hrec['etag'],object_total_bytes=total,
                prefix_reused=reuse is not None,prefix_source=reuse['header'] if reuse is not None else None,
                raw_metadata={'header':record(hpath),'ifd_count':record(cpath),'ifd':record(ipath)})


def inspect_serving(client, source, contract, request):
    name,url = source['fragment'],source['url']
    path,mrec = client.get(url,request['crop_row_ranges'],name+'-multipart.bin',mode='multipart')
    # Comparator is reached only if all eight multipart parts were accepted.
    compare_path,crec = client.get(url,[request['comparator_range']],name+'-comparator.bin')
    body,band = path.read_bytes(),compare_path.read_bytes()
    parts = {tuple(p['range_inclusive']):p for p in mrec['parts']}
    comparisons=[]
    for i,span in enumerate(request['crop_row_ranges']):
        part = parts[tuple(span)]
        crop = body[part['payload_offset']:part['payload_offset']+part['payload_bytes']]
        offset = i*contract['width']*2+request['comparator_crop_byte_offset_per_row']
        expected = band[offset:offset+request['crop_bytes_per_row']]
        require(len(expected) == len(crop) == 2304 and expected == crop,'Multipart/comparator raw bytes differ')
        comparisons.append(dict(row_native_y=source['bbox_source_xywh'][1]+i,range_inclusive=span,
                               comparator_body_offset=offset,payload_bytes=len(crop),bitwise_equal=True,
                               sha256=hashlib.sha256(crop).hexdigest()))
    return dict(fragment=name,url=url,etag=contract['etag'],object_total_bytes=contract['object_total_bytes'],
                multipart=record(path),comparator=record(compare_path),parts=mrec['parts'],
                rows=comparisons,numeric_pixel_decodes=0,
                limitation='Only eight declared rows on this object; no large-range or production support claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    started=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(2147483648,)*2)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Probe wall cap')))
    signal.alarm(LIMITS['wall_seconds'])
    plan=load_plan(args.plan)
    root=Path(args.output_dir).resolve()
    pins={c['url']:{'etag':c['etag'],'object_total_bytes':c['object_total_bytes']}
          for c in plan.get('tiff_contracts',[])}
    if plan['stage']=='headers':
        reuse=plan['prefix_reuse']
        pins[reuse['url']]={'etag':reuse['etag'],'object_total_bytes':reuse['object_total_bytes']}
    client=RangeClient(root,[s['url']for s in plan['sources']],LIMITS[plan['stage']],record(args.plan),pins=pins)
    result=dict(schema_version=1,status='STARTED',stage=plan['stage'],plan=record(args.plan),
                design=plan['design'],sources=plan['sources'],tiff_contracts=[],serving_comparisons=[],
                prefix_reuse=plan['prefix_reuse'],reused_metadata_bytes=8 if plan['stage']=='headers' else 0,
                prior_attempt_history={'result':plan['prefix_reuse']['result'],'http_attempts':1,
                                       'charged_response_bytes':8,'retained_body_bytes':8},
                numeric_pixel_decodes=0,model_calls=0,production_acquisition=False)
    try:
        if plan['stage']=='headers':
            for index,source in enumerate(plan['sources']):
                result['tiff_contracts'].append(inspect_header(client,source,plan['prefix_reuse'] if index==0 else None))
        else:
            result['tiff_contracts']=plan['tiff_contracts']
            for source,contract,request in zip(plan['sources'],plan['tiff_contracts'],plan['requests'],strict=True):
                result['serving_comparisons'].append(inspect_serving(client,source,contract,request))
        load_plan(args.plan)
        result['status']='PASS'
    except Exception as error:
        result.update(status='STOP',failure_type=type(error).__name__)
        if isinstance(error,ContractError):
            result['stop_reason']=str(error)
        raise
    finally:
        signal.alarm(0)
        result.update(wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      request_ledger=record(root/'request-ledger.json'),http_attempts=len(client.ledger['requests']),
                      charged_response_bytes=client.ledger['charged_response_bytes'],retained_body_bytes=client.ledger['retained_body_bytes'])
        save(root/'result.json',result)
    require(sum(p.stat().st_size for p in root.rglob('*')if p.is_file()) <= LIMITS['output_bytes'],'Probe output exceeds cap')
    print(json.dumps({'status':'PASS','stage':plan['stage'],'result':record(root/'result.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'STOP','failure_type':type(error).__name__}))
        raise SystemExit(1)
