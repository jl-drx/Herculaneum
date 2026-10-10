"""Prepare one fixed 256-row serving probe; no network or numeric CT decoding."""
import argparse
import json
from pathlib import Path
from large_common import *


def large_requests(source, contract):
    x,y,w,h = source['bbox_source_xywh']
    width,base = contract['width'],contract['strip_offset']
    require(w == h == 1152 and y == 1152 and y+LIMITS['rows'] <= contract['height']
            and x+w <= width and contract['url'] == source['url'], 'Fixed crop/layout mismatch')
    spans = [[base+((y+i)*width+x)*2,base+((y+i)*width+x+w)*2-1] for i in range(LIMITS['rows'])]
    contiguous = [base+y*width*2,base+(y+LIMITS['rows'])*width*2-1]
    require(spans[-1][1] < base+contract['strip_bytes'], 'Requested crop exceeds strip')
    return dict(fragment=source['fragment'],case_id=source['case_id'],url=source['url'],
                native_row_start=y,row_count=LIMITS['rows'],bbox_source_xywh=source['bbox_source_xywh'],
                crop_row_ranges=spans,comparator_range=contiguous,
                comparator_crop_byte_offset_per_row=x*2,crop_bytes_per_row=w*2,
                multipart_cap_bytes=sum(b-a+1 for a,b in spans)+LIMITS['framing_bytes_per_part']*LIMITS['rows']+LIMITS['framing_base_bytes'],
                comparator_cap_bytes=contiguous[1]-contiguous[0]+2)


def prerequisites():
    old=ROOT.parent/'retrieval-v2'
    hr,sr=record(old/'header-run/result.json'),record(old/'serving-run/result.json')
    require(hr['sha256'] == HEADER_RESULT_SHA and sr['sha256'] == SERVING_RESULT_SHA,
            'Exact completed V2 receipts required')
    h,s=json.loads(verify(hr).read_text()),json.loads(verify(sr).read_text())
    require(h['status'] == s['status'] == 'PASS' and h['stage'] == 'headers' and s['stage'] == 'serving'
            and h['design']['sha256'] == s['design']['sha256'] == V2_DESIGN_SHA,
            'V2 headers and serving must both pass')
    require(h['sources'] == s['sources'] and h['tiff_contracts'] == s['tiff_contracts']
            and len(h['sources']) == len(h['tiff_contracts']) == len(s['serving_comparisons']) == 3,
            'V2 complete source/layout correspondence required')
    expected=[('frag1','PHercParis2Fr47',[8181,6330],[2304,1152,1152,1152],'Paris2','evaluation','>u2',260),
              ('frag2','PHercParis2Fr143',[14830,9506],[1152,1152,1152,1152],'Paris2','evaluation','<u2',8),
              ('frag3','PHercParis1Fr34',[7606,5249],[2304,1152,1152,1152],'Paris1','calibration','<u2',8)]
    for source,contract,comparison,e in zip(h['sources'],h['tiff_contracts'],s['serving_comparisons'],expected,strict=True):
        name,identity,shape,bbox,parent,role,dtype,offset=e
        url=f'https://dl.ash2txt.org/fragments/{name.capitalize()}/{identity}.volpkg/working/54keV_exposed_surface/surface_volume/07.tif'
        require((source['fragment'],source['identity'],source['shape_yx'],source['bbox_source_xywh'],
                 source['parent_papyrus'],source['role'],source['url'],source['case_id'])
                == (name,identity,shape,bbox,parent,role,url,name+'-context1'), 'Fixed source/context/role changed')
        require(contract['fragment'] == name and contract['url'] == url and contract['dtype'] == dtype
                and contract['strip_offset'] == offset and [contract['height'],contract['width']] == shape
                and contract['strip_bytes'] == shape[0]*shape[1]*2 and contract['numeric_pixel_decodes'] == 0,
                'Per-object observed layout changed')
        require(comparison['fragment'] == name and comparison['url'] == url
                and comparison['etag'] == contract['etag'] and comparison['object_total_bytes'] == contract['object_total_bytes']
                and comparison['numeric_pixel_decodes'] == 0 and len(comparison['rows']) == len(comparison['parts']) == 8
                and all(r['bitwise_equal'] is True and r['payload_bytes'] == 2304
                        and r['row_native_y'] == 1152+i for i,r in enumerate(comparison['rows'])),
                'All eight V2 rows on all three objects must match')
    records=[hr,sr]
    for result in (h,s):
        records += [record(verify(r)) for r in nested_records(result)]
        previous_plan=json.loads(verify(result['plan']).read_text())
        records += [record(verify(r)) for r in previous_plan['input_pins']]
        ledger=json.loads(verify(result['request_ledger']).read_text())
        require(ledger['verified_tls'] is True and ledger['automatic_retries'] == ledger['redirects_followed'] == 0
                and all(r['success'] is True and r['http_status'] == 206 for r in ledger['requests']),
                'V2 completed transport must have no failures/retries/redirects')
        records += [record(verify(r)) for r in nested_records(ledger)]
    return dict(header_result=hr,serving_result=sr,sources=h['sources'],tiff_contracts=h['tiff_contracts']),records


def check_go(reviews,code):
    bound=False
    for r in reviews:
        receipt=json.loads(verify(r).read_text())
        if receipt.get('status') == 'GO':
            pins=list(nested_records(receipt))
            require(all(any(Path(p['path']).resolve() == Path(c['path']) and p['sha256'] == c['sha256']
                            for p in pins) for c in code), 'GO must bind all five exact source files')
            bound=True
    require(bound,'Source-bound technical GO required')


def load_plan(path,ready=True):
    plan=json.loads(Path(path).read_text())
    require(plan['schema_version'] == 1 and plan['stage'] == 'large-serving'
            and (not ready or plan['ready'] is True),'Large probe plan not ready')
    require(plan['python'] == sys.version.split()[0] == '3.12.14' and plan['limits'] == LIMITS,
            'Pinned runtime/limits changed')
    require(plan['design']['sha256'] == DESIGN_SHA,'Large-range design changed')
    for r in plan['input_pins']:
        verify(r)
    code=[record(ROOT/n) for n in CODE_NAMES]
    require(plan['code'] == code,'Executed source changed')
    proof,_=prerequisites()
    require(plan['prerequisites'] == proof,'Completed prerequisites changed')
    requests=[large_requests(s,c) for s,c in zip(proof['sources'],proof['tiff_contracts'],strict=True)]
    reserve=sum(r['multipart_cap_bytes']+r['comparator_cap_bytes'] for r in requests)
    require(plan['requests'] == requests and reserve == plan['maximum_reserved_body_bytes'] == MAXIMUM_RESERVATION
            and reserve <= LIMITS['http']['body_bytes'] and all(max(r['multipart_cap_bytes'],r['comparator_cap_bytes'])
            <= LIMITS['http']['per_response'] for r in requests),'Exact request/cap derivation changed')
    if ready:
        check_go(plan['reviews'],code)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--review',action='append',default=[])
    parser.add_argument('--ready',action='store_true')
    args=parser.parse_args()
    root=Path(args.output_dir).resolve()
    require(not root.exists(),'Fresh plan directory required')
    design=record(ROOT.parent/'LARGE_RANGE_DESIGN.md')
    require(design['sha256'] == DESIGN_SHA,'Frozen design changed')
    proof,records=prerequisites()
    code,reviews=[record(ROOT/n)for n in CODE_NAMES],[record(p)for p in args.review]
    if args.ready:
        check_go(reviews,code)
    records += [design]+code+reviews
    for review in reviews:
        records += [record(verify(r))for r in nested_records(json.loads(verify(review).read_text()))]
    unique={}
    for r in records:
        require(r['path'] not in unique or unique[r['path']] == r,'Conflicting input pin')
        unique[r['path']]=r
    requests=[large_requests(s,c)for s,c in zip(proof['sources'],proof['tiff_contracts'],strict=True)]
    plan=dict(schema_version=1,stage='large-serving',ready=args.ready,python='3.12.14',design=design,
              limits=LIMITS,prerequisites=proof,requests=requests,code=code,reviews=reviews,
              maximum_reserved_body_bytes=MAXIMUM_RESERVATION,input_pins=list(unique.values()),
              numeric_pixel_decodes=0,model_calls=0,production_acquisition=False,
              scope='Only fixed256rows on three plane07 objects; no production authorization')
    root.mkdir(parents=True)
    path=root/'probe-plan.json'
    save(path,plan)
    load_plan(path,ready=args.ready)
    print(json.dumps({'plan':record(path),'ready':args.ready}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'STOP','failure_type':type(error).__name__,
                          **({'stop_reason':str(error)}if isinstance(error,ContractError)else {})}))
        raise SystemExit(1)
