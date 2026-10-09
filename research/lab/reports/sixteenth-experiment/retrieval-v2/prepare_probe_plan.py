"""Prepare separate source-bound header and serving plans; no network/pixel decoding."""
from pathlib import Path
import argparse
import json
import re
from probe_common import *


def sources_from_contract(source, benchmark):
    expected = [('Frag1',[8181,6330],[2304,1152,1152,1152],'Paris2','evaluation'),
                ('Frag2',[14830,9506],[1152,1152,1152,1152],'Paris2','evaluation'),
                ('Frag3',[7606,5249],[2304,1152,1152,1152],'Paris1','calibration')]
    output = []
    for c, (alias,shape,bbox,parent,role) in zip(source['eligible_candidates'], expected, strict=True):
        require((c['legacy_alias'],c['shape_yx'],c['parent_papyrus'],c['role']) == (alias,shape,parent,role), 'Frozen source identity changed')
        case = next(r for r in benchmark['selected_contexts'] if r['case_id'] == alias.lower()+'-context1')
        require(case['bbox_source_xywh'] == bbox and case['fragment'] == c['id']
                and case['role'] == role and case['parent_papyrus'] == parent, 'First selected context changed')
        index = ROOT.parent.parent/'fifteenth-experiment'/'source-audit'/(alias.lower()+'-surface-index.html')
        require('href="07.tif"' in index.read_text(), 'Plane07 not source-discovered')
        output.append(dict(fragment=alias.lower(), identity=c['id'], shape_yx=shape, role=role,
                           parent_papyrus=parent, case_id=case['case_id'], bbox_source_xywh=bbox,
                           url=c['surface_volume_url']+'07.tif', source_index=record(index)))
    return output


def serving_requests(source, contract):
    require(contract['url'] == source['url'] and contract['width'] == source['shape_yx'][1]
            and contract['height'] == source['shape_yx'][0], 'Audited TIFF identity changed')
    x,y,w,h = source['bbox_source_xywh']
    width, base = contract['width'], contract['strip_offset']
    spans = [[base+((y+i)*width+x)*2, base+((y+i)*width+x+w)*2-1] for i in range(8)]
    contiguous = [base+y*width*2, base+(y+8)*width*2-1]
    return dict(fragment=source['fragment'], url=source['url'], crop_row_ranges=spans,
                comparator_range=contiguous, bbox_source_xywh=source['bbox_source_xywh'],
                comparator_crop_byte_offset_per_row=x*2, crop_bytes_per_row=w*2,
                multipart_cap_bytes=sum(b-a+1 for a,b in spans)+LIMITS['multipart_framing_bytes'],
                comparator_cap_bytes=contiguous[1]-contiguous[0]+2)


def validate_prefix_receipt(result,ledger,prefix,expected_url):
    require(result['status'] == 'STOP' and result['stage'] == 'headers' and result['http_attempts'] == 1
            and result['retained_body_bytes'] == 8 and result['numeric_pixel_decodes'] == 0,'Unexpected original attempt scope')
    require(len(ledger['requests']) == 1 and ledger['verified_tls'] is True
            and ledger['automatic_retries'] == ledger['redirects_followed'] == 0,
            'Only exact original prefix transport may be reused')
    request = ledger['requests'][0]
    require(request['success'] and request['http_status'] == 206 and request['ranges_inclusive'] == [[0,7]]
            and request['url'] == expected_url and request['retained_body_bytes'] == 8
            and request['file']['sha256'] == request['body_sha256'] == PRIOR_PREFIX_SHA
            and hashlib.sha256(prefix).hexdigest() == PRIOR_PREFIX_SHA,'Prior HTTP prefix receipt mismatch')
    require(content_range(request['http_identity']['content-range']) == (0,7,request['object_total_bytes'])
            and request['http_identity']['etag'] == request['etag']
            and request['http_identity']['content-length'] == '8'
            and request['http_identity'].get('content-encoding','identity').lower() in ('','identity')
            and re.fullmatch(r'"[\x21\x23-\x7e]+"',request['etag']) is not None,'Prior object identity mismatch')
    require(prefix == bytes.fromhex('4d4d002a00000008'),'Prior prefix changed')
    return request


def prefix_reuse():
    prior = ROOT.parent/'retrieval'/'header-run'
    result_r = record(prior/'result.json')
    require(result_r['sha256'] == PRIOR_RESULT_SHA,'Original stopped result changed')
    result = json.loads(verify(result_r).read_text())
    ledger_r = result['request_ledger']
    require(ledger_r['sha256'] == PRIOR_LEDGER_SHA,'Original request ledger changed')
    ledger = json.loads(verify(ledger_r).read_text())
    request = ledger['requests'][0]
    prefix = verify(request['file']).read_bytes()
    expected_url='https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/surface_volume/07.tif'
    request=validate_prefix_receipt(result,ledger,prefix,expected_url)
    return dict(result=result_r,ledger=record(verify(ledger_r)),header=record(verify(request['file'])),
                url=request['url'],etag=request['etag'],object_total_bytes=request['object_total_bytes'],
                prior_http_request=request,reused_bytes=8,new_GETs_for_prefix=0)


def load_plan(path, ready=True):
    plan = json.loads(Path(path).read_text())
    require(plan['schema_version'] == 1 and plan['stage'] in ('headers','serving')
            and (not ready or plan['ready'] is True), 'Probe plan is not ready')
    require(plan['limits'] == LIMITS and plan['python'] == sys.version.split()[0] == '3.12.14', 'Runtime/limits changed')
    require(plan['design']['sha256'] == DESIGN_SHA and plan['source_contract']['sha256'] == SOURCE_SHA
            and plan['benchmark_contract']['sha256'] == BENCHMARK_SHA, 'Scientific input contract changed')
    for r in plan['input_pins']:
        verify(r)
    require(plan['code'] == [record(ROOT/n) for n in CODE_NAMES], 'Executed source changed')
    source = json.loads(verify(plan['source_contract']).read_text())
    benchmark = json.loads(verify(plan['benchmark_contract']).read_text())
    require(plan['sources'] == sources_from_contract(source,benchmark), 'Source schedule changed')
    require(plan['prefix_reuse'] == prefix_reuse() and plan['prefix_reuse']['url'] == plan['sources'][0]['url'],
            'Pinned original prefix reuse changed')
    if plan['stage'] == 'serving':
        header = json.loads(verify(plan['header_result']).read_text())
        require(header['status'] == 'PASS' and header['stage'] == 'headers', 'All header audits must pass')
        require(header['sources'] == plan['sources'] and header['design'] == plan['design'], 'Header/source contract changed')
        require(plan['tiff_contracts'] == header['tiff_contracts'] and len(plan['tiff_contracts']) == 3, 'Observed layout/identity changed')
        expected = [serving_requests(s,c) for s,c in zip(plan['sources'],plan['tiff_contracts'],strict=True)]
        require(plan['requests'] == expected, 'Serving ranges/caps changed')
        require(sum(r['multipart_cap_bytes']+r['comparator_cap_bytes'] for r in expected)
                <= LIMITS['serving']['body_bytes'], 'Derived serving caps exceed stage budget')
    if ready:
        require(plan['reviews'], 'Source review missing')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['headers','serving'])
    parser.add_argument('--header-result')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--review', action='append', default=[])
    parser.add_argument('--ready', action='store_true')
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    require(not root.exists(), 'Fresh plan directory required')
    prior = ROOT.parent.parent/'fifteenth-experiment'
    design, source_r, benchmark_r = record(ROOT.parent/'PROBE_DESIGN_V2.md'),record(prior/'SOURCE_CONTRACT.json'),record(prior/'BENCHMARK_CONTRACT.json')
    require(design['sha256'] == DESIGN_SHA and source_r['sha256'] == SOURCE_SHA
            and benchmark_r['sha256'] == BENCHMARK_SHA, 'Root source/design pins changed')
    source, benchmark = json.loads(verify(source_r).read_text()), json.loads(verify(benchmark_r).read_text())
    code, reviews = [record(ROOT/n) for n in CODE_NAMES],[record(p) for p in args.review]
    if args.ready:
        bound_go = False
        for r in reviews:
            receipt = json.loads(verify(r).read_text())
            if receipt.get('status') == 'GO':
                pins = list(nested_records(receipt))
                require(all(any(Path(p['path']).resolve() == Path(c['path']) and p['sha256'] == c['sha256']
                                for p in pins) for c in code), 'GO must bind all five exact source files')
                bound_go = True
        require(bound_go, 'Source-bound technical GO required')
    records = [design,source_r,benchmark_r]+code+reviews
    # Source text/JSON only: do not read scientific labels, arrays or checkpoints during preparation.
    records += [record(verify(r)) for r in nested_records(source)]
    for r in reviews:
        records += [record(verify(p)) for p in nested_records(json.loads(verify(r).read_text()))
                    if Path(p['path']).suffix in ('.py','.json','.md','.txt','.html')]
    plan = dict(schema_version=1,stage=args.stage,ready=args.ready,design=design,
                source_contract=source_r,benchmark_contract=benchmark_r,python='3.12.14',limits=LIMITS,
                code=code,reviews=reviews,sources=sources_from_contract(source,benchmark),
                prefix_reuse=prefix_reuse(),
                numeric_pixel_decodes=0,model_calls=0,production_acquisition=False)
    records += [record(verify(r)) for r in nested_records(plan['sources'])]
    records += [record(verify(r)) for r in nested_records(plan['prefix_reuse'])]
    if args.stage == 'serving':
        require(args.header_result, 'Serving preparation requires completed header result')
        header_r = record(args.header_result)
        header = json.loads(verify(header_r).read_text())
        require(header['status'] == 'PASS' and header['stage'] == 'headers', 'Header audit did not pass')
        plan.update(header_result=header_r,tiff_contracts=header['tiff_contracts'],
                    requests=[serving_requests(s,c)for s,c in zip(plan['sources'],header['tiff_contracts'],strict=True)])
        records += [header_r]+[record(verify(r)) for r in nested_records(header)]
    plan['input_pins'] = list({r['path']:r for r in records}.values())
    root.mkdir(parents=True)
    path = root/'probe-plan.json'
    save(path,plan)
    load_plan(path,ready=args.ready)
    print(json.dumps({'plan':record(path),'ready':args.ready,'stage':args.stage}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'STOP','failure_type':type(error).__name__,
                          **({'stop_reason':str(error)} if isinstance(error,ContractError) else {})}))
        raise SystemExit(1)
