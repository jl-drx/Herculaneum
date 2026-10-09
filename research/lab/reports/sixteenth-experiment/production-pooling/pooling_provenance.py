"""Small receipt-chain validators; no CT decoding, transport or model operations."""
import math
from pathlib import Path
import time

from pooling_common import identity,require,row_range

PRODUCER_ROOT=Path(__file__).resolve().parent.parent/'production-acquisition'
JSON_PLAN_CAP=64*1024**2


def shared_elapsed(epoch,now=None):
    now=time.time() if now is None else now
    require(type(epoch) in (int,float) and math.isfinite(epoch) and epoch>0
        and 0<=now-epoch<=1500,'Shared M/P/C1500-second window exhausted or invalid')
    return now-epoch


def required_root_coverage(roots,paths):
    roots=[Path(p).resolve() for p in roots]
    require(PRODUCER_ROOT.resolve() in roots,'Entire producer package root must be counted')
    require(all(any(Path(p).resolve().is_relative_to(root) for root in roots) for p in paths),
        'Required production plans/runs excluded from storage accounting')


def nested_records(value):
    if isinstance(value,dict):
        if {'path','sha256'}<=value.keys() and any(k in value for k in ('size_bytes','bytes','byte_length')):yield identity(value)
        for v in value.values():yield from nested_records(v)
    elif isinstance(value,list):
        for v in value:yield from nested_records(v)


def validate_ready_pixel_plan(raw,pixel,source,producer_freeze,read):
    require(pixel['schema_version']==1 and pixel['stage']=='pixels' and pixel['ready'] is True,
        'Reviewed pixels-ready plan required')
    refs=pixel['fixed_inputs']['references']
    for key,role in [('design','production_design'),('schema','schema'),('benchmark','benchmark')]:
        require(identity(refs[key])==identity(source[role]),'Pixel scientific lineage differs: '+key)
    benchmark=read(source['benchmark']);require(identity(refs['source'])==identity(benchmark['source_contract']),
        'Pixel source catalog lineage differs')
    require(raw['fixed_inputs']==pixel['fixed_inputs'] and raw['tiff_contracts']==pixel['tiff_contracts']
        and identity(raw['raw_schema'])==identity(source['schema']),'Raw/pixel plan lineage differs')
    require(producer_freeze['no_further_edits'] is True,'Producer source must remain sealed')
    expected=[identity(r) for r in producer_freeze['code'] if Path(r['path']).suffix=='.py']
    require([identity(r) for r in pixel['code']]==expected,'Executed producer code differs from seal')
    bound=False
    for item in pixel['reviews']:
        review=read(item)
        if review.get('status')=='GO':
            pins=list(nested_records(review))
            if all(r in pins for r in expected):bound=True
    require(bound,'Pixel plan lacks source-bound technical GO')
    metadata=read(pixel['metadata_result']);metadata_plan=read(pixel['metadata_plan'],JSON_PLAN_CAP)
    require(metadata['status']=='PASS' and metadata['stage']=='metadata' and metadata['http_attempts']==450
        and metadata['tiff_contracts']==raw['tiff_contracts'] and metadata['fixed_inputs']==pixel['fixed_inputs']
        and identity(metadata['plan'])==identity(pixel['metadata_plan']),'Complete metadata stage linkage differs')
    require(metadata_plan['stage']=='metadata' and metadata_plan['ready'] is True
        and metadata_plan['fixed_inputs']==pixel['fixed_inputs'] and metadata_plan['code']==pixel['code'],
        'Metadata ready plan lineage differs')
    require(raw['production_started_epoch_seconds']==pixel['production_started_epoch_seconds']==metadata['production_started_epoch_seconds'],
        'Shared production epoch differs')
    shared_elapsed(raw['production_started_epoch_seconds'])
    for p in (pixel,metadata_plan):
        for r in p['input_pins']:read.remember(r)
    metadata_ledger=read(metadata['request_ledger']);read(metadata['terminal_receipts'])
    require(metadata_ledger['stage']=='metadata' and identity(metadata_ledger['plan'])==identity(pixel['metadata_plan'])
        and len(metadata_ledger['requests'])==450 and all(r['status']=='PASS' for r in metadata_ledger['requests'])
        and metadata_ledger['automatic_retries']==metadata_ledger['redirects_followed']==0
        and metadata_ledger['verified_tls'] is True,'Metadata request ledger incomplete')
    return metadata


def validate_global_receipts(raw,pixel,ledger,terminal_index):
    require(ledger['stage']=='pixels' and identity(ledger['plan'])==identity(raw['plan'])
        and ledger['automatic_retries']==ledger['redirects_followed']==0 and ledger['verified_tls'] is True,
        'Pixel ledger lineage/transport policy differs')
    requests={r['request_id']:r for r in pixel['requests']}
    entries={r['request_id']:r for r in ledger['requests']}
    terminals={r['request_id']:r['terminal'] for r in terminal_index['terminals']}
    require(terminal_index['schema_version']==1 and len(requests)==len(pixel['requests'])==1530
        and len(entries)==len(ledger['requests'])==1530 and len(terminals)==len(terminal_index['terminals'])==1530
        and set(requests)==set(entries)==set(terminals) and raw['http_attempts']==1530,
        'Missing/duplicate/unplanned production request or terminal')
    require(all(r['status']=='PASS' and r['returncode']==0 for r in entries.values()),'All1530requests must succeed')
    for key in requests:require(identity(entries[key]['terminal'])==identity(terminals[key]),'Global terminal index mismatch')
    require(sum(r['charged_response_bytes'] for r in entries.values())==ledger['charged_response_bytes']==raw['charged_response_bytes']
        and sum(r['retained_body_bytes'] for r in entries.values())==ledger['retained_body_bytes']==raw['retained_body_bytes'],
        'Global successful charge/retained totals differ')
    return requests,entries,terminals


def validate_terminal_chain(batch,request,entry,terminal,config,transport,pixel_record,contract):
    identifier=batch['request_id'];body=identity(batch['multipart']);size=body['size_bytes']
    require(terminal['status']=='PASS' and terminal['request_id']==identifier==request['request_id']
        and identity(terminal['request'])==identity(entry['request']) and identity(terminal['body'])==body
        and identity(terminal['transport_ledger'])==identity(entry['transport_ledger'])
        and terminal['status_code']==206 and terminal['content_type']==batch['content_type']
        and terminal['etag']==contract['etag'] and terminal['object_total_bytes']==contract['object_total_bytes']
        and terminal['charged_response_bytes']==terminal['retained_body_bytes']==size
        and entry['charged_response_bytes']==entry['retained_body_bytes']==size,'Terminal/source/body/charge linkage differs')
    require(config['stage']=='pixels' and config['request']==request and identity(config['plan'])==identity(pixel_record),
        'Request config belongs to another plan or route')
    require(entry['url']==request['url'] and entry['mode']==request['mode'],'Global entry source/mode differs')
    expected_ranges=[row_range(contract,batch['native_x_start'],y,batch['native_crop_width'])
        for y in range(batch['native_row_start'],batch['native_row_start']+batch['row_count'])]
    require(request['url']==batch['source_url'] and request['mode']=='multipart' and request['ranges']==expected_ranges
        and request['pin']==dict(etag=contract['etag'],object_total_bytes=contract['object_total_bytes'])
        and request['reserved_cap_bytes']==entry['reserved_cap_bytes']==batch['row_count']*3072+1024,
        'Fixed request ranges/pin/reservation differ')
    for k in ('case_id','fragment','plane_index','native_row_start','row_count','native_x_start','native_crop_width'):
        require(request[k]==batch[k],'Request/batch routing differs: '+k)
    require(identity(transport['plan'])==identity(terminal['request']) and len(transport['requests'])==1
        and transport['automatic_retries']==transport['redirects_followed']==0 and transport['verified_tls'] is True,
        'Child ledger policy/config lineage differs')
    http=transport['requests'][0];headers=http['request_headers'];response=http['http_identity']
    require(http['success'] is True and http['url']==request['url'] and http['mode']=='multipart'
        and http['ranges_inclusive']==expected_ranges and http['http_status']==206 and identity(http['file'])==body
        and http['parts']==batch['parts'] and http['etag']==contract['etag']
        and http['object_total_bytes']==contract['object_total_bytes'] and http['body_sha256']==body['sha256']
        and http['charged_response_bytes']==http['retained_body_bytes']==size
        and http['reserved_cap_bytes']==request['reserved_cap_bytes']
        and transport['charged_response_bytes']==transport['retained_body_bytes']==size,
        'Child successful transport/body evidence differs')
    require(headers['Range']=='bytes='+','.join(f'{a}-{b}' for a,b in expected_ranges)
        and headers['If-Match']==contract['etag'] and headers['Accept-Encoding']=='identity'
        and response['etag']==contract['etag'] and response['content-type']==batch['content_type']
        and response.get('content-encoding','identity').lower() in ('','identity')
        and 'content-range' not in response,'Child HTTP validators/framing differ')
    if 'content-length' in response:require(response['content-length']==str(size),'Child response length differs')
