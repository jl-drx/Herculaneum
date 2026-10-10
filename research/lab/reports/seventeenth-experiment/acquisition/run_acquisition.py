"""Execute one externally frozen HTTP acquisition stage; never advance automatically."""
import argparse
import json
import time
from pathlib import Path
from acquisition_common import *
from prepare_acquisition import load_plan, parse_ppm_prefix, format_source
from regional_http import AcquisitionClient


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();require(args.execute,'Explicit root execution flag required')
    plan=load_plan(args.plan);pr=record(args.plan);out=Path(args.output_dir).resolve()
    client=AcquisitionClient(out,plan['requests'],plan['limits'],pr);started=time.monotonic()
    result=dict(schema_version=1,status='RUNNING',stage=plan['stage'],plan=pr,design=plan['design'],
        code=plan['code'],fixed_inventory=plan['fixed_inventory'],objects=[],ppm_headers=[],numeric_image_decodes=0,
        numeric_ppm_decodes=0,model_calls=0,raw_CT_requests=0)
    save(out/'result.json',result)
    try:
        for request in plan['requests']:
            require(time.monotonic()-started<plan['limits']['wall_seconds'],'Stage wall cap reached')
            path,entry=client.get(request)
            result['objects'].append(dict(object_id=request['object_id'],request_id=request['request_id'],
                url=request['url'],file=record(path),etag=entry['etag'],object_total_bytes=entry['object_total_bytes'],
                **({'parts':entry['parts'],'case_id':request['case_id'],'variant':request['variant']}if'multipart'==request['mode']else{})))
            if request['mode']=='single':
                obj=next(o for o in plan['fixed_inventory']['objects']if o['object_id']==request['object_id'])
                contract=parse_ppm_prefix(path.read_bytes(),entry['object_total_bytes'],obj['shape_yx'])
                result['ppm_headers'].append(dict(contract,object_id=obj['object_id'],url=obj['url'],etag=entry['etag'],
                    object_total_bytes=entry['object_total_bytes'],prefix=record(path),format_source=format_source()))
            save(out/'result.json',result)
        result['status']='PASS'
    except Exception as error:
        result.update(status='STOP',failure_type=type(error).__name__)
        if isinstance(error,ContractError):result['stop_reason']=str(error)
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,http_attempts=len(client.ledger['requests']),
            retained_body_bytes=client.ledger['retained_body_bytes'],charged_response_bytes=client.ledger['charged_response_bytes'],
            ledger=record(out/'request-ledger.json'))
        save(out/'result.json',result)
    print(json.dumps(dict(result=record(out/'result.json'),status=result['status'],attempts=result['http_attempts'],retained_bytes=result['retained_body_bytes'])))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status='STOP',failure_type=type(error).__name__,**({'reason':str(error)}if isinstance(error,ContractError)else{}))))
        raise SystemExit(1)
