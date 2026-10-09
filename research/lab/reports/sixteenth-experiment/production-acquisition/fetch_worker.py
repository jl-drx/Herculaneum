"""One root-scheduled exact request; isolated raw ledger and terminal receipt, no retries."""
import argparse
import json
import re
import resource
import signal
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from production_common import *
from range_http import RangeClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Request cancelled by controller')))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Child wall cap')))
    signal.alarm(LIMITS['child_timeout_seconds'])
    config_path=Path(args.request).resolve()
    require(sha_file(config_path)==args.sha256,'Scheduled request changed')
    config=json.loads(config_path.read_text())
    require(config['python']==sys.version.split()[0]=='3.12.14'and config['limits']==LIMITS,'Runtime/limits changed')
    require(config['code']==[record(ROOT/n)for n in CODE_NAMES],'Worker source changed')
    request=config['request']
    require(re.fullmatch(r'[a-z0-9-]+',request['request_id'])is not None,'Invalid deterministic request ID')
    stage=config['stage'];require(stage in ('metadata','pixels'),'Invalid acquisition stage')
    ranges=request['ranges'];cap=request['reserved_cap_bytes']
    expected=sum(b-a+1 for a,b in ranges)+(1 if request['mode']=='single'else 768*len(ranges)+1024)
    require(cap==expected and cap<=LIMITS[stage]['per_response'],'Scheduled cap mismatch')
    client=RangeClient(config_path.parent/'response',[request['url']],
                       {'attempts':1,'body_bytes':cap,'per_response':LIMITS[stage]['per_response']},
                       record(config_path),pins={request['url']:request['pin']}if request.get('pin')else {})
    terminal=dict(schema_version=1,status='STARTED',request_id=request['request_id'],request=record(config_path),
                  numeric_pixel_decodes=0,model_calls=0)
    try:
        path,entry=client.get(request['url'],ranges,'body.bin',mode=request['mode'])
        terminal.update(status='PASS',body=record(path),etag=entry['etag'],object_total_bytes=entry['object_total_bytes'],
                        status_code=entry['http_status'],content_type=entry['http_identity'].get('content-type'),
                        charged_response_bytes=entry['charged_response_bytes'],retained_body_bytes=entry['retained_body_bytes'])
    except Exception as error:
        terminal.update(status='STOP',failure_type=type(error).__name__)
        if isinstance(error,ContractError):
            terminal['stop_reason']=str(error)
        raise
    finally:
        signal.alarm(0)
        terminal.update(transport_ledger=record(client.root/'request-ledger.json'),
                        charged_response_bytes=client.ledger['charged_response_bytes'],
                        retained_body_bytes=client.ledger['retained_body_bytes'])
        save(config_path.parent/'terminal.json',terminal)
    print(json.dumps({'status':'PASS','terminal':record(config_path.parent/'terminal.json')}))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'STOP','failure_type':type(error).__name__,
                          **({'stop_reason':str(error)}if isinstance(error,ContractError)else {})}))
        raise SystemExit(1)
