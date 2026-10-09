"""Research metadata adapter using the retained immutable explicit-redirect handler."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

HERE=Path(__file__).resolve().parent
OLD=Path('/workspace/herculaneum-lab/reports/sixth-experiment/data-contract/bounded_http_v2.py')
SPEC=importlib.util.spec_from_file_location('retained_no_auto_redirect',OLD)
RETAINED=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(RETAINED)
OLD_SHA=hashlib.sha256(OLD.read_bytes()).hexdigest()
CAP=6*1024*1024
REQUEST_CAP=60


def fetch(url,label,maximum=128*1024):
    initial=urllib.parse.urlsplit(url)
    if not initial.path.endswith(('/.zattrs','/.zarray','.md','.txt')):
        raise ValueError('Only public docs/array metadata permitted')
    ledger_path=HERE/'network-ledger.json'
    ledger=json.loads(ledger_path.read_text())
    for old in ledger['requests']:
        if old.get('initial_url')==url and old['status']==200 and old['complete']:
            body=(HERE/old['local_file']).read_bytes()
            if len(body)!=old['body_bytes'] or hashlib.sha256(body).hexdigest()!=old['sha256']:
                raise ValueError('Cached metadata hash changed')
            return body,old
    opener=urllib.request.build_opener(RETAINED.NoAutomaticRedirect())
    current=url
    for hop in range(6):
        RETAINED.permitted_url(current)
        if len(ledger['requests'])>=REQUEST_CAP:raise ValueError('Metadata request cap reached')
        remaining=CAP-ledger['response_body_bytes']-ledger.get('initial_attempt_conservative_budget_reservation_bytes',0)
        limit=min(maximum,remaining)
        if limit<=0:raise ValueError('Metadata body cap reached')
        content=bytearray()
        record=dict(initial_url=url,request_url_without_query=RETAINED.clean_url(current),hop_index=hop,
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),status=None,body_bytes=0,complete=False,
            transport='immutable urllib NoAutomaticRedirect handler',handler_source_sha256=OLD_SHA)
        ledger['requests'].append(record)
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        try:
            request=urllib.request.Request(current,headers={'Accept-Encoding':'identity'})
            try:response=opener.open(request,timeout=30)
            except urllib.error.HTTPError as error:response=error
            with response:
                record['status']=response.status
                length=response.headers.get('Content-Length')
                if response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('Encoded response')
                if length is not None and int(length)>limit:raise ValueError('Declared metadata length exceeds cap')
                eof=False
                while len(content)<limit:
                    block=response.read(min(65536,limit-len(content)))
                    if not block:eof=True;break
                    content.extend(block)
                    ledger['response_body_bytes']+=len(block)
                    record['body_bytes']=len(content)
                complete=eof or (length is not None and len(content)==int(length))
                if length is not None and len(content)!=int(length):complete=False
                digest=hashlib.sha256(content).hexdigest();name=f'{digest}-{label}-hop{hop}.body'
                (HERE/name).write_bytes(content)
                record.update(body_bytes=len(content),sha256=digest,local_file=name,complete=complete,
                    content_length=length,etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'))
                ledger['budgeted_bytes']=ledger['response_body_bytes']+ledger.get('initial_attempt_conservative_budget_reservation_bytes',0)
                ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
                print(json.dumps(dict(label=label,status=response.status,body_bytes=len(content),complete=complete)))
                if not complete:raise ValueError('Metadata stopped at cap/truncation')
                if response.status in (301,302,303,307,308):
                    if hop==5 or not response.headers.get('Location'):raise ValueError('Metadata redirect limit')
                    current=urllib.parse.urljoin(current,response.headers['Location']);continue
                return bytes(content),record
        except Exception as error:
            if content:
                digest=hashlib.sha256(content).hexdigest();name=f'{digest}-{label}-hop{hop}-incomplete.body'
                (HERE/name).write_bytes(content);record.update(body_bytes=len(content),sha256=digest,local_file=name)
            record['stop_type']=type(error).__name__
            ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
            raise ValueError('Metadata request stopped; transport URL/credentials suppressed') from None
    raise ValueError('Metadata redirect limit')
