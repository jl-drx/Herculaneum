"""Bounded public docs/array metadata only; counts every redirect/error response body."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin,urlsplit
import requests

HERE = Path(__file__).resolve().parent
CAP = 6*1024*1024
REQUEST_CAP = 60


class Anonymous(requests.auth.AuthBase):
    def __call__(self,request):
        request.headers.pop('Authorization',None)
        return request


def permitted(url):
    p=urlsplit(url);h=p.hostname or ''
    return p.scheme=='https' and not p.username and not p.password and (
        h=='huggingface.co' or h.endswith('.huggingface.co') or h.endswith('.hf.co') or
        h.endswith('.amazonaws.com') or h=='raw.githubusercontent.com')


def fetch(url,label,maximum=128*1024):
    p=urlsplit(url)
    if not (p.path.endswith(('/.zattrs','/.zarray','.md','.txt')) or '/api/buckets/' in p.path):
        raise ValueError('Only public documentation/metadata/inventory allowed')
    ledger_path=HERE/'network-ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
        max_response_body_bytes=CAP,max_http_requests=REQUEST_CAP,response_body_bytes=0,requests=[],automatic_retries=0)
    for prior in ledger['requests']:
        if prior.get('initial_url')==url and prior['status']==200 and prior['complete']:
            body=(HERE/prior['local_file']).read_bytes()
            if len(body)!=prior['body_bytes'] or hashlib.sha256(body).hexdigest()!=prior['sha256']:
                raise ValueError('Cached metadata hash differs')
            return body,prior
    session=requests.Session();session.auth=Anonymous()
    session.mount('https://',requests.adapters.HTTPAdapter(max_retries=0))
    current=url
    try:
        for hop in range(6):
            if not permitted(current) or len(ledger['requests'])>=REQUEST_CAP:
                raise ValueError('Metadata URL/request limit')
            limit=min(maximum,CAP-ledger['response_body_bytes'])
            if limit<=0:raise ValueError('Metadata byte cap exhausted')
            record=dict(initial_url=url,request_host=urlsplit(current).hostname,request_path=urlsplit(current).path,
                retrieved_at_utc=datetime.now(timezone.utc).isoformat(),status=None,body_bytes=0,complete=False)
            ledger['requests'].append(record)
            ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
            try:
                with session.get(current,allow_redirects=False,stream=True,timeout=(15,30),
                                 headers={'Accept-Encoding':'identity'}) as response:
                    record['status']=response.status_code
                    length=response.headers.get('Content-Length')
                    if response.headers.get('Content-Encoding','identity')!='identity':
                        raise ValueError('Encoded metadata response rejected')
                    if length is not None and int(length)>limit:
                        raise ValueError('Declared metadata size exceeds bounded cap')
                    content=bytearray()
                    eof=False
                    while len(content)<limit:
                        block=response.raw.read(min(65536,limit-len(content)),decode_content=False)
                        if not block:eof=True;break
                        content.extend(block)
                    complete=eof or (length is not None and len(content)==int(length))
                    if length is not None and len(content)!=int(length):complete=False
                    body=bytes(content);digest=hashlib.sha256(body).hexdigest()
                    name=f'{digest}-{label}-hop{hop}.body'
                    (HERE/name).write_bytes(body)
                    ledger['response_body_bytes']+=len(body)
                    record.update(body_bytes=len(body),sha256=digest,local_file=name,complete=complete,
                        etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'))
                    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
                    print(json.dumps(dict(label=label,status=record['status'],body_bytes=len(body),complete=complete)))
                    if not complete:raise ValueError('Metadata reached cap or truncated; retained incomplete')
                    if response.status_code in (301,302,303,307,308):
                        if hop==5 or not response.headers.get('Location'):raise ValueError('Metadata redirect limit')
                        current=urljoin(current,response.headers['Location']);continue
                    return body,record
            except Exception as error:
                record['stop_type']=type(error).__name__
                ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
                raise ValueError('Metadata request stopped; transport URL/credential text suppressed') from None
        raise ValueError('Metadata redirect limit')
    finally:
        session.close()
