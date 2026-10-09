import email.message
import io
import json
import tempfile
from pathlib import Path
import bounded_metadata as m

checks = []
def check(name, fn, stop=False):
    try:
        fn()
        assert not stop, 'Expected STOP'
    except m.Stop:
        assert stop
    checks.append({'name': name, 'passed': True})

query = {'id': 'synthetic-ink-root', 'url': 'https://' + m.HOST + '/fake/zarr.json', 'max_body_bytes': 64}
class Response:
    status = 200
    def __init__(self, headers=None, url=None):
        self.headers = email.message.Message()
        for k, v in (headers or [('Content-Length', '2'), ('Content-Type', 'application/json')]):
            self.headers[k] = v
        self.url = url or query['url']
    def geturl(self): return self.url

check('exact-host', lambda: m.validate_url(query['url']))
for name, url in [('http', query['url'].replace('https:', 'http:')), ('host', 'https://example.org/x'),
                  ('credentials', 'https://user:pass@'+m.HOST+'/x'), ('fragment', query['url']+'#x'),
                  ('port', 'https://'+m.HOST+':444/x')]:
    check('reject-'+name, lambda url=url: m.validate_url(url), True)
check('valid-headers', lambda: m.check_headers(Response(), query))
for name, headers in [
    ('oversize', [('Content-Length','65'),('Content-Type','application/json')]),
    ('duplicate-length', [('Content-Length','2'),('Content-Length','2'),('Content-Type','application/json')]),
    ('encoding', [('Content-Length','2'),('Content-Type','application/json'),('Content-Encoding','gzip')]),
    ('chunked', [('Content-Length','2'),('Content-Type','application/json'),('Transfer-Encoding','chunked')]),
    ('binary', [('Content-Length','2'),('Content-Type','application/octet-stream')]),
]:
    check('reject-'+name, lambda headers=headers: m.check_headers(Response(headers), query), True)
check('bounded-no-length', lambda: m.require(m.check_headers(Response([('Content-Type','application/json')]),query) is None,'Missing unframed mode'))
check('bounded-chunked', lambda: m.require(m.check_headers(Response([('Content-Type','application/json'),('Transfer-Encoding','chunked')]),query) is None,'Missing chunked mode'))
check('reject-redirect', lambda: m.check_headers(Response(url='https://example.org/x'), query), True)
check('json-group', lambda: m.parse_body(query, b'{"zarr_format":3,"node_type":"group"}'))
check('reject-duplicate-json', lambda: m.parse_body(query, b'{"zarr_format":3,"zarr_format":3,"node_type":"group"}'), True)
check('reject-array', lambda: m.parse_body(query, b'{"zarr_format":3,"node_type":"array"}'), True)
inventory = {'id': 'synthetic-package-inventory', 'url': 'https://'+m.HOST+'/?prefix=abc%2F'}
xml = '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>vesuvius-challenge-open-data</Name><Prefix>abc/</Prefix><KeyCount>0</KeyCount><IsTruncated>false</IsTruncated></ListBucketResult>'
check('complete-inventory', lambda: m.parse_body(inventory, xml.encode()))
check('reject-pagination', lambda: m.parse_body(inventory, xml.replace('false','true').encode()), True)
check('reject-wrong-prefix', lambda: m.parse_body(inventory, xml.replace('abc/','def/').encode()), True)
check('reject-wrong-count', lambda: m.parse_body(inventory, xml.replace('<KeyCount>0','<KeyCount>1').encode()), True)
check('reject-entity', lambda: m.parse_body(inventory, ('<!ENTITY x "y">'+xml).encode()), True)
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
    p = Path(temp)/'partial'
    with p.open('wb') as f:
        check('partial-short-body', lambda: m.copy_body(io.BytesIO(b'abc'), 4, f), True)
    assert p.read_bytes() == b'abc'
    checks.append({'name': 'partial-body-retained', 'passed': True})
    with p.open('wb') as f:
        check('unframed-eof', lambda: m.require(m.copy_body(io.BytesIO(b'abc'),None,f,4)==3,'Wrong byte count'))
    with p.open('wb') as f:
        check('unframed-cap-stop', lambda: m.copy_body(io.BytesIO(b'abcdef'),None,f,3),True)
    assert p.read_bytes()==b'abc'
check('no-auto-redirect', lambda: m.require(m.NoRedirect().redirect_request(None,None,302,None,None,'https://example.org') is None,'redirect followed'))
result = {'status':'PASS','checks':checks,'count':len(checks),'network_calls':0}
m.atomic_json(Path(__file__).parent/'fixtures.json',result)
print(json.dumps({'status':'PASS','checks':len(checks)}))
