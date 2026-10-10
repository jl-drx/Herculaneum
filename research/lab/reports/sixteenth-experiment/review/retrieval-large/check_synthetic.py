"""Independent 256-part byte-only review; no real HTTP or numeric image decoding."""
from pathlib import Path
from email.message import Message
import hashlib
import http.client
import io
import json
import socket
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
IMPL = HERE.parent.parent / 'retrieval-large'
sys.path.insert(0, str(IMPL))
def forbidden(*a, **k):
    raise AssertionError('No real network permitted')
socket.socket.connect = socket.socket.connect_ex = socket.create_connection = forbidden
import large_common as c
import range_http as h

CASES = []
def check(name, f):
    f()
    CASES.append(name)
def eq(a, b):
    assert a == b, (a, b)
def rejects(f):
    try:
        f()
    except (ValueError, OSError):
        return
    raise AssertionError('Expected rejection')

RANGES = [[260 + i*257 + 7, 260 + i*257 + 102] for i in range(256)]
TOTAL = 100000
BOUND = b'review256'
PAYLOADS = [(bytes([i])*11 + b'\r\n--review256\r\nContent-Range: bytes 1-2/3\r\n\r\n' + bytes([255-i])*96)[:96] for i in range(256)]
def body(spans=RANGES, payloads=PAYLOADS, total=TOTAL, bound=BOUND, extra=b''):
    parts = []
    for (a,b), payload in zip(spans, payloads):
        parts.append(b'--'+bound+b'\r\nContent-Range: bytes %d-%d/%d\r\n'%(a,b,total)+b'Content-Length: '+str(len(payload)).encode()+b'\r\n'+extra+b'\r\n'+payload+b'\r\n')
    return b''.join(parts)+b'--'+bound+b'--\r\n'
def parse(raw, content_type='multipart/byteranges; boundary=review256'):
    return c.parse_multipart(raw, content_type, RANGES, TOTAL)
def payloads(raw):
    return {tuple(p['range_inclusive']): raw[p['payload_offset']:p['payload_offset']+p['payload_bytes']] for p in parse(raw)}

class Response:
    def __init__(self, raw, *, status=206, etag='"fixed"', length=True, mime=True, failure=None):
        self.status = status
        self.headers = Message()
        self.headers['ETag'] = etag
        if length:
            self.headers['Content-Length'] = str(len(raw))
        if mime:
            self.headers['Content-Type'] = 'multipart/byteranges; boundary=review256'
        self.stream = io.BytesIO(raw)
        self.failure = failure
        self.reads = 0
        self.closed = False
    def read(self, n):
        self.reads += 1
        if self.failure:
            raise self.failure
        return self.stream.read(n)
    def close(self):
        self.closed = True
class Opener:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.request = None
    def open(self, request, **kwargs):
        self.request = request
        self.calls += 1
        return self.response

def main():
    out = HERE/'synthetic-fixture-result.json'
    assert not out.exists(), 'Preserve earlier receipt'
    expected = dict(zip(map(tuple, RANGES), PAYLOADS))
    raw = body()
    check('256-binary-delimiter-collisions', lambda: eq(payloads(raw), expected))
    check('256-reversed-wire-order', lambda: eq(payloads(body(RANGES[::-1],PAYLOADS[::-1])),expected))
    check('256-one-leading-CRLF',lambda:eq(payloads(b'\r\n'+raw),expected))
    check('70-byte-quoted-boundary',lambda:eq(len(parse(body(bound=b'x'*69+b'='),'multipart/byteranges; boundary="'+('x'*69)+'="')),256))
    for name, f in [
        ('255-parts',lambda:parse(body(RANGES[:-1],PAYLOADS[:-1]))),
        ('257-parts',lambda:parse(body(RANGES+[RANGES[0]],PAYLOADS+[PAYLOADS[0]]))),
        ('duplicate-part',lambda:parse(body(RANGES[:-1]+[RANGES[0]]))),
        ('unrequested-part',lambda:parse(body(RANGES[:-1]+[[90000,90095]]))),
        ('changed-total',lambda:parse(body(total=TOTAL+1))),
        ('truncated-payload',lambda:parse(raw[:-60])),
        ('trailing-data',lambda:parse(raw+b'x')),
        ('multiple-leading-CRLF',lambda:parse(b'\r\n\r\n'+raw)),
        ('text-preamble',lambda:parse(b'x\r\n'+raw)),
        ('duplicate-range-header',lambda:parse(body(extra=b'Content-Range: bytes 0-1/3\r\n'))),
        ('transfer-encoding',lambda:parse(body(extra=b'Content-Transfer-Encoding: base64\r\n'))),
        ('oversized-header',lambda:parse(body(extra=b'Content-Type: '+b'x'*513+b'\r\n'))),
        ('framing-cap',lambda:parse(raw+b'x'*197633)),
        ('boundary-over70',lambda:parse(body(bound=b'x'*71),'multipart/byteranges; boundary='+('x'*71))),
        ('wrong-request-count',lambda:c.parse_multipart(raw,'multipart/byteranges; boundary=review256',RANGES[:8],TOTAL)),
        ('overlapping-request-spans',lambda:c.parse_multipart(raw,'multipart/byteranges; boundary=review256',RANGES[:-1]+[[268,363]],TOTAL)),
        ('coalesced-first-parts',lambda:parse(body([[RANGES[0][0],RANGES[1][1]]]+RANGES[2:],[PAYLOADS[0]+b'x'*161+PAYLOADS[1]]+PAYLOADS[2:]))),
    ]:
        check(name,lambda f=f:rejects(f))
    cap = 256*96 + 197632
    url = 'https://dl.ash2txt.org/synthetic/07.tif'
    with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic-') as tmp:
        cases = [('valid',Response(raw),True,len(raw)),('unknown-length',Response(raw,length=False),True,len(raw)),
                 ('HTTP200',Response(raw,status=200),False,0),('changed-ETag',Response(raw,etag='"stale"'),False,0),
                 ('weak-ETag',Response(raw,etag='W/"fixed"'),False,0),
                 ('changed-part-total',Response(body(total=TOTAL+1)),False,len(body(total=TOTAL+1))),
                 ('missing-part',Response(body(RANGES[:-1],PAYLOADS[:-1])),False,len(body(RANGES[:-1],PAYLOADS[:-1]))),
                 ('partial-read',Response(raw,failure=http.client.IncompleteRead(b'abc',100)),False,3),
                 ('opaque-read-failure',Response(raw,failure=OSError('synthetic')),False,0)]
        duplicate = Response(raw);duplicate.headers['ETag']='"fixed"'
        cases.append(('duplicate-ETag',duplicate,False,0))
        for name,response,ok,retained in cases:
            op=Opener(response)
            client=h.RangeClient(Path(tmp)/name,[url],c.LIMITS['http'],{},pins={url:{'etag':'"fixed"','object_total_bytes':TOTAL}},opener=op)
            f=lambda:client.get(url,RANGES,'body.bin',mode='multipart')
            if ok:f()
            else:rejects(f)
            assert op.calls==1 and response.closed
            assert client.ledger['charged_response_bytes']==(retained if ok else cap)
            assert client.ledger['retained_body_bytes']==retained
            assert op.request.get_header('If-match')=='"fixed"'
            assert len(op.request.get_header('Range').split(','))==256
            if name in ('HTTP200','changed-ETag','weak-ETag','duplicate-ETag'):
                assert response.reads==0
            CASES.append('transport-'+name)
        for name,limits in [('attempts',dict(c.LIMITS['http'],attempts=0)),('bytes',dict(c.LIMITS['http'],body_bytes=cap-1)),('per-response',dict(c.LIMITS['http'],per_response=cap-1))]:
            op=Opener(Response(raw));client=h.RangeClient(Path(tmp)/name,[url],limits,{},pins={url:{'etag':'"fixed"','object_total_bytes':TOTAL}},opener=op)
            rejects(lambda:client.get(url,RANGES,'body.bin',mode='multipart'))
            assert op.calls==0
            CASES.append('budget-precheck-'+name)
    check('literal-six-response-budget',lambda:eq(3*(256*2304+197632)+sum(w*256*2+1 for w in (6330,9506,5249)),13157891))
    import prepare_plan
    import run_large_probe
    source=dict(fragment='synthetic',case_id='synthetic-context1',url=url,bbox_source_xywh=[37,1152,1152,1152])
    contract=dict(width=1301,height=3000,strip_offset=260,strip_bytes=1301*3000*2,
                  url=url,etag='"fixed"',object_total_bytes=1301*3000*2+260,dtype='>u2',byte_order='big')
    req=prepare_plan.large_requests(source,contract)
    assert req['crop_row_ranges']==[[260+2*((1152+i)*1301+37),260+2*((1152+i)*1301+1189)-1]for i in range(256)]
    assert req['comparator_range']==[260+2*1152*1301,260+2*1408*1301-1]
    CASES.append('256-row-independent-crop-addressing')
    band=b''.join(bytes((i*13+j*7)%256 for j in range(2602))for i in range(256))
    crops=[band[i*2602+74:i*2602+74+2304]for i in range(256)]
    wire=body(req['crop_row_ranges'][::-1],crops[::-1],contract['object_total_bytes'])
    parts=c.parse_multipart(wire,'multipart/byteranges; boundary=review256',req['crop_row_ranges'],contract['object_total_bytes'])
    with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic-runner-') as tmp:
        class Client:
            def __init__(self,fail=False):self.calls=[];self.fail=fail
            def get(self,u,ranges,name,mode='single'):
                self.calls.append((u,ranges,mode))
                if self.fail:raise c.ContractError('Synthetic multipart rejection')
                p=Path(tmp)/name
                p.write_bytes(wire if mode=='multipart' else band)
                return p,dict(parts=parts)if mode=='multipart' else {}
        client=Client()
        actual=run_large_probe.inspect_serving(client,source,contract,req)
        assert len(client.calls)==2 and client.calls[0][2]=='multipart' and client.calls[1][2]=='single'
        assert len(actual['rows'])==256 and all(r['sha256']==hashlib.sha256(crops[i]).hexdigest() and r['comparator_body_offset']==i*2602+74 and r['bitwise_equal'] is True for i,r in enumerate(actual['rows']))
        CASES.append('256-row-runner-byte-equality-reordered-parts')
        failed=Client(True)
        rejects(lambda:run_large_probe.inspect_serving(failed,source,contract,req))
        assert len(failed.calls)==1
        CASES.append('rejected-multipart-never-starts-comparator')
    c.save(out,dict(status='PASS',count=len(CASES),cases=CASES,source=[c.record(IMPL/n)for n in c.CODE_NAMES],reviewer=c.record(__file__),real_network_requests=0,numeric_pixel_decodes=0,model_calls=0))
    print(json.dumps(c.record(out)))
if __name__=='__main__':main()
