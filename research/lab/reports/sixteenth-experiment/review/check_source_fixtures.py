"""Synthetic-only independent TIFF/MIME/HTTP probes; never reads real TIFF payloads."""
from pathlib import Path
from email.message import Message
import hashlib
import http.client
import io
import json
import socket
import struct
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parent
IMPL=ROOT.parent/'retrieval'
sys.path.insert(0,str(IMPL))
def forbidden(*a,**k):
    raise AssertionError('Network forbidden in synthetic review')
socket.socket.connect=socket.socket.connect_ex=socket.create_connection=forbidden
import probe_common as common
import range_http
CASES=[]
def check(name,fn):
    fn(); CASES.append(name)
def equal(a,b):
    assert a==b,(a,b)
def rejects(fn):
    try:fn()
    except (ValueError,OSError):return
    raise AssertionError('Expected rejection')

TAGS=[(256,4,1,16),(257,4,1,16),(258,3,1,16),(262,3,1,1),(273,4,1,8),(279,4,1,512)]
def tiff(tags=TAGS,offset=520,next_ifd=0):
    return b'II*\0'+struct.pack('<I',offset),struct.pack('<H',len(tags))+b''.join(struct.pack('<HHII',*x) for x in tags)+struct.pack('<I',next_ifd)
def parse(tags=TAGS,offset=520,next_ifd=0,total=1024,shape=(16,16)):
    return common.parse_tiff(*tiff(tags,offset,next_ifd),total,shape)

RANGES=[[i*128,i*128+63] for i in range(8)]
PAYLOADS=[(bytes([i])*5+b'\r\n--review-boundary\r\n'+bytes([i])*64)[:64] for i in range(8)]
def multipart(spans=RANGES,total=2048,boundary=b'review-boundary',payloads=PAYLOADS,extra_header=b''):
    body=b''
    for span,payload in zip(spans,payloads):
        body+=b'--'+boundary+b'\r\nContent-Range: bytes %d-%d/%d\r\n'%(span[0],span[1],total)
        body+=b'Content-Length: '+str(len(payload)).encode()+b'\r\n'+extra_header+b'\r\n'+payload+b'\r\n'
    return body+b'--'+boundary+b'--\r\n'
def mime(body,content_type='multipart/byteranges; boundary=review-boundary'):
    return common.parse_multipart(body,content_type,RANGES,2048)

class Response:
    def __init__(self,body=b'12345678',status=206,length=True,etag='"stable"',total=100,failure=None):
        self.status=status;self.headers=Message();self.headers['ETag']=etag
        self.headers['Content-Range']=f'bytes 0-7/{total}'
        if length:self.headers['Content-Length']=str(len(body))
        self.stream=io.BytesIO(body);self.failure=failure;self.reads=0;self.closed=False
    def read(self,size):
        self.reads+=1
        if self.failure:raise self.failure
        return self.stream.read(size)
    def close(self):self.closed=True
class Opener:
    def __init__(self,response):self.response=response;self.calls=0;self.request=None
    def open(self,request,**kwargs):self.calls+=1;self.request=request;return self.response

def main():
    receipt=ROOT/'synthetic-fixture-result.json'
    assert not receipt.exists(),'Preserve earlier review receipt'
    check('TIFF-specification-defaults',lambda:equal(parse()['specification_defaults_used'],{'259':1,'266':1,'274':1,'277':1,'284':1,'317':1,'339':1,'278':4294967295}))
    check('TIFF-inline-SHORT-dimension',lambda:equal(parse([(256,3,1,16)]+TAGS[1:])['width'],16))
    check('TIFF-bounded-auxiliary-value',lambda:equal(parse(TAGS+[(282,5,1,800)])['auxiliary_metadata_extents'],[{'tag':282,'extent_half_open':[800,808],'dereferenced':False}]))
    for name,fn in [
        ('duplicate-tag',lambda:parse(TAGS+[TAGS[0]])),
        ('unsupported-type',lambda:parse(TAGS+[(282,13,1,800)])),
        ('zero-field-count',lambda:parse(TAGS+[(282,5,0,800)])),
        ('auxiliary-pixel-overlap',lambda:parse(TAGS+[(282,5,1,32)])),
        ('auxiliary-table-overlap',lambda:parse(TAGS+[(282,5,1,540)])),
        ('auxiliary-header-overlap',lambda:parse(TAGS+[(282,5,1,0)])),
        ('auxiliary-out-of-bounds',lambda:parse(TAGS+[(282,5,1,1020)])),
        ('IFD-pixel-overlap',lambda:parse(offset=500)),
        ('next-page',lambda:parse(next_ifd=900)),
        ('wrong-canvas',lambda:parse(shape=(16,15))),
        ('compression',lambda:parse(TAGS+[(259,3,1,5)])),
        ('orientation',lambda:parse(TAGS+[(274,3,1,2)])),
        ('multi-strip',lambda:parse(TAGS+[(278,4,1,8)])),
        ('tile-tag',lambda:parse(TAGS+[(322,4,1,16)])),
        ('BigTIFF',lambda:common.parse_tiff(b'II+\0\10\0\0\0',tiff()[1],1024,(16,16))),
    ]:check('TIFF-'+name,lambda fn=fn:rejects(fn))
    body=multipart(); parts=mime(body)
    check('multipart-delimiter-collision-is-payload',lambda:equal([body[p['payload_offset']:p['payload_offset']+p['payload_bytes']] for p in parts],PAYLOADS))
    prefixed=b'\r\n'+body
    check('multipart-one-initial-CRLF',lambda:equal([prefixed[p['payload_offset']:p['payload_offset']+p['payload_bytes']] for p in mime(prefixed)],PAYLOADS))
    check('multipart-reordered-parts',lambda:equal(len(mime(multipart(RANGES[::-1],payloads=PAYLOADS[::-1]))),8))
    check('quoted-tspecial-boundary',lambda:equal(len(mime(multipart(boundary=b'a=b'),'multipart/byteranges; boundary="a=b"')),8))
    for name,fn in [
        ('unquoted-tspecial-boundary',lambda:mime(multipart(boundary=b'a=b'),'multipart/byteranges; boundary=a=b')),
        ('missing-part',lambda:mime(multipart(RANGES[:-1],payloads=PAYLOADS[:-1]))),
        ('duplicate-part',lambda:mime(multipart(RANGES[:-1]+[RANGES[0]]))),
        ('changed-total',lambda:mime(multipart(total=2049))),
        ('unrequested-range',lambda:mime(multipart(RANGES[:-1]+[[1200,1263]]))),
        ('trailing-data',lambda:mime(body+b'junk')),
        ('arbitrary-preamble',lambda:mime(b'preamble\r\n'+body)),
        ('bad-CRLF',lambda:mime(body.replace(b'\r\n',b'\n',1))),
        ('transfer-encoding',lambda:mime(multipart(extra_header=b'Content-Transfer-Encoding: base64\r\n'))),
        ('duplicate-range-header',lambda:mime(multipart(extra_header=b'Content-Range: bytes 0-63/2048\r\n'))),
        ('header-expansion',lambda:mime(multipart(extra_header=b'Content-Type: '+b'a'*513+b'\r\n'))),
    ]:check('multipart-'+name,lambda fn=fn:rejects(fn))
    url='https://dl.ash2txt.org/synthetic/07.tif'
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='synthetic-http-') as tmp:
        scenarios=[('complete',Response(),True,8),('unknown-length-eof',Response(length=False),True,8),
                   ('HTTP200',Response(status=200),False,0),('weak-etag',Response(etag='W/"stable"'),False,0),
                   ('changed-etag',Response(etag='"changed"'),False,0),('changed-total',Response(total=101),False,0),
                   ('incomplete-partial',Response(failure=http.client.IncompleteRead(b'abc',5)),False,3),
                   ('opaque-failure',Response(failure=OSError('synthetic')),False,0),
                   ('unknown-length-overflow',Response(body=b'123456789',length=False),False,9)]
        repeated=Response();repeated.headers['ETag']='"conflicting"'
        scenarios.append(('duplicate-etag',repeated,False,0))
        for name,response,ok,retained in scenarios:
            op=Opener(response);client=range_http.RangeClient(Path(tmp)/name,[url],common.LIMITS['headers'],{},pins={url:{'etag':'"stable"','object_total_bytes':100}},opener=op)
            call=lambda:client.get(url,[[0,7]],'range.bin')
            if ok:call()
            else:rejects(call)
            assert response.closed and op.calls==1
            assert client.ledger['retained_body_bytes']==retained
            assert client.ledger['charged_response_bytes']==(8 if ok else 9)
            assert op.request.get_header('If-match')=='"stable"'
            if name in ('HTTP200','weak-etag','changed-etag','changed-total','duplicate-etag'):assert response.reads==0
            CASES.append('HTTP-'+name)
    result={'status':'PASS','count':len(CASES),'cases':CASES,'source':[common.record(IMPL/n) for n in common.CODE_NAMES],
            'reviewer':common.record(__file__),'scientific_arrays_read':0,'real_network_requests':0,'model_calls':0}
    common.save(receipt,result);print(json.dumps({'status':'PASS','count':len(CASES),'receipt':common.record(receipt)}))
if __name__=='__main__':main()
