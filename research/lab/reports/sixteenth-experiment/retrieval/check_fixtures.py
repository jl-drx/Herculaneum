"""Synthetic TIFF/MIME/HTTP contracts; no real TIFFs, network, arrays or models."""
from pathlib import Path
import argparse
from email.message import Message
import hashlib
import http.client
import io
import json
import struct
import tempfile
from probe_common import *
from range_http import RangeClient
from prepare_probe_plan import serving_requests


def rejected(fn):
    try:
        fn()
    except (ValueError,OSError):
        return
    raise AssertionError('Expected rejection')


def tiff_fields(changes=None, auxiliary=False, omit_defaults=False):
    width,height,strip_offset=40,50,8
    fields={256:(4,1,width),257:(4,1,height),258:(3,1,16),259:(3,1,1),262:(3,1,1),
            273:(4,1,strip_offset),274:(3,1,1),277:(3,1,1),278:(4,1,height),279:(4,1,width*height*2)}
    if omit_defaults:
        for tag in (259,274,277,278):
            fields.pop(tag)
    if changes:
        fields.update(changes)
    offset=4008
    if auxiliary:
        fields[282]=(5,1,4300)
    ifd=struct.pack('<H',len(fields))+b''.join(struct.pack('<HHII',tag,*value)for tag,value in sorted(fields.items()))+b'\x00'*4
    return b'II*\x00'+struct.pack('<I',offset),ifd,4400,[height,width]


def multipart_fixture(boundary=b'synthetic-boundary', prefix=b'', spans=None, payloads=None, total=1000, extra_header=b''):
    spans=spans if spans is not None else [[100+i*80,147+i*80]for i in range(8)]
    payloads=payloads if payloads is not None else [(b'row'+bytes([i])+b'\r\n--synthetic-boundary\r\n').ljust(48,b'Q')for i in range(8)]
    body=prefix
    for (a,b),payload in zip(spans,payloads,strict=True):
        body+=b'--'+boundary+b'\r\nContent-Type: image/tiff\r\nContent-Range: bytes '+f'{a}-{b}/{total}'.encode()+b'\r\n'+extra_header+b'\r\n'+payload+b'\r\n'
    return body+b'--'+boundary+b'--\r\n',spans,payloads


class FakeResponse:
    def __init__(self,body,headers,status=206,failure=None):
        self.status,self.headers,self.body,self.failure=status,headers,io.BytesIO(body),failure
        self.read_calls,self.closed=0,False
    def read(self,size):
        self.read_calls+=1
        if self.failure:
            raise self.failure
        return self.body.read(size)
    def close(self):
        self.closed=True


class FakeOpener:
    def __init__(self,response):
        self.response,self.calls,self.last_headers=response,0,None
    def open(self,request,timeout):
        self.calls+=1
        self.last_headers=dict(request.header_items())
        return self.response


def headers(**overrides):
    m=Message()
    for name,value in {'ETag':'"fixture-etag"','Content-Range':'bytes 0-7/1000','Content-Length':'8',**overrides}.items():
        if value is not None:
            m[name]=value
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    output=Path(args.output_dir).resolve()
    require(not output.exists(),'Fresh fixture output required')
    checks=[]
    h,i,total,shape=tiff_fields(auxiliary=True)
    parsed=parse_tiff(h,i,total,shape)
    assert parsed['strip_offset']==8 and parsed['auxiliary_metadata_extents']==[{'tag':282,'extent_half_open':[4300,4308],'dereferenced':False}]
    checks.append('classicLEu16 valid single strip and bounded undereferenced rational extent')
    h,i,total,shape=tiff_fields(omit_defaults=True)
    parsed=parse_tiff(h,i,total,shape)
    assert parsed['specification_defaults_used']['259']==1 and parsed['specification_defaults_used']['278']==4294967295
    checks.append('TIFF defaults are explicit and recorded')
    for label,changes in [('shape',{256:(4,1,41)}),('compression',{259:(3,1,5)}),
                          ('signed',{339:(3,1,2)}),('orientation',{274:(3,1,2)}),
                          ('multiple strips',{278:(4,1,10)}),('strip count',{279:(4,1,3999)}),
                          ('unknown type',{282:(13,1,4300)}),('aux in pixels',{282:(5,1,100)}),
                          ('aux in IFD',{282:(5,1,4010)}),('aux in header',{282:(5,1,0)}),
                          ('aux beyond object',{282:(5,1,4399)}),('offset overflow',{282:(5,4294967295,4300)}),
                          ('tile',{324:(4,1,100)})]:
        h,i,total,shape=tiff_fields(changes)
        rejected(lambda:parse_tiff(h,i,total,shape))
        checks.append('TIFF reject '+label)
    h,i,total,shape=tiff_fields()
    rejected(lambda:parse_tiff(b'II+\x00'+h[4:],i,total,shape))
    rejected(lambda:parse_tiff(h,i[:-1],total,shape))
    rejected(lambda:parse_tiff(h,i[:-4]+b'\x01\x00\x00\x00',total,shape))
    duplicate=struct.pack('<H',11)+i[2:-4]+i[2:14]+b'\x00'*4
    rejected(lambda:parse_tiff(h,duplicate,total,shape))
    checks+=['rejectBigTIFF','rejecttruncatedIFD','rejectnextpage','rejectduplicatetag']
    body,spans,payloads=multipart_fixture()
    parts=parse_multipart(body,'multipart/byteranges; boundary=synthetic-boundary',spans,1000)
    assert [body[p['payload_offset']:p['payload_offset']+p['payload_bytes']]for p in parts]==payloads
    checks.append('length parser preserves delimiter-like binary pixel bytes')
    prefixed=b'\r\n'+body
    pref_parts=parse_multipart(prefixed,'multipart/byteranges; boundary=synthetic-boundary',spans,1000)
    assert pref_parts[0]['payload_offset']==parts[0]['payload_offset']+2
    checks.append('one RFC/nginx leadingCRLF accepted')
    rejected(lambda:parse_multipart(b'\r\n\r\n'+body,'multipart/byteranges; boundary=synthetic-boundary',spans,1000))
    rejected(lambda:parse_multipart(b'arbitrary'+body,'multipart/byteranges; boundary=synthetic-boundary',spans,1000))
    checks+=['multipleleadingCRLFrejected','arbitrarypreamblerejected']
    quoted,_,_=multipart_fixture(boundary=b'a=b')
    assert len(parse_multipart(quoted,'multipart/byteranges; boundary="a=b"',spans,1000))==8
    rejected(lambda:parse_multipart(quoted,'multipart/byteranges; boundary=a=b',spans,1000))
    checks.append('MIME boundary tspecial quoted only')
    for label,damaged in [('missing',multipart_fixture(spans=spans[:-1],payloads=payloads[:-1])[0]),
                          ('duplicate',multipart_fixture(spans=spans[:-1]+[spans[0]])[0]),
                          ('total',multipart_fixture(total=1001)[0]),
                          ('truncated',body[:-1]),('trailing',body+b'x'),
                          ('CRLF',body.replace(b'Content-Type: image/tiff\r\n',b'Content-Type: image/tiff\n',1)),
                          ('encoding',multipart_fixture(extra_header=b'Content-Transfer-Encoding: binary\r\n')[0]),
                          ('expansion',multipart_fixture(extra_header=b'X-Padding: '+b'x'*600+b'\r\n')[0]),
                          ('part length',multipart_fixture(extra_header=b'Content-Length: 49\r\n')[0])]:
        rejected(lambda:parse_multipart(damaged,'multipart/byteranges; boundary=synthetic-boundary',spans,1000))
        checks.append('multipart reject '+label)
    url='https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/surface_volume/07.tif'
    with tempfile.TemporaryDirectory(prefix='synthetic-range-')as temporary:
        base=Path(temporary)
        for label,response,ok,retained in [('single',FakeResponse(b'abcdefgh',headers()),True,8),
            ('unknownEOF',FakeResponse(b'abcdefgh',headers(**{'Content-Length':None})),True,8),
            ('HTTP200',FakeResponse(b'unrequestedfullobject',headers(),status=200),False,0),
            ('changedETag',FakeResponse(b'abcdefgh',headers(ETag='"changed"')),False,0),
            ('changedTotal',FakeResponse(b'abcdefgh',headers(**{'Content-Range':'bytes 0-7/1001'})),False,0),
            ('wrongRange',FakeResponse(b'abcdefgh',headers(**{'Content-Range':'bytes 1-8/1000'})),False,0),
            ('incompletePartial',FakeResponse(b'abcdefgh',headers(),failure=http.client.IncompleteRead(b'abc',5)),False,3),
            ('opaqueFailure',FakeResponse(b'abcdefgh',headers(),failure=OSError('synthetic')),False,0)]:
            opener=FakeOpener(response)
            client=RangeClient(base/label,[url],LIMITS['headers'],{},pins={url:{'etag':'"fixture-etag"','object_total_bytes':1000}},opener=opener)
            if ok:
                path,entry=client.get(url,[[0,7]],'body.bin')
                assert path.read_bytes()==b'abcdefgh' and client.ledger['charged_response_bytes']==8
            else:
                rejected(lambda:client.get(url,[[0,7]],'body.bin'))
                assert client.ledger['charged_response_bytes']==9
            assert client.ledger['retained_body_bytes']==retained and opener.calls==1 and response.closed
            assert opener.last_headers['If-match']=='"fixture-etag"'
            if label in ('HTTP200','changedETag','changedTotal','wrongRange'):
                assert response.read_calls==0
            checks.append('boundedHTTP '+label)
        dup=headers();dup['ETag']='"other"'
        response=FakeResponse(b'abcdefgh',dup);opener=FakeOpener(response)
        client=RangeClient(base/'duplicateHTTP',[url],LIMITS['headers'],{},opener=opener)
        rejected(lambda:client.get(url,[[0,7]],'body.bin'))
        assert response.read_calls==0
        checks.append('duplicate HTTP singleton rejected before read')
        mh=headers(**{'Content-Range':None,'Content-Length':str(len(prefixed)),
                      'Content-Type':'multipart/byteranges; boundary=synthetic-boundary'})
        response=FakeResponse(prefixed,mh);opener=FakeOpener(response)
        client=RangeClient(base/'multipart',[url],LIMITS['serving'],{},pins={url:{'etag':'"fixture-etag"','object_total_bytes':1000}},opener=opener)
        path,entry=client.get(url,spans,'multipart.bin',mode='multipart')
        assert len(entry['parts'])==8 and path.read_bytes()==prefixed
        checks.append('bounded multipart transport preserves raw framing')
    widths=[6330,9506,5249]
    cap_total=sum(8*1152*2+8192+8*w*2+1 for w in widths)
    assert cap_total==417235 and cap_total<=524288
    checks.append('exact derived6request cap within512KiB')
    output.mkdir(parents=True)
    receipt=dict(status='PASS',checks=checks,check_count=len(checks),code=[record(ROOT/n)for n in CODE_NAMES],
                 actual_network_requests=0,real_CT_reads=0,numeric_pixel_decodes=0,model_calls=0,
                 scope='In-memory synthetic binary TIFF/MIME/HTTP fixtures; no numeric arrays')
    save(output/'fixture-receipt.json',receipt)
    print(json.dumps({'status':'PASS','checks':len(checks),'receipt':record(output/'fixture-receipt.json')}))


if __name__=='__main__':
    main()
