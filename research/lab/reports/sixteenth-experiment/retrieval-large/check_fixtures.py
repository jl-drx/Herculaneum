"""Synthetic binary/HTTP fixtures only; never open a network connection or source CT array."""
import argparse
from email.message import Message
import hashlib
import http.client
import json
from pathlib import Path
import socket
import tempfile
from large_common import *
from prepare_plan import large_requests
from range_http import RangeClient
from run_large_probe import inspect_serving

URL='https://dl.ash2txt.org/fragments/Frag1/synthetic.volpkg/working/54keV_exposed_surface/surface_volume/07.tif'
TOTAL=2000000
RANGES=[[1000+i*3000,1000+i*3000+2303] for i in range(256)]
BOUNDARY='bounded-test'


def multipart(ranges=RANGES,total=TOTAL,reverse=False,prefix=b'',boundary=BOUNDARY):
    chunks=[]
    selected=list(reversed(ranges)) if reverse else ranges
    for index,(a,b) in enumerate(selected):
        payload=(bytes([index%251])*20+b'\r\n--bounded-test\r\n'+bytes([255-index%251])*2304)[:b-a+1]
        chunks.append(b'--'+boundary.encode()+b'\r\nContent-Type: application/octet-stream\r\nContent-Range: bytes '+f'{a}-{b}/{total}'.encode()+b'\r\n\r\n'+payload+b'\r\n')
    return prefix+b''.join(chunks)+b'--'+boundary.encode()+b'--\r\n'


class Response:
    def __init__(self,body,status=206,headers=None,incomplete=None):
        self.body,self.status,self.position,self.incomplete=body,status,0,incomplete
        self.read_calls=0
        self.headers=Message()
        for k,v in (headers or {}).items():
            self.headers[k]=str(v)
    def read(self,n):
        self.read_calls+=1
        if self.incomplete is not None:
            partial,self.incomplete=self.incomplete,None
            raise http.client.IncompleteRead(partial,len(self.body))
        result=self.body[self.position:self.position+n]
        self.position+=len(result)
        return result
    def close(self):
        pass


class Opener:
    def __init__(self,response):
        self.response,self.calls=response,[]
    def open(self,request,timeout):
        self.calls.append(request)
        return self.response


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    root=Path(args.output_dir).resolve()
    require(not root.exists(),'Fresh fixture output required')
    root.mkdir(parents=True)
    socket.socket.connect=lambda *_: (_ for _ in ()).throw(AssertionError('Network forbidden in fixtures'))
    results=[]
    def passed(name,condition=True):
        require(condition,'Synthetic fixture failed:'+name)
        results.append({'name':name,'status':'PASS'})
    def rejected(name,callback):
        try:
            callback()
        except ContractError:
            passed(name)
            return
        raise AssertionError('Expected rejection:'+name)
    body=multipart()
    parsed=parse_multipart(body,'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL)
    passed('256 binary payloads survive delimiter collisions',len(parsed)==256 and all(
        hashlib.sha256(body[p['payload_offset']:p['payload_offset']+p['payload_bytes']]).hexdigest()==p['payload_sha256']
        for p in parsed))
    reverse=multipart(reverse=True,prefix=b'\r\n')
    parsed_reverse=parse_multipart(reverse,'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL)
    passed('reordered parts and one leading CRLF',parsed_reverse[0]['range_inclusive']==RANGES[-1])
    quoted=multipart(boundary='a=b')
    passed('quoted MIME special boundary',len(parse_multipart(quoted,'multipart/byteranges; boundary="a=b"',RANGES,TOTAL))==256)
    rejected('unquoted MIME special boundary',lambda:parse_multipart(quoted,'multipart/byteranges; boundary=a=b',RANGES,TOTAL))
    rejected('255 requested parts',lambda:parse_multipart(multipart(RANGES[:-1]),'multipart/byteranges; boundary='+BOUNDARY,RANGES[:-1],TOTAL))
    rejected('257 requested parts',lambda:parse_multipart(body,'multipart/byteranges; boundary='+BOUNDARY,RANGES+[RANGES[0]],TOTAL))
    rejected('missing returned part',lambda:parse_multipart(multipart(RANGES[:-1]),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('duplicate returned part',lambda:parse_multipart(multipart(RANGES[:-1]+[RANGES[0]]),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('unrequested/coalesced part',lambda:parse_multipart(multipart([[RANGES[0][0],RANGES[1][1]]]+RANGES[2:]),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('object total changed',lambda:parse_multipart(multipart(total=TOTAL+1),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('overlapping requested spans',lambda:parse_multipart(body,'multipart/byteranges; boundary='+BOUNDARY,[RANGES[0]]+[[RANGES[0][0]+1,RANGES[0][1]+1]]+RANGES[2:],TOTAL))
    rejected('truncated binary body',lambda:parse_multipart(body[:-10],'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('trailing data',lambda:parse_multipart(body+b'x','multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('arbitrary preamble',lambda:parse_multipart(b'ignored'+body,'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('two leading CRLF',lambda:parse_multipart(b'\r\n\r\n'+body,'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('part header above512',lambda:parse_multipart(body.replace(b'Content-Type: application/octet-stream',b'Content-Type: '+b'x'*520,1),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('duplicate MIME header',lambda:parse_multipart(body.replace(b'Content-Type:',b'Content-Length: 2304\r\nContent-Length: 2304\r\nContent-Type:',1),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('framing exceeds allowance',lambda:parse_multipart(body+b'x'*(768*256+1024),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    source={'fragment':'frag1','case_id':'frag1-context1','url':URL,'bbox_source_xywh':[17,1152,1152,1152]}
    contracts=[]
    for dtype,offset in [('>u2',260),('<u2',8)]:
        c={'url':URL,'width':6330,'height':8181,'strip_offset':offset,'strip_bytes':6330*8181*2,'dtype':dtype}
        request=large_requests(source,c)
        passed('address formula independent of '+dtype,request['crop_row_ranges'][0]==[offset+(1152*6330+17)*2,offset+(1152*6330+1169)*2-1]
               and request['crop_row_ranges'][-1][0]==offset+(1407*6330+17)*2
               and request['comparator_range']==[offset+1152*6330*2,offset+1408*6330*2-1])
        contracts.append(c)
    widths=[6330,9506,5249]
    reserve=3*(256*2304+768*256+1024)+sum(256*w*2+1 for w in widths)
    passed('literal six-response reserve',reserve==MAXIMUM_RESERVATION==13157891 and reserve<LIMITS['http']['body_bytes'])
    limits=LIMITS['http']
    pin={URL:{'etag':'"stable"','object_total_bytes':TOTAL}}
    count=0
    def client(response):
        nonlocal count
        count+=1
        opener=Opener(response)
        return RangeClient(root/('http-'+str(count)),[URL],limits,{'synthetic':True},pins=dict(pin),opener=opener),opener
    single_headers={'ETag':'"stable"','Content-Range':f'bytes 1000-1007/{TOTAL}','Content-Length':'8'}
    c,op=client(Response(b'abcdefgh',headers=single_headers))
    path,rec=c.get(URL,[[1000,1007]],'single.bin')
    passed('single success exact charged bytes and IfMatch',path.read_bytes()==b'abcdefgh' and c.ledger['charged_response_bytes']==8 and op.calls[0].get_header('If-match')=='"stable"')
    resp=Response(b'x'*100,status=200,headers=single_headers)
    c,_=client(resp)
    rejected('HTTP200 rejected before body',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    passed('HTTP200 full failure reservation no consumed body',resp.read_calls==0 and c.ledger['charged_response_bytes']==9 and c.ledger['retained_body_bytes']==0)
    resp=Response(b'abcdefgh',headers={**single_headers,'ETag':'"changed"'})
    c,_=client(resp)
    rejected('stale ETag before body',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    passed('stale ETag no body',resp.read_calls==0)
    resp=Response(b'abcdefgh',headers=single_headers)
    resp.headers['ETag']='"other"'
    c,_=client(resp)
    rejected('duplicate identity header',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    resp=Response(b'abcdefgh',headers={**single_headers,'Content-Range':f'bytes 1000-1007/{TOTAL+1}'})
    c,_=client(resp)
    rejected('changed total before body',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    resp=Response(b'abcdefgh',headers=single_headers,incomplete=b'abc')
    c,_=client(resp)
    rejected('partial failed read preserved',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    passed('partial charge separated from retained',c.ledger['charged_response_bytes']==9 and c.ledger['retained_body_bytes']==3 and (c.root/'raw/single.bin.partial').read_bytes()==b'abc')
    resp=Response(b'abcdefgh',headers={k:v for k,v in single_headers.items()if k!='Content-Length'})
    c,_=client(resp)
    path,rec=c.get(URL,[[1000,1007]],'single.bin')
    passed('unknown length bounded EOF',path.read_bytes()==b'abcdefgh' and rec['complete_eof'])
    resp=Response(b'abcdefghi',headers={k:v for k,v in single_headers.items()if k!='Content-Length'})
    c,_=client(resp)
    rejected('unknown length sentinel overflow',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    c,_=client(Response(body,headers={'ETag':'"stable"','Content-Type':'multipart/byteranges; boundary='+BOUNDARY,'Content-Length':len(body)}))
    path,rec=c.get(URL,RANGES,'multi.bin',mode='multipart')
    passed('256 multipart HTTP payload accepted',len(rec['parts'])==256 and c.ledger['charged_response_bytes']==len(body))
    c,_=client(Response(body,headers={'ETag':'"stable"','Content-Type':'multipart/byteranges; boundary='+BOUNDARY,'Content-Length':len(body)}))
    rejected('8part network count forbidden',lambda:c.get(URL,RANGES[:8],'multi.bin',mode='multipart'))
    passed('bad count consumes no attempt',len(c.ledger['requests'])==0)
    # End-to-end row routing uses synthetic raw byte bands, never a numeric CT decoder.
    src={'fragment':'frag1','case_id':'synthetic','url':URL,'bbox_source_xywh':[17,1152,1152,1152]}
    contract={'url':URL,'width':1200,'height':3000,'strip_offset':260,'strip_bytes':1200*3000*2,
              'dtype':'>u2','byte_order':'big','etag':'"stable"','object_total_bytes':8000000}
    req=large_requests(src,contract)
    band=bytes((i*37+9)%256 for i in range(256*1200*2))
    blob=b''
    for i,span in reversed(list(enumerate(req['crop_row_ranges']))):
        a,b=span
        payload=band[i*2400+34:i*2400+34+2304]
        blob+=b'--'+BOUNDARY.encode()+b'\r\nContent-Range: bytes '+f'{a}-{b}/8000000'.encode()+b'\r\n\r\n'+payload+b'\r\n'
    blob+=b'--'+BOUNDARY.encode()+b'--\r\n'
    class FakeClient:
        def __init__(self):
            self.calls=[]
        def get(self,url,ranges,name,mode='single'):
            self.calls.append(mode)
            path=root/name
            path.write_bytes(blob if mode=='multipart'else band)
            return path,{'parts':parse_multipart(blob,'multipart/byteranges; boundary='+BOUNDARY,req['crop_row_ranges'],8000000)}if mode=='multipart'else {}
    fake=FakeClient()
    result=inspect_serving(fake,src,contract,req)
    passed('256 comparator rows exact under reordered response',len(result['rows'])==256 and all(r['bitwise_equal']for r in result['rows']) and fake.calls==['multipart','single'])
    class FailedClient:
        calls=[]
        def get(self,*a,**k):
            self.calls.append(k.get('mode','single'))
            raise ContractError('synthetic first request failure')
    failed=FailedClient()
    rejected('first failure prevents comparator',lambda:inspect_serving(failed,src,contract,req))
    passed('only multipart attempted on first failure',failed.calls==['multipart'])
    receipt=dict(status='PASS',schema_version=1,synthetic_only=True,cases=len(results),checks=results,
                 code=[record(ROOT/n)for n in CODE_NAMES],network_requests=0,numeric_pixel_decodes=0,
                 actual_source_arrays_read=0,model_calls=0)
    save(root/'fixture-receipt.json',receipt)
    print(json.dumps({'status':'PASS','cases':len(results),'receipt':record(root/'fixture-receipt.json')}))


if __name__=='__main__':
    main()
