"""Synthetic safety checks; poisoned image resources must never be requested."""
import copy
from email.message import Message
import json
from pathlib import Path
import struct
import sys
import tempfile
from psd_metadata import Client, inspect, MAX_METADATA, MAX_RESOURCES, save


def block(ident, data, name=b''):
    name_field = bytes([len(name)]) + name
    name_field += b'\0'*(len(name_field)%2)
    return b'8BIM'+struct.pack('>H',ident)+name_field+struct.pack('>I',len(data))+data+b'\0'*(len(data)%2)


def psd(resources, color=b''):
    return (b'8BPS'+struct.pack('>H',1)+b'\0'*6+struct.pack('>HIIHH',3,8853,6205,8,3)
            +struct.pack('>I',len(color))+color+struct.pack('>I',len(resources))+resources
            +struct.pack('>I',0)+b'COMPOSITE_IMAGE_POISON')


def parse(data, forbidden=()):
    reads=[]
    def get(start,length,kind):
        for low,high in forbidden:
            assert start+length <= low or start >= high, 'Forbidden payload requested'
        reads.append((start,length,kind))
        assert 0 <= start and start+length <= len(data)
        return data[start:start+length]
    return inspect(get,len(data)),reads


def rejected(data):
    try:
        parse(data)
    except ValueError:
        return
    raise AssertionError('Malformed PSD accepted')


class Response:
    def __init__(self,status,headers,body=b''):
        self.status=status;self.headers=Message();self.body=body;self.reads=0
        for key,value in headers:self.headers[key]=value
    def read(self,n):
        self.reads+=1; result=self.body[:n];self.body=self.body[n:];return result
    def close(self):pass


class Opener:
    def __init__(self,responses):self.responses=responses;self.requests=[]
    def open(self,request,timeout):
        self.requests.append(request);return self.responses.pop(0)


def main():
    target=Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=False)
    checks=[]
    def run(name,fn):fn();checks.append(name)
    xmp=b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><author>fixture</author></x:xmpmeta>'
    def valid():
        thumbnail=block(1036,b'THUMBNAIL_IMAGE_POISON',b'ab')
        resources=thumbnail+block(1060,xmp,b'abc')+block(1028,b'IPTC')
        data=psd(resources,b'PALETTE_POISON')
        start=34+len(b'PALETTE_POISON'); thumb_start=start+14
        result,reads=parse(data,[(30,30+len(b'PALETTE_POISON')),
            (thumb_start,thumb_start+len(b'THUMBNAIL_IMAGE_POISON')),
            (start+len(resources),len(data))])
        assert result['header']['height']==8853 and result['header']['width']==6205
        assert result['selected_metadata']['XMP']['text_utf8']==xmp.decode()
        assert len(result['resources'])==3 and not result['resources'][0]['payload_read']
    run('skip palette thumbnail and all layer/composite payloads; odd names and payload padding',valid)
    run('empty image resources valid',lambda:parse(psd(b'')))
    run('zero length selected metadata valid',lambda:parse(psd(block(1060,b''))))
    valid_data=psd(block(1060,xmp))
    for name,offset,value in [('bad signature',0,b'BAD!'),('PSB version',4,b'\0\2'),
        ('nonzero reserved',6,b'1'),('zero channels',12,b'\0\0'),
        ('zero height',14,b'\0'*4),('unsupported depth',22,b'\0\3'),
        ('resource section overflow',30,b'\xff'*4),('resource signature',34,b'BAD!')]:
        edited=bytearray(valid_data);edited[offset:offset+len(value)]=value
        run(name,lambda d=bytes(edited):rejected(d))
    run('duplicate XMP rejected',lambda:rejected(psd(block(1060,xmp)*2)))
    run('metadata cap rejects before payload',lambda:rejected(psd(block(1060,b'a'*(MAX_METADATA+1)))))
    run('resource cap rejects',lambda:rejected(psd(block(1000,b'')*(MAX_RESOURCES+1))))
    edited=bytearray(valid_data);edited[42:46]=struct.pack('>I',len(xmp)+1000)
    run('resource payload crosses section',lambda:rejected(bytes(edited)))
    edited=bytearray(valid_data);edited[40]=255
    run('resource name crosses section',lambda:rejected(bytes(edited)))
    common=[('Content-Type','application/octet-stream'),('ETag','"fixture"')]
    def transport_case(name,status=206,extra=(),body=b'abc',expected=True):
        head=Response(200,common+[('Content-Length','100')])
        headers=common+[('Content-Length','3'),('Content-Range','bytes 0-2/100')]
        for key,value in extra:
            if key.startswith('replace:'):
                actual=key.split(':',1)[1];headers=[h for h in headers if h[0]!=actual];headers.append((actual,value))
            else:headers.append((key,value))
        ranged=Response(status,headers,body)
        op=Opener([head,ranged]);client=Client(target/name,opener=op);client.request()
        try:
            result=client.request(0,3,'fixture')
        except ValueError:
            assert not expected
        else:
            assert expected and result==b'abc'
        assert head.reads==0
        assert op.requests[-1].get_header('If-match')=='"fixture"'
        assert len(client.log['requests'])==2
        return ranged,client
    run('conditional exact range valid; HEAD body unread',lambda:transport_case('valid'))
    for name,status,extra in [
        ('full object refused',200,()),('redirect refused',302,()),
        ('wrong ETag',206,(('replace:ETag','"changed"'),)),
        ('wrong range',206,(('replace:Content-Range','bytes 1-3/100'),)),
        ('wrong total',206,(('replace:Content-Range','bytes 0-2/101'),)),
        ('duplicate length',206,(('Content-Length','3'),)),
        ('encoded body',206,(('Content-Encoding','gzip'),)),
        ('chunked body',206,(('Transfer-Encoding','chunked'),)),
        ('wrong MIME',206,(('replace:Content-Type','text/html'),))]:
        def test(name=name,status=status,extra=extra):
            response,_=transport_case(name.replace(' ','-'),status,extra,expected=False)
            assert response.reads==0
        run(name+' before body read',test)
    run('truncation retained and no retry',lambda:transport_case('truncated',body=b'ab',expected=False))
    run('excess retained and no retry',lambda:transport_case('excess',body=b'abcd',expected=False))
    save(target/'TESTS.json',dict(status='PASS',count=len(checks),checks=checks,network_calls=0,
        scientific_pixels=0,fixture_note='Synthetic bytes and injected HTTP responses; no independent reviewer.'))
    print(json.dumps(dict(status='PASS',checks=len(checks),network_calls=0)))


if __name__=='__main__':main()
