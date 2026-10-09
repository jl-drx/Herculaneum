import io,json,tempfile,unittest
from email.message import Message
from pathlib import Path
from transport import Client
from common import parse_multipart
URL='https://dl.ash2txt.org/test/01.tif'

class Response:
    def __init__(self,body=b'abcdefgh',status=206,etag='"a"',cr='bytes 0-7/100'):
        self.status=status;self.headers=Message();self.headers['ETag']=etag
        self.headers['Content-Length']=str(len(body));self.headers['Content-Range']=cr
        self.body=io.BytesIO(body);self.read_calls=0
    def read(self,n):self.read_calls+=1;return self.body.read(n)
    def close(self):pass
class Opener:
    def __init__(self,r):self.r=r
    def open(self,*args,**kwargs):return self.r

class Tests(unittest.TestCase):
    def client(self,root,response,cap=1000000):
        return Client(root,dict(urls=[URL],limits=dict(requests=800,reserved_body_bytes=cap,acquisition_seconds=600)),opener=Opener(response))
    def test_200_rejected_without_body(self):
        r=Response(status=200)
        with tempfile.TemporaryDirectory() as d:
            c=self.client(d,r)
            with self.assertRaises(ValueError):c.get(URL,[[0,7]],'a.bin')
            self.assertEqual(r.read_calls,0);self.assertEqual(c.ledger['reserved_body_bytes'],9)
    def test_changed_etag_rejected_without_body(self):
        r=Response()
        with tempfile.TemporaryDirectory() as d:
            c=self.client(d,r)
            with self.assertRaises(ValueError):c.get(URL,[[0,7]],'a.bin',dict(etag='"b"',total_bytes=100))
            self.assertEqual(r.read_calls,0)
    def test_success_does_not_refund_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.client(d,Response());p,e=c.get(URL,[[0,7]],'a.bin')
            self.assertTrue(e['success']);self.assertEqual(p.read_bytes(),b'abcdefgh')
            self.assertEqual(c.ledger['reserved_body_bytes'],9)
            self.assertEqual(c.ledger['returned_body_bytes'],8)
    def test_budget_prevents_http(self):
        r=Response()
        with tempfile.TemporaryDirectory() as d:
            c=self.client(d,r,cap=8)
            with self.assertRaises(ValueError):c.get(URL,[[0,7]],'a.bin')
            self.assertEqual(r.read_calls,0);self.assertEqual(len(c.ledger['requests']),0)
    def test_reordered_binary_multipart_and_duplicate_rejection(self):
        spans=[[i*20,i*20+7] for i in range(256)]
        def body(seq):
            return b''.join(b'--x\r\nContent-Range: bytes '+f'{a}-{b}/10000'.encode()+b'\r\n\r\n--x\r\nabc\r\n' for a,b in seq)+b'--x--\r\n'
        data=body(list(reversed(spans)))
        parts=parse_multipart(data,'multipart/byteranges; boundary=x',spans,10000)
        self.assertEqual(len(parts),256)
        self.assertEqual(data[parts[0]['payload_offset']:parts[0]['payload_offset']+8],b'--x\r\nabc')
        with self.assertRaises(ValueError):parse_multipart(body(spans[:-1]+[spans[0]]),'multipart/byteranges; boundary=x',spans,10000)

if __name__=='__main__':unittest.main()
