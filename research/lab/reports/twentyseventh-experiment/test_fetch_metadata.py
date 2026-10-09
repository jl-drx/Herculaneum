import io
import json
from email.message import Message
from pathlib import Path
import tempfile
import unittest

from fetch_metadata import Fetcher, acquire, decode_json, first_dataset, metadata_name


class Response(io.BytesIO):
    def __init__(self, body=b'{}', status=200, headers=()):
        super().__init__(body); self.status=status; self.headers=Message()
        self.headers['Content-Length']=str(len(body)); self.headers['Content-Type']='application/json'
        for key,value in headers:self.headers[key]=value


class Opener:
    def __init__(self, responses):self.responses=iter(responses); self.urls=[]
    def open(self, request, timeout):
        self.urls.append(request.full_url)
        return next(self.responses)


class Tests(unittest.TestCase):
    def fetcher(self, *responses):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        return Fetcher(Path(t.name)/'out', Opener(responses))

    def test_metadata_names(self):
        for name in ('.zattrs','0/.zarray','foo/bar/zarr.json'):self.assertEqual(metadata_name(name),name)
        for name in ('../.zarray','/0/.zarray','0/0/0','0/.zarray?x','https://x/.zarray','%2e%2e/.zattrs'):
            with self.subTest(name=name), self.assertRaises(ValueError):metadata_name(name)

    def test_json_rejections(self):
        for body in (b'{"a":1,"a":2}',b'{"a":NaN}',b'[]',b'null',b'\xff',b'{'):
            with self.subTest(body=body),self.assertRaises((ValueError,UnicodeError)):decode_json(body)

    def test_first_dataset(self):
        self.assertEqual(first_dataset({'multiscales':[{'datasets':[{'path':'0'},{'path':'1'}]}]}),'0')
        for path in ('../0','0?x','/0',''):
            with self.assertRaises(ValueError):first_dataset({'multiscales':[{'datasets':[{'path':path}]}]})

    def test_v2(self):
        f=self.fetcher(Response(b'{"multiscales":[{"datasets":[{"path":"0"}]}]}'),Response(),Response(b'{"zarr_format":2}'),Response())
        r=acquire(f);self.assertEqual(r['array_path'],'0');self.assertEqual(len(f.log['requests']),4)
        self.assertEqual(f.log['returned_body_bytes'],sum(x['returned_bytes'] for x in f.log['requests']))

    def test_v3(self):
        f=self.fetcher(Response(b'absent',404),Response(b'{"zarr_format":3,"node_type":"array","shape":[2,2,2]}'))
        self.assertEqual(acquire(f)['storage_format'],'zarr_v3')

    def test_404_retained(self):
        f=self.fetcher(Response(b'absent',404));self.assertIsNone(f.get('.zattrs',100))
        self.assertEqual(f.log['returned_body_bytes'],6);self.assertTrue(f.log['requests'][0]['body']['sha256'])

    def test_status_encoding_duplicate_oversize(self):
        variants=[Response(status=302),Response(headers=[('Content-Encoding','gzip')]),
                  Response(headers=[('Content-Length','2')]),Response(headers=[('Transfer-Encoding','chunked')]),
                  Response(b'123456'),Response(headers=[('ETag','a'),('ETag','b')])]
        for response in variants:
            f=self.fetcher(response)
            with self.assertRaises(ValueError):f.get('.zattrs',4)
            self.assertFalse(f.log['requests'][0]['success'])

    def test_truncated_excess_malformed(self):
        a=Response(b'{');a.headers.replace_header('Content-Length','2')
        b=Response(b'{}x');b.headers.replace_header('Content-Length','2')
        for response,retained in ((a,1),(b,3),(Response(b'no'),2)):
            f=self.fetcher(response)
            with self.assertRaises(ValueError):f.get('.zattrs',100)
            self.assertEqual(f.log['returned_body_bytes'],retained)

    def test_no_retry_and_reservation(self):
        f=self.fetcher(Response());f.get('.zattrs',65536)
        with self.assertRaises(ValueError):f.get('.zattrs',65536)
        f.log['reserved_body_bytes']=262144
        with self.assertRaises(ValueError):f.get('.zgroup',1)
        self.assertEqual(len(f.opener.urls),1)


if __name__=='__main__':unittest.main(verbosity=2)
