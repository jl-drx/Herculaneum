import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import numcodecs
import zarr

from fetch_metadata import pin
from local_zarr_reader import LocalReader,COMPRESSOR,sample_checked
from reference_mechanics import sample_trilinear

HERE=Path(__file__).resolve().parent
FIXTURE_IDENTITIES=[]


def bind(root):
    def identity(p):
        item=pin(p);item.pop('path');return item
    result=dict(metadata={n:identity(root/n) for n in ('.zattrs','.zgroup','0/.zarray')},
                objects={str(p.relative_to(root)):identity(p) for p in (root/'0').rglob('*')
                         if p.is_file() and not p.name.startswith('.')})
    FIXTURE_IDENTITIES.append(copy.deepcopy(result))
    return result


def fixture(root,shape=(5,6,7),chunks=(4,4,4)):
    attrs=json.loads((HERE/'metadata-run-v1/01-metadata.body').read_text())
    group=zarr.open_group(str(root),mode='w',zarr_format=2,attributes=attrs)
    a=group.create_array('0',shape=shape,chunks=chunks,dtype='<u2',
        compressors=numcodecs.Blosc(**{k:v for k,v in COMPRESSOR.items() if k!='id'}),filters=None,
        order='C',fill_value=0,chunk_key_encoding={'name':'v2','separator':'/'},config={'write_empty_chunks':True})
    z,y,x=np.indices(shape,dtype=np.uint16);known=3*z+5*y+7*x
    a[:]=known
    np.testing.assert_array_equal(a[:],known)
    return known,bind(root)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'store';self.cache=Path(self.temp.name)/'cache'
        self.known,self.manifest=fixture(self.root)

    def test_reference_roundtrip_and_edges(self):
        r=LocalReader(self.root,self.manifest)
        for key in np.ndindex(2,2,2):
            slices=tuple(slice(k*4,min(k*4+4,s)) for k,s in zip(key,self.known.shape))
            np.testing.assert_array_equal(r(key),self.known[slices])
        self.assertEqual(r.stats['decoded'],8)

    def test_sampler_known_linear_values(self):
        r=LocalReader(self.root,self.manifest)
        points=np.array([[0,0,0],[1.2,2.3,3.4],[3.7,4.1,2.6],[6,5,4],[6.1,5,4]])
        values,valid,stats=sample_trilinear(points,r.spec,r)
        self.assertEqual(valid.tolist(),[True,True,True,True,False])
        np.testing.assert_allclose(values[:4],points[:4]@np.array([7,5,3]),rtol=0,atol=1e-12)
        self.assertTrue(np.isnan(values[-1]));self.assertEqual(values[0],0)

    def test_missing_is_not_fill_zero(self):
        del self.manifest['objects']['0/0/0/0'];r=LocalReader(self.root,self.manifest)
        values,valid,_=sample_trilinear(np.array([[0,0,0],[6,5,4]]),r.spec,r)
        self.assertEqual(valid.tolist(),[False,True]);self.assertTrue(np.isnan(values[0]))

    def test_expected_missing_fails(self):
        (self.root/'0/0/0/0').unlink()
        with self.assertRaises(ValueError):LocalReader(self.root,self.manifest)((0,0,0))

    def test_changed_object_fails(self):
        p=self.root/'0/0/0/0';b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
        with self.assertRaises(ValueError):LocalReader(self.root,self.manifest)((0,0,0))

    def test_truncated_object_fails(self):
        p=self.root/'0/0/0/0';p.write_bytes(p.read_bytes()[:20])
        with self.assertRaises(ValueError):LocalReader(self.root,self.manifest)((0,0,0))

    def test_changed_metadata_after_construction(self):
        r=LocalReader(self.root,self.manifest);p=self.root/'0/.zarray';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises(ValueError):r((0,0,0))

    def test_semantic_schema_mismatches(self):
        p=self.root/'0/.zarray';original=p.read_bytes()
        for key,value in [('dtype','>u2'),('order','F'),('dimension_separator','.'),('filters',[]),
                          ('chunks',[256,256,256]),('shape',[0,6,7]),('fill_value',None),
                          ('compressor',dict(COMPRESSOR,cname='lz4'))]:
            a=json.loads(original);a[key]=value;p.write_text(json.dumps(a))
            with self.subTest(key=key),self.assertRaises(ValueError):LocalReader(self.root,bind(self.root))
        p.write_bytes(original)

    def test_axes_mismatch(self):
        p=self.root/'.zattrs';a=json.loads(p.read_text());a['multiscales'][0]['axes'].reverse();p.write_text(json.dumps(a))
        with self.assertRaises(ValueError):LocalReader(self.root,bind(self.root))

    def test_decompression_bomb_header(self):
        p=self.root/'0/0/0/0';b=bytearray(p.read_bytes());struct.pack_into('<I',b,4,2**31);p.write_bytes(b)
        with self.assertRaises(ValueError):LocalReader(self.root,bind(self.root))((0,0,0))

    def test_rebound_corrupt_compressed_payload(self):
        # Larger compressible fixture ensures a real compressed stream, not memcpy.
        root=Path(self.temp.name)/'corrupt';_,manifest=fixture(root,(16,16,16),(16,16,16))
        p=root/'0/0/0/0';b=bytearray(p.read_bytes());b[16:]=b'\xff'*(len(b)-16);p.write_bytes(b)
        r=LocalReader(root,bind(root))
        with self.assertRaises((ValueError,RuntimeError)):r((0,0,0))
        self.assertEqual(r.stats['decoded'],0)

    def test_rebound_header_mismatch(self):
        p=self.root/'0/0/0/0';original=p.read_bytes()
        for byte,value in ((0,1),(2,149),(3,4)):
            b=bytearray(original);b[byte]=value;p.write_bytes(b)
            with self.subTest(byte=byte),self.assertRaises(ValueError):LocalReader(self.root,bind(self.root))((0,0,0))

    def test_encoded_size_budget(self):
        m=copy.deepcopy(self.manifest);m['objects']['0/0/0/0']['size_bytes']=1000000
        r=LocalReader(self.root,m)
        with self.assertRaises(ValueError):r((0,0,0))
        self.assertEqual(r.stats['source_reads'],0)

    def test_sampler_budget_before_source_read(self):
        r=LocalReader(self.root,self.manifest)
        with self.assertRaises(ValueError):sample_trilinear(np.array([[1,1,1]]),r.spec,r,max_bytes=1)
        self.assertEqual(r.stats['source_reads'],0)

    def test_physical_edge_budget_before_source_read(self):
        root=Path(self.temp.name)/'tiny';_,manifest=fixture(root,(1,1,1),(128,128,128))
        r=LocalReader(root,manifest)
        with self.assertRaises(ValueError):sample_checked(np.array([[0,0,0]]),r,max_bytes=1024**2)
        self.assertEqual(r.stats['source_reads'],0)

    def test_cache_reuse_without_source(self):
        r=LocalReader(self.root,self.manifest,cache=self.cache);expected=r((0,0,0))
        (self.root/'0/0/0/0').unlink()
        fresh=LocalReader(self.root,self.manifest,cache=self.cache)
        np.testing.assert_array_equal(fresh((0,0,0)),expected)
        self.assertEqual(fresh.stats['source_reads'],0);self.assertEqual(fresh.stats['cache_hits'],1)

    def test_cache_corruption_stops(self):
        r=LocalReader(self.root,self.manifest,cache=self.cache);r((0,0,0))
        p=next(self.cache.iterdir());p.write_bytes(b'bad')
        with self.assertRaises(ValueError):r((0,0,0))
        self.assertEqual(r.stats['source_reads'],1)

    def test_cache_byte_budget(self):
        r=LocalReader(self.root,self.manifest,cache=self.cache,cache_limit=1)
        with self.assertRaises(ValueError):r((0,0,0))
        self.assertEqual(list(self.cache.iterdir()),[])

    def test_interrupted_publication_and_explicit_resume(self):
        r=LocalReader(self.root,self.manifest,cache=self.cache,interrupt_before_publish=True)
        with self.assertRaises(InterruptedError):r((0,0,0))
        self.assertTrue(all(p.suffix=='.partial' for p in self.cache.iterdir()))
        fresh=LocalReader(self.root,self.manifest,cache=self.cache);fresh((0,0,0))
        self.assertEqual(fresh.stats['source_reads'],1);self.assertEqual(fresh.stats['cache_hits'],0)
        self.assertFalse(any(p.suffix=='.partial' for p in self.cache.iterdir()))

    def test_bad_key_and_symlink(self):
        r=LocalReader(self.root,self.manifest)
        with self.assertRaises(ValueError):r((-1,0,0))
        p=self.root/'0/0/0/0';p.unlink();p.symlink_to(self.root/'0/0/0/1')
        with self.assertRaises(ValueError):r((0,0,0))


if __name__=='__main__':unittest.main(verbosity=2)
