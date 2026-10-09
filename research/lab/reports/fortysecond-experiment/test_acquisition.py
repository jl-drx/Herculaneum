import io,unittest
import numpy as np
import tifffile
from common import *
from acquire_ct import assemble
from acquire_geometry import Remote,values
class Memory:
 def __init__(self,data):self.data=data;self.requests=[]
 def get(self,url,tag,size,start,length):self.requests.append((start,length));return self.data[start:start+length]
class Tests(unittest.TestCase):
 def test_ranged_stripped_tiff_coordinates(self):
  for endian in ['<','>']:
   a=np.arange(145*170,dtype=np.float32).reshape(145,170)/7;f=io.BytesIO();tifffile.imwrite(f,a,compression=None,rowsperstrip=13,byteorder=endian);raw=f.getvalue();memory=Memory(raw);remote=Remote(memory,'memory',len(raw),'test')
   with tifffile.TiffFile(remote) as tf:
    y=[0,12,13,97,144];x=[0,169,2,161,155];v=values(remote,tf.pages[0],y,x)
    np.testing.assert_array_equal(v,a[y,x])
   self.assertTrue(all(n<=65536 for s,n in memory.requests))
 def test_chunk_axis_assembly(self):
  base=np.array([7,11,19]);pieces={}
  for z in range(2):
   for y in range(2):
    for x in range(2):pieces[tuple(base+[z,y,x])]=np.full((128,128,128),100*z+10*y+x,np.uint8)
  result=assemble(pieces,base*128)
  for z in range(2):
   for y in range(2):
    for x in range(2):self.assertTrue((result[z*128:(z+1)*128,y*128:(y+1)*128,x*128:(x+1)*128]==100*z+10*y+x).all())
 def test_missing_chunk_rejected(self):
  with self.assertRaises(KeyError):assemble({},[0,0,0])
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));save(HERE/'ACQUISITION_TESTS.json',dict(tests_run=r.testsRun,passed=r.testsRun-len(r.failures)-len(r.errors),failed=len(r.failures),errors=len(r.errors),new_network_reads=0,new_model_calls=0));raise SystemExit(not r.wasSuccessful())
