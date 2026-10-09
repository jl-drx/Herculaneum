"""Synthetic classic TIFF JSON-key adapter checks; no real CT body or sample decoding."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import socket
import struct
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
IMPL=BASE/'production-pooling-v2'
def blocked(*a,**k):raise AssertionError('Network forbidden')
socket.socket.connect=socket.socket.connect_ex=socket.create_connection=blocked
sys.path.insert(0,str(IMPL))
from pooling_common import persisted_layout_matches,record
p=BASE/'retrieval-v2/probe_common.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='1aecc2ada825d60c67fe4f99db6ae629626ba7446de028456fd430e3bb6e2539'
s=importlib.util.spec_from_file_location('synthetic_TIFF_parser',p);parser=importlib.util.module_from_spec(s);s.loader.exec_module(parser)
def main():
 out=HERE/'synthetic-fixture-result.json';assert not out.exists()
 cases=[]
 for endian,marker in [('<',b'II'),('>',b'MM')]:
  header=marker+struct.pack(endian+'HI',42,8)
  fields=[(256,3,17),(257,4,3),(258,3,16),(262,3,1),(273,4,256),(279,4,102)]
  ifd=struct.pack(endian+'H',len(fields))
  for tag,kind,value in fields:
   raw=struct.pack(endian+'H',value)+b'\xa5\x5a'if kind==3 else struct.pack(endian+'I',value)
   ifd+=struct.pack(endian+'HHI',tag,kind,1)+raw
  ifd+=struct.pack(endian+'I',0)
  layout=parser.parse_tiff(header,ifd,1024,[3,17]);saved=json.loads(json.dumps(layout))
  assert [k for k,v in layout.items()if saved.get(k)!=v]==['tags']
  cases.append(endian+' historical mismatch reproduced')
  assert persisted_layout_matches(layout,saved)
  cases.append(endian+' complete saved JSON roundtrip accepted')
  for name,edit in [
   ('tag value',lambda x:x['tags']['256'].__setitem__('value_or_offset',x['tags']['256']['value_or_offset']+1)),
   ('tag count',lambda x:x['tags']['256'].__setitem__('count',2)),
   ('tag bytes',lambda x:x['tags']['256'].__setitem__('value_bytes_hex','00000000')),
   ('missing tag',lambda x:x['tags'].pop('256')),
   ('extra tag',lambda x:x['tags'].__setitem__('999',dict(x['tags']['256']))),
   ('byte order',lambda x:x.__setitem__('dtype','>u2'if endian=='<'else'<u2')),
   ('strip offset',lambda x:x.__setitem__('strip_offset',257)),
  ]:
   bad=copy.deepcopy(saved);edit(bad);assert not persisted_layout_matches(layout,bad)
   cases.append(endian+' rejects changed '+name)
  try:persisted_layout_matches(saved,saved)
  except ValueError:cases.append(endian+' rejects noninteger reconstructed keys')
  else:raise AssertionError('Raw parser key type guard missing')
  assert layout['tags'][256]['value_bytes_hex']==saved['tags']['256']['value_bytes_hex']
  assert layout['dtype']==endian+'u2'and layout['strip_bytes']==102
 result=dict(status='PASS',count=len(cases),cases=cases,source=[record(IMPL/'pooling_common.py'),record(IMPL/'run_pooling.py')],retained_parser=record(p),reviewer=record(__file__),actual_CT_reads=0,numeric_pixel_decodes=0,model_calls=0,network_calls=0)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(record(out)))
if __name__=='__main__':main()
