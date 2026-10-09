import json,pathlib,struct,math,socket
import tiff_metadata as T
import run_structural as R
ROOT=pathlib.Path(__file__).resolve().parent;out=ROOT/'tile-fixtures';out.mkdir(exist_ok=False)
def no(*a,**k):raise AssertionError('no network')
socket.socket.connect=no;socket.socket.connect_ex=no
cases=[]
def tile(order='<',counts=(1024,1024),offsets=(1024,2048),external=True):
 tags={256:(4,1,17 if external else 16),257:(4,1,16),258:(3,1,32),259:(3,1,1),262:(3,1,1),277:(3,1,1),284:(3,1,1),339:(3,1,3),322:(4,1,16),323:(4,1,16),324:(4,2 if external else 1,300 if external else 1024),325:(4,2 if external else 1,308 if external else counts[0])}
 raw=b''
 for tag,(typ,n,v) in tags.items():raw+=struct.pack(order+'HHI',tag,typ,n)+(struct.pack(order+'H',v)+b'\x11\x22' if typ==3 else struct.pack(order+'I',v))
 head=T.header((b'II' if order=='<' else b'MM')+struct.pack(order+'HI',42,8),4096);n=T.count(struct.pack(order+'H',len(tags)),head);doc=T.parse_ifd(raw+struct.pack(order+'I',0),head,n);lay=T.layout(doc);extra={324:struct.pack(order+'II',*offsets),325:struct.pack(order+'II',*counts)} if external else {};return T.finish(doc,lay,extra)
def check(name,fn,bad=False):
 try:fn()
 except T.ContractError:
  if not bad:raise
 else:
  if bad:raise AssertionError(name)
 cases.append({'name':name,'status':'PASS'})
for o in '<>':
 check(o+' inline tile',lambda o=o:tile(o,external=False))
 check(o+' external two tiles with edge padding',lambda o=o:tile(o))
check('overlapping tiles',lambda:tile(offsets=(1024,1536)),True)
check('tile count wrong padded size',lambda:tile(counts=(1024,64)),True)
check('tile outside object',lambda:tile(offsets=(1024,3500)),True)
scale=.05000000074505806
assert [math.floor(1444/scale+.5),math.floor(1404/scale+.5)]==[28880,28080]
cases.append({'name':'candidate mesh-to-raster arithmetic only','status':'PASS'})
r={'status':'PASS','count':len(cases),'cases':cases,'source':R.rec(ROOT/'tiff_metadata.py'),'fixture':R.rec(pathlib.Path(__file__)),'scope':'Synthetic metadata only; candidate1444x1404 is not asserted observed'}
f=out/'receipt.json';f.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(R.rec(f)))
