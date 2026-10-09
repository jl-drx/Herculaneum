"""Synthetic numerical contract checks only. No network or retained arrays."""
import hashlib,json,pathlib,socket
import numpy as np
from derive_plan import axis_dependencies,rec
ROOT=pathlib.Path(__file__).resolve().parent;F=np.float32
def no_network(*a,**k):raise AssertionError('network forbidden')
socket.socket.connect=no_network;socket.socket.connect_ex=no_network
cases=[]
def case(name,fn):fn();cases.append({'name':name,'status':'PASS'})
def assert_(x):
 if not x:raise AssertionError('fixture assertion')
def vertex_ok(p):return bool(np.isfinite(p).all() and p[2]>F(0) and p[0]!=F(-1))
def projected(P,x,y):
 x0=int(np.floor(x));y0=int(np.floor(y));nx=int(np.floor(float(x)+.5));ny=int(np.floor(float(y)+.5))
 if x0<0 or y0<0 or x0+1>=P.shape[1] or y0+1>=P.shape[0] or nx<1 or ny<1 or nx>=P.shape[1]-1 or ny>=P.shape[0]-1:return None
 coords=[P[y0,x0],P[y0,x0+1],P[y0+1,x0],P[y0+1,x0+1]];normcoords=[P[ny,nx],P[ny,nx-1],P[ny,nx+1],P[ny-1,nx],P[ny+1,nx]]
 if not all(vertex_ok(p) for p in coords+normcoords):return None
 wx=F(x-F(x0));wy=F(y-F(y0));ix=F(1)-wx;iy=F(1)-wy
 b=F(F(F(coords[0]*ix)+F(coords[1]*wx))*iy)+F(F(F(coords[2]*ix)+F(coords[3]*wx))*wy)
 u=F(normcoords[2]-normcoords[1]);v=F(normcoords[4]-normcoords[3]);n=np.array([F(F(u[1]*v[2])-F(u[2]*v[1])),F(F(u[2]*v[0])-F(u[0]*v[2])),F(F(u[0]*v[1])-F(u[1]*v[0]))],dtype=np.float32)
 l2=F(F(F(n[0]*n[0])+F(n[1]*n[1]))+F(n[2]*n[2]))
 if not np.isfinite(l2) or l2<=0:return None
 n=F(n*F(F(1)/F(np.sqrt(l2))));d=F(-n);l2=F(F(F(d[0]*d[0])+F(d[1]*d[1]))+F(d[2]*d[2]));d=F(d/F(np.sqrt(l2)))
 if not np.isfinite(b).all() or not np.isfinite(d).all():return None
 return b,d
DIMS=np.array([100,100,100],dtype=np.float64)
def depths(b,d):return np.stack([F(b+F(d*F(o))) for o in range(-33,35)])
def endpoint_valid(b,d,dims=DIMS):
 e=np.stack([F(b+F(d*F(o))) for o in (-33,34)])
 return bool(np.isfinite(e).all() and np.all(e.astype(np.float64)>=0) and np.all(e.astype(np.float64)<dims-1))
def full_valid(b,d,dims=DIMS):
 a=depths(b,d);return bool(np.isfinite(a).all() and np.all(a.astype(np.float64)>=0) and np.all(a.astype(np.float64)<dims-1))
Y,X=np.mgrid[:9,:9];P=np.stack([X+40,Y+40,np.full_like(X,50)],axis=-1).astype(np.float32)
case('planar bilinear position',lambda:assert_(np.array_equal(projected(P,F(3.25),F(4.75))[0],np.array([43.25,44.75,50],dtype=np.float32))))
case('planar normal sign flipped once',lambda:assert_(np.array_equal(projected(P,F(3.25),F(4.75))[1],np.array([0,0,-1],dtype=np.float32))))
case('planar all68depth support',lambda:assert_(endpoint_valid(*projected(P,F(3.25),F(4.75)))))
for name,idx,value in [('zeroZbilinear',(4,3,2),0),('minusOneXbilinear',(4,3,0),-1),('NaNbilinear',(4,3,1),np.nan),('Infnearestnormal',(5,2,0),np.inf),('invalidnormalneighboroutsidebilinear',(6,3,2),0)]:
 Q=P.copy();Q[idx]=value;case(name,lambda Q=Q:assert_(projected(Q,F(3.25),F(4.75)) is None))
Q=P.copy();Q[:,:,0]=F(40);case('degenerate normal',lambda:assert_(projected(Q,F(3.25),F(4.75)) is None))
case('normal grid border rejected',lambda:assert_(projected(P,F(.1),F(.1)) is None))
case('strict full CTcorner boundary rejects nminus1',lambda:assert_(not endpoint_valid(np.array([99,50,50],dtype=np.float32),np.zeros(3,dtype=np.float32))))
case('CTboundary exact zero with zero direction accepted',lambda:assert_(endpoint_valid(np.array([0,50,50],dtype=np.float32),np.zeros(3,dtype=np.float32))))
case('CTboundary nextfloat below zero rejected',lambda:assert_(not endpoint_valid(np.array([np.nextafter(F(0),F(-1)),50,50],dtype=np.float32),np.zeros(3,dtype=np.float32))))
case('CTboundary nextfloat below nminus1 accepted',lambda:assert_(endpoint_valid(np.array([np.nextafter(F(99),F(0)),50,50],dtype=np.float32),np.zeros(3,dtype=np.float32))))
def endpoint_equivalence():
 rng=np.random.default_rng(210139)
 for _ in range(2048):
  b=rng.uniform(-5,105,3).astype(np.float32);d=rng.uniform(-1,1,3).astype(np.float32);a=depths(b,d);ends=a[[0,-1]]
  assert np.all(a>=ends.min(axis=0)) and np.all(a<=ends.max(axis=0));assert endpoint_valid(b,d)==full_valid(b,d)
case('all68endpoint equivalence2048fixedsyntheticcases',endpoint_equivalence)
def coarse():
 v=np.ones((8,8),dtype=bool);v[3,3]=False;q=v.reshape(2,4,2,4).all(axis=(1,3));assert np.array_equal(q,np.array([[False,True],[True,True]]))
case('all16fineXYconstituents not center-only',coarse)
def row_decode():
 # Synthetic marker bytes in every skipped gap are never passed to float decoding.
 w,h,stride=3,4,11;raw=bytearray([255]*((h-1)*stride*4+w*4))
 for j in range(h):raw[j*stride*4:j*stride*4+w*4]=np.array([j,j+.25,j+.5],dtype='<f4').tobytes()
 got=np.stack([np.frombuffer(memoryview(raw)[j*stride*4:j*stride*4+w*4],dtype='<f4') for j in range(h)])
 assert got.shape==(h,w) and np.all(np.isfinite(got)) and got[-1,-1]==F(3.5)
case('decode only exact row windows gaps opaque',row_decode)
for name,mesh,canvas,first,expected in [('rank01X',1444,28880,7168,[357,436]),('rank01Y',1404,28080,19456,[972,1051]),('rank02X',1444,28880,14336,[716,795]),('rank02Y',1404,28080,20480,[1023,1102])]:
 d=axis_dependencies(mesh,canvas,.05000000074505806,first,1536);case(name+'exactFP32stencil',lambda d=d,e=expected:assert_(d['dependency_inclusive']==e and d['fused_vs_separate_nearest_indices_equal'] and d['fused_vs_separate_bilinear_indices_equal']))
out=ROOT/'synthetic-fixtures.json';out.write_text(json.dumps({'status':'PASS','scope':'Synthetic coordinates/index arithmetic only, sockets forbidden; no real coordinate/CT/model/label arrays read','case_count':len(cases),'cases':cases,'sources':[rec(pathlib.Path(__file__)),rec(ROOT/'derive_plan.py')],'numpy_version':np.__version__},indent=2)+'\n');print(json.dumps(rec(out)))
