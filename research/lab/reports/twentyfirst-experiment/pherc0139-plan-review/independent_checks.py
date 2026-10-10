"""Independent proposal arithmetic checks; no scientific payload or transport."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import math
import platform
import socket
import struct
from fractions import Fraction
from pathlib import Path
import numpy as np

def denied(*args, **kwargs):
    raise AssertionError('Network forbidden in independent proposal review')
socket.socket = denied
socket.create_connection = denied
ROOT = Path('/workspace/herculaneum-lab/reports/twentyfirst-experiment')
BASE = ROOT / 'pherc0139-geometry'
PROP = BASE / 'public-geometry-plan'
OUT = ROOT / 'pherc0139-plan-review'
records = []
def check(name, condition, **details):
    if not bool(condition):
        raise AssertionError(name)
    records.append(dict(name=name, status='PASS', **details))
def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
def read_json(path):
    return json.loads(path.read_text())
plan = read_json(PROP / 'COORDINATE_QUERY_PLAN.json')
check('root specified proposal identity', pin(PROP / 'COORDINATE_QUERY_PLAN.json')['sha256'] == '4f180f4507e77c8460c5b3a57c0683e818c8b29957c1a6d75c70dc9d53f6623d')
manifest = read_json(PROP / 'PROPOSAL_MANIFEST.json')
for p in manifest['sources_and_evidence'] + plan['input_pins']:
    check('text/structural pin: ' + str(Path(p['path']).relative_to(BASE)), pin(Path(p['path'])) == p)
check('proposal cannot execute', plan['ready'] is False and plan['status'] == 'PROPOSAL_ONLY_NO_ACQUISITION_AUTHORIZED')

# Use struct rounding, independently of the author's NumPy implementation.
def f(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]
axis_receipts = []
for ctx in plan['contexts']:
    check(ctx['id'] + ' exact two reductions', ctx['fine_raster_bbox_xywh'] == [4*v for v in ctx['level2_bbox_xywh']])
    for axis in ('x', 'y'):
        declared = ctx['axis_' + axis]
        s = f(declared['float32_scale'])
        c = f(declared['mesh_count'] / 2.0 / s)
        origin = f(f(-0.5) * f(f(declared['canvas_count']) - f(1)))
        first, length = declared['fine_first'], declared['fine_length']
        check(ctx['id'] + axis + ' aligned source tile extent', first % 128 == length % 128 == 0)
        separate, fused = [], []
        exact_double = True
        for start in range(first, first+length, 128):
            tile_offset = f(origin + f(start))
            upper_left = f(f(tile_offset + c) * s)
            scratch_origin = f(float(upper_left) - 4.0 * s)
            for j in range(128):
                dx = f(j+4)
                separate.append(f(scratch_origin + f(dx*s)))
                unrounded = float(scratch_origin) + float(dx)*float(s)
                exact_double &= Fraction(unrounded) == Fraction(scratch_origin) + Fraction(dx)*Fraction(s)
                fused.append(f(unrounded))
        check(ctx['id'] + axis + ' exact double intermediary for one-FMA simulation', exact_double)
        bil = [math.floor(v) for v in separate + fused]
        nearest = [math.floor(v+0.5) for v in separate + fused]
        dependency = [min(min(bil), min(nearest)-1), max(max(bil)+1, max(nearest)+1)]
        expected = dict(float32_center=c, coordinate_min=min(separate+fused), coordinate_max=max(separate+fused),
                        bilinear_grid_min=min(bil), bilinear_grid_max=max(bil)+1,
                        nearest_grid_min=min(nearest), nearest_grid_max=max(nearest),
                        dependency_inclusive=dependency, dependency_count=dependency[1]-dependency[0]+1,
                        fused_vs_separate_position_max_abs=max(abs(a-b) for a,b in zip(separate,fused)),
                        fused_vs_separate_bilinear_indices_equal=all(math.floor(a)==math.floor(b) for a,b in zip(separate,fused)),
                        fused_vs_separate_nearest_indices_equal=all(math.floor(a+.5)==math.floor(b+.5) for a,b in zip(separate,fused)))
        check(ctx['id'] + axis + ' every declared footprint value', all(declared[k] == v for k,v in expected.items()))
        check(ctx['id'] + axis + ' full stencil interior', 0 <= dependency[0] < dependency[1] < declared['mesh_count'])
        axis_receipts.append(dict(context=ctx['id'], axis=axis, positions_per_path=len(separate), **expected))
    a,b = ctx['axis_x']['dependency_inclusive'],ctx['axis_y']['dependency_inclusive']
    check(ctx['id'] + ' Cartesian dependency window', ctx['mesh_window_xywh'] == [a[0], b[0], a[1]-a[0]+1, b[1]-b[0]+1])
    check(ctx['id'] + ' raw depth to normal offset mapping', list(range(*ctx['selected_source_depth_half_open'])) == list(range(21,89)) and [21-54,88-54] == ctx['normal_offsets_inclusive'] and ctx['all_depth_count'] == 68)

contracts = {a:read_json(BASE / 'structural/run' / (a+'.tif.contract.json')) for a in 'xyz'}
for req in plan['requests']:
    c = contracts[req['source']['name'][0]]
    check(req['id'] + ' structural source identity', req['source'] == c['source'])
    check(req['id'] + ' FP32 single strip layout', (c['byte_order'],c['width'],c['height'],c['bits_per_sample'],c['compression'],c['sample_format'],c['samples_per_pixel'],c['orientation'],c['storage_offsets'],c['storage_byte_counts']) == ('<',1444,1404,32,1,3,1,1,[8],[1444*1404*4]))
    x,y,w,h=req['window_xywh']
    lo=8+4*(y*1444+x)
    hi=8+4*((y+h-1)*1444+x+w)-1
    body=hi-lo+1
    check(req['id'] + ' byte range and reservation', req['range_inclusive']==[lo,hi] and req['expected_body_bytes']==body and req['reserved_body_bytes']==body+1)
    check(req['id'] + ' useful/opaque split', req['row_stride_bytes']==5776 and req['useful_coordinate_bytes']==w*h*4 and req['uninterpreted_gap_bytes']==body-w*h*4)
    check(req['id'] + ' within payload not metadata', 8 <= lo <= hi < 8+1444*1404*4 < c['object_size_bytes'])
    check(req['id'] + ' exactly last selected row ends response', (h-1)*5776+w*4 == body)
check('six requests, all quotas consistent', len(plan['requests'])==plan['maximum_attempts']==6 and plan['additional_header_requests']==0 and sum(r['expected_body_bytes'] for r in plan['requests'])==plan['derived_total_expected_body_bytes']==2739744 and sum(r['reserved_body_bytes'] for r in plan['requests'])==plan['derived_total_reserved_body_bytes']==2739750 < plan['maximum_total_reserved_response_body_bytes']==4*1024**2)
check('total coordinate/gap bytes', sum(r['useful_coordinate_bytes'] for r in plan['requests'])==153600 and sum(r['uninterpreted_gap_bytes'] for r in plan['requests'])==2586144)
check('finite declared resource caps', plan['decode_budget']['maximum_peak_rss_bytes']==512*1024**2 and plan['decode_budget']['maximum_output_bytes']==32*1024**2 and plan['external_wall_seconds']==240 and plan['per_request_timeout_seconds']==30)

# A decoder audit fixture that refuses all byte reads outside allowed rows.
class SyntheticRows:
    def __init__(self):
        self.raw = bytearray([255])*456624
        self.calls=[]
        for row in range(80):
            self.raw[row*5776:row*5776+320] = np.arange(row*80,(row+1)*80,dtype='<f4').tobytes()
    def __getitem__(self, key):
        assert isinstance(key,slice) and key.step is None
        assert key.start % 5776 == 0 and key.stop-key.start == 320 and 0 <= key.start//5776 < 80
        self.calls.append((key.start,key.stop))
        return self.raw[key]
opaque = SyntheticRows()
decoded = np.vstack([np.frombuffer(opaque[j*5776:j*5776+320],dtype='<f4') for j in range(80)])
check('gap bytes never passed to float decoder', len(opaque.calls)==80 and np.array_equal(decoded,np.arange(6400,dtype=np.float32).reshape(80,80)))

# Independent finite toy geometry implementing the conservative contract.
F=np.float32
def reconstruct(points, x, y):
    if not np.isfinite([x,y]).all(): return None
    iy,ix=math.floor(y),math.floor(x)
    ny,nx=math.floor(y+.5),math.floor(x+.5)
    h,w=points.shape[:2]
    if not (0<=ix< w-1 and 0<=iy<h-1 and 0<nx<w-1 and 0<ny<h-1): return None
    stencil={(iy,ix),(iy,ix+1),(iy+1,ix),(iy+1,ix+1),(ny,nx),(ny,nx-1),(ny,nx+1),(ny-1,nx),(ny+1,nx)}
    if any(not np.isfinite(points[r,c]).all() or points[r,c,2]<=0 or points[r,c,0]==-1 for r,c in stencil): return None
    wx,wy=F(F(x)-F(ix)),F(F(y)-F(iy))
    base=F(F(F(points[iy,ix]*F(1-wx))+F(points[iy,ix+1]*wx))*F(1-wy))+F(F(F(points[iy+1,ix]*F(1-wx))+F(points[iy+1,ix+1]*wx))*wy)
    u=F(points[ny,nx+1]-points[ny,nx-1]);v=F(points[ny+1,nx]-points[ny-1,nx])
    n=np.array([F(F(u[1]*v[2])-F(u[2]*v[1])),F(F(u[2]*v[0])-F(u[0]*v[2])),F(F(u[0]*v[1])-F(u[1]*v[0]))],dtype=F)
    norm2=F(F(F(n[0]*n[0])+F(n[1]*n[1]))+F(n[2]*n[2]))
    if not np.isfinite(norm2) or norm2<=0: return None
    n=-F(n*F(F(1)/F(np.sqrt(norm2))))
    second=F(F(F(n[0]*n[0])+F(n[1]*n[1]))+F(n[2]*n[2]))
    n=F(n/F(np.sqrt(second)))
    return (base,n) if np.isfinite(base).all() and np.isfinite(n).all() else None
yy,xx=np.indices((10,10),dtype=F)
plane=np.stack([xx+F(200),yy+F(300),np.full_like(xx,F(500))],axis=-1)
b,d=reconstruct(plane,F(4.25),F(5.75))
check('toy planar base and one normal flip', np.array_equal(b,[204.25,305.75,500]) and np.array_equal(d,[0,0,-1]))
for name,where,value in [('zero z',(5,4,2),0),('negative z',(5,4,2),-1),('sentinel x',(5,4,0),-1),('NaN',(5,4,1),np.nan),('infinity',(6,4,2),np.inf),('normal neighbor outside bilinear',(6,3,0),-1),('nearest center',(6,4,0),-1)]:
    bad=plane.copy();bad[where]=value
    check('reject '+name,reconstruct(bad,F(4.25),F(5.75)) is None)
check('reject normal grid border',reconstruct(plane,F(.1),F(5.75)) is None)
bad=plane.copy();bad[:,:,0]=F(200)
check('reject degenerate normal', reconstruct(bad,F(4.25),F(5.75)) is None)
reduced=np.ones((8,8),dtype=bool)
for j in range(4):
    for i in range(4):
        toy=reduced.copy();toy[j,i]=False
        pooled=toy.reshape(2,4,2,4).all(axis=(1,3))
        check('fine contributor '+str((j,i))+' participates',np.array_equal(pooled,[[False,True],[True,True]]))

# Enumerate all depths in independent random and exact native-boundary cases.
rng=np.random.default_rng(210139)
dims=np.array([26511,26511,76953],dtype=F)
bases=rng.uniform(-50,dims+50,size=(4096,3)).astype(F)
directions=rng.uniform(-1,1,size=(4096,3)).astype(F)
boundary=[np.zeros(3,dtype=F),dims-F(1),np.nextafter(dims-F(1),F(0)),np.full(3,np.nextafter(F(0),F(-1)),dtype=F)]
for base in boundary:
    for direction in [np.zeros(3,dtype=F),np.full(3,F(-0.0)),np.full(3,np.nextafter(F(0),F(1))),np.full(3,np.nextafter(F(0),F(-1))),np.array([1,-1,0],dtype=F)]:
        bases=np.vstack([bases,base]);directions=np.vstack([directions,direction])
offsets=np.arange(-33,35,dtype=F)
coords=(bases[:,None,:]+(directions[:,None,:]*offsets[None,:,None]).astype(F)).astype(F)
valid=np.isfinite(coords)&(coords>=F(0))&(coords<dims-F(1))
all_depths=valid.all(axis=(1,2));endpoints=valid[:,[0,-1],:].all(axis=(1,2))
check('all 68 depths equivalent to endpoints',np.array_equal(all_depths,endpoints), synthetic_cases=len(bases), depth_count=len(offsets), accepted_cases=int(all_depths.sum()))
check('every computed component between endpoints',np.all(coords>=np.minimum(coords[:,0,:],coords[:,-1,:])[:,None,:]) and np.all(coords<=np.maximum(coords[:,0,:],coords[:,-1,:])[:,None,:]))
check('declared frame fixed and not successor',plan['native_frame']['shape_xyz']==[26511,26511,76953] and plan['native_frame']['volume_id']=='20260102150214' and plan['native_frame']['automatic_successor_substitution'] is False)

receipt=dict(status='PASS',scope='Independent text/structural identity, synthetic index/geometry/bounds arithmetic only; no XYZ payload, CT, labels, models or network.', checks=len(records), records=records, axis_receipts=axis_receipts, python=platform.python_version(),numpy=np.__version__,review_source=pin(Path(__file__)),proposal_manifest=pin(PROP/'PROPOSAL_MANIFEST.json'),query_plan=pin(PROP/'COORDINATE_QUERY_PLAN.json'))
target=OUT/'independent-fixtures.json'
with target.open('x') as h: json.dump(receipt,h,indent=2);h.write('\n')
print(json.dumps(dict(status='PASS',checks=len(records),receipt=pin(target))))
