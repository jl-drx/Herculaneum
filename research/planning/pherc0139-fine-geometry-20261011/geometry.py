"""Pinned-source group0/group1 geometry; no CT/labels/predictions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,importlib.util,itertools,json
import numpy as np
import tifffile
R=Path(__file__).resolve().parent
I=R/'intake'
F=np.float32
S=F(.05000000074505806)

def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
N=load_module('retained_native_geometry',I/'sources/native_geometry.py')

def axis(coords,count,canvas,render_scale):
    """Exact staged FP32 branch and its multiply-add index ambiguity check."""
    coords=np.asarray(coords,dtype=np.int64);tile=(coords//128)*128;local=coords-tile
    r=F(render_scale);s=F(float(S)/float(r))
    center=F(float(count)/2/float(S));origin=F(F(-.5)*F(F(canvas)-F(1)))
    off=F(origin+tile.astype(np.float32));ul=F(F(F(off/r)+center)*S)
    ox=F(ul.astype(float)-4*float(s));q=F(ox+F((local+4).astype(np.float32)*s))
    alt=F(ox.astype(float)+(local+4).astype(float)*float(s))
    i=np.floor(q).astype(int);near=np.floor(q.astype(float)+.5).astype(int)
    stable=(i==np.floor(alt).astype(int))&(near==np.floor(alt.astype(float)+.5).astype(int))
    return q,i,near,stable

def render(mesh,cache,xs,ys,canvas,group):
    r=2.**(-group);h,w=mesh.shape[:2]
    xx,ix,nx,ux=axis(xs,w,canvas[0],r);yy,iy,ny,uy=axis(ys,h,canvas[1],r)
    safe_x=(ix>=0)&(ix+1<w)&(nx>0)&(nx<w-1)&ux
    safe_y=(iy>=0)&(iy+1<h)&(ny>0)&(ny<h-1)&uy
    # Clipping protects indexing only. All out-of-contract samples remain invalid.
    ix=np.clip(ix,0,w-2);iy=np.clip(iy,0,h-2);nx=np.clip(nx,0,w-1);ny=np.clip(ny,0,h-1)
    valid,normal,nok=cache
    p00=mesh[iy[:,None],ix[None,:]];p01=mesh[iy[:,None],(ix+1)[None,:]]
    p10=mesh[(iy+1)[:,None],ix[None,:]];p11=mesh[(iy+1)[:,None],(ix+1)[None,:]]
    wx=F(xx-ix.astype(np.float32))[None,:,None];wy=F(yy-iy.astype(np.float32))[:,None,None]
    top=F(F(p00*F(F(1)-wx))+F(p01*wx));bottom=F(F(p10*F(F(1)-wx))+F(p11*wx))
    # The renderer scales coordinates to the source level after interpolation;
    # unit normals are NOT multiplied by that factor.
    base=F(F(F(top*F(F(1)-wy))+F(bottom*wy))*F(r))
    direction=normal[ny[:,None],nx[None,:]]
    good=safe_y[:,None]&safe_x[None,:]&valid[iy[:,None],ix[None,:]]&valid[iy[:,None],(ix+1)[None,:]]
    good &= valid[(iy+1)[:,None],ix[None,:]]&valid[(iy+1)[:,None],(ix+1)[None,:]]&nok[ny[:,None],nx[None,:]]
    good &= np.isfinite(base).all(-1)&np.isfinite(direction).all(-1)
    return base.astype(float),direction.astype(float),good

def asset(cat,w,kind,vol):
    matches=[a for a in cat['segments'][w]['data'] if a['type']==kind and (a.get('parameters') or {}).get('target_volume')==vol]
    assert len(matches)==1,(w,kind,vol,len(matches));return matches[0]

def load(w,kind):
    vol={'native':'20250728140407','fine78':'20260102150214','fine59':'20260413113053'}[kind]
    cat=read(I/'CATALOG_EXTRACT.json');metadata=I/'metadata'
    attrs=read(metadata/f'{w}-{kind}-attrs.json');arr=read(metadata/f'{w}-{kind}-array.json')
    meta=read(metadata/f'{w}-{kind}-mesh-meta.json');assert meta['scale']==[float(S)]*2
    group=attrs['source_group'];assert group==(1 if kind=='fine59' else 0)
    re=asset(cat,w,'layers-zarr',vol);me=asset(cat,w,'tifxyz-transformed',vol)
    ps=re['creation_info']['provenance']['parameters']
    expected={'scale':'1','group-idx':str(group),'rotate':'0','flip':'-1','flip-normals':'true','crop-x':'0','crop-y':'0','crop-width':'0','crop-height':'0','affine-matrix':'','affine-transform':'','affines':'','scale-segmentation':''}
    assert all(ps.get(k)==v for k,v in expected.items())
    commit='650bd56663efb7225fdba73fe267ba86b8e29368' if kind=='fine78' else '530584554c031c982ba3aa243e1ea39c4bef9818'
    assert ps['commit']==commit
    d=I/'coordinates' if kind=='native' else R/'coordinates'
    xyz=[tifffile.imread(d/f'{w}-{kind}-{a}.tif') for a in 'xyz']
    assert all(a.dtype==np.float32 and a.shape==xyz[0].shape for a in xyz)
    mesh=np.stack(xyz,-1);canvas=attrs['canvas_size'];r=2**(-group)
    assert canvas==[int(np.floor(mesh.shape[1]*r/float(S)+.5)),int(np.floor(mesh.shape[0]*r/float(S)+.5))]
    assert arr['shape'][1:]==canvas[::-1] and arr['shape'][0]==attrs['num_slices'] and attrs['slice_step']==1
    cache=N.normals(mesh)
    normerror=float(np.max(np.abs(np.linalg.norm(cache[1][cache[2]].astype(float),axis=-1)-1)))
    assert normerror<1e-5
    return dict(mesh=mesh,cache=cache,canvas=canvas,group=group,slices=attrs['num_slices'],volume=vol,
                render_asset=re,mesh_asset=me,normal_norm_max_error=normerror,
                source_shape_xyz_conservative=(np.array(cat['volumes'][vol]['properties']['shape'][::-1])//(2**group)).tolist())

def to_native(cat,vol):
    if vol=='20250728140407':return np.eye(4)
    t=cat['volumes'][vol]['properties']['transforms'];assert len(t)==1
    m=np.eye(4);m[:3]=np.asarray(t[0]['transformation_matrix'])
    return to_native(cat,t[0]['to_volume_id'])@m

def mapped_axes(native,fine,xywh,naive=False):
    x,y,w,h=xywh;out=[]
    for j,coords in enumerate([np.arange(x,x+w),np.arange(y,y+h)]):
        nc=native['mesh'].shape[1-j];fc=fine['mesh'].shape[1-j]
        if naive:
            z=(coords+.5)*fine['canvas'][j]/native['canvas'][j]-.5
        else:
            q=axis(coords,nc,native['canvas'][j],1)[0].astype(float)
            qf=(q+.5)*fc/nc-.5
            z=(qf-fc/2)*(2**(-fine['group']))/float(S)+(fine['canvas'][j]-1)/2
        out.append(z)
    return out

def sampled_field(fine,fx,fy):
    ix=np.floor(fx).astype(int);iy=np.floor(fy).astype(int)
    ux=np.unique(np.r_[ix,ix+1]);uy=np.unique(np.r_[iy,iy+1])
    base,normal,good=render(fine['mesh'],fine['cache'],ux,uy,fine['canvas'],fine['group'])
    ax=np.searchsorted(ux,ix);bx=np.searchsorted(ux,ix+1);ay=np.searchsorted(uy,iy);by=np.searchsorted(uy,iy+1)
    wx=(fx-ix)[None,:,None];wy=(fy-iy)[:,None,None]
    def mix(z):return (z[ay[:,None],ax[None,:]]*(1-wx)+z[ay[:,None],bx[None,:]]*wx)*(1-wy)+(z[by[:,None],ax[None,:]]*(1-wx)+z[by[:,None],bx[None,:]]*wx)*wy
    ok=good[ay[:,None],ax[None,:]]&good[ay[:,None],bx[None,:]]&good[by[:,None],ax[None,:]]&good[by[:,None],bx[None,:]]
    return mix(base),mix(normal),ok

def grid(length,tile=256,stride=128):
    values=list(range(0,max(1,length-tile+1),stride));end=max(0,length-tile)
    if not values or values[-1]!=end:values.append(end)
    return values

def footprint(fine,fx,fy,verified_grid):
    samples=[np.unique(np.r_[np.floor(v).astype(int),np.floor(v).astype(int)+1]) for v in [fx,fy]]
    if verified_grid:
        starts=[]
        for v,size in zip(samples,fine['canvas']):
            starts.append([p for p in grid(size) if ((v>=p)&(v<p+256)).any()])
        if not all(starts):return None
        rect=[min(starts[0]),min(starts[1]),max(starts[0])+256,max(starts[1])+256]
        return dict(rect_xyxy=rect,tiles_xy=[[x,y] for y in starts[1] for x in starts[0]],kind='source_v42_conservative_full_tile_union')
    rect=[int(samples[0].min())-255,int(samples[1].min())-255,int(samples[0].max())+256,int(samples[1].max())+256]
    return dict(rect_xyxy=rect,tiles_xy=None,kind='metadata_only_any_phase_tile_envelope_v43_unverified')

def context_bound(fine,rect,depth_range,matrix):
    x0,y0,x1,y1=rect;h,w=fine['mesh'].shape[:2];canvas=fine['canvas'];group=fine['group'];r=2.**(-group)
    if x0<0 or y0<0 or x1>canvas[0] or y1>canvas[1]:return dict(complete_geometry=False,reason='outside_render_canvas')
    xx,ix,nx,ux=axis(np.arange(x0,x1),w,canvas[0],r);yy,iy,ny,uy=axis(np.arange(y0,y1),h,canvas[1],r)
    bx=np.unique(np.r_[ix,ix+1]);by=np.unique(np.r_[iy,iy+1]);vx=np.unique(nx);vy=np.unique(ny)
    if not(ux.all() and uy.all() and bx.min()>=0 and bx.max()<w and by.min()>=0 and by.max()<h and vx.min()>0 and vx.max()<w-1 and vy.min()>0 and vy.max()<h-1):
        return dict(complete_geometry=False,reason='index_or_rounding_ambiguity')
    valid,normal,nok=fine['cache']
    if not(valid[np.ix_(by,bx)].all() and nok[np.ix_(vy,vx)].all()):
        return dict(complete_geometry=False,reason='incomplete_mesh_or_normal_stencil')
    points=fine['mesh'][np.ix_(by,bx)].astype(float).reshape(-1,3)*r
    centre=(fine['slices']-1)/2;d=max(abs(depth_range[0]-centre),abs(depth_range[1]-1-centre))
    pad=d*1.00001+1.01+64*np.finfo(np.float32).eps*np.max(np.abs(points),axis=0)
    box=np.array([points.min(0)-pad,points.max(0)+pad]);shape=np.asarray(fine['source_shape_xyz_conservative'])
    extent=bool((box[0]>=0).all() and (box[1]<shape).all())
    corners=np.array(list(itertools.product(*zip(box[0],box[1]))))*(2**group)
    native=corners@matrix[:3,:3].T+matrix[:3,3]
    return dict(complete_geometry=True,within_catalog_source_extent=extent,
                source_level_shape_xyz_conservative=shape.tolist(),bbox_source_level_xyz=box.tolist(),
                bbox_native_xyz=[native.min(0).tolist(),native.max(0).tolist()],depth_half_open=depth_range,
                centre_depth=centre,source_group=group,margin_per_axis_source_level=pad.tolist())

def intersects(a,b):
    a,b=np.array(a),np.array(b);return bool((a[0]<=b[1]).all() and (b[0]<=a[1]).all())
