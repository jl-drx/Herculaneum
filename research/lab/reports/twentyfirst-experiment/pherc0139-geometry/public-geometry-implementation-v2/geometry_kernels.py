"""Bounded public XYZ computational-support kernels; staged FP32, no CT sampling."""
import numpy as np
F=np.float32
class ContractError(ValueError):pass
def require(x,message):
 if not x:raise ContractError(message)
def axis_map(mesh_count,canvas_count,scale,first,length):
 require(first%128==0 and length%128==0,'render tile alignment');s=F(scale);center=F(float(mesh_count)/2/float(s));a=[];b=[]
 for tile in range(first,first+length,128):
  origin=F(F(-.5)*F(F(canvas_count)-F(1)));offset=F(F(origin+F(0))+F(tile));ul=F(F(F(offset/F(1))+center)*s);ox=F(float(ul)-4*float(s))
  for j in range(128):
   dx=F(j+4);a.append(F(ox+F(dx*s)));b.append(F(float(ox)+float(dx)*float(s)))
 a=np.array(a,dtype=np.float32);b=np.array(b,dtype=np.float32)
 require(np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite raster map')
 i0=np.floor(a.astype(np.float64)).astype(np.int64);i1=i0+1;near=np.floor(a.astype(np.float64)+.5).astype(np.int64)
 require(np.array_equal(i0,np.floor(b.astype(np.float64)).astype(np.int64)) and np.array_equal(near,np.floor(b.astype(np.float64)+.5).astype(np.int64)),'FMA dependency ambiguity')
 require(i0.min()>=0 and i1.max()<mesh_count and near.min()>=1 and near.max()<mesh_count-1,'global mesh stencil boundary')
 lo=min(int(i0.min()),int(near.min())-1);hi=max(int(i1.max()),int(near.max())+1)
 return {'positions':a,'floor':i0,'upper':i1,'nearest':near,'weight':F(a-i0.astype(np.float32)),'dependency_inclusive':[lo,hi]}
def decode_window(body,request):
 x,y,w,h=request['window_xywh'];stride=request['row_stride_bytes'];require(len(body)==request['expected_body_bytes'],'body length for decoding');require(w==h==80 and stride==5776,'literal decode window')
 require((h-1)*stride+w*4==len(body),'strided range formula');view=memoryview(body);rows=[]
 for j in range(h):
  first=j*stride;last=first+w*4;require(0<=first<last<=len(body),'window slice outside range')
  # This is the only byte-to-coordinate conversion; gaps never enter frombuffer.
  rows.append(np.frombuffer(view[first:last],dtype='<f4').copy())
 result=np.stack(rows);require(result.shape==(80,80) and result.dtype==np.dtype('float32'),'decoded window dtype/shape');return result

def valid_vertices(xyz):
 return np.isfinite(xyz).all(axis=2)&(xyz[:,:,2]>F(0))&(xyz[:,:,0]!=F(-1))
def squared_norm(v):
 return F(F(F(v[...,0]*v[...,0])+F(v[...,1]*v[...,1]))+F(v[...,2]*v[...,2]))
def normal_cache(xyz,valid):
 require(xyz.dtype==np.float32 and xyz.ndim==3 and xyz.shape[2]==3,'XYZ dtype/shape')
 height,width=xyz.shape[:2];normal=np.zeros_like(xyz);ok=np.zeros((height,width),dtype=bool)
 center=valid[1:-1,1:-1];stencil=center&valid[1:-1,:-2]&valid[1:-1,2:]&valid[:-2,1:-1]&valid[2:,1:-1]
 with np.errstate(invalid='ignore',over='ignore',divide='ignore',under='ignore'):
  u=F(xyz[1:-1,2:]-xyz[1:-1,:-2]);v=F(xyz[2:,1:-1]-xyz[:-2,1:-1]);n=np.empty_like(u)
  n[...,0]=F(F(u[...,1]*v[...,2])-F(u[...,2]*v[...,1]));n[...,1]=F(F(u[...,2]*v[...,0])-F(u[...,0]*v[...,2]));n[...,2]=F(F(u[...,0]*v[...,1])-F(u[...,1]*v[...,0]));l2=squared_norm(n)
  good=stencil&np.isfinite(n).all(axis=2)&np.isfinite(l2)&(l2>F(0));safe=np.where(good,l2,F(1)).astype(np.float32)
  n=F(n*F(F(1)/F(np.sqrt(safe)))[...,None]);d=F(-n);l2b=squared_norm(d);good &= np.isfinite(d).all(axis=2)&np.isfinite(l2b)&(l2b>F(0));safe2=np.where(good,l2b,F(1)).astype(np.float32);d=F(d/F(np.sqrt(safe2))[...,None]);good &= np.isfinite(d).all(axis=2)
 normal[1:-1,1:-1]=np.where(good[...,None],d,F(0));ok[1:-1,1:-1]=good
 return normal,ok

def endpoint_support(base,direction,dims):
 dims=np.asarray(dims,dtype=np.float64);require(dims.shape==(3,) and np.all(dims>1),'native frame dimensions')
 with np.errstate(invalid='ignore',over='ignore',under='ignore'):
  first=F(base+F(direction*F(-33)));last=F(base+F(direction*F(34)))
 ok=np.isfinite(first).all(axis=2)&np.isfinite(last).all(axis=2)
 ok &= ((first.astype(np.float64)>=0)&(first.astype(np.float64)<dims-1)).all(axis=2)
 ok &= ((last.astype(np.float64)>=0)&(last.astype(np.float64)<dims-1)).all(axis=2)
 return ok

def fine_tile(xyz,valid,normals,normal_ok,ax,ay,window,tx,ty):
 x0,y0,w,h=window;slx=slice(tx,tx+128);sly=slice(ty,ty+128)
 ix0=ax['floor'][slx]-x0;ix1=ax['upper'][slx]-x0;iy0=ay['floor'][sly]-y0;iy1=ay['upper'][sly]-y0;nx=ax['nearest'][slx]-x0;ny=ay['nearest'][sly]-y0
 require(all(np.all((v>=0)&(v<w)) for v in (ix0,ix1,nx)) and all(np.all((v>=0)&(v<h)) for v in (iy0,iy1,ny)),'support expansion/index out of frozen window')
 require(np.all((nx>=1)&(nx<w-1)) and np.all((ny>=1)&(ny<h-1)),'normal stencil escaped frozen window')
 v00=valid[iy0[:,None],ix0[None,:]];v01=valid[iy0[:,None],ix1[None,:]];v10=valid[iy1[:,None],ix0[None,:]];v11=valid[iy1[:,None],ix1[None,:]];position_ok=v00&v01&v10&v11
 p00=xyz[iy0[:,None],ix0[None,:]];p01=xyz[iy0[:,None],ix1[None,:]];p10=xyz[iy1[:,None],ix0[None,:]];p11=xyz[iy1[:,None],ix1[None,:]]
 wx=ax['weight'][slx][None,:,None];wy=ay['weight'][sly][:,None,None];ix=F(F(1)-wx);iy=F(F(1)-wy)
 with np.errstate(invalid='ignore',over='ignore',under='ignore'):
  top=F(F(p00*ix)+F(p01*wx));bottom=F(F(p10*ix)+F(p11*wx));base=F(F(top*iy)+F(bottom*wy))
 position_ok &= np.isfinite(base).all(axis=2);direction=normals[ny[:,None],nx[None,:]];geom=position_ok&normal_ok[ny[:,None],nx[None,:]]&np.isfinite(direction).all(axis=2)
 return base,direction,position_ok,geom

def reconstruct(xyz,context,scale,dims,guard=lambda:None):
 require(xyz.shape==(80,80,3) and xyz.dtype==np.float32,'frozen XYZ window');require(context['selected_source_depth_half_open']==[21,89] and context['normal_offsets_inclusive']==[-33,34] and context['all_depth_count']==68,'depth contract')
 fine=context['fine_raster_bbox_xywh'];require(fine[2:]==[1536,1536],'fixed fine extent');window=context['mesh_window_xywh'];require(window[2:]==[80,80],'fixed mesh window')
 ax=axis_map(1444,28880,scale[0],fine[0],1536);ay=axis_map(1404,28080,scale[1],fine[1],1536)
 require(ax['dependency_inclusive']==[window[0],window[0]+79] and ay['dependency_inclusive']==[window[1],window[1]+79],'derived window differs from frozen plan')
 valid=valid_vertices(xyz);normal,nok=normal_cache(xyz,valid);mesh=np.zeros((384,384),dtype=bool);bounded=np.zeros_like(mesh);counts={'fine_total':0,'invalid_position_stencil_or_base':0,'invalid_normal_stencil_or_direction':0,'outside_declared_native_interpolation_bounds':0,'mesh_and_bounds_valid':0};tiles=0
 for ty in range(0,1536,128):
  for tx in range(0,1536,128):
   guard();base,direction,pok,geom=fine_tile(xyz,valid,normal,nok,ax,ay,window,tx,ty);good=geom&endpoint_support(base,direction,dims)
   counts['fine_total']+=128*128;counts['invalid_position_stencil_or_base']+=int(np.count_nonzero(~pok));counts['invalid_normal_stencil_or_direction']+=int(np.count_nonzero(pok&~geom));counts['outside_declared_native_interpolation_bounds']+=int(np.count_nonzero(geom&~good));counts['mesh_and_bounds_valid']+=int(np.count_nonzero(good))
   sy=slice(ty//4,(ty+128)//4);sx=slice(tx//4,(tx+128)//4);mesh[sy,sx]=geom.reshape(32,4,32,4).all(axis=(1,3));bounded[sy,sx]=good.reshape(32,4,32,4).all(axis=(1,3));tiles+=1
 require(tiles==144 and counts['fine_total']==2359296,'fine tile accounting');require(sum(v for k,v in counts.items() if k!='fine_total')==counts['fine_total'],'rejection partition');require(np.all(~bounded|mesh),'bounded support not subset')
 counts.update({'mesh_supported_level2':int(mesh.sum()),'mesh_and_bounds_supported_level2':int(bounded.sum()),'level2_total':147456,'tiles':144,'depth_bound_method':'two endpoints, staged FP32 monotonicity, all68 depths','native_voxels_observed':0})
 return mesh,bounded,counts
