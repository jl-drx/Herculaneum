"""Offline PHerc0800 renderer geometry and conservative support operations."""
import numpy as np
from scipy.ndimage import minimum_filter


def reconstruct(mesh, scale, shape):
    """Archived centered canvas, 128 tile, halo4 and nearest-grid normal.

    No raw intensities. Float32 arithmetic preserves the retained point checker;
    a separate 0.01-voxel envelope accounts for renderer contraction differences.
    """
    h,w=shape;scale=np.asarray(scale,np.float32)
    xs=np.arange(w);ys=np.arange(h);c0=xs//128*128;r0=ys//128*128
    center=np.asarray([mesh.shape[1]/2/float(scale[0]),mesh.shape[0]/2/float(scale[1])],np.float32)
    def axis(p,tile,dim,k):
        offset=np.asarray(-.5*(dim-1)+tile,np.float32)
        ul=np.asarray(np.asarray(offset+center[k],np.float32)*scale[k],np.float32)
        origin=np.asarray(ul.astype(float)-4*float(scale[k]),np.float32)
        return np.asarray(origin+np.asarray(p-tile+4,np.float32)*scale[k],np.float32)
    gx=axis(xs,c0,w,0);gy=axis(ys,r0,h,1)
    base=np.full((h,w,3),np.nan,np.float32);normal=np.full_like(base,np.nan)
    bilinear=np.zeros((h,w),bool);valid=np.zeros_like(bilinear)
    good=np.isfinite(mesh).all(-1)&(mesh>0).all(-1)
    for start in range(0,h,128):
        X,Y=np.meshgrid(gx,gy[start:start+128]);ix=np.floor(X).astype(int);iy=np.floor(Y).astype(int)
        inside=(ix>=0)&(iy>=0)&(ix+1<mesh.shape[1])&(iy+1<mesh.shape[0])
        ixc=np.clip(ix,0,mesh.shape[1]-2);iyc=np.clip(iy,0,mesh.shape[0]-2)
        ok=inside&good[iyc,ixc]&good[iyc,ixc+1]&good[iyc+1,ixc]&good[iyc+1,ixc+1]
        wx=(X-ix)[...,None].astype(np.float32);wy=(Y-iy)[...,None].astype(np.float32)
        p00,p01,p10,p11=mesh[iyc,ixc],mesh[iyc,ixc+1],mesh[iyc+1,ixc],mesh[iyc+1,ixc+1]
        b=(p00*(np.float32(1)-wx)+p01*wx)*(np.float32(1)-wy)+(p10*(np.float32(1)-wx)+p11*wx)*wy
        nx=np.floor(X.astype(float)+.5).astype(int);ny=np.floor(Y.astype(float)+.5).astype(int)
        ni=(nx>=1)&(ny>=1)&(nx+1<mesh.shape[1])&(ny+1<mesh.shape[0])
        nx=np.clip(nx,1,mesh.shape[1]-2);ny=np.clip(ny,1,mesh.shape[0]-2)
        nok=ok&ni&good[ny,nx]&good[ny,nx-1]&good[ny,nx+1]&good[ny-1,nx]&good[ny+1,nx]
        u=mesh[ny,nx+1]-mesh[ny,nx-1];v=mesh[ny+1,nx]-mesh[ny-1,nx]
        n=np.cross(u,v);length=np.sqrt(n[...,0]**2+n[...,1]**2+n[...,2]**2)
        nok &= np.isfinite(length)&(length>0)
        with np.errstate(invalid='ignore',divide='ignore'):
            n=n*(np.float32(1)/length)[...,None];n=-n
            length=np.sqrt(n[...,0]**2+n[...,1]**2+n[...,2]**2);n=n/length[...,None]
        sl=np.s_[start:start+len(X)]
        base[sl]=np.where(ok[...,None],b,np.nan);normal[sl]=np.where(nok[...,None],n,np.nan)
        bilinear[sl]=ok;valid[sl]=nok
    return base,normal,bilinear,valid


def corner_envelope(base,normal):
    ends=np.stack([base.astype(float)-32.5*normal,base.astype(float)+32.5*normal])
    lo=np.floor(ends.min(0)-.01)
    hi=np.floor(ends.max(0)+.01)+1
    return lo,hi  # Inclusive integer source-corner bounds.


def context_core(support,radius=128):
    return minimum_filter(support.astype(np.uint8),size=2*radius+1,mode='constant',cval=0).astype(bool)


def largest_square(mask):
    yy,xx=np.where(mask)
    if not len(yy):return {'side':0,'box_xywh':None}
    y,x=int(yy.min()),int(xx.min());a=mask[y:yy.max()+1,x:xx.max()+1]
    sums=np.pad((~a).astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    def locations(n):return np.argwhere((sums[n:,n:]-sums[:-n,n:]-sums[n:,:-n]+sums[:-n,:-n])==0)
    low,high=1,min(a.shape)
    while low<high:
        mid=(low+high+1)//2
        if len(locations(mid)):low=mid
        else:high=mid-1
    iy,ix=locations(low)[0]
    return {'side':int(low),'box_xywh':[int(x+ix),int(y+iy),int(low),int(low)]}
