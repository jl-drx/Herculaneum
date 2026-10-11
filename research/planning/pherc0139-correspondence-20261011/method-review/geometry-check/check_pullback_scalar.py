#!/usr/bin/env python3
"""Independent scalar pooled-geometry evaluation of saved inverse queries."""
from pathlib import Path
from functools import lru_cache
import hashlib,json,math
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1];F=np.float32;S=F(.05)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def axis(pixel,count,canvas):
    # Scalar replay of renderer tile-origin construction and scratchhalo.
    tile=128*(pixel//128);within=pixel-tile
    centre=F((count/2)/float(S))
    first=F(F(-.5)*F(canvas-1))
    tilepos=F(F(F(first+F(tile))+centre)*S)
    start=F(float(tilepos)-4*float(S))
    p=F(start+F(F(within+4)*S))
    alternate=F(float(start)+(within+4)*float(S))
    assert math.floor(float(p))==math.floor(float(alternate))
    assert math.floor(float(p)+.5)==math.floor(float(alternate)+.5)
    return float(p)

def isvalid(p):return np.isfinite(p).all() and np.all(p[...,0]!=-1) and np.all(p[...,2]>0)

def ray(mesh,x,y,canvas):
    height,width=mesh.shape[:2];px=axis(x,width,canvas[0]);py=axis(y,height,canvas[1]);ix,iy=math.floor(px),math.floor(py)
    assert 0<=ix<width-1 and 0<=iy<height-1
    p=mesh[iy:iy+2,ix:ix+2];assert isvalid(p)
    wx,wy=F(px-ix),F(py-iy)
    top=F(F(p[0,0]*F(F(1)-wx))+F(p[0,1]*wx))
    bottom=F(F(p[1,0]*F(F(1)-wx))+F(p[1,1]*wx))
    base=F(F(top*F(F(1)-wy))+F(bottom*wy))
    nx,ny=math.floor(px+.5),math.floor(py+.5)
    assert 1<=nx<width-1 and 1<=ny<height-1
    assert isvalid(mesh[[ny,ny,ny,ny-1,ny+1],[nx,nx-1,nx+1,nx,nx]])
    ux=F(mesh[ny,nx+1]-mesh[ny,nx-1]);uy=F(mesh[ny+1,nx]-mesh[ny-1,nx])
    cross=np.array([F(F(ux[1]*uy[2])-F(ux[2]*uy[1])),F(F(ux[2]*uy[0])-F(ux[0]*uy[2])),F(F(ux[0]*uy[1])-F(ux[1]*uy[0]))],dtype=np.float32)
    def length(a):return F(np.sqrt(F(F(F(a[0]*a[0])+F(a[1]*a[1]))+F(a[2]*a[2]))))
    norm=F(cross*F(F(1)/length(cross)));norm=F(-norm);norm=F(norm/length(norm))
    assert np.isfinite(base).all() and np.isfinite(norm).all()
    return base.astype(np.float64),norm.astype(np.float64)

def main():
    inputs={}
    for entry in json.loads((BASE/'geometry/REQUEST_LEDGER.json').read_text()):
        path=BASE/'geometry/responses'/entry['name']
        assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256']
        if path.suffix=='.tif':inputs[str(path.relative_to(BASE))]=entry['sha256']
    fine=np.stack([tifffile.imread(BASE/'geometry/responses'/f'20260102150214-{a}.tif') for a in 'xyz'],axis=-1)
    native=np.stack([tifffile.imread(BASE/'geometry/responses'/f'20250728140407-{a}.tif') for a in 'xyz'],axis=-1)
    catalog_path=BASE/'metadata/CATALOG_EXTRACT.json';cat=json.loads(catalog_path.read_text());inputs[str(catalog_path.relative_to(BASE))]=digest(catalog_path)
    tr=[t for t in cat['volumes']['20260102150214']['properties']['transforms'] if t['to_volume_id']=='20250728140407'];assert len(tr)==1
    T=np.asarray(tr[0]['transformation_matrix']);A,b=T[:,:3],T[:,3]
    @lru_cache(maxsize=5000)
    def pooled(u,v):
        pp=[];nn=[]
        for dy in range(4):
            for dx in range(4):
                p,n=ray(fine,4*u+dx,4*v+dy,(31040,24880));pp.append(p);nn.append(n)
        # Explicit summean of16 individual scalar rendered rays, noauthorcode.
        p=sum(pp)/16;n=sum(nn)/16
        return A@p+b,A@n
    def point(q):
        u,v,k=q;iu,iv=math.floor(u),math.floor(v);du,dv=u-iu,v-iv
        bb=np.zeros(3);nn=np.zeros(3)
        for dx,dy,w in ((0,0,(1-du)*(1-dv)),(1,0,du*(1-dv)),(0,1,(1-du)*dv),(1,1,du*dv)):
            p,n=pooled(iu+dx,iv+dy);bb+=w*p;nn+=w*n
        return bb+(k-54)*nn
    records=[];failures=[]
    for site,cx,cy in (('site1',2432,1920),('site2',3968,3200),('site3',5632,4480)):
        directory=BASE/'geometry-review/pullback'/site
        receipt=json.loads((directory/'RESULTS.json').read_text())
        for e in receipt['output_files']:
            path=directory/e['name']
            assert path.stat().st_size==e['bytes'] and digest(path)==e['sha256'],path
            inputs[str(path.relative_to(BASE))]=e['sha256']
        q=np.load(directory/'queries_uvk.npy',mmap_mode='r');ok=np.load(directory/'support.npy',mmap_mode='r');rr=np.load(directory/'residual_native_voxels.npy',mmap_mode='r')
        for dz in (0,9,19):
            for iy in (8,136,263):
                for ix in (8,136,263):
                    assert ok[dz,iy,ix]
                    query=np.asarray(q[dz,iy,ix]);j=4+dz;x=cx-136+ix;y=cy-136+iy
                    p,n=ray(native,x,y,(7980,6400));target=p+(j-13.5)*n
                    pp=point(query);d=pp-target;norm=float(np.linalg.norm(d));stored=float(rr[dz,iy,ix]);difference=abs(norm-stored)
                    r={'site':site,'array_zyx':[dz,iy,ix],'native_jyx':[j,y,x],'fine_uvk':query.tolist(),'independent_residual_xyz':d.tolist(),'independent_residual_native_voxels':norm,'stored_residual_native_voxels':stored,'residual_difference':difference,'pass':norm<=.011 and difference<1e-8}
                    records.append(r)
                    if not r['pass']:failures.append(r)
    result={'status':'PASS' if not failures else 'FAIL','input_hashes':inputs,'selection':'Fixedarrayindices XY8,136,263; depth0,9,19, Cartesianproduct27per site;3sites81total. NoCT.',
      'n_checked':len(records),'n_failed':len(failures),'maximum_independent_residual_native_voxels':max(r['independent_residual_native_voxels'] for r in records),'maximum_residual_difference_from_author':max(r['residual_difference'] for r in records),'records':records,'failures':failures,'limitations':['Independent scalar evaluation ofrepresentative geometry,notCTsampling equivalence orphysical tissue registration.','Finite81-point check supplements authoredfull4.439million screen; itdoesnotindependently recheckeveryquery.','Scalar implementation shares declared source equations; itdoesnotimport theauthor evaluator.']}
    (ROOT/'PULLBACK_SCALAR_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','n_checked','n_failed','maximum_independent_residual_native_voxels','maximum_residual_difference_from_author')},indent=2))
    assert not failures
if __name__=='__main__':main()
