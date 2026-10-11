#!/usr/bin/env python3
"""Independent w040 geometry diagnostic. No CT, network, labels or model calls."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import tifffile

F = np.float32

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def valid(v):
    return bool(np.isfinite(v).all() and np.all(v[..., 0] != -1) and np.all(v[..., 2] > 0))

def bilinear(xyz, x, y):
    ix, iy = math.floor(x), math.floor(y)
    if not (0 <= ix < xyz.shape[1]-1 and 0 <= iy < xyz.shape[0]-1):
        return None
    b = xyz[iy:iy+2,ix:ix+2].astype(np.float64)
    if not valid(b):
        return None
    tx, ty = x-ix, y-iy
    return ((b[0,0]*(1-tx)+b[0,1]*tx)*(1-ty)
            +(b[1,0]*(1-tx)+b[1,1]*tx)*ty)

def render_normal(xyz, x, y):
    # Renderer nearest-cache convention, sign reversed by flip-normals=true.
    ix,iy=math.floor(x+.5),math.floor(y+.5)
    if not (1 <= ix < xyz.shape[1]-1 and 1 <= iy < xyz.shape[0]-1):
        return None
    support=xyz[[iy,iy,iy,iy-1,iy+1],[ix,ix-1,ix+1,ix,ix]]
    if not valid(support):
        return None
    u=F(xyz[iy,ix+1]-xyz[iy,ix-1])
    v=F(xyz[iy+1,ix]-xyz[iy-1,ix])
    n=np.array([F(F(u[1]*v[2])-F(u[2]*v[1])),
                F(F(u[2]*v[0])-F(u[0]*v[2])),
                F(F(u[0]*v[1])-F(u[1]*v[0]))],dtype=np.float32)
    l2=F(F(F(n[0]*n[0])+F(n[1]*n[1]))+F(n[2]*n[2]))
    if not np.isfinite(l2) or l2 <= 0:
        return None
    n=F(n*F(F(1)/F(np.sqrt(l2))))
    n=F(-n)
    length=F(np.sqrt(F(F(F(n[0]*n[0])+F(n[1]*n[1]))+F(n[2]*n[2]))))
    return F(n/length).astype(np.float64)

def angle(a,b):
    return float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))

def summary(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),
            'max':float(a.max())} if a.size else {'n':0}

def run(base):
    response=base/'geometry/responses'
    source=base/'geometry-review/sources'
    files={}
    receipts=json.loads((base/'geometry/REQUEST_LEDGER.json').read_text())
    for record in receipts:
        p=response/record['name']
        assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],p
        files[str(p.relative_to(base))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    for name in ('650bd56-AffineTransform.cpp','d9a472e-AffineTransform.cpp'):
        p=source/name
        files[str(p.relative_to(base))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    contract=Path(__file__).resolve().parent/'CONTRACT_SNAPSHOT.md'
    files[str(contract.relative_to(base))]={'bytes':contract.stat().st_size,'sha256':sha(contract)}
    catalog_path=base/'metadata/CATALOG_EXTRACT.json'
    files[str(catalog_path.relative_to(base))]={'bytes':catalog_path.stat().st_size,'sha256':sha(catalog_path)}
    catalog=json.loads(catalog_path.read_text())
    T=np.asarray(catalog['volumes']['20260102150214']['properties']['transforms'][0]['transformation_matrix'],dtype=np.float64)
    A,b=T[:,:3],T[:,3]
    fine=np.stack([tifffile.imread(response/f'20260102150214-{a}.tif') for a in 'xyz'],axis=-1)
    native=np.stack([tifffile.imread(response/f'20250728140407-{a}.tif') for a in 'xyz'],axis=-1)
    assert fine.shape==(1244,1552,3) and native.shape==(320,399,3)
    scale=float(np.float32(.05))
    nf=np.asarray(fine.shape[:2][::-1],dtype=np.float64)
    nn=np.asarray(native.shape[:2][::-1],dtype=np.float64)
    # Contract dimensions are exact published canvases, not int(mesh/FP32(.05)).
    Wn,Hn=7980,6400
    Wf,Hf=31040,24880
    ratios=nf/nn
    positions=[]
    for vy in (.1,.3,.5,.7,.9):
        for vx in (.1,.3,.5,.7,.9):
            # Nearest native output pixel to fixed normalized canvas fractions.
            x=math.floor(vx*Wn); y=math.floor(vy*Hn)
            # Real-arithmetic form of renderer scale/center equation. It preserves
            # stored FP32 scale and declared dimensions; no CT footprint is used.
            qn=np.array([scale*(x-(Wn-1)/2)+nn[0]/2,
                         scale*(y-(Hn-1)/2)+nn[1]/2])
            qf=(qn+.5)*ratios-.5
            # Alternative naive same-canvas resize: usual half-pixel resampling.
            xf=(x+.5)*Wf/Wn-.5; yf=(y+.5)*Hf/Hn-.5
            qbad=np.array([scale*(xf-(Wf-1)/2)+nf[0]/2,
                           scale*(yf-(Hf-1)/2)+nf[1]/2])
            pn=bilinear(native,*qn);pf=bilinear(fine,*qf);pbad=bilinear(fine,*qbad)
            r={'site':f'u{vx:.1f}-v{vy:.1f}','normalized_xy':[vx,vy],
               'native_render_xy':[x,y],'native_mesh_xy':qn.tolist(),
               'fine_mesh_xy_source_derived':qf.tolist(),
               'fine_mesh_xy_naive_canvas_resize':qbad.tolist(),
               'status':'invalid_coordinate_stencil'}
            if pn is None or pf is None:
                positions.append(r); continue
            delta=A@pf+b-pn
            r.update({'status':'coordinate_valid','native_xyz':pn.tolist(),
                      'mapped_fine_xyz':(A@pf+b).tolist(),
                      'source_derived_residual_native_voxels_xyz':delta.tolist(),
                      'source_derived_residual_norm_native_voxels':float(np.linalg.norm(delta))})
            if pbad is not None:
                d=A@pbad+b-pn
                r['naive_resize_residual_norm_native_voxels']=float(np.linalg.norm(d))
                r['naive_resize_residual_native_voxels_xyz']=d.tolist()
            fn=render_normal(fine,*qf);nnorm=render_normal(native,*qn)
            if fn is None or nnorm is None:
                r['normal_status']='invalid_normal_stencil';positions.append(r);continue
            cov=np.linalg.inv(A).T@fn;cov/=np.linalg.norm(cov)
            # det(A)>0, so no cross-product orientation reversal.
            assert np.linalg.det(A)>0
            mapped_per_native_voxel=(A@fn)*(9.362/2.399)
            discrepancy=mapped_per_native_voxel-nnorm
            rays=[]
            for d in (-13.5,-6,0,6,13.5):
                pointdiff=delta+d*discrepancy
                # Ray term excludes the base interpolation/regridding difference.
                rays.append({'native_offset_voxels':d,
                             'fine_layer_continuous_nominal':54+d*9.362/2.399,
                             'ray_only_discrepancy_native_voxels':float(abs(d)*np.linalg.norm(discrepancy)),
                             'base_plus_ray_discrepancy_native_voxels':float(np.linalg.norm(pointdiff)),
                             'base_plus_ray_vector_native_voxels':pointdiff.tolist()})
            r.update({'normal_status':'valid','fine_render_normal':fn.tolist(),
                      'native_render_normal':nnorm.tolist(),
                      'geometric_covector_vs_native_normal_angle_degrees':angle(cov,nnorm),
                      'affine_ray_vs_native_normal_angle_degrees':angle(A@fn,nnorm),
                      'affine_ray_length_per_nominal_native_voxel':float(np.linalg.norm(mapped_per_native_voxel)),
                      'residual_normal_component_native_voxels':float(delta@nnorm),
                      'residual_tangent_norm_native_voxels':float(np.linalg.norm(delta-(delta@nnorm)*nnorm)),
                      'ray_positions':rays})
            positions.append(r)
    coord=[p for p in positions if 'source_derived_residual_norm_native_voxels' in p]
    normals=[p for p in positions if p.get('normal_status')=='valid']
    result={'purpose':'geometry only; public mesh pipeline correspondence, not physical CT registration',
            'input_files':files,'affine_fine_to_native':T.tolist(),
            'source_mapping_derivation':'Both meshes resample the same source with half-pixel bilinear mapping. Composing source coordinates cancels unknown common grid dimensions: q_f=(q_n+0.5)*(N_f/N_n)-0.5. Points then interpolate each saved grid independently; two resamplings need not commute on curved bilinear geometry.',
            'normal_implementation':'FP32 centered +/-1 mesh differences, cross product, normalize, flip normal, renormalize; nearest mesh vertex floor(q+0.5). Geometry point interpolation is independent float64, not claimed bit-exact renderer arithmetic.',
            'fixed_sites':'native pixels nearest 5x5 normalized fractions .1,.3,.5,.7,.9; all25 retained, invalid stencils are unknown',
            'fine_mesh_zyx_channels_shape':list(fine.shape),
            'native_mesh_zyx_channels_shape':list(native.shape),
            'mesh_ratio_xy':ratios.tolist(),
            'extra_fine_render_pixels_vs_naive_resize_xy':((ratios-1)*.5/scale).tolist(),
            'extra_native_equivalent_pixels_vs_naive_resize_xy':(((ratios-1)*.5/scale)/ratios).tolist(),
            'summary':{'coordinate_valid_sites':len(coord),'normal_valid_sites':len(normals),
              'source_derived_residual_native_voxels':summary([p['source_derived_residual_norm_native_voxels'] for p in coord]),
              'naive_resize_residual_native_voxels':summary([p['naive_resize_residual_norm_native_voxels'] for p in coord if 'naive_resize_residual_norm_native_voxels' in p]),
              'geometric_normal_angle_degrees':summary([p['geometric_covector_vs_native_normal_angle_degrees'] for p in normals]),
              'affine_ray_angle_degrees':summary([p['affine_ray_vs_native_normal_angle_degrees'] for p in normals]),
              'ray_only_error_at_native_plus13_5':summary([p['ray_positions'][-1]['ray_only_discrepancy_native_voxels'] for p in normals]),
              'total_error_at_native_plus13_5':summary([p['ray_positions'][-1]['base_plus_ray_discrepancy_native_voxels'] for p in normals])},
            'sites':positions,
            'limitations':['No CT samples were read. No claim of physical tissue/ink registration.',
              'Common normalized source grid was not acquired; resampling composition follows exact pinned callers and half-pixel conventions. Observed residuals include curved-grid interpolation and float quantization.',
              '25 deterministic sites measure this finite coordinate check only; they do not qualify all surface pixels.',
              'Native and fine nearest-grid normal caches have different physical stencil widths. A scalar depth ratio does not guarantee matching 3D rays.',
              'A pooled fine XY pixel has a16-sample footprint; these representative point calculations do not turn it into an exact CT point.']}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('/workspace/herculaneum-lab/reports/pherc0139-correspondence-20261011'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.base);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2))
