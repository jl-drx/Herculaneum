"""Native geometry at fixed structural-score supports; no CT or predictions."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import json
import numpy as np
import tifffile
import pullback as pb

POSITIONS=[(48,48),(208,48),(48,208),(208,208),(128,48),(48,128),(208,128),(128,208)]

def basis(mesh,cache,x,y):
    p,n,ok=pb.render(mesh,cache,np.arange(x-1,x+2),np.arange(y-1,y+2),(7980,6400))
    assert ok.all()
    return np.stack((n[1,1],(p[2,1]-p[0,1])/2,(p[1,2]-p[1,0])/2),axis=1)

def main():
    mesh=np.stack([tifffile.imread(pb.GEO/f'20250728140407-{a}.tif') for a in 'xyz'],axis=-1);cache=pb.normals(mesh);results=[]
    for name,cx,cy in pb.SITES:
        central=basis(mesh,cache,cx,cy);samples=[]
        for lx,ly in POSITIONS:
            x,y=cx-128+lx,cy-128+ly;J=basis(mesh,cache,x,y)
            c=central/np.linalg.norm(central,axis=0)[None,:];j=J/np.linalg.norm(J,axis=0)[None,:]
            samples.append({'core_local_xy':[lx,ly],'global_native_xy':[x,y],'basis_columns_depth_y_x_nativeXYZ':J.tolist(),'basis_column_angles_to_core_degrees':np.degrees(np.arccos(np.clip((c*j).sum(0),-1,1))).tolist(),'basis_column_lengths_nativevox':np.linalg.norm(J,axis=0).tolist(),'basis_column_length_ratios_to_core':(np.linalg.norm(J,axis=0)/np.linalg.norm(central,axis=0)).tolist(),'signed_det':float(np.linalg.det(J))})
        item={'site':name,'core_center_xy':[cx,cy],'core_basis_columns_depth_y_x_nativeXYZ':central.tolist(),'supports':samples,'normal_angle_max_degrees':max(s['basis_column_angles_to_core_degrees'][0] for s in samples),'interpretation':'each score support should use its local native basis for physical shift norm and normal/tangent projection; central basis is only a summary'}
        results.append(item)
    out={'kind':'geometry_only_fixed_support_bases','code_sha256':pb.sha(Path(__file__)),'pullback_code_sha256':pb.sha(Path(pb.__file__)),'support_positions_fixed_core_xy':POSITIONS,'input_mesh_hashes':{a:pb.sha(pb.GEO/f'20250728140407-{a}.tif') for a in 'xyz'},'sites':results}
    pb.write(pb.ROOT/'LOCAL_BASES.json',out)
    print(json.dumps([{'site':s['site'],'normal_angle_max_degrees':s['normal_angle_max_degrees']} for s in results],indent=2))

if __name__=='__main__':main()
