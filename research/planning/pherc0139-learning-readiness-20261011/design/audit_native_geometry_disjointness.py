"""Retained mesh geometry only; no CT/teacher intensity bytes or network."""
import importlib.util,pathlib,json,itertools
import numpy as np,tifffile
P=pathlib.Path('/workspace/herculaneum-lab/reports/pherc0139-correspondence-20261011/geometry-review/pullback/pullback.py')
spec=importlib.util.spec_from_file_location('pb',P);pb=importlib.util.module_from_spec(spec);spec.loader.exec_module(pb)
mesh=np.stack([tifffile.imread(pb.GEO/f'20250728140407-{a}.tif') for a in 'xyz'],axis=-1);cache=pb.normals(mesh)
out={'meaning':'Conservative axis-aligned native-XYZ boxes of all source voxel centres in each retained 272-square native domain, over z0..27; geometry only, not a CT registration certificate. Boxes expanded by 0.5 XYZ voxel to include interpolation-cell margin; this is not continuous-mesh self-intersection certification.','sites':[],'pairs':[]}
for name,x,y in pb.SITES:
 p,n,ok=pb.render(mesh,cache,np.arange(x-136,x+136),np.arange(y-136,y+136),(7980,6400));assert ok.all()
 ends=np.stack([p-13.5*n,p+13.5*n])
 out['sites'].append({'site':name,'native_xyz_min':(ends.min((0,1,2))-.5).tolist(),'native_xyz_max':(ends.max((0,1,2))+.5).tolist(),'all_geometry_valid':bool(ok.all())})
for a,b in itertools.combinations(out['sites'],2):
 gap=np.maximum(0,np.maximum(np.array(a['native_xyz_min'])-b['native_xyz_max'],np.array(b['native_xyz_min'])-a['native_xyz_max']))
 out['pairs'].append({'sites':[a['site'],b['site']],'axis_box_gaps_native_voxels':gap.tolist(),'box_gap_lower_bound_native_voxels':float(np.linalg.norm(gap)),'sampled_native_domains_disjoint':bool((gap>0).any())})
out['source_mesh_sha256']={a:pb.sha(pb.GEO/f'20250728140407-{a}.tif') for a in 'xyz'}
pathlib.Path(__file__).with_name('NATIVE_GEOMETRY_DISJOINTNESS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['pairs'],indent=2))
