"""Geometry-only prospective intake; offline and relocatable."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text())

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')

def module(name, p):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

G = module('geometry', ROOT / 'sources/native_geometry.py')
A = module('audit', ROOT / 'sources/audit_geometry.py')

def hits(box, reserved):
    return [r['core_id'] for r in reserved if A.intersects(box, r['bbox_conservative_xyz'])]

def fast_box(mesh, cache, x, y, canvas, size=256):
    """Conservative vertex hull, complete stencils; no CT or label values."""
    h, w = mesh.shape[:2]
    xs, ys = np.arange(x, x+size), np.arange(y, y+size)
    xx, ix, nx, ux = G.axis(xs, w, canvas[0])
    yy, iy, ny, uy = G.axis(ys, h, canvas[1])
    ax = (xs-.5*(canvas[0]-1)+w/(2*float(G.S)))*float(G.S)
    ay = (ys-.5*(canvas[1]-1)+h/(2*float(G.S)))*float(G.S)
    jx, jy = np.floor(ax).astype(int), np.floor(ay).astype(int)
    bx = np.unique(np.r_[ix, ix+1, jx, jx+1])
    by = np.unique(np.r_[iy, iy+1, jy, jy+1])
    if not (ux.all() and uy.all() and bx.min() >= 0 and bx.max() < w
            and by.min() >= 0 and by.max() < h and nx.min() > 0 and nx.max() < w-1
            and ny.min() > 0 and ny.max() < h-1):
        return None
    valid, normals, nok = cache
    if not (valid[np.ix_(by, bx)].all() and nok[np.ix_(np.unique(ny), np.unique(nx))].all()):
        return None
    points = mesh[np.ix_(by, bx)].astype(float).reshape(-1, 3)
    # Bilinear convex hull plus64 single-precision epsilons at source magnitude
    # safely exceeds this renderer's weighted-sum roundoff. Exact selected boxes
    # and all28 rays are separately checked below.
    pad = 14.51 + 64*np.finfo(np.float32).eps*np.max(np.abs(points), axis=0)
    return [(points.min(0)-pad).tolist(), (points.max(0)+pad).tolist()]

def exact_box(mesh, cache, x, y, canvas, size=256):
    xs, ys = np.arange(x, x+size), np.arange(y, y+size)
    base, normal, valid = G.render(mesh, cache, xs, ys, canvas)
    alt, av = A.alternative_base(mesh, xs, ys, canvas, float(G.S))
    valid &= av
    assert valid.all(), 'Selected context fails exact complete support'
    box, rays, _ = A.boxes(base, alt, normal, valid)
    return dict(bbox_conservative_xyz=box, bbox_endpoint_ray_plus1p01_xyz=rays,
                geometry_valid_pixels=int(valid.sum()),
                staged_vs_float64_max_absolute_difference=float(np.max(np.abs(base-alt))))

def load(w):
    cat = read(ROOT / 'CATALOG_EXTRACT.json')
    render = next(a for a in cat['segments'][w]['data'] if a['type'] == 'layers-zarr'
                  and (a.get('parameters') or {}).get('target_volume') == '20250728140407')
    params = render['creation_info']['provenance']['parameters']
    expected = dict(commit='530584554c031c982ba3aa243e1ea39c4bef9818', scale='1',
                    **{'group-idx':'0','num-slices':'28','rotate':'0','flip':'-1',
                       'flip-normals':'true','crop-x':'0','crop-y':'0','crop-width':'0',
                       'crop-height':'0','affine-matrix':'','affine-transform':'',
                       'affines':'','scale-segmentation':''})
    assert all(params.get(k) == v for k, v in expected.items())
    attrs = read(ROOT / 'metadata' / f'{w}-native-attrs.json')
    array = read(ROOT / 'metadata' / f'{w}-native-array.json')
    meta = read(ROOT / 'metadata' / f'{w}-native-mesh-meta.json')
    assert attrs['num_slices'] == 28 and attrs['source_group'] == 0 and attrs['slice_step'] == 1
    assert array['chunks'] == [28,128,128] and array['compressor'] is None and array['dtype'] == '|u1'
    assert meta['scale'] == [float(G.S)]*2
    xyz = [tifffile.imread(ROOT / 'coordinates' / f'{w}-native-{axis}.tif') for axis in 'xyz']
    assert all(a.shape == xyz[0].shape and a.dtype == np.float32 for a in xyz)
    mesh = np.stack(xyz, -1)
    canvas = attrs['canvas_size']
    assert array['shape'][1:] == canvas[::-1]
    assert canvas == [int(round(mesh.shape[1]/float(G.S))), int(round(mesh.shape[0]/float(G.S)))]
    cache = G.normals(mesh)
    assert np.max(np.abs(np.linalg.norm(cache[1][cache[2]].astype(float),axis=1)-1)) < 1e-5
    return mesh, cache, canvas, render

def select(w, count, reserved):
    mesh, cache, canvas, _ = load(w)
    candidates = []
    for y in range(0, canvas[1]-255, 256):
        for x in range(0, canvas[0]-255, 256):
            box = fast_box(mesh, cache, x, y, canvas)
            matches = [] if box is None else hits(box, reserved)
            candidates.append(dict(core_id=f'{w}-x{x+64}-y{y+64}', winding=w,
                core_xywh=[x+64,y+64,128,128], context_xywh=[x,y,256,256],
                vertex_bound_xyz=box, native_geometry_eligible=box is not None,
                intersecting_reserved_ids=matches, eligible=box is not None and not matches))
    eligible = [r for r in candidates if r['eligible']]
    if not eligible:
        return [], candidates
    centres = np.array([np.mean(r['vertex_bound_xyz'], axis=0) for r in eligible])
    min_distance = np.full(len(eligible), np.inf)
    available = np.ones(len(eligible), bool)
    selected = []
    for k in range(min(count,len(eligible))):
        if not k:
            i = int(np.argmin(np.sum((centres-centres.mean(0))**2,axis=1)))
        else:
            i = int(np.argmax(np.where(available, min_distance, -1)))
        available[i] = False
        row = dict(eligible[i], selection_order=k)
        x,y,_,_ = row['context_xywh']
        exact = exact_box(mesh,cache,x,y,canvas)
        vertex = np.array(row['vertex_bound_xyz']); box = np.array(exact['bbox_conservative_xyz'])
        assert np.all(box[0] >= vertex[0]) and np.all(box[1] <= vertex[1])
        assert not hits(exact['bbox_conservative_xyz'],reserved)
        row.update(exact)
        selected.append(row)
        min_distance = np.minimum(min_distance,np.sum((centres-centres[i])**2,axis=1))
    return selected,candidates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output', type=Path, required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    for r in read(ROOT/'METADATA_LEDGER.json')['responses']:
        if r['method']=='GET':assert sha(ROOT/'metadata'/(r['id']+'.json'))==r['sha256']
    for r in read(ROOT/'COORDINATE_LEDGER.json')['requests']:
        assert sha(ROOT/'coordinates'/(r['id']+'.tif'))==r['sha256']
    reserved=read(ROOT/'sources/reserved-native-contexts.json')['cores']
    assert len(reserved)==265 and all(r['geometry_qualified'] for r in reserved)
    mesh,cache,canvas,_=load('w040')
    for site,cx,cy in [('site1',2432,1920),('site2',3968,3200),('site3',5632,4480)]:
        reserved.append(dict(core_id='old-w040-'+site,winding='w040',context_xywh=[cx-136,cy-136,272,272],
                             **exact_box(mesh,cache,cx-136,cy-136,canvas,272)))
    selected=[];candidate_rows=[];counts={}
    for w,n,role in [('w024',32,'alternate_reference_reservation'),('w053',32,'blank_reference_reservation'),
                     ('w040',64,'training_candidate'),('w041',64,'training_candidate'),('w043',64,'training_candidate')]:
        rows,candidates=select(w,n,reserved)
        for row in rows:row['role']=role
        selected.extend(rows);candidate_rows.extend(candidates)
        if role!='training_candidate':reserved.extend(rows)
        counts[w]=dict(grid=len(candidates),geometry_eligible=sum(r['native_geometry_eligible'] for r in candidates),
                       passes_native_exclusion=sum(r['eligible'] for r in candidates),selected=len(rows),requested=n)
        print(w,counts[w],flush=True)
    save(out/'CANDIDATES.json',candidate_rows)
    save(out/'RESERVED_CONTEXTS.json',reserved)
    result=dict(status='PROSPECTIVE_NATIVE_GEOMETRY_ONLY',counts=counts,selected=selected,
        coordinate_payload_bytes=read(ROOT/'COORDINATE_LEDGER.json')['coordinate_bytes'],
        CT_teacher_annotation_bytes_fetched=0,model_forwards=0,GPU_spend_GBP=0,
        fine_teacher_context_exclusions='PENDING',local_CT_registration='PENDING',
        common_supervision_support='PENDING',paired_training_admitted=False,
        protocol_sha256=sha(ROOT/'PROTOCOL.md'),code_sha256=sha(Path(__file__)))
    save(out/'RESULTS.json',result)
    with (out/'SELECTED_REGIONS.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['core_id','winding','role','core_x','core_y','core_size','context_x','context_y','context_size'])
        for r in selected:writer.writerow([r['core_id'],r['winding'],r['role'],*r['core_xywh'][:3],*r['context_xywh'][:3]])
    requests=[]
    for w in counts:
        _,_,_,render=load(w)
        base='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'+render['origins'][0]['path']+'0/'
        chunks=set()
        for r in selected:
            if r['winding']!=w:continue
            x,y,_,_=r['context_xywh']
            chunks.update((cy,cx) for cy in range(y//128,(y+255)//128+1) for cx in range(x//128,(x+255)//128+1))
        arr=read(ROOT/'metadata'/f'{w}-native-array.json');sep=arr.get('dimension_separator','.')
        for cy,cx in sorted(chunks):
            key=sep.join(map(str,(0,cy,cx)))
            requests.append(dict(id=f'{w}-0-{cy}-{cx}',winding=w,key=key,url=base+key,expected_bytes=28*128*128,
                                 etag=None,sha256=None,array_metadata_sha256=sha(ROOT/'metadata'/f'{w}-native-array.json')))
    save(out/'NEXT_NATIVE_CT_PLAN.json',dict(status='NOT_EXECUTED',purpose='Native geometry-selected contexts; all28 planes',
        requests=requests,max_payload_bytes=sum(r['expected_bytes'] for r in requests),max_workers=2,
        on_missing_or_changed_source='STOP; never synthesize zeros or replace regions',
        prerequisites=['Recheck and pin source metadata','HEAD and freeze chunk ETags/lengths before GET',
                       'Save payload SHA256 receipts','Common data support checks before any training']))
    inputs=[ROOT/'PROTOCOL.md',ROOT/'CATALOG_EXTRACT.json',ROOT/'METADATA_PLAN.json',ROOT/'METADATA_LEDGER.json',
            ROOT/'COORDINATE_PLAN.json',ROOT/'COORDINATE_LEDGER.json',Path(__file__)]+list((ROOT/'sources').glob('*'))
    save(out/'COMPUTE_BINDING.json',dict(files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(inputs) if p.is_file()]))
    print(json.dumps(dict(selected=len(selected),future_native_chunks=len(requests),future_native_bytes=sum(r['expected_bytes'] for r in requests))))

if __name__=='__main__':main()
