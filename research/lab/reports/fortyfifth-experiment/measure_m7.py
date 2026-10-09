"""Sparse, geometry-fixed mesh-to-M7 disagreement; never treated as ground truth."""
import ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import tifffile
P=Path(__file__).resolve().parent
LAB=P.parents[1]
sys.path.insert(0,str(P.parent/'fortyfourth-experiment'))
from pixel_http import pin,save,need,now

def nearest_run(profile):
    v=np.asarray(profile,dtype=bool);d=np.diff(np.r_[False,v,False].astype(np.int8))
    starts=np.flatnonzero(d==1);ends=np.flatnonzero(d==-1)
    if not len(starts):return dict(status='no_predicted_run',runs=0,offset=None)
    centres=(starts+ends-1)/2-15
    ids=np.flatnonzero(np.abs(centres)==np.min(np.abs(centres)))
    if len(ids)>1:return dict(status='equal_nearest_runs',runs=len(starts),offset=None)
    j=int(ids[0]);censored=starts[j]==0 or ends[j]==31
    return dict(status='censored_nearest_run' if censored else 'measured',runs=int(len(starts)),offset=None if censored else float(centres[j]))

def prepare():
    out=P/'geometry';out.mkdir(exist_ok=False)
    # Read the pinned author's normal function without executing its CLI/module.
    source=P/'source-evidence/pherc343/mesh/snap_meshes.py'
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normals')
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
    fixtures=[]
    for interval,expected in [((17,20),3.),((10,13),-4.),((14,17),0.)]:
        a=np.zeros(31,bool);a[slice(*interval)]=True;need(nearest_run(a)['offset']==expected,'Known normal shift');fixtures.append(interval)
    a=np.zeros(31,bool);a[12]=a[18]=True;need(nearest_run(a)['status']=='equal_nearest_runs','Equidistant sheets stay ambiguous')
    a=np.zeros(31,bool);a[:3]=True;need(nearest_run(a)['status']=='censored_nearest_run','Endpoint censorship')
    need(nearest_run(np.zeros(31))['status']=='no_predicted_run','No sheet is not zero offset')
    records=[];pins=[pin(__file__),pin(source)];points=[]
    for seg in ['20251029010146','20251028222030','20251028220955']:
        root=LAB/'data/experiments/pherc0800-surface-geometry' if seg=='20251029010146' else LAB/'data/experiments/pherc0800-larger-geometry'/seg
        files=[root/f'mesh.{axis}.tif' for axis in 'xyz'];pins.extend(pin(f) for f in files)
        xyz=np.stack([tifffile.imread(f) for f in files],-1).astype(np.float32)
        valid=np.isfinite(xyz).all(-1)&(xyz>-.5).all(-1)
        normals,ok=env['normals'](*np.moveaxis(xyz,-1,0),valid)
        ys=np.rint(np.linspace(1,xyz.shape[0]-2,9)).astype(int);xs=np.rint(np.linspace(1,xyz.shape[1]-2,9)).astype(int)
        for y in ys:
            for x in xs:
                row=dict(segment=seg,node_yx=[int(y),int(x)],valid_normal=bool(ok[y,x]),profile_index=None)
                if ok[y,x]:
                    row.update(profile_index=len(points),xyz=xyz[y,x].tolist(),normal_xyz=normals[y,x].tolist())
                    points.append((xyz[y,x,None,:]+np.arange(-15,16,dtype=np.float32)[:,None]*normals[y,x,None,:])[:,::-1])
                records.append(row)
    meta_path=P/'source-evidence/0800-m7-0_.zarray';meta=json.loads(meta_path.read_text());pins.append(pin(meta_path))
    need(meta['dtype']=='|u1' and meta['order']=='C' and meta['dimension_separator']=='/','M7 array contract')
    coords=np.rint(np.stack(points)).astype(np.int64)
    need((coords>=0).all() and (coords<np.array(meta['shape'])).all(),'M7 index bounds')
    chunks=np.unique(coords.reshape(-1,3)//meta['chunks'],axis=0);need(len(chunks)<=256,'Sparse chunk cap')
    np.save(out/'sample-indices-zyx.npy',coords);pins.append(pin(out/'sample-indices-zyx.npy'))
    metadata=json.loads((P/'source-evidence/M7_METADATA_PINS.json').read_text());url=metadata[-1]['url'].removesuffix('.zarray')
    save(out/'PLAN.json',dict(created_at_utc=now(),pins=pins,records=records,chunks=chunks.tolist(),array=meta,root_url=url,maximum_body_bytes=128*1024**2,fixture_groups=6,interpretation='Nearest binary M7 run centre disagreement, not independent surface error; no sheet correction or ink inference. Missing chunks stay unknown. All 243 selected nodes retained; no replacement.'))
    print(json.dumps(dict(nodes=len(records),valid_profiles=len(points),chunks=len(chunks),status='FROZEN')))

def acquire():
    import requests,numcodecs
    from concurrent.futures import ThreadPoolExecutor
    out=P/'geometry';plan=json.loads((out/'PLAN.json').read_text())
    for p in plan['pins']:need(pin(p['path'])==p,'Geometry pin')
    raw=out/'chunks';raw.mkdir(exist_ok=False)
    def fetch(c):
        key='/'.join(map(str,c));url=plan['root_url']+key
        with requests.get(url,timeout=(10,25),stream=True,allow_redirects=False) as r:
            need(r.status_code in [200,404],'M7 HTTP status')
            body=bytearray()
            for part in r.iter_content(65536):
                body.extend(part);need(len(body)<=8*1024**2,'Chunk size cap')
            dest=raw/('-'.join(map(str,c))+'.bin');dest.write_bytes(body)
            return dict(chunk=c,url=url,status=r.status_code,body=pin(dest),etag=r.headers.get('ETag'))
    rows=list(ThreadPoolExecutor(4).map(fetch,plan['chunks']))
    save(out/'REQUESTS.json',rows);need(sum(r['body']['size_bytes'] for r in rows)<=plan['maximum_body_bytes'],'Total M7 body cap')
    coords=np.load(out/'sample-indices-zyx.npy');ci=coords//plan['array']['chunks'];profile=np.zeros(coords.shape[:2],np.uint8);observed=np.zeros_like(profile,bool)
    codec=numcodecs.get_codec(plan['array']['compressor'])
    for row in rows:
        if row['status']!=200:continue
        rawbytes=codec.decode(Path(row['body']['path']).read_bytes());need(len(rawbytes)==int(np.prod(plan['array']['chunks'])),'Decoded chunk length')
        a=np.frombuffer(rawbytes,np.uint8).reshape(plan['array']['chunks']);sel=(ci==row['chunk']).all(-1);q=coords[sel]-np.array(row['chunk'])*plan['array']['chunks'];profile[sel]=a[tuple(q.T)];observed[sel]=True
    np.save(out/'profiles.npy',profile);np.save(out/'observed.npy',observed)
    results=[]
    for record in plan['records']:
        row=record.copy();i=row['profile_index']
        if i is None:row['measurement']=dict(status='invalid_mesh_normal',offset=None)
        elif not observed[i].all():row['measurement']=dict(status='unobserved_m7_support',offset=None)
        else:row['measurement']=nearest_run(profile[i]>0)
        results.append(row)
    summaries=[]
    from collections import Counter
    for seg in sorted(set(r['segment'] for r in results)):
        group=[r['measurement'] for r in results if r['segment']==seg];vals=np.array([r['offset'] for r in group if r['status']=='measured'])
        summaries.append(dict(segment=seg,status_counts=dict(Counter(r['status'] for r in group)),median_absolute_offset=float(np.median(abs(vals))) if len(vals) else None,p90_absolute_offset=float(np.quantile(abs(vals),.9)) if len(vals) else None,measured_over_two=int((abs(vals)>2).sum()),multiple_runs=int(sum(r.get('runs',0)>1 for r in group)),denominator_measured=len(vals)))
    save(out/'RESULTS.json',dict(created_at_utc=now(),nodes=results,summaries=summaries,requests=len(rows),body_bytes=sum(r['body']['size_bytes'] for r in rows),geometry_accuracy_established=False))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__': {'prepare':prepare,'acquire':acquire}[sys.argv[1]]()
