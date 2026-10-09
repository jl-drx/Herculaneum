"""Model-free bounded selection and acquisition, gated by ACQUISITION_GO.json."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.request,urllib.error,urllib.parse,time
import numpy as np
import numcodecs

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'data/experiments/known-ink-control'
def pin(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),sha256=digest,size_bytes=path.stat().st_size)
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
def need(condition,message):
    if not condition:raise ValueError(message)
def main():
    go=json.loads((HERE/'ACQUISITION_GO.json').read_text())
    for item in go['pins']:need(pin(item['path'])==item,'Frozen acquisition input/code')
    plan=json.loads((HERE/'ACQUISITION_PLAN.json').read_text());out=HERE/'acquisition';out.mkdir(exist_ok=False)
    started=time.monotonic();records=[];cache={};wire=0
    def fetch(url,name,limit):
        nonlocal wire
        if name in cache:return cache[name]
        need(len(records)<plan['request_cap'] and time.monotonic()-started<plan['wall_seconds'],'Acquisition request/time bound')
        request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=45) as response:
            cap=min(limit,plan['wire_byte_cap']-wire);advertised=response.headers.get('Content-Length')
            need(not advertised or int(advertised)<=cap,'Advertised size limit')
            body=response.read(cap+1);need(len(body)<=cap and response.status==200,'Delivered body bound and status')
            split=urllib.parse.urlsplit(response.url)
            record=dict(url=url,final_origin_path=urllib.parse.urlunsplit((split.scheme,split.netloc,split.path,'','')),status=response.status,etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'),retrieved_at_utc=datetime.now(timezone.utc).isoformat())
        path=out/name
        with path.open('xb') as stream:stream.write(body)
        wire+=len(body);record.update(pin(path));records.append(record);cache[name]=body
        with (out/'request-ledger.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        return body
    try:
        metadata={}
        for item in plan['metadata']:
            body=fetch(item['url'],item['name'],65536)
            need(hashlib.sha256(body).hexdigest()==item['prior']['sha256'],'Unchanged public source/label metadata')
            metadata[item['name']]=json.loads(body)
        attrs=metadata['inklabels.zattrs.json'];la=metadata['inklabels.level0.zarray.json'];sup=metadata['supervision.level0.zarray.json'];sa=metadata['surface.level2.zarray.json']
        need(attrs==metadata['supervision.zattrs.json'] and attrs['online_validation'] is False and attrs['source_z_slice']==[13,97] and attrs['annotation_center_channel']==10,'Same operational training reference')
        need(la==sup and la['chunks']==[21,128,128] and sa['chunks']==[109,128,128] and sa['dtype']==la['dtype']=='|u1','Same input decoding contract')
        def chunk(role,y,x):
            spec=sa if role=='surface' else la
            base=plan['surface_base']+'/2' if role=='surface' else plan['label_base']+'/'+plan['case']+('_inklabels.zarr/0' if role=='inklabels' else '_supervision_mask.zarr/0')
            key=spec.get('dimension_separator','.').join(map(str,(0,y,x)))
            raw=fetch(base+'/'+key,f'{role}-0-{y}-{x}.bin',3*1024**2 if role=='surface' else 512*1024)
            decoded=numcodecs.get_codec(spec['compressor']).decode(raw) if spec['compressor'] else raw
            need(len(decoded)==int(np.prod(spec['chunks'])),'Exact declared decoded byte count')
            return np.frombuffer(decoded,np.uint8).reshape(spec['chunks'])
        probes=[];chosen=None
        for bbox in plan['candidates_xywh']:
            x,y,w,h=bbox;labels=np.empty((256,256),np.uint8);mask=np.empty_like(labels)
            for dy in range(2):
                for dx in range(2):
                    target=(slice(128*dy,128*(dy+1)),slice(128*dx,128*(dx+1)))
                    for role,array in [('inklabels',labels),('supervision',mask)]:
                        decoded=chunk(role,y//128+dy,x//128+dx)
                        need(np.isin(decoded,[0,1]).all() and not np.delete(decoded,10,axis=0).any(),'Central-plane binary annotation contract')
                        array[target]=decoded[10]
            counts=[dict(x=xx,y=yy,positive=int(((labels[yy:yy+128,xx:xx+128]==1)&(mask[yy:yy+128,xx:xx+128]>0)).sum()),negative=int(((labels[yy:yy+128,xx:xx+128]==0)&(mask[yy:yy+128,xx:xx+128]>0)).sum())) for yy in range(0,129,32) for xx in range(0,129,32)]
            feasible=all(any(t['positive']>=r['positive'] and t['negative']>=r['negative'] for t in counts) for r in plan['replacement_requirements'])
            row=dict(bbox_source_xywh=bbox,positive=int(((labels==1)&(mask>0)).sum()),negative=int(((labels==0)&(mask>0)).sum()),unknown=int((mask==0).sum()),all_32_replacements_feasible=feasible,tile_counts=counts)
            probes.append(row);print(json.dumps({k:v for k,v in row.items() if k!='tile_counts'}),flush=True)
            if feasible:
                chosen=row;break
        need(chosen is not None,'No admissible crop in frozen candidate inventory; stop without relaxing criteria')
        # Selection is durable before any CT body or model output is read.
        save(out/'SELECTION.json',dict(selected=chosen,probes=probes,selection_before_surface_chunks=True,selection_before_model_calls=True,plan=pin(HERE/'ACQUISITION_PLAN.json')))
        x,y,w,h=chosen['bbox_source_xywh'];raw=np.empty((109,256,256),np.uint8)
        for dy in range(2):
            for dx in range(2):raw[:,128*dy:128*(dy+1),128*dx:128*(dx+1)]=chunk('surface',y//128+dy,x//128+dx)
        pooled=np.stack([np.rint(raw[z:z+4].astype(np.float32).mean(0)).astype(np.uint8) for z in range(13,97,4)])
        for name,array in [('surface',pooled),('inklabels',labels),('supervision-mask',mask)]:np.save(out/(name+'.npy'),array,allow_pickle=False)
        need(np.isfinite(pooled).all() and pooled.shape==(21,256,256),'Fixed pooled input shape')
        save(out/'COMPLETE.json',dict(status='PASS',selected=chosen,selection=pin(out/'SELECTION.json'),requests=len(records),wire_bytes=wire,scan_chunks=4,scan_body_bytes=sum(r['size_bytes'] for r in records if Path(r['path']).name.startswith('surface-0-')),records=records,arrays={n:pin(out/(n+'.npy')) for n in ['surface','inklabels','supervision-mask']},elapsed_seconds=time.monotonic()-started,new_model_forwards=0))
        print(json.dumps(dict(status='PASS',requests=len(records),wire_bytes=wire,selected=chosen['bbox_source_xywh'])),flush=True)
    except Exception as exc:
        save(out/'FAILURE.json',dict(status='STOP',error=type(exc).__name__+': '+str(exc),requests=len(records),wire_bytes=wire));raise

if __name__=='__main__':main()
