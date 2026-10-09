"""Bounded metadata-only continuation after stage1 admission failure."""
from pathlib import Path
import hashlib,json,re,urllib.request,time
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[2]/'data/experiments/known-ink-control'
def pin(p):
    p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def main():
    frozen=json.loads((HERE/'INVENTORY_GO.json').read_text())
    assert all(pin(r['path'])==r for r in frozen['pins'])
    first=OLD/'label-chunk-listing-page-0.json';entries=json.loads(first.read_text());old=json.loads((OLD/'acquisition-sources.json').read_text());r=next(r for r in old if r['local_file']==first.name)
    link=next(v for k,v in r['headers'].items() if k.lower()=='link');records=[];wire=0;start=time.monotonic()
    for page in range(1,16):
        match=re.search(r'<([^>]+)>; rel="next"',link)
        if not match:break
        url=match.group(1);assert url.startswith('https://huggingface.co/api/buckets/scrollprize/datasets/tree/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/')
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=45) as response:
            raw=response.read(1024**2+1);assert response.status==200 and len(raw)<=1024**2;link=response.headers.get('Link','')
        path=HERE/f'inventory-page-{page}.json';path.write_bytes(raw);wire+=len(raw);assert wire<=10*1024**2 and time.monotonic()-start<300
        rows=json.loads(raw);entries.extend(rows);records.append(dict(url=url,**pin(path)));print(json.dumps(dict(page=page,objects=len(rows),wire_bytes=wire)),flush=True)
    assert not re.search(r'<([^>]+)>; rel="next"',link),'Incomplete inventory at bound; stop'
    candidates=set()
    for row in entries:
        key=row['path'].split('/')[-1]
        if row.get('type')!='file' or key.startswith('.') or row['size']<=78:continue
        z,cy,cx=map(int,key.split('.'));x,y=cx*128,cy*128
        if z!=0 or x+256>19900 or y+256>10400:continue
        if x+256<=3968-256 or x>=3968+512 or y+256<=1408-256 or y>=1408+512:candidates.add((y,x))
    stage1=json.loads((HERE.parent/'ACQUISITION_PLAN.json').read_text());excluded={(b[1],b[0]) for b in stage1['candidates_xywh']}
    selected=[[x,y,256,256] for y,x in sorted(candidates-excluded)[:64]]
    save(HERE/'INVENTORY_COMPLETE.json',dict(status='PASS',metadata_requests=len(records),wire_bytes=wire,records=records,first_page=pin(first),all_inventory_objects=len(entries),geometry_candidates=len(candidates),stage1_candidates_excluded=len(excluded),candidate_order=selected,new_model_forwards=0,new_label_bodies=0,new_CT_chunks=0))
    stage1.update(candidates_xywh=selected,request_cap=540,wall_seconds=1800,selection_rule='Stage2 first feasible numeric Y/X candidate from expanded metadata inventory, exclude prior eight; same class counts/crop/gap',expanded_inventory=pin(HERE/'INVENTORY_COMPLETE.json'))
    save(HERE/'ACQUISITION_PLAN.json',stage1)
    files=[HERE/'acquire_crop.py',HERE/'ACQUISITION_PLAN.json',HERE/'INVENTORY_GO.json',HERE/'INVENTORY_COMPLETE.json',HERE/'PROTOCOL.md']+[Path(x['prior']['path']) for x in stage1['metadata']]
    save(HERE/'ACQUISITION_GO.json',dict(status='FROZEN_BEFORE_STAGE2_LABELS_AND_CT',pins=[pin(p) for p in files],new_model_forwards=0))
    print(json.dumps(dict(status='FROZEN',candidate_count=len(selected),first_candidates=selected[:4])))
if __name__=='__main__':main()
