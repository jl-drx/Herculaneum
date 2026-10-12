"""Assemble only the frozen native development contexts and fixed CT panels."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from registry import require_admitted


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--chunks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    registry=json.loads((a.packet/'registry/REGISTRY.json').read_text())
    freeze=json.loads((a.packet/'geometry/CT_INTAKE_FREEZE.json').read_text())
    plan=a.packet/'source-plans/native-ct-chunks.json'
    cases=a.packet/'geometry/NATIVE_CANDIDATES.json'
    assert digest(plan)==freeze['chunk_plan_sha256']
    assert digest(cases)==freeze['selection_sha256']
    assert digest(a.packet/'registry/REGISTRY.json')==freeze['registry_sha256']
    rows=json.loads(plan.read_text())
    lookup={}
    for row in rows:
        require_admitted(row['sample'],'development',registry)
        path=a.chunks/(row['id']+'.raw')
        receipt=json.loads(path.with_suffix('.receipt.json').read_text())
        assert receipt['url']==row['url'] and path.stat().st_size==row['decoded_bytes']==receipt['bytes']
        assert digest(path)==receipt['sha256']
        lookup[row['url']]=path
    a.output.mkdir(parents=True)
    results=[]; panels={}; counts={}
    for case in json.loads(cases.read_text())['cases']:
        require_admitted(case['sample'],'development',registry)
        lo,hi=map(np.asarray,case['ct_context_box_zyx'])
        shape=tuple(hi-lo)
        assert shape==(192,192,192)
        array=np.zeros(shape,np.uint8); covered=np.zeros(shape,bool)
        source=case['source_chunks']['ct']
        base=source['root'].replace('s3://vesuvius-challenge-open-data','https://vesuvius-challenge-open-data.s3.amazonaws.com')
        for key in source['keys']:
            path=lookup[base+'/'+key]
            origin=np.asarray(list(map(int,key.split('/')[1:]))) * 128
            begin=np.maximum(lo,origin); end=np.minimum(hi,origin+128)
            src=tuple(slice(int(x),int(y)) for x,y in zip(begin-origin,end-origin))
            dst=tuple(slice(int(x),int(y)) for x,y in zip(begin-lo,end-lo))
            chunk=np.memmap(path,mode='r',dtype=np.uint8,shape=(128,128,128),order='C')
            assert not covered[dst].any()
            array[dst]=chunk[src]; covered[dst]=True
            del chunk
        assert covered.all()
        # Independent coordinate spot-checks use global // and % rather than
        # the block-intersection copy code above.
        probes=[(0,0,0),(191,191,191),(32,32,32),(96,96,96),(159,159,159),(31,160,97)]
        for point in probes:
            global_point=lo+point
            key='0/'+'/'.join(map(str,global_point//128))
            chunk=np.memmap(lookup[base+'/'+key],mode='r',dtype=np.uint8,shape=(128,128,128))
            assert array[point]==chunk[tuple(global_point%128)]
            del chunk
        path=a.output/(case['id']+'.npy');np.save(path,array,allow_pickle=False)
        results.append({'id':case['id'],'sample':case['sample'],'ct_context_box_zyx':case['ct_context_box_zyx'],
                        'shape':list(array.shape),'dtype':str(array.dtype),'zero_fraction':float((array==0).mean()),
                        'minimum':int(array.min()),'maximum':int(array.max()),'file':path.name,
                        'sha256':digest(path),'bytes':path.stat().st_size,'coordinate_spot_checks':len(probes),
                        'stratum':'pending_CT_only_review'})
        sample=case['sample']
        if sample not in panels:
            panels[sample]=Image.new('RGB',(618,1332),'white');counts[sample]=0
        canvas=panels[sample];draw=ImageDraw.Draw(canvas);row=counts[sample];counts[sample]+=1
        for col,(plane,label) in enumerate([(array[96,:,:],'Z=96'),(array[:,96,:],'Y=96'),(array[:,:,96],'X=96')]):
            left,top=6+204*col,26+220*row
            canvas.paste(Image.fromarray(plane).convert('RGB'),(left,top))
            draw.text((left,top-20),case['id']+' '+label,fill='black')
    for sample,canvas in panels.items():canvas.save(a.output/(sample+'-CT_ONLY.png'))
    out={'status':'PASS','utc':dt.datetime.now(dt.timezone.utc).isoformat(),'cases':results,
         'verified_requests':len(rows),'verified_response_bytes':sum(r['decoded_bytes'] for r in rows),
         'assembled_context_bytes':sum(r['bytes'] for r in results),'array_shape_each':[192,192,192],
         'panels':'All 12 cases and all three central orthogonal slices; grayscale 0..255',
         'reserve_payloads_read':False,'model_prediction_payloads_read':False,
         'scientific_claim':'Source/context qualification and CT-only development review, not surface accuracy or ink detection.'}
    (a.output/'RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
