"""Reconstruct both label-only selection stages without models or new acquisition."""
import json,math
import numpy as np
import numcodecs
from common import *
def main():
    initial=json.loads((HERE/'ACQUISITION_PLAN.json').read_text());expanded=json.loads((HERE/'admission-stage2/ACQUISITION_PLAN.json').read_text())
    inventory=json.loads(verify(initial['inventory']).read_text())
    inv=json.loads((HERE/'admission-stage2/INVENTORY_COMPLETE.json').read_text())
    for record in inv['records']:
        r={k:record[k] for k in ['path','sha256','size_bytes']};inventory.extend(json.loads(verify(r).read_text()))
    candidates=set()
    for r in inventory:
        name=r['path'].split('/')[-1]
        if r.get('type')!='file' or name.startswith('.') or r['size']<=78:continue
        depth,cy,cx=map(int,name.split('.'));x,y=128*cx,128*cy
        if depth or x+256>19900 or y+256>10400:continue
        if max(3968-(x+256),x-4224,1408-(y+256),y-1664)>=256:candidates.add((y,x))
    excluded={(b[1],b[0]) for b in initial['candidates_xywh']}
    required=[[x,y,256,256] for y,x in sorted(candidates-excluded)[:64]]
    require(required==expanded['candidates_xywh'],'Independent inventory ordering/exclusions')
    probes=[]
    def reconstruct(folder,bbox):
        x,y,w,h=bbox;spec=json.loads((folder/'inklabels.level0.zarray.json').read_text());codec=numcodecs.get_codec(spec['compressor']);labels=np.empty((256,256),np.uint8);mask=np.empty_like(labels)
        for dy in range(2):
            for dx in range(2):
                for role,target in [('inklabels',labels),('supervision',mask)]:
                    a=np.frombuffer(codec.decode((folder/f'{role}-0-{y//128+dy}-{x//128+dx}.bin').read_bytes()),np.uint8).reshape(21,128,128)
                    target[dy*128:(dy+1)*128,dx*128:(dx+1)*128]=a[10]
        tiles=[]
        for yy in range(0,129,32):
            for xx in range(0,129,32):
                m=mask[yy:yy+128,xx:xx+128]>0;t=labels[yy:yy+128,xx:xx+128];tiles.append((int((m&(t==1)).sum()),int((m&(t==0)).sum())))
        failed=[r['step'] for r in initial['replacement_requirements'] if not any(p>=r['positive'] and n>=r['negative'] for p,n in tiles)]
        return dict(bbox=bbox,positive=int(((labels==1)&(mask>0)).sum()),negative=int(((labels==0)&(mask>0)).sum()),unknown=int((mask==0).sum()),failing_replacement_steps=failed,maximum_tile_positive=max(p for p,n in tiles),maximum_tile_negative=max(n for p,n in tiles))
    for bbox in initial['candidates_xywh']:
        r=reconstruct(HERE/'acquisition',bbox);require(r['failing_replacement_steps'],'Every stage1 crop independently infeasible');probes.append(r)
    selected=json.loads((HERE/'admission-stage2/acquisition/SELECTION.json').read_text())
    require(selected['selected']['bbox_source_xywh']==required[0] and len(selected['probes'])==1,'First stage2 candidate only')
    accepted=reconstruct(HERE/'admission-stage2/acquisition',required[0]);require(not accepted['failing_replacement_steps'],'Every replacement independently feasible')
    new=json.loads((HERE/'admission-stage2/acquisition/COMPLETE.json').read_text());old=json.loads((HERE/'acquisition/FAILURE.json').read_text())
    totals=dict(requests=old['requests']+inv['metadata_requests']+new['requests'],wire_bytes=old['wire_bytes']+inv['wire_bytes']+new['wire_bytes'],scan_chunks=4)
    require(totals==dict(requests=68,wire_bytes=10707142,scan_chunks=4),'Actual acquisition totals')
    save(HERE/'ADMISSION_VERIFICATION.json',dict(status='PASS',verifier=pin(__file__),stage1_probes_independently_reconstructed=probes,stage2_first_candidate=accepted,expanded_candidate_order_verified=True,geometry_candidates=len(candidates),inventory_objects=len(inventory),acquisition_totals=totals,new_model_forwards=0))
    print(json.dumps(dict(status='PASS',stage1_failures_verified=8,stage2_first_candidate_verified=True,**totals)))
if __name__=='__main__':main()
