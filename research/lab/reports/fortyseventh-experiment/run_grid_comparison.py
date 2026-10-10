"""Frozen global-cohort subset of the qualified Villa inference; no target/GPU."""
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

P=Path(__file__).resolve().parent
LAB=P.parents[1]
PREV=P.parent/'fortysixth-experiment'
sys.path.insert(0,str(PREV/'runtime-source'))
import numpy as np
import torch
import tifffile
import zarr
from vesuvius.ink_detection.inference import infer

ROI=(768,2816,1280,3328)
OFFSETS=[(0,0),(0,32),(32,0),(32,32)]
SHAPE=(2560,6720)

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}

def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def check(r):
    p=Path(r['path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)

def intersects(b):
    y,x,Y,X=ROI
    return b.y0<Y and b.y0+b.valid_h>y and b.x0<X and b.x0+b.valid_w>x

def prepare():
    assert not (P/'GO.json').exists()
    seal=PREV/'final-verification.json'
    assert sha(seal)=='c91572b8763e74331638468208a7d15a5d9f01e2c4e38bbc0b4891352cf0e271'
    retained={r['path']:r for r in json.loads(seal.read_text())['records']}
    q=json.loads((PREV/'INFERENCE_QUALIFICATION.json').read_text())
    ck=Path(next(r['checkpoint']['path'] for r in q['models'] if r['model']=='hybrid_3d2d-seed42'))
    baseline=next((PREV/'work/maps_plain').glob('*__hybrid_3d2d-seed42__*.tif'))
    volume=PREV/'work/concat_w047-w048_R5B2_z9500-11000_66.zarr'
    paths=[ck,baseline,Path(str(baseline)+'.receipt.json'),PREV/'work/valid.npy',PREV/'INFERENCE_QUALIFICATION.json']
    paths += [f for f in volume.rglob('*') if f.is_file()]
    paths += [f for f in (PREV/'runtime-source').rglob('*') if f.is_file() and f.suffix in ['.py','.json']]
    pins=[]
    for f in sorted(set(paths)):
        r=pin(f);old=retained[str(f)]
        assert (r['bytes'],r['sha256'])==(old['size_bytes'],old['sha256'])
        pins.append(r)
    root=zarr.open_group(str(PREV/'work/plain-level0.zarr'),mode='r')
    assert tuple(root['0'].shape)==(66,*SHAPE)
    occupancy,scale,level=infer.build_lowres_block_mask(root,height=SHAPE[0],width=SHAPE[1],user_mask=None)
    assert scale==(1,1) and level=='0'
    schedule=[]
    for dy,dx in OFFSETS:
        def starts(n,offset):return sorted({0,n-128,*range(offset,n-128+1,64)})
        blocks=[infer.Block(y,x,128,128) for y in starts(SHAPE[0],dy) for x in starts(SHAPE[1],dx) if occupancy[y:y+128,x:x+128].any()]
        if (dy,dx)==(0,0):
            assert blocks==infer.iter_blocks(SHAPE,128,64,occupancy,scale)
            assert len(blocks)==2797
        batches=[list(range(i,min(i+4,len(blocks)))) for i in range(0,len(blocks),4) if any(intersects(b) for b in blocks[i:i+4])]
        schedule.append({'offset_yx':[dy,dx],'blocks':[[b.y0,b.x0,b.valid_h,b.valid_w] for b in blocks],
                         'retained_batches':batches,'roi_contributing_tiles':sum(intersects(b) for b in blocks),
                         'cohort_tiles':sum(map(len,batches))})
    sched=P/'SCHEDULE.json';save(sched,schedule)
    identity={'created_at_utc':datetime.now(timezone.utc).isoformat(),'phase46_seal':pin(seal),
              'roi_yxYX':ROI,'offsets_yx':OFFSETS,'baseline':str(baseline),'checkpoint':str(ck),
              'occupancy':{'level':level,'scale':scale,'sha256':hashlib.sha256(occupancy.tobytes()).hexdigest()},
              'pins':pins+[pin(sched),pin(Path(__file__)),pin(P/'PROTOCOL.md')],
              'torch':torch.__version__,'numpy':np.__version__,'gpu_spend_GBP':0}
    save(P/'GO.json',identity)
    print(json.dumps({'status':'FROZEN','pins':len(identity['pins']),'schedules':[{k:v for k,v in s.items() if k not in ['blocks','retained_batches']} for s in schedule]}),flush=True)

class Recorder(torch.nn.Module):
    def __init__(self,model,out):super().__init__();self.model=model;self.out=out;self.records=[]
    def forward(self,x):
        logits=self.model(x)
        assert torch.isfinite(logits).all()
        probability=infer.logits_to_probabilities(logits,image_hw=tuple(x.shape[-2:])).detach().cpu().numpy()
        f=self.out/f'batch-{len(self.records):03}.npy';np.save(f,probability)
        self.records.append({'probabilities':pin(f),'normalized_input_sha256':hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),'batch':int(x.shape[0])})
        return logits

def execute():
    go=json.loads((P/'GO.json').read_text())
    check(go['phase46_seal'])
    for r in go['pins']:check(r)
    assert torch.__version__==go['torch'] and np.__version__==go['numpy']
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    infer.load_checkpoint=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    infer.load_flat_inference_state=lambda m,s:m.load_state_dict(s,strict=True)
    ck=Path(go['checkpoint'])
    args=infer.parse_args(['unused.zarr',str(ck),'unused.tif','--no-compile'])
    conf=infer.configure_model(args)
    assert conf.patch_size==128 and conf.input_depth==17 and conf.preprocessing=='tifxyz_robust'
    reader=infer.FlatPatchReader(input_path=PREV/'work/plain-level0.zarr',resolution='0',depth_axis_first=True,
        height=SHAPE[0],width=SHAPE[1],layer_indices=np.arange(42,25,-1),output_depth=17,preprocessing=conf.preprocessing)
    weights=infer.compute_importance_map_2d(patch_size=(128,128),mode='hann').numpy()
    y,x,Y,X=ROI;reference=tifffile.imread(go['baseline'])[y:Y,x:X]
    outputs=[]
    for schedule in json.loads((P/'SCHEDULE.json').read_text()):
        dy,dx=schedule['offset_yx'];out=P/f'offset-{dy:02}-{dx:02}';out.mkdir(exist_ok=False)
        blocks=[infer.Block(*b) for b in schedule['blocks']]
        dataset=infer.FlatBlockDataset(reader=reader,blocks=blocks,patch_size=128,preprocessing=conf.preprocessing)
        # Complete original global cohorts, with identical raw-empty filtering.
        loader=torch.utils.data.DataLoader(dataset,batch_sampler=schedule['retained_batches'],num_workers=0,pin_memory=False)
        used=[blocks[j] for batch in schedule['retained_batches'] for j in batch]
        prob=np.zeros(SHAPE,np.float32);weight=np.zeros(SHAPE,np.float32)
        accumulator=infer.ChunkAccumulator(shape=SHAPE,chunk_shape=(128,128),prob_sum_store=prob,weight_sum_store=weight,
            contribution_counts=infer.compute_chunk_contribution_counts(used,(128,128)))
        recorder=Recorder(conf.model,out);start=time.monotonic()
        infer.run_block_inference(loader=loader,model=recorder,accumulator=accumulator,weight_map=weights,
            mask=None,device=torch.device('cpu'),amp_dtype=None,tta_axes=(),tta_batch_size=None)
        accumulator.flush_remaining()
        raw=prob[y:Y,x:X].copy();mass=weight[y:Y,x:X].copy()
        np.save(out/'probability_sum.npy',raw);np.save(out/'weight_sum.npy',mass)
        np.divide(raw,mass,out=raw,where=mass>1e-6);np.clip(raw,0,1,out=raw);raw*=255
        result=raw.astype(np.uint8);np.save(out/'response.npy',result)
        rec={'status':'PASS','offset_yx':[dy,dx],'batches':recorder.records,
             'seconds':time.monotonic()-start,'tile_forwards':sum(r['batch'] for r in recorder.records),
             'response':pin(out/'response.npy'),'decoded_sha256':hashlib.sha256(result.tobytes()).hexdigest()}
        if (dy,dx)==(0,0):
            difference=np.abs(result.astype(np.int16)-reference.astype(np.int16))
            rec['baseline_gate']={'exact':bool(np.array_equal(result,reference)),'different_pixels':int(np.count_nonzero(difference)),'max_difference':int(difference.max())}
            if not rec['baseline_gate']['exact']:
                rec['status']='STOP_BASELINE_MISMATCH';save(out/'RECEIPT.json',rec)
                save(P/'RUN_RESULT.json',{'status':rec['status'],'offsets':[rec],'shifted_inference_run':False})
                print(json.dumps(rec['baseline_gate']),flush=True);return
        save(out/'RECEIPT.json',rec);outputs.append(rec)
        print(json.dumps({k:v for k,v in rec.items() if k!='batches'}),flush=True)
        del prob,weight,accumulator,recorder
    save(P/'RUN_RESULT.json',{'status':'PASS','offsets':outputs,'gpu_spend_GBP':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);a=ap.parse_args()
    prepare() if a.action=='prepare' else execute()
