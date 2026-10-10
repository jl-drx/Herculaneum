"""Reassemble the fixed strip from retained d0 logits without a neural model.

Use the original Villa probability/accumulator functions and Hecate arithmetic.
Do not instantiate a model, load a checkpoint, or write into the prior study.
"""
from pathlib import Path
import json,sys,time
import numpy as np,cv2,zarr,torch
from check_feasibility import P,W,PREV,GO_SHA,pin,save,array_pin,axis_starts
sys.path.insert(0,str(PREV.parent/'fortysixth-experiment/runtime-source'))
from vesuvius.ink_detection.inference import infer

def main():
    start=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert json.loads((P/'FEASIBILITY.json').read_text())['status']=='PASS_INPUT_AND_SCHEDULE_FEASIBILITY_ONLY'
    assert pin(PREV/'INFERENCE_GO.json')['sha256']==GO_SHA
    schedule=json.loads((P/'SCHEDULE.json').read_text());x,y,width,height=schedule['scoring_xywh']
    old_sl=np.s_[y-512:y-512+height,x-512:x-512+width]
    rows=json.loads((PREV/'INFERENCE_RESULTS.json').read_text())['maps'];results=[]
    for row in rows:
        phase=row['phase'];family='hecate' if row['model']=='hecate' else 'villa'
        g=next(g for g in schedule['grids'] if (g['family'],g['phase'])==(family,phase))
        py,px=(0,0) if phase=='plain' else (40,200);native_shape=(2580+py,2580+px)
        sl=np.s_[y+py:y+py+height,x+px:x+px+width];shape=tuple(g['grid_shape_yx'])
        num=np.zeros(shape,np.float32);den=np.zeros(shape,np.float32)
        source=Path(row['outputs']['aligned']['path']).parent;extra={}
        if family=='villa':
            weight=infer.compute_importance_map_2d(patch_size=(128,128),mode='hann').numpy()
            blocks=[infer.Block(yy,xx,128,128) for batch in g['batches'] for yy,xx in batch['cohort_yx']]
            acc=infer.ChunkAccumulator(shape=shape,chunk_shape=(128,128),prob_sum_store=num,weight_sum_store=den,contribution_counts=infer.compute_chunk_contribution_counts(blocks,chunk_shape=(128,128)))
        else:
            win=torch.hann_window(64,periodic=False);win=win[:,None]*win[None,:];win=(win/win.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001)
            for yy in axis_starts(shape[0],64,32):
                for xx in axis_starts(shape[1],64,32):den[yy:yy+64,xx:xx+64]+=win.numpy()
            volume=np.asarray(zarr.open_group(str(W/f'hecate-{phase}-d+00.zarr'),mode='r')['0'])
        for batch in g['batches']:
            path=source/f"batch-{batch['original_batch_index']:05}.json";r=json.loads(path.read_text())
            assert r['binding']==dict(go_sha256=GO_SHA,job=[row['model'],phase,row['direction']])
            assert r['index']==batch['original_batch_index'] and r['cohort_yx']==batch['cohort_yx']
            assert pin(r['logits']['path'])==r['logits'];logits=torch.from_numpy(np.load(r['logits']['path']))
            assert logits.shape[0]==len(batch['cohort_yx'])
            if family=='villa':
                probability=infer.logits_to_probabilities(logits,image_hw=(128,128)).numpy()[:,0]
                for tile,(yy,xx) in zip(probability,batch['cohort_yx']):acc.add_tile(y0=yy,x0=xx,tile=tile,tile_weights=weight)
            else:
                raw=np.stack([volume[:,yy:yy+64,xx:xx+64] for yy,xx in batch['cohort_yx']])
                if row['direction']=='reverse':raw=raw[:,::-1].copy()
                image=(raw.astype(np.float32)/255)[:,None];valid=np.broadcast_to(raw.any(1)[:,None,None],(len(raw),1,16,64,64))
                assert array_pin(image)==r['inputs'][0] and array_pin(valid)==r['inputs'][1]
                probability=logits.float().sigmoid()[:,0];weighted=(probability*torch.from_numpy(raw).any(1)*win).numpy()
                for tile,(yy,xx) in zip(weighted,batch['cohort_yx']):num[yy:yy+64,xx:xx+64]+=tile
        if family=='villa':
            acc.flush_remaining();p=num[sl].copy();mass=den[sl]
            for name,a in [('numerator',p),('denominator',mass)]:
                rec=row['outputs'][name];assert pin(rec['path'])==rec
                assert np.array_equal(a,np.load(rec['path'],mmap_mode='r')[old_sl]),name
            np.divide(p,mass,out=p,where=mass>1e-6);np.clip(p,0,1,out=p)
            aligned=(p*255).astype(np.uint8).astype(np.float32)/255
            extra['original_numerator_and_denominator_exact']=True
        else:
            raw=np.rint((num/den).clip(0,1)*255).astype(np.uint8)
            aligned=cv2.resize(raw.astype(np.float32)/255,(native_shape[1],native_shape[0]),interpolation=cv2.INTER_LINEAR)[sl]
            extra['global_denominator_and_output_interpolation_preserved']=True
        display=np.rint(aligned*255).astype(np.uint8)
        for key,a in [('aligned',aligned),('display',display)]:
            rec=row['outputs'][key];assert pin(rec['path'])==rec
            expected=np.load(rec['path'],mmap_mode='r')[old_sl]
            assert a.shape==(height,width) and np.array_equal(a,expected),(row['model'],phase,row['direction'],key,int((a!=expected).sum()))
        result=dict(status='PASS',model=row['model'],phase=phase,direction=row['direction'],original_batches_replayed=len(g['batches']),entire_crop_values_exact=int(aligned.size),aligned=array_pin(aligned),display=array_pin(display),**extra)
        results.append(result);print('BASELINE_REPLAY_PASS',row['model'],phase,row['direction'],flush=True)
    report=dict(status='PASS',maps=results,exact_maps=len(results),exact_values_per_output=sum(r['entire_crop_values_exact'] for r in results),old_batches_replayed=sum(r['original_batches_replayed'] for r in results),new_neural_model_forwards=0,model_checkpoints_loaded=0,elapsed_seconds=time.monotonic()-start,source_pins=[pin(p) for p in [Path(__file__),P/'SCHEDULE.json',P/'FEASIBILITY_GO.json',PREV/'INFERENCE_RESULTS.json']],scope='Reduced-schedule saved-logit reconstruction, exact aligned and display values for the entire strip including unscored pixels; not a shifted-input inference qualification.')
    save(P/'BASELINE_REPLAY.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['maps','source_pins']},indent=2),flush=True)

if __name__=='__main__':main()
