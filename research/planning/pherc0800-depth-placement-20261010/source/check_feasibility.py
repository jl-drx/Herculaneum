"""Freeze and verify depth-input/schedule feasibility; no neural model is loaded.

Write only this new study's work and reports. Prior data are opened read-only.
Run with the existing ph erc343 scientific environment (NumPy, OpenCV, Zarr).
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,itertools,json,subprocess,sys,time
import numpy as np,cv2,zarr

P=Path(__file__).resolve().parent;W=P/'work';PREV=P.parent/'fiftyfirst-experiment'
GO_SHA='bd10e35e367fa982a3db8bd14be581766df9e06b7629186e3dba54f318cfc05e'
def pin(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=h)
def save(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def array_pin(a):
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def axis_starts(n,p,s):return list(range(0,max(0,n-p),s))+[max(0,n-p)]
def overlaps(y,x,p,b):
    x0,y0,x1,y1=b
    return x<x1 and x+p>x0 and y<y1 and y+p>y0
def interpolation_bounds(src,dst,start,end):
    # Analytic positive support under half-pixel bilinear interpolation.
    c=(np.arange(start,end,dtype=np.float64)+.5)*src/dst-.5
    lo=np.floor(c).astype(int);hi=np.ceil(c).astype(int)
    lo=np.clip(lo,0,src-1);hi=np.clip(hi,0,src-1)
    analytic=(int(lo.min()),int(hi.max())+1)
    # Independent OpenCV ramp: output values equal source coordinates under
    # linear interpolation. Every point is >=.05 from an integer in this grid.
    ramp=cv2.resize(np.arange(src,dtype=np.float32)[None],(dst,1),interpolation=cv2.INTER_LINEAR)[0,start:end]
    assert np.min(np.abs(ramp-np.rint(ramp)))>.01
    actual=(int(np.floor(ramp.min())),int(np.ceil(ramp.max()))+1)
    assert actual==analytic,(src,dst,start,end,analytic,actual)
    return analytic

def main():
    started=time.monotonic();plan=json.loads((P/'PLAN.json').read_text())
    assert plan['scoring_xywh']==[512,1024,1536,512] and plan['nominal_depth_offsets']==[-15,0,15]
    assert not (P/'FEASIBILITY.json').exists()
    W.mkdir(exist_ok=True)
    assert pin(PREV/'INFERENCE_GO.json')['sha256']==GO_SHA
    go=json.loads((PREV/'INFERENCE_GO.json').read_text())
    pins=go['pins']+go['runtime_pins']+[m['checkpoint'] for m in go['models']]
    for r in pins:assert pin(r['path'])==r,r['path']
    assert [m['name'] for m in go['models']]==plan['models']
    old_results=json.loads((PREV/'INFERENCE_RESULTS.json').read_text());assert old_results['status']=='PASS'
    prep=PREV.parent/'fortyfifth-experiment/source-evidence/pherc343/ink/scripts/hecate_prep.py'
    save(P/'FEASIBILITY_GO.json',dict(status='FROZEN_NO_MODEL_EXECUTION',created_at_utc=datetime.now(timezone.utc).isoformat(),pins=[pin(p) for p in [P/'PROTOCOL.md',P/'PLAN.json',Path(__file__),PREV/'INFERENCE_GO.json',PREV/'INFERENCE_RESULTS.json',prep]],inherited_pins_verified=len(pins),prior_go_sha256=GO_SHA))
    x,y,width,height=plan['scoring_xywh'];sl=np.s_[y-512:y-512+height,x-512:x-512+width]
    old_masks=np.load(PREV/'work/output-support.npz');common=old_masks['common'][sl]
    assert common.shape==(height,width) and common.sum()==786240
    known=np.load(PREV/'work/observed-mask.npy');grids=[];support=[]
    prior_grid_plan=json.loads((PREV/'RENDER_GO.json').read_text())['grids']
    for old in go['grids']:
        family,phase=old['family'],old['phase'];py,px=(0,0) if phase=='plain' else (40,200)
        shape=(2580+py,2580+px);h,w=shape;hh,ww=old['grid_shape_yx'];patch=64 if family=='hecate' else 128
        matching=next(g for g in prior_grid_plan if (g['family'],g['phase'])==(family,phase));stride=matching['stride']
        b=(x+px,y+py,x+px+width,y+py+height)
        if family=='hecate':
            x0,x1=interpolation_bounds(ww,w,b[0],b[2]);y0,y1=interpolation_bounds(hh,h,b[1],b[3]);b=(x0,y0,x1,y1)
        all_origins=[(yy,xx) for yy in axis_starts(hh,patch,stride) for xx in axis_starts(ww,patch,stride)]
        contributors=[(yy,xx) for yy,xx in all_origins if overlaps(yy,xx,patch,b)]
        padded=np.pad(known,((py,0),(px,0)),constant_values=True)
        km=padded if family=='villa' else cv2.resize(padded.astype(np.float32),(ww,hh),interpolation=cv2.INTER_AREA)>=1
        bad=np.zeros((hh,ww),bool);covered=np.zeros_like(bad);valid=[]
        for yy,xx in contributors:
            covered[yy:yy+patch,xx:xx+patch]=True
            if km[yy:yy+patch,xx:xx+patch].all():valid.append([yy,xx])
            else:bad[yy:yy+patch,xx:xx+patch]=True
        if family=='hecate':
            bad=cv2.resize(bad.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>0
            covered=cv2.resize(covered.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>=1
        this=(~bad&covered)[y+py:y+py+height,x+px:x+px+width]
        assert np.array_equal(this,old_masks[family+'_'+phase][sl])
        support.append(this)
        chosen=set(map(tuple,valid));original=set((yy,xx) for batch in old['cohorts_yx'] for yy,xx in batch)
        assert chosen<=original
        batches=[dict(original_batch_index=i,cohort_yx=batch) for i,batch in enumerate(old['cohorts_yx']) if any(tuple(t) in chosen for t in batch)]
        retained=set(tuple(t) for batch in batches for t in batch['cohort_yx'])
        assert chosen<=retained<=original
        assert all(km[yy:yy+patch,xx:xx+patch].all() for yy,xx in retained)
        grids.append(dict(family=family,phase=phase,grid_shape_yx=[hh,ww],patch_size=patch,output_dependency_bbox_xyxy=list(b),all_contributing_tiles=len(contributors),fully_known_contributing_tiles=len(valid),complete_cohorts=len(batches),retained_model_tiles=sum(len(v['cohort_yx']) for v in batches),extra_cohort_tiles=len(retained-chosen),source_study_batches=len(old['cohorts_yx']),contributors_yx=valid,batches=batches,support_pixels=int(this.sum())))
    assert np.array_equal(np.logical_and.reduce(support),common)
    np.savez_compressed(W/'output-support.npz',common=common,**{g['family']+'_'+g['phase']:a for g,a in zip(grids,support)})
    save(P/'SCHEDULE.json',dict(status='FROZEN_BEFORE_SHIFTED_PREPARATION',scoring_xywh=plan['scoring_xywh'],grids=grids))
    print('GRID_AND_CONTEXT_PASS',json.dumps([{k:g[k] for k in ['family','phase','complete_cohorts','retained_model_tiles','support_pixels']} for g in grids]),flush=True)
    inputs=[];baseline_checks=[];positions=[];occupancy=[]
    for d in plan['nominal_depth_offsets']:
        villa=np.arange(25+d,42+d)
        pos=32.5+d+(np.arange(16)-7.5)*(9.6/8.64);lo=np.floor(pos).astype(int);hi=lo+1
        assert villa.min()>=0 and villa.max()<66 and lo.min()>=0 and hi.max()<66
        # Independent rational derivation, avoiding the production float formula.
        from fractions import Fraction
        exact=[Fraction(65,2)+d+(Fraction(i)-Fraction(15,2))*Fraction(10,9) for i in range(16)]
        assert np.max(np.abs(pos-np.array(list(map(float,exact)))))<1e-12
        positions.append(dict(d=d,villa_indices=villa.tolist(),villa_centre_native_voxels=d+.5,villa_range_native_voxels=[float(villa.min()-32.5),float(villa.max()-32.5)],hecate_positions=pos.tolist(),hecate_centre_native_voxels=d,hecate_range_native_voxels=[float(pos.min()-32.5),float(pos.max()-32.5)],required_native_indices_min_max=[int(min(villa.min(),lo.min())),int(max(villa.max(),hi.max()))]))
        for phase in plan['phases']:
            src=PREV/'work/model-inputs'/(phase+'.zarr');out=W/f'hecate-{phase}-d{d:+03}.zarr'
            assert not out.exists();log=W/f'prep-{phase}-d{d:+03}.log'
            with log.open('x') as f:subprocess.run([sys.executable,'-B',str(prep),'--src',str(src),'--out',str(out),'--native-um','8.64','--native-um-z','8.64','--target-um','9.6','--z-center',str(32.5+d),'--nz','16'],check=True,stdout=f,stderr=subprocess.STDOUT)
            a=np.asarray(zarr.open_group(str(out),mode='r')['0']);assert a.dtype==np.uint8 and a.shape[0]==16
            if d==0:
                old=zarr.open_group(str(PREV/'work/model-inputs'/('hecate-'+phase+'.zarr')),mode='r')['0']
                assert np.array_equal(a,np.asarray(old));baseline_checks.append(dict(phase=phase,all_prepared_values_exact=True,values=int(a.size),array=array_pin(a)))
            native=np.asarray(zarr.open_group(str(src),mode='r')['0'][int(villa[0]):int(villa[-1])+1])
            for family,volume in [('villa',native),('hecate',a)]:
                g=next(g for g in grids if (g['family'],g['phase'])==(family,phase));patch=g['patch_size'];retained=[t for batch in g['batches'] for t in batch['cohort_yx']]
                empty=[];zero_profiles=total_profiles=0
                for yy,xx in retained:
                    v=volume[:,yy:yy+patch,xx:xx+patch];assert v.shape==(17 if family=='villa' else 16,patch,patch)
                    if not v.any():empty.append([yy,xx])
                    zero_profiles+=int((~v.any(axis=0)).sum());total_profiles+=patch*patch
                assert not empty,'Depth changes active batch occupancy; stop rather than regroup'
                occupancy.append(dict(d=d,phase=phase,family=family,retained_tiles=len(retained),empty_tiles=empty,zero_depth_profiles_counted_per_tile=zero_profiles,profiles_counted_per_tile=total_profiles))
            inputs.append(dict(d=d,phase=phase,hecate_volume=array_pin(a),native_source=str(src),native_indices=villa.tolist(),prepared_files=[pin(f) for f in sorted(out.rglob('*')) if f.is_file()]))
            print('PREPARED_AND_OCCUPANCY_PASS',d,phase,flush=True)
    baselines=[];estimates=[]
    for row in old_results['maps']:
        family='hecate' if row['model']=='hecate' else 'villa';g=next(g for g in grids if (g['family'],g['phase'])==(family,row['phase']))
        for output in ['aligned','display']:
            rec=row['outputs'][output];assert pin(rec['path'])==rec
        a=np.load(row['outputs']['aligned']['path'],mmap_mode='r')[sl]
        folder=W/'baseline-crops';folder.mkdir(exist_ok=True);dest=folder/('-'.join([row['model'],row['phase'],row['direction']])+'.npy');assert not dest.exists();np.save(dest,a)
        oldfolder=Path(row['outputs']['aligned']['path']).parent
        for batch in g['batches']:
            meta=oldfolder/f"batch-{batch['original_batch_index']:05}.json";rec=json.loads(meta.read_text())
            assert rec['binding']==dict(go_sha256=GO_SHA,job=[row['model'],row['phase'],row['direction']])
            assert rec['cohort_yx']==batch['cohort_yx'] and pin(rec['logits']['path'])==rec['logits']
        baselines.append(dict(model=row['model'],phase=row['phase'],direction=row['direction'],source=row['outputs']['aligned'],crop=pin(dest),crop_values=int(a.size),read_only_reassembly_pending=True))
        # Includes study51 per-map cached replay/I/O time. Cohort ratio is only
        # an estimate; full-grid preprocessing and fixed overhead do not scale.
        seconds=row['seconds']*g['complete_cohorts']/row['counts']['model_calls']
        estimates.append(dict(model=row['model'],phase=row['phase'],direction=row['direction'],source_seconds=row['seconds'],source_batches=row['counts']['model_calls'],new_batches_per_offset=g['complete_cohorts'],estimated_seconds_per_offset=seconds))
    save(P/'PREPARED_INPUTS.json',dict(inputs=inputs,depth_positions=positions,occupancy_checks=occupancy))
    save(P/'BASELINES.json',dict(maps=baselines,d0_Hecate_preparation_checks=baseline_checks))
    for r in pins:assert pin(r['path'])==r,'Prior frozen input changed'
    assert 'torch' not in sys.modules
    result=dict(status='PASS_INPUT_AND_SCHEDULE_FEASIBILITY_ONLY',created_at_utc=datetime.now(timezone.utc).isoformat(),model_forwards=0,new_CT_fetches=0,gpu_spend_GBP=0,prior_pins_verified=len(pins),common_support_pixels=int(common.sum()),scoring_pixels=int(common.size),unknown_pixels=int((~common).sum()),prepared_Hecate_volumes=len(inputs),exact_baseline_Hecate_preparations=len(baseline_checks),baseline_map_crops=len(baselines),all_retained_cohorts_nonempty_at_all_depths=True,new_production_maps=32,shifted_forward_batches=2*sum(r['new_batches_per_offset'] for r in estimates),estimated_shifted_seconds=2*sum(r['estimated_seconds_per_offset'] for r in estimates),timing_estimates=estimates,feasibility_wall_seconds=time.monotonic()-started,worker_launch_admitted=False,remaining_gates=['Exact d0 saved-logit reassembly under reduced schedule','Shifted distinct-tile batch qualification and original-framework/cache checks','Depth-bound receipt mismatch and interruption/restart qualification','Frozen worker/input identities and private checkpoint setup before launch'],limitations=['Timing extrapolates study51 per-map wall times by retained cohort count; fixed overhead, qualifying and backup time remain.','No shifted prediction, ink accuracy, surface-location truth or independent scientific review has been obtained.','Prepared inputs are reproducible derivatives of the retained source; this is not a standalone full inference bundle.'])
    save(P/'FEASIBILITY.json',result);print(json.dumps({k:v for k,v in result.items() if k!='timing_estimates'},indent=2),flush=True)

if __name__=='__main__':main()
