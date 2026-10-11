"""Synthetic end-to-end qualification, including deliberate artifact faults.

Uses the frozen producer only to construct a synthetic reference result. Model
construction and every inference function are replaced by errors. No real run,
CT, annotation packet, checkpoint tensor, or held-out prediction is read.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import zarr
import verify_evaluation as v


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pin(path, root):
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=v.digest(path))


def forbidden(*args, **kwargs):
    raise RuntimeError('Inference or map access forbidden in this fixture')


def make_fixture(root, producer_path):
    spec = importlib.util.spec_from_file_location('frozen_producer_fixture_only', producer_path)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    producer.NativeNet = forbidden
    producer.predict = forbidden
    producer.profile_maps = forbidden
    plan = v.read(producer_path.with_name('PLAN.json'))
    run, source = root / 'run', root / 'source'
    run.mkdir();source.mkdir()
    write(run / 'PLAN.json', plan)
    rows, masks, labels, sites = [], [], [], []
    for fold in plan['fold_order']:
        canvas_mask = np.zeros((128, 512), np.uint8)
        canvas_label = np.zeros_like(canvas_mask)
        region_rows = []
        for i, (count, positive) in enumerate([(4, 0), (2, 0), (4, 4), (4, 2)]):
            mask = np.zeros((128, 128), bool)
            label = np.zeros_like(mask)
            coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for j, (y, x) in enumerate(coords[:count]):
                mask[y, x] = True;label[y, x] = j < positive
            label[5, 5] = True  # Ink outside supervision must not become a label.
            core_id = f'{fold}-fixture-{i}'
            row = dict(core_id=core_id, x=i*128, y=0, width=128, height=128,
                       eligible=True, supervised_pixels=count, ink_supervised=positive)
            region_rows.append(row);rows.append(dict(row, winding=fold))
            masks.append(mask);labels.append(label)
            canvas_mask[:, i*128:(i+1)*128] = mask.astype(np.uint8)*255
            canvas_label[:, i*128:(i+1)*128] = label.astype(np.uint8)*255
        sites.append(dict(winding=fold, rows=region_rows))
        for name, data in [('supervision', canvas_mask), ('inklabels', canvas_label)]:
            array = zarr.open_array(str(source/'arrays'/fold/name), mode='w', shape=data.shape,
                                    chunks=(128, 128), dtype='uint8', zarr_format=3)
            array[:] = data
    folds = []
    for fold in plan['fold_order']:
        folds.append(dict(held_out=fold, original_evaluation_core_ids=[r['core_id'] for r in rows if r['winding']==fold],
                          retained_training_core_ids=[r['core_id'] for r in rows if r['winding']!=fold], fold_geometry_admitted=True))
    write(source/'RESULTS.json', dict(sites=sites))
    write(source/'geometry/RESULTS.json', dict(folds=folds))
    go = dict(plan=plan, code_pins=[pin(run/'PLAN.json',run)],
              source_pins=[pin(p,source) for p in sorted(source.rglob('*')) if p.is_file()])
    write(run/'GO.json', go);go_sha = v.digest(run/'GO.json')
    data = dict(rows=rows, y=np.stack(labels), mask=np.stack(masks))
    prevalence = 3/7
    for fold in folds:
        w = fold['held_out'];directory = run/'work'/w
        normalization = dict(mean=[0.]*28, std=[1.]*28, context_occurrences=8, pixels=8*256**2)
        write(directory/'NORMALIZATION.json',dict(go_sha256=go_sha, training_core_ids=fold['retained_training_core_ids'], normalization=normalization))
        beta = [0.]*28 + [math.log(prevalence/(1-prevalence))]
        write(directory/'PROFILE.json',dict(go_sha256=go_sha,normalization_sha256=v.digest(directory/'NORMALIZATION.json'),
                                           fit=dict(prevalence=prevalence,beta=beta,converged=True,gradient_infinity=0.)))
        for seed in plan['seeds']:
            arm = run/'work'/f'{w}-seed{seed}';arm.mkdir()
            binding = dict(go_sha256=go_sha,fold=w,seed=seed,normalization_sha256=v.digest(directory/'NORMALIZATION.json'))
            history = [dict(step=step,lr=plan['learning_rate'],bce=.67-.004*(step//256),ap=.95,auc=.96,
                            prevalence=prevalence,positives=12,negatives=16)
                       for step in range(0,plan['maximum_updates']+1,256)]
            step = plan['maximum_updates'];checkpoint=arm/f'step-{step:06d}.pt'
            checkpoint.write_bytes(b'SYNTHETIC CHECKPOINT PLACEHOLDER; NEVER DESERIALIZE\n'+f'{w}-{seed}'.encode())
            receipt = arm/f'step-{step:06d}.json'
            write(receipt,dict(binding=binding,checkpoint=pin(checkpoint,arm),step=step,best_step=step,
                               train=history[-1],termination='ceiling_still_improving'))
            write(arm/'COMPLETE.json',dict(binding=binding,history=history,terminal_step=step,endpoint_step=step,
                                          endpoint_receipt=pin(receipt,arm),termination='ceiling_still_improving',
                                          draws=step*plan['batch_size'],supervised_pixel_exposures=1))
        indices = [i for i,r in enumerate(rows) if r['winding']==w]
        for model in ['constant','profile']+[f'cnn-seed{s}' for s in plan['seeds']]:
            if model in ['constant','profile']:
                maps = np.full((4,128,128),beta[-1],np.float64)
            else:
                seed = int(model.split('seed')[1]);maps=np.full((4,128,128),9,np.float32)
                for i,index in enumerate(indices):
                    mask, label = data['mask'][index],data['y'][index]
                    maps[i][mask] = np.where(label[mask],4,-4).astype(np.float32)
                maps[0][data['mask'][indices[0]]] = -2
                maps[1][data['mask'][indices[1]]] = -6
                if seed==17:maps[0,0,0]=2
                if seed==43:maps[3,0,0]=-4
                if seed==89:maps[2,0,0]=-.1
            output = run/'evaluation'/f'{w}-{model}.npy';output.parent.mkdir(exist_ok=True)
            np.save(output,maps,allow_pickle=False)
            write(output.with_suffix('.json'),dict(go_sha256=go_sha,fold=w,model=model,
                core_ids=fold['original_evaluation_core_ids'],maps=pin(output,output.parent),
                **producer.score_maps(data,indices,maps)))
    # The producer sees only synthetic precomputed maps; neural construction fails if called.
    producer.evaluate(run,source,data,folds,plan,go_sha)
    return run, source, go_sha, plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    # Analytical ties, single-class convention, permutations and extreme logits.
    y=np.array([0,1,0,1],bool);scores=np.array([0.,1.,1.,2.]);m=v.metrics(y,scores)
    check('grouped-tie AP',abs(m['ap']-5/6)<1e-14)
    check('half-credit AUC ties',abs(m['auc']-.875)<1e-14)
    check('constant AP equals prevalence and AUC equals half',v.metrics(y,np.zeros(4))['ap']==.5 and v.metrics(y,np.zeros(4))['auc']==.5)
    check('both single-class AP and AUC undefined',all(v.metrics(np.full(4,k),scores)['ap'] is None and v.metrics(np.full(4,k),scores)['auc'] is None for k in [False,True]))
    check('zero logit uses inclusive fixed-half response',v.metrics(y,np.zeros(4))['fraction_probability_ge_half']==1.)
    check('extreme-logit BCE remains finite',v.metrics([0,1],[-1000,1000])['bce']==0.)
    for perm in itertools.permutations(range(4)):
        ix=list(perm);v.compare(v.metrics(y[ix],scores[ix]),m,'tie permutation')
    check('all 24 tied-score permutations agree',True)
    rng=np.random.default_rng(831);a=rng.integers(0,2,41).astype(bool);s=rng.integers(-3,4,41)
    wins=np.array([float(x>z)+.5*float(x==z) for x in s[a] for z in s[~a]])
    check('AUC matches explicit positive-negative pair counting',abs(v.metrics(a,s)['auc']-wins.mean())<1e-14)
    with tempfile.TemporaryDirectory(prefix='native-evaluation-review-') as temp:
        root=Path(temp);run,source,go,plan=make_fixture(root,args.producer)
        report,rows=v.verify_evaluation(run,source,go)
        check('complete 9-endpoint / 15-map fixture matches frozen producer',report['status']=='PASS' and report['held_out_map_bodies_opened']==15 and len(rows)==60)
        check('positive advancement can pass with budget-limited fits',report['recomputed']['decision']=='advance_to_broader_paired_supervision_design')
        check('all supervised-negative and supervised-positive single-class cores retained',sum(r['ap'] is None for r in rows)==45)
        check('out-of-mask ink excluded',all(r['positives']+r['negatives'] in [2,4] for r in rows))
        # Unequal negative-core sizes must be pixel-weighted, not core-averaged.
        cnn=[r for r in rows if r['fold']=='w035' and r['model']=='cnn-seed17' and not r['positives']]
        weighted=sum(r['negatives']*r['mean_probability'] for r in cnn)/sum(r['negatives'] for r in cnn)
        unweighted=float(np.mean([r['mean_probability'] for r in cnn]))
        check('negative-response guard uses supervised-pixel weighting',abs(weighted-unweighted)>.01 and abs(report['recomputed']['folds'][0]['cnn_blank_mean_probabilities'][0]-weighted)<1e-14)
        fold_labels=v.load_labels(source,v.read(source/'geometry/RESULTS.json'),plan)['w035']
        ensemble=np.mean([np.load(run/'evaluation'/f'w035-cnn-seed{s}.npy') for s in plan['seeds']],axis=0)
        ensemble_metrics=v.metrics(np.concatenate([y[m] for _,y,m in fold_labels]),
                                   np.concatenate([ensemble[i][m] for i,(_,_,m) in enumerate(fold_labels)]))
        check('primary mean of individual AP differs from ensemble AP',abs(report['recomputed']['folds'][0]['cnn_mean_ap']-ensemble_metrics['ap'])>.01)
        originals={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
        def restore():
            for p in root.rglob('*'):
                if p.is_file() and p not in originals:p.unlink()
            for p,body in originals.items():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
        def refresh_result_pin(receipt):
            result=v.read(run/'RESULTS.json')
            for i,r in enumerate(result['outputs']):
                if r['path']==str(receipt.relative_to(run)):result['outputs'][i]=pin(receipt,run)
            write(run/'RESULTS.json',result)
        def edit_receipt(name,edit):
            p=run/'evaluation'/f'w035-{name}.json';r=v.read(p);edit(r);write(p,r);refresh_result_pin(p)
        def bad(name,mutate):
            restore();mutate()
            try:v.verify_evaluation(run,source,go)
            except (v.AuditError,FileNotFoundError):checks.append(name)
            else:raise AssertionError('Accepted fault: '+name)
        endpoint=run/'work/w035-seed17/COMPLETE.json'
        # A genuine incomplete run has no evaluation outputs; np.load must never be reached.
        restore();endpoint.unlink()
        for p in (run/'evaluation').iterdir():p.unlink()
        (run/'RESULTS.json').unlink()
        with patch.object(v.np,'load',forbidden):waiting,_=v.verify_evaluation(run,source,go)
        check('partial live-shaped run refuses all map opens',waiting['status']=='WAITING_FOR_TRAINING' and waiting['held_out_map_bodies_opened']==0)
        restore()
        for p in (run/'evaluation').iterdir():p.unlink()
        (run/'RESULTS.json').unlink()
        with patch.object(v.np,'load',forbidden):waiting,_=v.verify_evaluation(run,source,go)
        check('nine endpoints still require their joint seal',waiting['status']=='WAITING_FOR_ENDPOINT_SEAL' and waiting['held_out_map_bodies_opened']==0)
        restore();(run/'RESULTS.json').unlink()
        with patch.object(v.np,'load',forbidden):waiting,_=v.verify_evaluation(run,source,go)
        check('sealed but unfinished evaluation opens no maps',waiting['status']=='WAITING_FOR_COMPLETE_EVALUATION' and waiting['held_out_map_bodies_opened']==0)
        bad('premature seal rejected',lambda:endpoint.unlink())
        bad('changed GO rejected before map access',lambda:(run/'GO.json').write_text('{}'))
        bad('corrupted checkpoint bytes rejected',lambda:(run/'work/w035-seed17/step-016384.pt').write_bytes(b'broken'))
        bad('changed normalization binding rejected',lambda:(run/'work/w035/NORMALIZATION.json').write_text('{}'))
        def corrupt_completion(key,value):
            d=v.read(endpoint);d[key]=value;write(endpoint,d)
        bad('wrong seed binding rejected',lambda:corrupt_completion('binding',dict(v.read(endpoint)['binding'],seed=89)))
        bad('non-minimum training endpoint rejected',lambda:corrupt_completion('endpoint_step',0))
        bad('ceiling falsely labelled plateau rejected',lambda:corrupt_completion('termination',v.PLATEAU))
        bad('unreceipted/missing model map rejected',lambda:(run/'evaluation/w035-cnn-seed17.npy').unlink())
        bad('partial evaluation output rejected',lambda:(run/'evaluation/unfinished.npy.partial').write_bytes(b'x'))
        bad('unexpected extra map rejected',lambda:(run/'evaluation/unexpected.npy').write_bytes(b'x'))
        bad('binary map corruption rejected',lambda:(run/'evaluation/w035-cnn-seed17.npy').write_bytes(b'broken'))
        bad('wrong core order rejected with valid receipt hash',lambda:edit_receipt('cnn-seed17',lambda r:r['core_ids'].reverse()))
        bad('duplicate core identity rejected',lambda:edit_receipt('cnn-seed17',lambda r:r['core_ids'].__setitem__(1,r['core_ids'][0])))
        bad('wrong model identity rejected',lambda:edit_receipt('cnn-seed17',lambda r:r.update(model='cnn-seed89')))
        bad('fabricated AP rejected despite refreshed hashes',lambda:edit_receipt('cnn-seed17',lambda r:r['overall'].update(ap=.12345)))
        bad('fabricated per-core negative count rejected',lambda:edit_receipt('cnn-seed17',lambda r:r['cores'][0].update(negatives=100)))
        bad('source annotation corruption rejected',lambda:(source/'arrays/w035/inklabels/zarr.json').write_text('{}'))
        def change_array(value):
            path=run/'evaluation/w035-cnn-seed17.npy';np.save(path,value,allow_pickle=False)
            edit_receipt('cnn-seed17',lambda r:r.update(maps=pin(path,path.parent)))
        bad('wrong map shape rejected with refreshed hashes',lambda:change_array(np.zeros((3,128,128),np.float32)))
        bad('wrong map dtype rejected with refreshed hashes',lambda:change_array(np.zeros((4,128,128),np.int16)))
        def nonfinite():
            a=np.load(run/'evaluation/w035-cnn-seed17.npy');a[0,5,6]=np.nan;change_array(a)
        bad('nonfinite unmasked logits rejected',nonfinite)
        bad('path escape rejected',lambda:edit_receipt('cnn-seed17',lambda r:r['maps'].update(path='../../escape.npy')))
        def wrong_decision():
            p=run/'RESULTS.json';r=v.read(p);r['decision']='ink authenticated';write(p,r)
        bad('wrong final research decision rejected',wrong_decision)
        def duplicate_output():
            p=run/'RESULTS.json';r=v.read(p);r['outputs'][1]=r['outputs'][0];write(p,r)
        bad('duplicate output receipt rejected',duplicate_output)
        restore()
        # Decision scenarios exercise aggregation and guard thresholds, not only a branch function.
        saved=[v.read(run/p['path']) for p in v.read(run/'RESULTS.json')['outputs']]
        _,_,train=v.endpoint_preflight(run,go)
        for value in train.values():value['termination']=v.PLATEAU
        base=copy.deepcopy(saved)
        for row in base:
            row['overall']['ap']=.5 if row['model']=='profile' else .53
        scenario=copy.deepcopy(base)
        for row in scenario:
            if row['fold']=='w044' and row['model'].startswith('cnn-'):row['overall']['ap']=.519
        result=v.summarize(scenario,train,plan)
        check('large macro gain cannot hide a failing winding',not result['practical_transfer_gate'] and result['decision']=='fit_without_consistent_transfer_prioritize_coverage_or_reference_mismatch')
        scenario=copy.deepcopy(base)
        for row in scenario:
            if row['fold']=='w035' and row['model']=='cnn-seed17':row['overall']['ap']=.5
            elif row['fold']=='w035' and row['model'].startswith('cnn-'):row['overall']['ap']=.55
        check('positive mean cannot hide a seed equal to profile',not v.summarize(scenario,train,plan)['practical_transfer_gate'])
        scenario=copy.deepcopy(base)
        for row in scenario:
            if row['model'].startswith('cnn-'):row['overall']['ap']=.52
        check('per-winding gain at representable practical margin advances',v.summarize(scenario,train,plan)['practical_transfer_gate'])
        scenario=copy.deepcopy(base)
        for row in scenario:
            if row['fold']=='w035' and row['model']=='cnn-seed17':
                for core in row['cores']:
                    if not core['positives']:core['mean_probability']=3/7+.050001
        check('negative-response guard blocks otherwise favorable transfer',v.summarize(scenario,train,plan)['decision']=='blank_response_guard_failed_review_calibration_and_transfer')
        poor=copy.deepcopy(train);poor[('w035',17)]['adequate_fit']=False
        check('poor plateaued fit routes to training diagnosis',v.summarize(base,poor,plan)['decision']=='fit_inadequate_one_training_only_representation_optimizer_diagnosis')
        limited=copy.deepcopy(train);limited[('w035',17)]['termination']='ceiling_without_confirmed_plateau'
        check('ceiling prevents a negative signal conclusion',v.summarize(scenario,limited,plan)['decision']=='optimization_not_demonstrated_saturated_no_native_signal_null')
    receipt=dict(status='PASS',checks=len(checks),check_names=checks,
                 frozen_producer_sha256=v.digest(args.producer),verifier_sha256=v.digest(Path(v.__file__)),
                 qualifier_sha256=v.digest(Path(__file__)),real_study_data_read=False,real_held_out_maps_read=0,
                 neural_forwards=0,scope='Synthetic qualification only; live completed evaluation remains pending')
    write(args.output,receipt);print(json.dumps({k:x for k,x in receipt.items() if k!='check_names'}))


if __name__=='__main__':
    main()
