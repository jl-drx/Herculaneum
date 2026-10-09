"""Independent retained-output arithmetic for the continuation and checkpoint panel."""
import json,sys,hashlib,math
from pathlib import Path
import numpy as np
from scipy.special import expit
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'thirtyseventh-experiment'))
import study_common as common
from verify_evaluation import independent_rank
read=lambda p:json.loads(p.read_text())
native=read(P/'TRAINING_RESULTS.json');old=P.parent/'fortyfourth-experiment'
assert read(P/'training/REPLAY_VERIFICATION.json')['all_equal']
assert native['updates']==8192 and native['new_continuation_updates']==6144
for pin in read(P/'RUN_GO.json')['pins']:common.verify(pin)
checks=0;maximum_rank_difference=0.
for row in native['training_fit']:
    if row['step'] not in [2048,4096,8192]:continue
    for c in row['cases']:
        cid=c['case'];score=np.load(P/f'training/step{row["step"]}-train-{cid}.npy');y=np.load(old/f'crops/{cid}-label.npy')!=0;m=np.load(old/f'training/{cid}-support.npy')
        rank=independent_rank(y[m],score[m].astype(np.float64))
        for key,target in [('average_precision','ap'),('roc_auc','auroc')]:
            delta=abs(rank[key]-c['metrics'][target]);assert delta<1e-12;maximum_rank_difference=max(maximum_rank_difference,delta);checks+=1
for row in native['reserved_development_evaluation']:
    score=np.load(P/f'training/step8192-reserved-{row["quality"]}.npy');cid='248_090925_PI';y=np.load(old/f'crops/{cid}-label.npy')!=0;m=np.load(old/f'training/{cid}-support.npy')
    rank=independent_rank(y[m],score[m].astype(np.float64))
    for key,target in [('average_precision','ap'),('roc_auc','auroc')]:
        delta=abs(rank[key]-row['metrics'][target]);assert delta<1e-12;maximum_rank_difference=max(maximum_rank_difference,delta);checks+=1
prior=read(old/'training/fold0-native-result.json')['training_fit']
assert native['training_fit'][:3]==prior
import torch
torch.set_num_threads(1)
tensor_pins={}
for step in [2048,4096,8192]:
    path=P/f'training/step{step}-continuation.pt';receipt=read(P/f'training/step{step}-checkpoint.json');common.verify(receipt['pin'])
    state=torch.load(path,map_location='cpu',weights_only=True);assert state['step']==step
    for row in state['optimizer']['state'].values():
        assert int(row['step'])==step
        assert torch.isfinite(row['exp_avg']).all() and torch.isfinite(row['exp_avg_sq']).all() and (row['exp_avg_sq']>=0).all()
    tensor_pins[str(step)]=[dict(name=k,shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in state['model'].items()]
common.save(P/'NATIVE_CHECKPOINT_TENSOR_HASHES.json',tensor_pins)

p=common.plan();masks={};labels={}
for c in p['cases']:
    if c['family']=='paris17':
        y=np.load(common.verify(c['arrays']['labels'])).astype(bool);known=np.load(common.verify(c['arrays']['known_mask']));m=np.zeros((384,384),bool);m[32:352,32:352]=known[32:352,32:352]
        assert np.array_equal(m,np.load(common.verify(c['arrays']['primary_mask'])))
    else:
        src=c['legacy_case']['inputs'];y=np.load(common.verify(src['inklabels'])).astype(bool);observed=np.load(common.verify(src['acquisition_observed_mask']));sup=np.load(common.verify(src['supervision_mask']))>0;m=np.zeros((384,384),bool);m[64:320,64:320]=(sup&observed)[64:320,64:320]
        assert np.array_equal(m,np.load(common.verify(c['primary_mask'])))
    masks[c['id']]=m;labels[c['id']]=y
panel=read(P/'checkpoint-panel/RESULTS.json');w=np.hanning(128);weights=np.maximum(np.outer(w,w)/np.outer(w,w).max(),.001);blend_max=0.;pixels=0
for arm in panel['results']:
    for row in arm['regions']:
        score=np.load(common.verify(row['scores']));logits=np.load(common.verify(row['logits']));m=masks[row['id']];y=labels[row['id']]
        # A second ranking implementation: descending grouped recall increments.
        s=score[m];truth=y[m];idx=np.argsort(-s,kind='stable');v=s[idx];t=truth[idx];ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1];tp=np.cumsum(t)[ends].astype(float);fp=ends+1-tp
        ap=float(np.sum(np.diff(np.r_[0,tp])/truth.sum()*tp/(ends+1)));auc=float(np.trapezoid(np.r_[0,tp]/truth.sum(),np.r_[0,fp]/(~truth).sum()))
        for key,val in [('average_precision',ap),('roc_auc',auc)]:
            delta=abs(val-row['metrics'][key]);assert delta<1e-12;maximum_rank_difference=max(maximum_rank_difference,delta);checks+=1
        numerator=np.zeros((384,384),float);denominator=np.zeros_like(numerator)
        for i,(yy,xx) in enumerate(( (yy,xx) for yy in range(0,257,64) for xx in range(0,257,64))):
            numerator[yy:yy+128,xx:xx+128]+=expit(logits[i].astype(float))*weights;denominator[yy:yy+128,xx:xx+128]+=weights
        error=float(abs(numerator/denominator-score).max());assert error<2e-6;blend_max=max(blend_max,error);pixels+=score.size
    for parent,values in arm['parents'].items():
        for key,val in values.items():assert abs(val-np.mean([r['metrics'][key] for r in arm['regions'] if r['parent']==parent]))<1e-15
common.save(P/'RESULT_VERIFICATION.json',dict(status='PASS',native_replay_tensors=42,native_optimizer_checkpoints=3,ranking_values=checks,maximum_ranking_difference=maximum_rank_difference,panel_supports_reconstructed=8,panel_raw_tiles=800,blended_score_pixels=pixels,maximum_blend_difference=blend_max,new_model_forwards=0,scope='Retained-output arithmetic and optimizer/model serialization; no replay of all optimizer transitions, no physical-reference authentication.'))
print(json.dumps(read(P/'RESULT_VERIFICATION.json')))
