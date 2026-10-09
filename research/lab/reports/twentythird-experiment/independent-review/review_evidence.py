"""Source-bound evidence helpers, independent of the phase23 producer."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from review_math import need, model_metrics, disagreement, selected

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REFERENCE=PHASE.parent/'twentysecond-experiment/parity-review/verify_parity.py'
REFERENCE_SHA='98a7356225792d26705d176ea53237531d7b4da960f56ff5d39aa5827cc768ee'
PROTOCOL_SHA='5a8fbf4762ed8dc03c3c13b0f5bcc7acaddaf146cce5987164463161eb76dc02'
IDS=['frag3-context1','frag3-context2','frag1-context1','frag1-context2','frag2-context1','frag2-context2']

def pin(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return dict(path=str(path),sha256=h.hexdigest(),size_bytes=path.stat().st_size)
def checked(record):
    need(pin(record['path'])=={k:record[k] for k in ('path','sha256','size_bytes')},'file_pin:'+record['path'])
    return Path(record['path'])
def read_json(record):return json.loads(checked(record).read_text())
def raw_hash(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def load_array(record):
    a=np.load(checked(record),allow_pickle=False)
    if 'shape' in record:need(list(a.shape)==record['shape'],'array_shape')
    if 'dtype' in record:need(a.dtype==np.dtype(record['dtype']),'array_dtype')
    if 'raw_c_order_sha256' in record:need(raw_hash(a)==record['raw_c_order_sha256'],'array_raw_hash')
    need(np.isfinite(a).all(),'array_finite')
    return a
def reference_helpers():
    need(pin(REFERENCE)['sha256']==REFERENCE_SHA,'reference_source_pin')
    spec=importlib.util.spec_from_file_location('phase23_independent_reference_helpers',REFERENCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def schedule():return [[x,y,128,128] for y in range(0,257,64) for x in range(0,257,64)]
def normalized_fingerprints(surface,normalizer):
    need(surface.dtype==np.float32 and surface.shape==(17,384,384),'surface_contract')
    records=[]
    for x,y,w,h in schedule():
        raw=surface[:,y:y+h,x:x+w]
        need(raw.any(),'unexpected_empty_tile')
        a=normalizer(raw)
        need(a.dtype==np.float32 and a.shape==(17,128,128) and np.isfinite(a).all(),'normalization_contract')
        records.append(dict(bbox_roi_xywh=[x,y,w,h],input_sha256=raw_hash(a)))
    return records
def reconstruct(probabilities,weights):
    need(probabilities.shape==(25,128,128) and probabilities.dtype==np.float32,'probability_shape_dtype')
    need(weights.shape==(128,128) and weights.dtype==np.float32,'hann_shape_dtype')
    sums=np.zeros((384,384),np.float32);counts=np.zeros_like(sums)
    for p,(x,y,w,h) in zip(probabilities,schedule(),strict=True):
        sums[y:y+h,x:x+w]+=p*weights;counts[y:y+h,x:x+w]+=weights
    raw=np.zeros_like(sums);np.divide(sums,counts,out=raw,where=counts>0)
    return sums,counts,raw
def check_arrays(native,helper):
    logits=native['tile-logits'];p=native['tile-probabilities'];weights=native['hann']
    need(logits.dtype==np.float32 and logits.shape==(25,128,128) and np.isfinite(logits).all(),'logits_contract')
    need(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'probability_contract')
    checks=dict(sigmoid=helper.compare(helper.sigmoid(logits),p,2e-6,2e-6,'native_sigmoid'),
        hann=helper.compare(helper.hann(),weights,2e-6,2e-6,'native_hann'))
    sums,counts,raw=reconstruct(p,weights)
    for key,expected in [('probability-sum',sums),('weight-sum',counts),('raw-scores',raw)]:
        need(native[key].dtype==np.float32 and native[key].shape==(384,384),'native_canvas_dtype')
        checks[key]=helper.compare(expected,native[key],2e-6,2e-6,'native_'+key)
    helper.exact(native['evaluated-mask'],counts>0,'native_evaluated')
    helper.exact(native['uint8'],helper.encode(native['raw-scores']),'native_uint8')
    helper.exact(native['decoded'],helper.decode(native['uint8']),'native_decode')
    need(((native['raw-scores']>=0)&(native['raw-scores']<=1)).all(),'native_raw_range')
    return checks
def load_context_sources(case,helper):
    source={k:load_array(v) for k,v in case['arrays'].items()}
    need(set(source)=={'surface','labels','primary_mask','known_mask','surface_mask'},'source_names')
    need(case['id'] in IDS and case['role']=='reused_development_diagnostic','case_role')
    need(case['selected_surface_planes']==list(range(17)) and case['layer_start']==0 and case['direction']=='forward','depth_contract')
    for k in ('primary_mask','known_mask','surface_mask'):
        need(source[k].dtype==np.bool_ and source[k].shape==(384,384),'mask_contract')
    need(source['labels'].dtype==np.uint8 and source['labels'].shape==(384,384),'label_contract')
    center=np.zeros((384,384),bool);center[32:352,32:352]=True
    helper.exact(source['primary_mask'],source['known_mask']&center,'primary_preserved')
    need(not np.any(source['known_mask']&~source['surface_mask']),'known_outside_surface')
    fingerprints=normalized_fingerprints(source['surface'],helper.normalize)
    need(fingerprints==case['tile_schedule_and_captured_inputs'],'audited_normalized_fingerprints')
    p=source['primary_mask'];labels=source['labels'];counts=case['counts']
    need(int(p.sum())==counts['primary'] and int((labels[p]==1).sum())==counts['primary_positive']
        and int((labels[p]==0).sum())==counts['primary_negative'],'primary_class_counts')
    baseline={k:load_array(case['seed42'][k]) for k in ('scores','logits','evaluated')}
    need(baseline['scores'].dtype==np.float32 and baseline['scores'].shape==(384,384),'baseline_score_shape')
    need(baseline['logits'].dtype==np.float32 and baseline['logits'].shape==(25,128,128),'baseline_logit_shape')
    helper.exact(baseline['evaluated'],np.ones((384,384),bool),'baseline_evaluated_full')
    selected(labels,baseline['scores'],p,baseline['evaluated'])
    sums,weights,recomputed=reconstruct(helper.sigmoid(baseline['logits']),helper.hann())
    score_check=helper.compare(recomputed,baseline['scores'],2e-6,2e-6,'baseline_blend')
    return source,baseline,fingerprints,score_check
def canonical_model_metrics(labels,scores,primary,evaluated):
    value=model_metrics(labels,scores,primary,evaluated)
    result={r:dict(roc_auc=value[r]['auc'],average_precision=value[r]['ap'],unique_scores=value[r]['unique_scores'])
        for r in ('raw','uint8','decoded')}
    result.update({k:value[k] for k in ('decoded_zero_fraction','decoded_one_fraction')})
    return result
def context_metrics(case,source,baseline,native):
    labels=source['labels'];primary=source['primary_mask']
    models=dict(seed42=canonical_model_metrics(labels,baseline['scores'],primary,baseline['evaluated']),
        native=canonical_model_metrics(labels,native['raw-scores'],primary,native['evaluated-mask']))
    return dict(parent_papyrus=case['parent_papyrus'],fragment=case['fragment'],models=models,
        native_minus_seed42={k:models['native']['raw'][k]-models['seed42']['raw'][k] for k in ('roc_auc','average_precision')},
        score_differences=disagreement(native['raw-scores'],baseline['scores'],primary))
def parent_metrics(contexts,parent):
    ids=[key for key in IDS if contexts[key]['parent_papyrus']==parent]
    need(len(ids)=={'Paris1':2,'Paris2':4}[parent],'parent_size')
    models={m:{k:float(np.mean([contexts[c]['models'][m]['raw'][k] for c in ids])) for k in ('roc_auc','average_precision')}
        for m in ('native','seed42')}
    return dict(context_ids=ids,models=models,native_minus_seed42={k:models['native'][k]-models['seed42'][k]
        for k in ('roc_auc','average_precision')})
