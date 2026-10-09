"""Paired reversal checks built on sealed independent phase22/23 mathematics."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
LAB=PHASE.parent.parent
IDS=['frag3-context1','frag3-context2','frag1-context1','frag1-context2','frag2-context1','frag2-context2']
PINS={
 'review_math':('twentythird-experiment/independent-review/review_math.py','dfedca658920efa32d67e6018cc6ecb24003af995b371ad398512c542bfcf514'),
 'phase24_old_evidence':('twentythird-experiment/independent-review/review_evidence.py','d6479d38afcf25389ee42cbd05fff1cea04f7bbeabbaa75243e194d30f154ffa'),
 'phase24_reference22':('twentysecond-experiment/parity-review/verify_parity.py','98a7356225792d26705d176ea53237531d7b4da960f56ff5d39aa5827cc768ee')}

def need(ok,code):
    if not ok:raise ValueError(code)
def pin(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return dict(path=str(path),sha256=h.hexdigest(),size_bytes=path.stat().st_size)
def checked(record):
    need(pin(record['path'])=={k:record[k] for k in ('path','sha256','size_bytes')},'file_pin:'+record['path'])
    return Path(record['path'])
def json_read(record):return json.loads(checked(record).read_text())
def inherited():
    modules={}
    for name,(relative,digest) in PINS.items():
        path=LAB/'reports'/relative;need(pin(path)['sha256']==digest,'inherited_source_pin')
        if name in sys.modules:
            module=sys.modules[name];need(Path(module.__file__).resolve()==path.resolve(),'inherited_namespace_collision')
        else:
            spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module;spec.loader.exec_module(module)
        modules[name]=module
    return modules['review_math'],modules['phase24_old_evidence'],modules['phase24_reference22']
def load_array(record):
    a=np.load(checked(record),allow_pickle=False)
    if 'shape' in record:need(list(a.shape)==record['shape'],'array_shape')
    if 'dtype' in record:need(a.dtype==np.dtype(record['dtype']),'array_dtype')
    if 'raw_c_order_sha256' in record:need(raw_hash(a)==record['raw_c_order_sha256'],'array_raw_hash')
    need(np.isfinite(a).all(),'array_finite');return a
def raw_hash(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def input_pairs(surface,normalizer):
    need(surface.dtype==np.float32 and surface.shape==(17,384,384),'source_surface')
    original=[];reverse=[]
    for y in range(0,257,64):
        for x in range(0,257,64):
            raw=surface[:,y:y+128,x:x+128];need(raw.any(),'empty_tile')
            value=normalizer(raw);need(value.dtype==np.float32 and value.shape==(17,128,128),'normalized_shape')
            reversed_value=np.ascontiguousarray(value[::-1])
            box=[x,y,128,128]
            original.append(dict(bbox_roi_xywh=box,input_sha256=raw_hash(value)))
            reverse.append(dict(bbox_roi_xywh=box,input_sha256=raw_hash(reversed_value)))
            need(np.ascontiguousarray(reversed_value[::-1]).tobytes()==value.tobytes(),'exact_permutation')
    return original,reverse
def verify_context_input(case,normalizer,helper):
    arrays={k:load_array(v) for k,v in case['arrays'].items()}
    need(set(arrays)=={'surface','primary_mask','known_mask','surface_mask','labels'},'input_arrays')
    for key in ('primary_mask','known_mask','surface_mask'):
        need(arrays[key].dtype==np.bool_ and arrays[key].shape==(384,384),'mask_contract')
    need(arrays['labels'].dtype==np.uint8 and arrays['labels'].shape==(384,384),'labels_contract')
    center=np.zeros((384,384),bool);center[32:352,32:352]=True
    helper.exact(arrays['primary_mask'],arrays['known_mask']&center,'primary_unchanged')
    need(not (arrays['known_mask']&~arrays['surface_mask']).any(),'known_subset_surface')
    y=arrays['labels'][arrays['primary_mask']];need(np.isin(y,[0,1]).all(),'binary_primary')
    need(len(y)==case['counts']['primary'] and int((y==1).sum())==case['counts']['primary_positive']
        and int((y==0).sum())==case['counts']['primary_negative'],'class_counts')
    f,r=input_pairs(arrays['surface'],normalizer)
    need(f==case['forward_inputs'] and r==case['reversed_inputs'],'normalized_input_pair_hashes')
    need(case['forward_surface_planes']==list(range(17)) and case['reversed_surface_planes']==list(range(16,-1,-1)),'plane_order')
    return arrays,f,r
def raw_metric(labels,scores,mask,evaluated,maths):
    need(scores.dtype==np.float32 and ((scores>=0)&(scores<=1)).all(),'raw_probability_contract')
    y,s=maths.selected(labels,scores,mask,evaluated);m=maths.ranking(y,s)
    return dict(roc_auc=m['auc'],average_precision=m['ap'],unique_scores=m['unique_scores'])
def metric_delta(a,b):return {k:a[k]-b[k] for k in ('roc_auc','average_precision')}
def verify_arrays(arrays,evidence,helper):
    need(set(arrays)=={'tile_logits','tile_probabilities','raw_scores','probability_sum','weight_sum','hann','evaluated_mask'},'reverse_array_set')
    logits=arrays['tile_logits'];p=arrays['tile_probabilities'];h=arrays['hann']
    need(logits.shape==p.shape==(25,128,128) and logits.dtype==p.dtype==np.float32,'reverse_tile_shape')
    need(np.isfinite(logits).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'reverse_tile_values')
    checks=dict(sigmoid=helper.compare(helper.sigmoid(logits),p,2e-6,2e-6,'reverse_sigmoid'),
        hann=helper.compare(helper.hann(),h,2e-6,2e-6,'reverse_hann'))
    sums,weights,raw=evidence.reconstruct(p,h)
    for key,value in [('probability_sum',sums),('weight_sum',weights),('raw_scores',raw)]:
        need(arrays[key].shape==(384,384) and arrays[key].dtype==np.float32,'reverse_canvas')
        checks[key]=helper.compare(value,arrays[key],2e-6,2e-6,'reverse_'+key)
    helper.exact(arrays['evaluated_mask'],weights>0,'reverse_evaluated')
    return checks
def forward_arrays(record,evidence,helper):
    arrays={k:load_array(record[k]) for k in ('scores','logits','evaluated')}
    need(arrays['scores'].shape==(384,384) and arrays['scores'].dtype==np.float32,'forward_canvas')
    need(arrays['logits'].shape==(25,128,128) and arrays['logits'].dtype==np.float32,'forward_tiles')
    helper.exact(arrays['evaluated'],np.ones((384,384),bool),'forward_full_evaluated')
    _,_,raw=evidence.reconstruct(helper.sigmoid(arrays['logits']),helper.hann())
    check=helper.compare(raw,arrays['scores'],2e-6,2e-6,'forward_blend')
    return arrays,check
def seed42_value_hashes(record,state_key,schema):
    # Qualified artifact identity precedes the only restricted pickle decode.
    path=checked(record)
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=True)
    need(isinstance(payload,dict) and state_key in payload,'seed42_payload_key')
    state=payload[state_key];need(isinstance(state,dict),'seed42_state_mapping')
    expected={v['name']:v for v in schema['tensor_schema']}
    need(set(state)==set(expected) and len(state)==508,'seed42_named_state')
    hashes={};semantic=hashlib.sha256()
    for name,value in sorted(state.items()):
        need(isinstance(value,torch.Tensor) and value.dtype==torch.float32 and value.device.type=='cpu'
            and list(value.shape)==expected[name]['shape'] and bool(torch.isfinite(value).all()),'seed42_tensor_schema')
        raw=value.detach().contiguous().numpy().tobytes()
        hashes[name]=hashlib.sha256(raw).hexdigest()
        semantic.update(json.dumps([name,list(value.shape),'torch.float32'],separators=(',',':')).encode()+b'\0')
        semantic.update(raw)
    for pair in schema['expected_factory_alias_groups']:
        a,b=pair['names'];need(hashes[a]==hashes[b],'seed42_alias_values')
    checked(record)
    return dict(hashes=hashes,legacy_semantic_sha256=semantic.hexdigest(),config=payload.get('config'),step=payload.get('step'))

def compare_model_identity(model,actual_role,actual_checkpoint,expected_checkpoint,actual_semantic,expected_semantic):
    need(model in ('seed42','native') and actual_role==model,'worker_model_identity')
    need(actual_checkpoint==expected_checkpoint,'worker_checkpoint_identity')
    need(actual_semantic==expected_semantic,'worker_semantic_identity')

def compare_forward_reference(actual,expected,model):
    need(model in ('seed42','native'),'forward_model_identity')
    need(actual==expected,'forward_reference_identity')
