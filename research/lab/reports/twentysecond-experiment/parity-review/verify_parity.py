"""Independent synthetic parity evidence verification; never imports model factories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PROTOCOL_SHA = 'ff6f4c236c884483c350115598ef303805c519da738629acde18783e62593d8f'
CASES = [
 dict(name='native-default', shape=[28,128,128], interval=[0,28], reverse=False, forwards=1),
 dict(name='native-window-0-20',shape=[28,128,128],interval=[0,20],reverse=False,forwards=1),
 dict(name='native-window-3-23',shape=[28,128,128],interval=[3,23],reverse=False,forwards=1),
 dict(name='native-window-5-25',shape=[28,128,128],interval=[5,25],reverse=False,forwards=1),
 dict(name='native-window-8-27',shape=[28,128,128],interval=[8,27],reverse=False,forwards=1),
 dict(name='aligned-default',shape=[21,128,128],interval=[0,21],reverse=False,forwards=1),
 dict(name='native-reverse',shape=[28,128,128],interval=[0,28],reverse=True,forwards=1),
 dict(name='native-tiled',shape=[28,192,192],interval=[0,28],reverse=False,forwards=4),
 dict(name='zero-skip',shape=[28,128,128],interval=[0,28],reverse=False,forwards=0,zero=True),
]
ARRAY_NAMES = {'selected_indices','hann','probability_sum','weight_sum','raw_score','uint8',
 'author_clipped_score','evaluated','blocks','normalized','logits','probabilities'}

class Reject(ValueError): pass
def need(value, code):
    if not value: raise Reject(code)
def pin(path):
    path=Path(path).resolve(); h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return dict(path=str(path),sha256=h.hexdigest(),size_bytes=path.stat().st_size)
def checked(record):
    need(pin(record['path'])=={k:record[k] for k in ('path','sha256','size_bytes')},'pin:'+record['path'])
    return Path(record['path'])
def json_checked(record): return json.loads(checked(record).read_text())
def array_record(a):
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
def exact(a,b,code):
    need(a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C'),code)
def compare(a,b,atol,rtol,code):
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),code+':shape_finiteness')
    aa=a.astype(np.float64); bb=b.astype(np.float64); diff=np.abs(aa-bb)
    tolerance=atol+rtol*np.abs(aa)
    need(bool(np.all(diff<=tolerance)),code+':tolerance')
    return dict(max_absolute=float(diff.max(initial=0)),max_relative=float(np.max(
        np.divide(diff,np.abs(aa),out=np.zeros_like(diff),where=aa!=0),initial=0)),
        reference_zero_mismatch_count=int(np.count_nonzero((aa==0)&(diff!=0))),
        exact_value_count=int(np.count_nonzero(a==b)),count=a.size,
        exact_bitwise=bool(a.dtype==b.dtype and a.tobytes()==b.tobytes()),atol=atol,rtol=rtol)
def volume(shape,zero=False):
    if zero: return np.zeros(shape,dtype=np.uint8)
    zz,yy,xx=np.indices(shape,dtype=np.int64)
    return (1+(17*zz+3*yy+5*xx+(yy*xx)%29+(zz*yy)%23)%251).astype(np.uint8)
def indices(bounds,reverse=False):
    a,b=bounds; need(type(a)is int and type(b)is int and 0<=a<b,'bounds')
    result=np.arange(a,b,dtype=np.int64)
    if len(result)>17:
        left=len(result)//2-17//2; result=result[left:left+17]
    return result[::-1].copy() if reverse else result
def positions(size):
    if size<=128: return [0]
    starts=list(range(0,size-127,64))
    if starts[-1]!=size-128: starts.append(size-128)
    return starts
def blocks(shape):
    h,w=shape[-2:]
    return np.asarray([[y,x,min(128,h-y),min(128,w-x)] for y in positions(h) for x in positions(w)],dtype=np.int64)
def normalize(raw):
    # Independent implementation of audited no-mask robust-MAD scalar equations.
    arr=raw.astype(np.float32,copy=True)
    flat=arr.reshape(-1); std=float(flat.std()); lo=float(np.percentile(flat,1)); hi=float(np.percentile(flat,99))
    np.clip(arr,lo,hi,out=arr)
    center=float(np.median(arr)); scale=1.4826*float(np.median(np.abs(arr-center)))
    if not np.isfinite(scale) or scale<1e-6:
        scale=next((v for v in (abs(std),abs(hi-lo)/2) if np.isfinite(v) and v>=1e-6),1.0)
    arr-=center; arr/=scale
    return np.nan_to_num(arr,nan=0,posinf=0,neginf=0,copy=False)
def hann():
    # Independent double-precision formula; compare with frozen FP32 tolerance.
    w=.5-.5*np.cos(2*np.pi*np.arange(128,dtype=np.float64)/127)
    result=np.outer(w,w); result/=result.max()
    return np.maximum(result,1e-3).astype(np.float32)
def sigmoid(logits):
    x=logits.astype(np.float64); e=np.exp(-np.abs(x))
    return np.where(x>=0,1/(1+e),e/(1+e)).astype(np.float32)
def accumulate(probabilities,weights,schedule,shape):
    ps=np.zeros(shape,dtype=np.float32); ws=np.zeros(shape,dtype=np.float32)
    for p,(y,x,h,w) in zip(probabilities,schedule,strict=True):
        ps[y:y+h,x:x+w]+=p.reshape(128,128)[:h,:w]*weights[:h,:w]
        ws[y:y+h,x:x+w]+=weights[:h,:w]
    raw=np.zeros(shape,dtype=np.float32)
    np.divide(ps,ws,out=raw,where=ws>0)
    return ps,ws,raw
def encode(raw): return (np.clip(raw,0,1)*np.float32(255)).astype(np.uint8)
def decode(encoded): return np.clip((encoded.astype(np.float32)-np.float32(64))/np.float32(128),0,1)
def case_contract(actual, expected):
    for key in ('name','shape','interval','forwards'): need(actual.get(key)==expected[key],'case_contract:'+key)
    for key in ('reverse','zero'): need(bool(actual.get(key,False))==bool(expected.get(key,False)),'case_contract:'+key)
def verify_case(directory,meta,case):
    case_contract(meta['case'],case)
    need(meta['forwards']==case['forwards'],'case_forwards')
    need(set(meta['arrays'])==ARRAY_NAMES,'array_set')
    values={}
    for name in sorted(ARRAY_NAMES):
        a=np.load(directory/(name+'.npy'),allow_pickle=False)
        need(array_record(a)==meta['arrays'][name],'array_record:'+name)
        need(np.isfinite(a).all(),'finite:'+name); values[name]=a
    raw_input=volume(case['shape'],case.get('zero',False)); ids=indices(case['interval'],case['reverse'])
    need(meta['input']==array_record(raw_input),'source_volume')
    exact(values['selected_indices'],ids,'selection'); schedule=blocks(case['shape'])
    exact(values['blocks'],schedule,'schedule')
    n=case['forwards']; h,w=case['shape'][-2:]
    for name,shape in [('normalized',(n,1,1,17,128,128)),('logits',(n,1,1,128,128)),('probabilities',(n,1,1,128,128))]:
        need(values[name].shape==shape and values[name].dtype==np.float32,'shape_dtype:'+name)
    for name in ('hann','probability_sum','weight_sum','raw_score','author_clipped_score'):
        need(values[name].dtype==np.float32 and values[name].shape==((128,128) if name=='hann' else (h,w)),'shape_dtype:'+name)
    need(values['uint8'].dtype==np.uint8 and values['uint8'].shape==(h,w),'uint8_shape_dtype')
    need(values['evaluated'].dtype==np.bool_ and values['evaluated'].shape==(h,w),'evaluated_shape_dtype')
    occupancy=np.array([np.any(raw_input[ids,y:y+vh,x:x+vw]) for y,x,vh,vw in schedule])
    need(int(occupancy.sum())==n,'raw_occupancy_calls')
    expected_inputs=[normalize(raw_input[ids,y:y+vh,x:x+vw])[None,None] for y,x,vh,vw in schedule[occupancy]]
    expected_normalized=np.stack(expected_inputs) if n else np.empty((0,1,1,17,128,128),dtype=np.float32)
    exact(values['normalized'],expected_normalized,'normalized_bytes')
    arithmetic={'hann':compare(hann(),values['hann'],2e-6,2e-6,'hann')}
    arithmetic['sigmoid']=compare(sigmoid(values['logits']),values['probabilities'],2e-6,2e-6,'sigmoid')
    need(bool(((values['probabilities']>=0)&(values['probabilities']<=1)).all()),'probability_range')
    ps,ws,raw=accumulate(values['probabilities'],values['hann'],schedule[occupancy],(h,w))
    for key,expected in [('probability_sum',ps),('weight_sum',ws),('raw_score',raw)]:
        arithmetic[key]=compare(expected,values[key],2e-6,2e-6,'arithmetic:'+key)
    exact(values['evaluated'],ws>0,'evaluated_mask')
    need(meta['evaluated_pixels']==int((ws>0).sum()),'evaluated_count')
    need(bool(((values['raw_score']>=0)&(values['raw_score']<=1)).all()),'raw_range')
    # Own recorded conversion arithmetic, exact including quantization boundaries.
    serialized_raw=np.zeros_like(values['raw_score'])
    np.divide(values['probability_sum'],values['weight_sum'],out=serialized_raw,where=values['weight_sum']>1e-6)
    exact(values['uint8'],encode(serialized_raw),'encoded_exact')
    exact(values['author_clipped_score'],decode(values['uint8']),'decoded_exact')
    if not n:
        need(all(np.count_nonzero(values[k])==0 for k in ('raw_score','uint8','author_clipped_score','probability_sum','weight_sum','evaluated')),'zero_skipped_output')
    return dict(case=case['name'],forwards=n,arithmetic=arithmetic),values
def verify_state(receipt,expected_hashes,schema):
    need(receipt['status']=='PASS','load_status')
    need(receipt['named_tensors']==508 and receipt['named_elements']==68175426 and receipt['unique_groups']==264
         and receipt['unique_parameters']==34546498,'state_counts')
    need(receipt['all_loaded_bytes_exact'] is True and receipt['all_finite'] is True,'state_bytes_claim')
    need(receipt['named_value_sha256']==expected_hashes,'state_value_hashes')
    expected_aliases=sorted(sorted(item['names']) for item in schema['expected_factory_alias_groups'])
    need(receipt['alias_pairs']==expected_aliases and len(expected_aliases)==244,'state_aliases')
    digest=hashlib.sha256(json.dumps(expected_hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    need(receipt['semantic_value_sha256']==digest,'semantic_state_hash')
def qualified_hashes(record,schema):
    path=checked(record)
    from safetensors import safe_open
    expected={item['name']:item for item in schema['tensor_schema']}; hashes={}
    need(len(expected)==508,'reference_count')
    with safe_open(str(path),framework='numpy') as reader:
        need(set(reader.keys())==set(expected),'qualified_keys')
        for name,item in expected.items():
            a=reader.get_tensor(name)
            need(a.dtype==np.float32 and list(a.shape)==item['shape'] and np.isfinite(a).all(),'qualified_schema:'+name)
            hashes[name]=hashlib.sha256(a.tobytes(order='C')).hexdigest()
    checked(record)
    return hashes
def verify_run(plan_path,run_path,release_path):
    plan_path=Path(plan_path); run=Path(run_path); release=json.loads(Path(release_path).read_text())
    need(release['status']=='GO' and release['plan']==pin(plan_path),'root_release')
    freeze=json.loads((HERE/'CHECKER_FREEZE.json').read_text())
    need(freeze['author_plan']==pin(plan_path),'checker_plan_binding')
    for item in freeze['code'].values(): checked(item)
    need(pin(PHASE/'SYNTHETIC_PARITY_PROTOCOL.md')['sha256']==PROTOCOL_SHA,'protocol_sha')
    plan=json.loads(plan_path.read_text())
    need(release['code_freeze']==plan['code_freeze'],'root_code_release')
    authorfreeze=json_checked(plan['code_freeze']); need(authorfreeze['code']==plan['code'],'author_code_freeze')
    for item in plan['code'].values(): checked(item)
    for item in plan['sources'].values(): checked(item)
    need(len(plan['cases'])==9 and plan['forwards_per_worker']==11,'plan_counts')
    for actual,case in zip(plan['cases'],CASES,strict=True): case_contract(actual,case)
    need(json_checked(plan['stage1']['author_receipt'])['status']=='FULL_FILE_INTEGRITY_AND_VALUES_QUALIFIED_LOAD_UNVERIFIED','stage1_author')
    need(json_checked(plan['stage1']['independent_receipt'])['status']=='PASS','stage1_independent')
    schema=json_checked(plan['sources']['reference_schema']); hashes=qualified_hashes(plan['stage1']['artifact'],schema)
    controller=json.loads((run/'controller-result.json').read_text())
    need(controller['status']=='PASS' and controller['model_forwards']==22 and controller['model_constructions']==2
         and controller['training_steps']==0 and controller['no_retry'] is True,'controller_contract')
    need(len(controller['workers'])==2 and [w['factory'] for w in controller['workers']]==['historical','current'],'worker_order')
    need(controller['wall_seconds']<=400,'controller_wall')
    reports=[]; allarrays={}; states={}
    for which,cw in zip(('historical','current'),controller['workers'],strict=True):
        need(cw['returncode']==0 and cw['stop'] is None,'worker_exit')
        need(cw['external_peak_rss_bytes']<=4*1024**3 and cw['elapsed_seconds']<=180,'external_worker_limits')
        result=json_checked(cw['result']); need(result['factory']==which and result['status']=='PASS','worker_identity')
        need(result['model_constructions']==1 and result['model_forwards']==11 and result['training_steps']==0
             and result['gradients']==0 and result['state_unchanged'] is True and result['source_hashes_unchanged'] is True,'worker_invariants')
        need(all(v==0 for v in result['forbidden_calls'].values()) and all(v==0 for v in result['stub_accounting']['runtime_calls'].values()),'forbidden_calls')
        need(result['wall_seconds']<=180 and result['cpu_seconds']<=360 and result['max_rss_bytes']<=4*1024**3,'worker_limits')
        loaded=json_checked(result['exact_strict_load']); final=json_checked(result['final_state_verification'])
        need(loaded['strict'] is True and loaded['missing_keys']==[] and loaded['unexpected_keys']==[]
             and loaded['dtype_conversion'] is False and loaded['checkpoint_format']=='safetensors','strict_load')
        verify_state(loaded,hashes,schema); verify_state(final,hashes,schema); states[which]=loaded
        need(len(result['cases'])==9,'worker_case_count')
        case_reports=[]; allarrays[which]={}
        for meta,case in zip(result['cases'],CASES,strict=True):
            report,arrays=verify_case(run/which/case['name'],meta,case)
            case_reports.append(report); allarrays[which][case['name']]=arrays
        reports.append(dict(factory=which,cases=case_reports,result=pin(cw['result']['path'])))
    cross=[]
    for case in CASES:
        a=allarrays['historical'][case['name']]; b=allarrays['current'][case['name']]
        for name in ('selected_indices','blocks','normalized','evaluated'): exact(a[name],b[name],'cross_exact:'+name)
        row=dict(case=case['name'])
        for name in ('logits','probabilities','raw_score'):
            tol=1e-5 if name=='logits' else 2e-6
            row[name]=compare(a[name],b[name],tol,tol,'cross_'+name)
        row['uint8_mismatch_count']=int(np.count_nonzero(a['uint8']!=b['uint8']))
        row['author_decode_mismatch_count']=int(np.count_nonzero(a['author_clipped_score']!=b['author_clipped_score']))
        cross.append(row)
    for item in plan['sources'].values(): checked(item)
    closure=json_checked(plan['sources']['historical_closure'])
    for item in closure['modules']: checked(item)
    for key in ('normalization_module_for_separate_input_parity','historical_padding_module_for_separate_input_parity',
                'historical_inference_source_for_reviewed_pure_extraction','authored_configuration'): checked(closure[key])
    vendor=json_checked(plan['sources']['current_vendor_manifest']); base=Path(plan['sources']['current_vendor_manifest']['path']).parent
    for item in vendor['files']: need(pin(base/item['path'])['sha256']==item['sha256'],'vendor_final_hash')
    need(sum(p.stat().st_size for p in run.parent.rglob('*') if p.is_file())<=128*1024**2,'stage2_output_size')
    need(sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file())<=768*1024**2,'phase_output_size')
    return dict(status='PASS',workers=reports,cross_factory=cross,plan=pin(plan_path),root_release=pin(release_path),
        independent_state_hashes_checked=len(hashes),state_alias_pairs_checked=244,model_forwards_verified=22,
        verifier_model_constructions=0,verifier_model_forwards=0,network_calls=0,
        scope='Strict state loading and frozen synthetic CPU compatibility only; no ink accuracy or CUDA equality claim.')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan',required=True); p.add_argument('--run',required=True)
    p.add_argument('--root-release',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2); resource.setrlimit(resource.RLIMIT_CPU,(120,)*2)
    def deny(event,args):
        if event.startswith('socket.'): raise Reject('network_forbidden')
    sys.addaudithook(deny); start=time.monotonic(); cpu=time.process_time()
    try: result=verify_run(a.plan,a.run,a.root_release)
    except Exception as exc:
        result=dict(status='STOPPED',failure_type=type(exc).__name__,failure_code=str(exc) if isinstance(exc,Reject) else 'unexpected_checker_exception')
    result.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    with Path(a.output).open('x') as f: json.dump(result,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(dict(status=result['status'],receipt=pin(a.output))))
    return 0 if result['status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
