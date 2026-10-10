"""Independent retained input/output audit. No model, checkpoint decoding or network."""
from pathlib import Path
import hashlib, importlib.util, json, platform, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;LAB=Path('/workspace/herculaneum-lab');REPORTS=LAB/'reports'
PLAN=REPORTS/'twentyfirst-experiment/allocation-implementation/final-plan/experiment-plan.json'
PLAN_SHA='2e1cc8c46d5f7aaade31916cdc0304cd6cb8268e49d8aea73927c730dd535f74'
SOURCE=LAB/'vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py'
NORMALIZER_SHA='52b9d950ad339ff251195bb0b62cc9f7ec9add8617294c7920b825f192ab700e'
verified={};checks=[]
def require(value,description):
 checks.append({'check':description,'pass':bool(value)})
 if not value:raise ValueError(description)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
 p=Path(p);return {'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size}
def verify(p):
 actual=pin(p['path']);require(all(actual[k]==p[k] for k in actual),'pin '+str(p['path']));verified[actual['path']]=actual;return Path(actual['path'])
def raw_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def loadarray(p):
 a=np.load(verify(p),allow_pickle=False);return a,{**p,'shape':list(a.shape),'dtype':a.dtype.str,'raw_c_order_sha256':raw_sha(a)}
def independent_normalize(raw):
 a=np.array(raw,dtype=np.float32,order='K',copy=True);v=a.reshape(-1)
 sd=float(np.std(v));lo=float(np.percentile(v,1.0));hi=float(np.percentile(v,99.0));np.clip(a,lo,hi,out=a)
 v=a.reshape(-1);median=float(np.median(v));mad=float(np.median(np.abs(v-median)));scale=1.4826*mad
 if not np.isfinite(scale) or scale<1e-6:
  candidates=[abs(sd),abs(hi-lo)/2.0];scale=next((x for x in candidates if np.isfinite(x) and x>=1e-6),1.0)
 a-=median;a/=scale;np.nan_to_num(a,copy=False,nan=0.,posinf=0.,neginf=0.);return np.ascontiguousarray(a,dtype=np.float32)
def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def main():
 start=time.monotonic();require(platform.python_version()=='3.12.14' and np.__version__=='2.5.3' and torch.__version__=='2.12.0+cpu','fixed runtime');require(not torch.cuda.is_available(),'CPU runtime')
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 require(sha(PLAN)==PLAN_SHA,'phase21 final plan SHA');plan=json.loads(PLAN.read_text());verified[str(PLAN)]=pin(PLAN)
 p16=json.loads(verify(plan['inference_base_plan']).read_text());runtime=json.loads(verify(p16['runtime']).read_text())
 require(runtime['threads']==4 and runtime['packages']['torch']==torch.__version__ and runtime['packages']['numpy']==np.__version__ and runtime['python']==[3,12,14],'phase16 runtime agreement')
 for key in ['run_ink_reference.py','normalization.py','model_contract_v2.py']:verify(plan['source_code'][key])
 vendor=LAB/'vendor/villa-ink-baseline';manifest=json.loads((vendor/'source-manifest.json').read_text());verified[str(vendor/'source-manifest.json')]=pin(vendor/'source-manifest.json')
 for r in manifest['files']:verify({'path':str(vendor/r['path']),'sha256':r['sha256'],'size_bytes':r['size_bytes']})
 require(sha(SOURCE)==NORMALIZER_SHA,'pure normalizer source SHA');spec=importlib.util.spec_from_file_location('audited_pure_normalization',SOURCE);normalizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(normalizer)
 fixtures=[]
 for name,a in [('zero',np.zeros((17,128,128),dtype=np.float32)),('constant',np.full((17,128,128),73.5,dtype=np.float32)),('ramp',(np.arange(17*128*128,dtype=np.uint32).reshape(17,128,128)%251).astype(np.float32)),('sparse',np.pad(np.ones((1,8,8),dtype=np.float32)*127,((8,8),(60,60),(60,60))))]:
  mine=independent_normalize(a);up=np.ascontiguousarray(normalizer.normalize_robust(a),dtype=np.float32);require(raw_sha(mine)==raw_sha(up),'independent normalization toy '+name);fixtures.append({'name':name,'status':'PASS','sha256':raw_sha(mine)})
 save('synthetic-checks.json',{'status':'PASS','cases':fixtures,'model_calls':0})
 roles={};config=None
 for role,record in plan['historical_inference_results'].items():
  result=json.loads(verify(record).read_text());roles[role]=result
  require(result['status']=='PASS' and result['role']=='frozen_seed42' and result['state_sha256']==plan['backbone_state_sha256'],'historical seed42 identity '+role)
  require(result['model_state_unchanged'] and result['parameter_gradients_absent'] and result['training_calls']==0 and result['network_calls']==0,'historical no-mutation '+role)
  ic=result['initializer_contract'];require(ic['sha256']==plan['checkpoint']['sha256'] and ic['size_bytes']==plan['checkpoint']['size_bytes'],'initializer artifact identity '+role)
  require(ic['strict_state_match'] and not ic['missing_keys'] and not ic['unexpected_keys'],'initializer strict-load receipt '+role)
  if config is None:config=ic['embedded_config']
  else:require(config==ic['embedded_config'],'same seed42 configuration both historical workers')
 require(config['mode']=='flat' and config['patch_size']==[17,128,128] and config['image_normalization']=={'mode':'robust_mad','percentile_lower':1.0,'percentile_upper':99.0},'seed42 config input contract')
 save('seed42-bound-config.json',config);configpin=pin(ROOT/'seed42-bound-config.json')
 h=torch.hann_window(128,periodic=False,dtype=torch.float32);w=torch.outer(h,h);w/=torch.clamp(w.max(),min=torch.finfo(w.dtype).eps);weights=torch.clamp(w,min=.001).numpy()
 hn=.5-.5*np.cos(2*np.pi*np.arange(128,dtype=np.float64)/127);wn=np.outer(hn,hn);wn/=wn.max();wn=np.maximum(wn,.001)
 boxes=[[x,y,128,128] for y in range(0,257,64) for x in range(0,257,64)]
 cases=[]
 for case in plan['cases']:
  arrays={};descriptors={}
  for name,record in case['arrays'].items():arrays[name],descriptors[name]=loadarray(record)
  s=arrays['surface'];require(s.shape==(17,384,384) and s.dtype==np.float32 and np.isfinite(s).all(),'surface17 float32 '+case['id'])
  for name in ['primary_mask','known_mask','surface_mask']:require(arrays[name].shape==(384,384) and arrays[name].dtype==np.bool_,'mask shape dtype '+case['id']+' '+name)
  labels=arrays['labels'];require(labels.shape==(384,384) and labels.dtype in [np.dtype('uint8'),np.dtype('bool')] and np.isin(labels,[0,1]).all(),'binary labels '+case['id'])
  primary=arrays['primary_mask'];known=arrays['known_mask'];surface=arrays['surface_mask'];positive=int(np.count_nonzero(primary&(labels>0)));negative=int(np.count_nonzero(primary&(labels==0)))
  require(positive==case['primary_positive'] and negative==case['primary_negative'],'class counts '+case['id']);require(not np.any(primary&~known) and not np.any(primary&~surface),'primary support inclusion '+case['id'])
  metadata=json.loads(verify(case['baseline_metadata']).read_text());require(metadata==roles[case['role']]['runs'][case['id']],'standalone metadata matches worker result '+case['id'])
  require(metadata['adapter_layer_start']==0 and metadata['adapter_direction']=='forward' and metadata['input_fingerprint_before_model_call'] and metadata['model_input_tensor_unchanged'],'captured input contract '+case['id'])
  require([r['bbox_roi_xywh'] for r in metadata['captured_tiles']]==boxes,'row-major25 capture schedule '+case['id'])
  require([r['bbox_roi_xywh'] for r in metadata['inference']['tiles']]==boxes and all(r['status']=='scored' for r in metadata['inference']['tiles']),'all25 tiles scored '+case['id'])
  captures=[]
  for i,(x,y,width,height) in enumerate(boxes):
   raw=s[:,y:y+height,x:x+width];require(np.any(raw),'nonempty tile '+case['id']+str(i));a=independent_normalize(raw);actual=raw_sha(a);wanted=metadata['captured_tiles'][i]['input_sha256'];require(actual==wanted,'captured normalized input SHA '+case['id']+str(i));captures.append({'bbox_roi_xywh':boxes[i],'input_sha256':actual})
  logits,lp=loadarray(metadata['logits']);scores,sp=loadarray(metadata['scores']);evaluated,ep=loadarray(metadata['evaluated'])
  require(logits.dtype==np.float32 and logits.shape==(25,128,128) and np.isfinite(logits).all(),'logits shape dtype finite '+case['id']);require(scores.dtype==np.float32 and scores.shape==(384,384) and np.isfinite(scores).all(),'score shape dtype finite '+case['id']);require(evaluated.dtype==np.bool_ and evaluated.shape==(384,384) and evaluated.all(),'complete evaluated support '+case['id'])
  total=np.zeros((384,384),dtype=np.float32);denom=np.zeros_like(total);ntotal=np.zeros((384,384),dtype=np.float64);ndenom=np.zeros_like(ntotal)
  for i,(x,y,width,height) in enumerate(boxes):
   probability=torch.from_numpy(logits[i][None,None]).float().sigmoid().numpy()[0,0];total[y:y+128,x:x+128]+=probability*weights;denom[y:y+128,x:x+128]+=weights
   probn=1/(1+np.exp(-logits[i].astype(np.float64)));ntotal[y:y+128,x:x+128]+=probn*wn;ndenom[y:y+128,x:x+128]+=wn
  reconstructed=np.zeros_like(total);np.divide(total,denom,out=reconstructed,where=denom>0);require(np.array_equal(denom>0,evaluated),'recomputed evaluated mask '+case['id']);bitexact=raw_sha(reconstructed)==raw_sha(scores);require(bitexact,'bitwise original FP32 sigmoid Hann reconstruction '+case['id']);numeric=float(np.max(np.abs(ntotal/ndenom-scores.astype(np.float64))));require(numeric<=2e-6,'independent float64 math reconstruction '+case['id'])
  require(not np.any(primary&~evaluated),'all primary support covered '+case['id'])
  cases.append({'id':case['id'],'fragment':case['fragment'],'parent_papyrus':case['parent_papyrus'],'role':'reused_development_diagnostic','historical_phase16_role':case['role'],'bbox_source_xywh':case['bbox_source_xywh'],'native_bbox_yxhw':case['native_bbox_yxhw'],'arrays':descriptors,'selected_surface_planes':list(range(17)),'direction':'forward','layer_start':0,'normalization':'robust_mad1/99 over full17x128x128 patch; no mask','tile_schedule_and_captured_inputs':captures,'seed42':{'metadata':case['baseline_metadata'],'result':case['baseline_result'],'logits':lp,'scores':sp,'evaluated':ep,'state_sha256':plan['backbone_state_sha256']},'counts':{'canvas':384*384,'surface':int(surface.sum()),'known':int(known.sum()),'primary':int(primary.sum()),'primary_positive':positive,'primary_negative':negative,'evaluated':int(evaluated.sum()),'unscored_primary':int((primary&~evaluated).sum()),'primary_equals_known_and_surface':bool(np.array_equal(primary,known&surface))},'reuse_checks':{'all25_normalized_input_SHA256_match':True,'original_FP32_sigmoid_Hann_score_reconstruction_bitwise':True,'independent_float64_reconstruction_max_abs_error':numeric,'scores_outside_0_1':int(((scores<0)|(scores>1)).sum())}})
 contract={'status':'PASS_RETAINED_SEED42_OUTPUTS_REUSABLE_UNDER_THIS_EXACT_INPUT_CONTRACT','phase21_plan':pin(PLAN),'phase16_plan':plan['inference_base_plan'],'runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'cpu_threads':4,'interop_threads':1,'deterministic_algorithms':True,'no_autocast':True},'seed42_artifact_metadata_binding':plan['backbone_checkpoint_lineage'],'seed42_bound_configuration':configpin,'checkpoint_values_read_or_decoded':False,'checkpoint_hash_provenance':'Existing phase21 plan/phase16 initializer receipts and root phase22 retention verification; this narrow audit does not reopen checkpoint bytes.','backbone_state_sha256':plan['backbone_state_sha256'],'inference':{'patch_shape':[17,128,128],'surface_shape':[17,384,384],'surface_dtype':'float32','layer_start':0,'selected_surface_planes':list(range(17)),'direction':'forward','stride':64,'tiles_per_context':25,'mask_passed_to_adapter':False,'sigmoid_before_blending':True,'blending':'FP32 Hann probability-space weights: periodic=False, normalized maximum, floor0.001','output':'raw FP32 sigmoid probabilities; no TIFF quantization or Nieuwlaar display scaling'},'source_pooling_contract':p16['preprocessing'],'cases':cases,'total_contexts':len(cases),'total_captured_tiles':sum(len(c['tile_schedule_and_captured_inputs']) for c in cases),'total_primary_positive':sum(c['counts']['primary_positive'] for c in cases),'total_primary_negative':sum(c['counts']['primary_negative'] for c in cases),'verified_source_and_artifact_pins':list(verified.values()),'scope_limits':['All six regions are reused DEVELOPMENT; no new reserve/independence claim.','Checks authenticate retained input bytes, pre-forward fingerprints and output arithmetic; no model execution or independent weight-to-logit recomputation occurs.','Surface validity and primary annotation support are inherited fixed masks; alignment is not revalidated here.','Root decides whether to reuse seed42 outputs and authorizes any native model execution separately.']}
 require(len(cases)==6 and contract['total_captured_tiles']==150,'six contexts150 tiles');save('INPUT_CONTRACT.json',contract)
 save('verification.json',{'status':'PASS','checks':checks,'check_count':len(checks),'source_code':pin(Path(__file__)),'contract':pin(ROOT/'INPUT_CONTRACT.json'),'synthetic_checks':pin(ROOT/'synthetic-checks.json'),'elapsed_seconds':time.monotonic()-start,'model_constructions':0,'model_calls':0,'network_requests':0,'checkpoint_decodes':0,'input_array_reads':30,'retained_output_array_reads':18,'transient_tool_note':'An earlier read-only source command failed to create due exec-server transport disconnection; no scientific verifier or model had run. Source reads resumed after tool recovery.'});print(json.dumps({'status':'PASS','checks':len(checks),'contexts':6,'tiles':150,'contract':pin(ROOT/'INPUT_CONTRACT.json'),'max_numpy_abs_error':max(c['reuse_checks']['independent_float64_reconstruction_max_abs_error'] for c in cases)}))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  save('STOP.json',{'status':'STOP','error_type':type(e).__name__,'message':str(e),'checks':checks,'source_code':pin(Path(__file__))});raise
