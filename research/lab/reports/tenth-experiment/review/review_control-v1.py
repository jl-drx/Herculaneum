"""One completed-output offline preservation review; no model forward/backward."""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import numpy as np

LAB=Path('/workspace/herculaneum-lab')
OLD=LAB/'reports/eighth-experiment'
ARMS=['frozen_seed42','A_smooth','B_smooth','A_preserve','B_preserve']
NEW=ARMS[-2:]
RANK=LAB/'reports/sixth-experiment/independent-metric-review/review_metrics.py'
RANK_SHA='e3cad20abca6f4daa82c34f25c3237ba4626fbda43c90cd36e0735f4923c0452'
OLD_PLAN_SHA='9d72d3177d5aa68cd5d59db6f316ed1a5c335f442b6c5c2439f45bfdaebe29cd'
NORMALIZER=LAB/'vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py'

def require(value,message):
    if not value: raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result); return result

def numpy_state_hash(values):
    digest=hashlib.sha256()
    for key,array in sorted(values.items()):
        array=np.ascontiguousarray(array); dtype={np.dtype('float32'):'torch.float32',np.dtype('bool'):'torch.bool'}[array.dtype]
        digest.update(json.dumps([key,list(array.shape),dtype],separators=(',',':')).encode()+b'\0');digest.update(array.tobytes())
    return digest.hexdigest()

def review(plan_path,expected_sha,run_dir,analysis_dir,output):
    require(not output.exists(),'Fresh independent output required')
    require(sha(plan_path)==expected_sha,'Final control plan SHA differs')
    pins={};max_error=0.
    def pin(path,expected=None):
        path=Path(path).resolve();actual=dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha(path))
        if expected is not None:
            require(actual['sha256']==expected['sha256'] and actual['size_bytes']==expected.get('size_bytes',expected.get('byte_length')),'Retained hash/size differs: '+path.name)
        require(str(path) not in pins or pins[str(path)]==actual,'Conflicting receipts');pins[str(path)]=actual;return path
    def collect(value):
        if isinstance(value,dict):
            if {'path','sha256'}<=value.keys() and ('size_bytes' in value or 'byte_length' in value):pin(value['path'],value)
            for item in value.values():collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
    def read(path):
        value=json.loads(pin(path).read_text());collect(value);return value
    def load(item):return np.load(pin(item['path'],item),allow_pickle=False)
    def close(actual,saved,label):
        nonlocal max_error
        require((actual is None)==(saved is None),'Undefined metric differs: '+label)
        if actual is not None:
            error=abs(actual-saved);max_error=max(max_error,error);require(error<=1e-12,'Metric differs: '+label)
    pin(__file__);pin(RANK);require(sha(RANK)==RANK_SHA,'Independent rank source changed');rank=module(RANK,'tenth_independent_rank').ranked_metrics
    plan=read(plan_path);base=read(plan['base_plan']['path'])
    require(plan['kind']=='preservation-continuation-v1' and plan['ready_for_execution'],'Ready preservation plan required')
    require(plan['base_plan']['sha256']==OLD_PLAN_SHA,'Ninth plan differs')
    for key in ('adaptation','development','initializer','teacher','schedule','exposure','helper','metrics','cutoff','calibration'):
        require(plan[key]==base[key],'Matched input/mechanics contract differs: '+key)
    require(plan['training']==base['training'] and plan['diagnostic']==base['diagnostic'],'Matched smoothed training and diagnostic contract differs')
    require(plan['preservation']==dict(coefficient=1.,temperature=1.,dtype='float64',teacher='unchanged seed42 initializer',
        support='A: M; B: M union P; separate masked means',pseudo_coefficient=.25,minimum_mean=-1e-6,
        first_logit_absolute_tolerance=1e-7,first_element_absolute_tolerance=1e-7,
        first_output_gradient_absolute_tolerance=1e-7,first_parameter_norm_absolute_tolerance=1e-6),'Preservation specification differs')
    require(plan['diagnostic_role']=='reused_development_retention_diagnostic' and plan['screens']==dict(specificity_numerator=95,specificity_denominator=100,mitigation_mean_sensitivity_gain_min=.02,per_context_sensitivity_delta_min=-.02,per_context_auc_delta_min=-.005,per_context_ap_delta_min=-.01),'Diagnostic role/screen contract differs')
    require(set(plan['historical_results'])==set(ARMS[:3]) and plan['new_arms']==NEW,'Fixed historical/new role scope differs')
    require(read(run_dir/'frozen-plan.json')==plan,'Executed plan differs')
    execution=read(run_dir/'execution.json');require(execution['plan']['sha256']==expected_sha and execution['retries']==0,'Execution provenance/retries differ')
    require([w['arm'] for w in execution['workers']]==NEW and all(w['exit_code']==0 for w in execution['workers']),'Exactly two successful new workers required')
    report=read(analysis_dir/'metrics.json');require(report['status']=='PASS' and report['plan']['sha256']==expected_sha,'Analysis provenance differs')
    require(report['screen_criteria']==plan['screens'],'Reported screen criteria differ')
    workers={a:read(plan['historical_results'][a]['path'] if a in plan['historical_results'] else run_dir/a/'result.json') for a in ARMS}
    for arm,w in workers.items():
        require(w['status']=='PASS' and w['arm']==arm and w['initializer']==plan['initializer'] and w['initial_state_sha256']==plan['historical_initial_state_sha256'],'Worker/initializer identity differs')
        require(w['plan']==(plan['historical_plan_records'][arm] if arm not in NEW else execution['plan']),'Worker plan identity differs')
        require(report['model_results'][arm]['sha256']==sha(plan['historical_results'][arm]['path'] if arm not in NEW else run_dir/arm/'result.json'),'Analysis role receipt differs')
        if arm not in NEW:require(Path(plan['historical_results'][arm]['path']).resolve()==(OLD if arm=='frozen_seed42' else LAB/'reports/ninth-experiment')/'implementation/run'/arm/'result.json','Historical worker replaced')
        else:
            require(w['threads']==4 and w['model_contract']['strict_state_match'] and w['model_contract']['sha256']==plan['initializer']['sha256'],'New model/runtime differs')
            require(w['wall_seconds']<=plan['limits']['wall_seconds'] and w['cpu_seconds']<=plan['limits']['cpu_seconds'] and w['peak_rss_bytes']<=plan['limits']['rss_acceptance_bytes'],'New worker resource cap exceeded')
            require(w['torch_version']==workers['A_smooth']['torch_version'],'Torch runtime changed')
    for directory in (run_dir,analysis_dir):
        for path in directory.rglob('*'):
            if path.is_file():pin(path)
    require(sum(p.stat().st_size for d in (run_dir,analysis_dir) for p in d.rglob('*') if p.is_file())<=plan['limits']['output_bytes'],'New output budget exceeded')

    def arrays(case):
        a={k:load(v) for k,v in case['inputs'].items()};shape=a['surface'].shape[1:]
        require(a['surface'].dtype==np.uint8 and a['surface'].shape[0]==21 and shape in ((256,256),(384,384)),'Source encoding differs')
        require(a['acquisition_observed_mask'].dtype==bool and a['acquisition_observed_mask'].shape==shape and a['acquisition_observed_mask'].all(),'Observed support differs')
        for key in ('inklabels','supervision_mask','validation_mask'):
            require(a[key].dtype==np.uint8 and a[key].shape==shape and set(np.unique(a[key]))<={0,1},'Binary labels/masks differ')
        require(np.array_equal(a['source_observed_nonzero_mask'],np.any(a['surface'][2:19]!=0,axis=0)),'Source occupancy differs');return a
    def domain(case):
        a=arrays(case);role=case['evaluation_participation_role'];require(role in ('public_validation','supplied_supervision'),'Reference role differs')
        part=a['validation_mask']>0 if role=='public_validation' else a['supervision_mask']>0
        require(np.array_equal(part,a['evaluation_participation_mask']>0),'Reference participation differs')
        mask=np.zeros(part.shape,bool);mask[64:-64,64:-64]=True;mask&=part&a['acquisition_observed_mask'];return a,a['inklabels']>0,mask
    loss_review=module(pin(Path(__file__).with_name('loss_receipts.py')),'tenth_loss_review').review_losses(plan,workers,pin)
    collect(loss_review);training=loss_review['arms'];first_hash=training['A_preserve']['first_step_input_target_sha256']

    def operating(y,s,threshold):
        predicted=s.astype(np.float64)>=threshold;p,q=int(y.sum()),int((~y).sum());tp,fp=int((y&predicted).sum()),int((~y&predicted).sum())
        return dict(threshold=threshold,pixel_count=int(y.size),supplied_positive_pixels=p,supplied_negative_pixels=q,confusion=dict(tp=tp,fp=fp,tn=q-fp,fn=p-tp),sensitivity=tp/p,specificity=(q-fp)/q,precision=tp/(tp+fp) if tp+fp else None,balanced_accuracy=(tp/p+(q-fp)/q)/2,specificity_target_met_exact=fp<=q//20)
    def check_op(value,saved):
        for key in ('threshold','pixel_count','supplied_positive_pixels','supplied_negative_pixels','confusion'):require(value[key]==saved[key],'Operating counts differ')
        for key in ('sensitivity','specificity','precision','balanced_accuracy'):close(value[key],saved[key],'operating/'+key)
        if 'specificity_target_met_exact' in saved:require(value['specificity_target_met_exact']==saved['specificity_target_met_exact'],'Transfer95 flag differs')
    def raw(maps,shape):
        scores,evaluated=load(maps['scores']),load(maps['evaluated']);require(scores.dtype==np.float32 and scores.shape==shape and evaluated.dtype==bool and evaluated.shape==shape,'Raw map encoding differs')
        require(np.all((scores[np.isfinite(scores)]>=0)&(scores[np.isfinite(scores)]<=1)),'Raw score range differs');return scores,evaluated
    locks={};calibration={}
    for arm,w in workers.items():
        lock=read(w['threshold_lock']['path']);require(lock['arm']==arm and lock['plan']==w['plan'],'Lock provenance differs')
        require(lock['fresh_inference_started' if arm=='frozen_seed42' else 'diagnostic_inference_started'] is False,'Pre-inference lock differs');selections={};calibration[arm]={}
        for case in plan['development']:
            _,y,mask=domain(case);maps=w['development'][case['id']];require(maps==lock['development'][case['id']]['maps'],'Lock maps differ');s,e=raw(maps,mask.shape);require(e[mask].all() and np.isfinite(s[mask]).all(),'Calibration evaluated support differs')
            neg=s[mask&~y];allowed=neg.size//20;boundary=np.partition(neg,neg.size-allowed-1)[neg.size-allowed-1];threshold=float(np.nextafter(boundary,np.float32(np.inf),dtype=np.float32));selection=lock['development'][case['id']]['selection']
            require(selection['threshold']==threshold and selection['excluded_negative_boundary']==float(boundary) and selection['negative_count']==neg.size and selection['maximum_false_positives']==allowed and selection['boundary_tie_count']==int((neg==boundary).sum()),'Local cutoff/tie budget differs')
            require(selection['development_false_positives']==int((neg.astype(np.float64)>=threshold).sum())<=allowed and int((neg>=boundary).sum())>allowed,'Local FP crossing bound differs');selections[case['id']]=threshold
            ranking=rank(y[mask],s[mask]);saved=report['calibration'][arm][case['id']];close(ranking['roc_auc'],saved['auc'],'calibrationAUC');close(ranking['average_precision'],saved['ap'],'calibrationAP')
            ops={rule:operating(y[mask],s[mask],lock[rule]) for rule in ('primary','secondary')}
            for rule,value in ops.items():check_op(value,saved['operating'][rule])
            calibration[arm][case['id']]=dict(ranking=ranking,operating=ops)
        require(lock['primary']==max(selections.values()) and lock['secondary']==selections['pherc1667-w029'],'Fixed cutoff rules differ');locks[arm]=dict(primary=lock['primary'],secondary=lock['secondary'],region_cutoffs=selections)
    diagnostic={}
    for case in plan['diagnostic']:
        a,y,mask=domain(case);before=mask.copy();maps={}
        require(case['physical_group']=='PHerc0814' and case['evaluation_participation_role']=='supplied_supervision','Diagnostic source/role differs')
        for arm,w in workers.items():
            s,e=raw(w['fresh' if arm=='frozen_seed42' else 'diagnostic'][case['id']],mask.shape);maps[arm]=s;mask&=e&np.isfinite(s)
        stored=np.load(pin(analysis_dir/case['id']/'primary-common-mask.npy'),allow_pickle=False);require(stored.dtype==bool and np.array_equal(stored,mask),'Diagnostic common mask differs');saved=report['diagnostic'][case['id']]
        require(saved['fixed_primary_count']==int(before.sum()) and saved['common_primary_count']==int(mask.sum()) and saved['positive']==int((mask&y).sum()) and saved['negative']==int((mask&~y).sum()),'Diagnostic class support differs');models={}
        require(saved['role']==plan['diagnostic_role'] and min(saved['positive'],saved['negative'])>=256,'Diagnostic role/class gate differs');close(float(y[mask].mean()),saved['prevalence'],'prevalence')
        for arm,s in maps.items():
            r=rank(y[mask],s[mask]);old=saved['models'][arm];close(r['roc_auc'],old['auc'],'diagnosticAUC');close(r['average_precision'],old['ap'],'diagnosticAP')
            ops={rule:operating(y[mask],s[mask],locks[arm][rule]) for rule in ('primary','secondary')}
            for rule,value in ops.items():check_op(value,old['operating'][rule])
            quantiles=dict(zip(('min','q05','median','q95','max'),map(float,np.quantile(s[mask],[0,.05,.5,.95,1]))))
            for key,value in quantiles.items():close(value,old['score_quantiles'][key],'quantile')
            models[arm]=dict(auc=r['roc_auc'],ap=r['average_precision'],midpoint=r,operating=ops,score_quantiles=quantiles)
        diagnostic[case['id']]=dict(models=models,counts=dict(positive=int((mask&y).sum()),negative=int((mask&~y).sum()),pixels=int(mask.sum())))
    def endpoints(models):return {arm:dict(auc=v['auc'],ap=v['ap'],**{rule+'_'+key:v['operating'][rule][key] for rule in ('primary','secondary') for key in ('sensitivity','specificity')}) for arm,v in models.items()}
    def contrasts(values):
        delta=lambda l,r:{key:values[l][key]-values[r][key] for key in values[l]}
        c=dict(preservation_with_manual_only=delta('A_preserve','A_smooth'),preservation_with_pseudo=delta('B_preserve','B_smooth'),pseudo_without_preservation=delta('B_smooth','A_smooth'),pseudo_with_preservation=delta('B_preserve','A_preserve'))
        c['interaction_difference_of_differences']={k:c['pseudo_with_preservation'][k]-c['pseudo_without_preservation'][k] for k in values['A_smooth']};return c
    def check_tree(a,b,label):
        require(set(a)==set(b),'Summary fields differ: '+label)
        for key,v in a.items():
            if isinstance(v,dict):check_tree(v,b[key],label+'/'+key)
            elif isinstance(v,(bool,str)):require(v==b[key],'Summary value differs: '+label+'/'+key)
            else:close(v,b[key],label+'/'+key)
    endpoints_by_case=[endpoints(v['models']) for v in diagnostic.values()];require(len(endpoints_by_case)==2,'Two reused contexts required')
    macro={arm:{key:math.fsum(row[arm][key] for row in endpoints_by_case)/2 for key in endpoints_by_case[0][arm]} for arm in ARMS};factorial=contrasts(macro)
    check_tree(macro,report['equal_context_macro'],'macro');check_tree(factorial,report['factorial_contrasts'],'factorial')
    for case,row in zip(plan['diagnostic'],endpoints_by_case):check_tree(contrasts(row),report['diagnostic'][case['id']]['factorial_contrasts'],'contextFactorial')
    screens={}
    for new,old in (('A_preserve','A_smooth'),('B_preserve','B_smooth')):
        deltas=[];mitigation=[];retention=[];sens=[]
        for context in diagnostic.values():
            m,o,f=[context['models'][a] for a in (new,old,'frozen_seed42')];p=context['counts']['positive'];d=Fraction(m['operating']['primary']['confusion']['tp']-o['operating']['primary']['confusion']['tp'],p);sens.append(d)
            kept=m['operating']['primary']['specificity_target_met_exact'];mitigation.append(kept and d>=Fraction(-1,50) and m['auc']-o['auc']>=-.005 and m['ap']-o['ap']>=-.01)
            baseline_delta=Fraction(m['operating']['primary']['confusion']['tp']-f['operating']['primary']['confusion']['tp'],p);retention.append(kept and baseline_delta>=Fraction(-1,50) and m['auc']-f['auc']>=-.005 and m['ap']-f['ap']>=-.01)
            deltas.append(dict(sensitivity_vs_unpreserved=float(d),auc_vs_unpreserved=m['auc']-o['auc'],ap_vs_unpreserved=m['ap']-o['ap'],sensitivity_vs_frozen=float(baseline_delta),auc_vs_frozen=m['auc']-f['auc'],ap_vs_frozen=m['ap']-f['ap'],specificity_target_met_exact=kept))
        screens[new]=dict(mitigation_pass=all(mitigation) and sum(sens,Fraction())/2>=Fraction(1,50),retention_margin_pass=all(retention),mean_primary_sensitivity_delta=float(sum(sens,Fraction())/2),per_context_deltas=deltas)
        for rule,comparator,passed in (('mitigation',old,screens[new]['mitigation_pass']),('retention','frozen_seed42',screens[new]['retention_margin_pass'])):
            saved=report['screens'][rule][new];require(saved['comparator']==comparator and saved['pass_descriptive_screen']==passed and saved['mean_gain_required']==(rule=='mitigation'),'Descriptive screen differs')
            if rule=='mitigation':require(saved['mean_gain_check']==(sum(sens,Fraction())/2>=Fraction(1,50)),'Mean mitigation check differs')
            else:require(saved['mean_gain_check'] is None,'Retention mean-gain scope differs')
            direct=[]
            for case_id,context in diagnostic.items():
                m,o=context['models'][new],context['models'][comparator];delta=dict(sensitivity=m['operating']['primary']['sensitivity']-o['operating']['primary']['sensitivity'],auc=m['auc']-o['auc'],ap=m['ap']-o['ap']);direct.append(delta['sensitivity'])
                checks=dict(specificity=m['operating']['primary']['specificity_target_met_exact'],sensitivity=delta['sensitivity']>=-.02,auc=delta['auc']>=-.005,ap=delta['ap']>=-.01);row=saved['per_context'][case_id]
                check_tree(delta,row['deltas'],'screenDelta');require(checks==row['checks'] and row['pass_context']==all(checks.values()),'Per-context screen checks differ')
            close(math.fsum(direct)/2,saved['mean_sensitivity_delta'],'screenMean')
    require(report['consistent_mitigation_both_pairs']==all(v['mitigation_pass'] for v in screens.values()),'Shared mitigation flag differs')
    costs=report['adaptation_cost'];require(costs['teacher_predictions_bitwise_identical'] is True,'Adaptation cross-arm teacher receipt differs')
    online={}
    for arm,old,coefficient in (('A_preserve','A_smooth',0.),('B_preserve','B_smooth',.25)):
        tr=workers[arm]['training'];rows=[json.loads(line) for line in pin(tr['steps_log']['path'],tr['steps_log']).read_text().splitlines()]
        old_tr=workers[old]['training'];old_rows=[json.loads(line) for line in pin(old_tr['steps_log']['path'],old_tr['steps_log']).read_text().splitlines()]
        keys=('loss_U','loss_P','loss_R','original_objective','loss_D_M','loss_D_P','total_loss','probability_drift_manual','probability_drift_pseudo')
        expected={key:np.array([r[key] for r in rows],dtype=np.float64) for key in keys}
        expected.update(step=np.arange(1,33),historical_U=np.array([r['loss_U'] for r in old_rows]),
            historical_P=np.array([r['loss_P'] for r in old_rows]),historical_original_objective=np.array([r['total_loss'] for r in old_rows]),
            teacher_U=np.array([r['teacher_U'] for r in rows]),teacher_P=np.array([r['teacher_P'] for r in rows]))
        expected['teacher_original_objective']=expected['teacher_U']+coefficient*expected['teacher_P']
        saved=costs['arms'][arm]
        require(saved['comparator']==old and saved['last_eight_steps']==[25,32] and saved['pseudo_coefficient']==coefficient
                and saved['preupdate_logits']==tr['preupdate_logits'],'Adaptation summary scope differs')
        with np.load(pin(saved['curves']['path'],saved['curves']),allow_pickle=False) as curves:
            require(set(curves.files)==set(expected),'Online curve set differs')
            for key,value in expected.items():require(np.array_equal(curves[key],value),'Saved online curve differs: '+key)
        means={key:math.fsum(value[-8:])/8 for key,value in expected.items() if key!='step'}
        check_tree(means,saved['last_eight_means'],'lastEightMeans')
        deltas=dict(original_objective_delta_vs_historical=means['original_objective']-means['historical_original_objective'],
            original_objective_delta_vs_initializer=means['original_objective']-means['teacher_original_objective'],
            manual_loss_delta_vs_historical=means['loss_U']-means['historical_U'],
            manual_loss_delta_vs_initializer=means['loss_U']-means['teacher_U'])
        for key,value in deltas.items():close(value,saved[key],'onlineDelta/'+key)
        online[arm]=dict(last_eight_means=means,**deltas)
    cp=module(pin(Path(__file__).with_name('checkpoint_receipts.py')),'tenth_checkpoint_review').verify_checkpoints(plan,workers);collect(cp)
    for path,item in pins.items():require(Path(path).stat().st_size==item['size_bytes'] and sha(path)==item['sha256'],'Input/output changed during independent review')
    output.mkdir(parents=True);result=dict(status='PASS',plan_sha256=expected_sha,actual_new_updates=64,historical_paired_updates_reused=64,historical_C_excluded=True,verified_unique_records=len(pins),verified_input_output_records_before_and_after=list(pins.values()),training_receipts=training,full_loss_review=loss_review,online_adaptation_review=online,first_step_input_target_sha256=first_hash,checkpoint_review=cp,development_thresholds=locks,calibration=calibration,diagnostic=diagnostic,equal_context_macro=macro,factorial_contrasts=factorial,independent_protocol_screens=screens,shared_preservation_mitigation=all(v['mitigation_pass'] for v in screens.values()),maximum_absolute_metric_difference=max_error,max_saved_loss_formula_difference=max(loss_review['maximum_absolute_reconstruction_errors'].values()),benchmark_metric_code_imported=False,forward_calls=0,backward_calls=0,network_requests=0,loss_limit='All64preupdate losses independently reconstructed from raw student/teacher logits and source-derived masks/targets; gradients proven only on artificial logits plus worker sentinel receipts',interpretation='One-seed development retention diagnostics on previously examined same-specimen contexts; historical/teacher exposure unresolved; no fresh holdout or accuracy claim')
    target=output/'independent-verification.json';target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='PASS',path=str(target),sha256=sha(target),records=len(pins),max_metric_difference=max_error,screens=screens)))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('plan','run-dir','analysis-dir','output-dir'):parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--plan-sha256',required=True);args=parser.parse_args()
    def blocked(*a,**k):raise RuntimeError('Independent review attempted network access')
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.socket.sendto=blocked;socket.create_connection=blocked;socket.getaddrinfo=blocked
    review(args.plan.resolve(),args.plan_sha256,args.run_dir.resolve(),args.analysis_dir.resolve(),args.output_dir.resolve())
