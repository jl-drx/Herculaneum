#!/usr/bin/env python3
"""Per-context and equal-context specimen summaries; no pooled pixel inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from operating_point import operating_metrics
from panel_common import ARMS, base, load_case, load_plan, metric_module


def plot_context(directory,case,arrays,labels,primary,secondary,scores):
    import os
    os.environ['MPLCONFIGDIR']=str(directory/'.matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    axes[0,0].imshow(arrays['surface'][case['data']['label_plane']],cmap='gray',vmin=0,vmax=255)
    axes[0,0].set_title('Pooled source center plane; intensity0–255')
    colors=np.zeros((*labels.shape,3),dtype=np.float32)
    colors[secondary & ~labels]=[.25,.25,.7];colors[secondary & labels]=[1,.8,0]
    axes[0,1].imshow(colors);axes[0,1].contour(primary,levels=[.5],colors=['white'],linewidths=.5)
    axes[0,1].set_title('Supplied annotations; white primary edge')
    axes[0,1].legend(handles=[Patch(color=[1,.8,0],label='Positive'),Patch(color=[.25,.25,.7],label='Negative'),
        Patch(color='black',label='Excluded/unsupported')],loc='lower left',fontsize=6,facecolor='#555555',labelcolor='white')
    for axis,role in zip([axes[0,2],*axes[1]],ARMS):
        image=axis.imshow(scores[role],cmap='viridis',vmin=0,vmax=1)
        axis.contour(primary,levels=[.5],colors=['white'],linewidths=.5);axis.set_title(role+'; raw0–1 score')
    for axis in axes.flat:axis.set_axis_off()
    fig.colorbar(image,ax=list(axes.flat),shrink=.65,label='Uncalibrated model score')
    data=case['data'];fig.suptitle(f"{case['id']} / {case['physical_group']}; sourceXYWH {data['bbox_source_xywh']}; {data['pixel_spacing_um']}µm/px\nReserved from our tuning; supplied-label agreement, no authenticated geometry or reading",fontsize=10)
    fig.savefig(directory/'comparison.png',dpi=130);plt.close(fig)


def mean_range(values):
    values=[v for v in values if v is not None]
    return dict(mean=float(np.mean(values)) if values else None,min=min(values) if values else None,max=max(values) if values else None)


def analyze(plan_path,output):
    plan,plan_path,lock=load_plan(plan_path,execute=True)
    output=Path(output).resolve(strict=True)
    base.require((output/'frozen-plan.json').read_bytes()==plan_path.read_bytes(),'Frozen panel plan differs')
    base.require((output/'threshold-lock.json').read_bytes()==Path(plan['threshold_lock']['path']).read_bytes(),'Locked thresholds changed')
    analysis=output/'analysis';analysis.mkdir(exist_ok=False)
    metric=metric_module(plan);contexts={};map_records=[]
    for case in plan['cases']:
        case_id=case['id'];directory=analysis/case_id;directory.mkdir(exist_ok=False)
        arrays,labels,primary,secondary,preflight=load_case(case)
        scores={};common=np.ones(labels.shape,dtype=bool)
        for role in ARMS:
            metadata_path=output/case_id/role/'metadata.json';metadata=json.loads(metadata_path.read_text())
            base.require(metadata['case_id']==case_id and metadata['role']==role
                and metadata['plan']['sha256']==base.record(plan_path)['sha256']
                and metadata['source_inputs']==case['inputs'] and metadata['threshold_lock']==plan['threshold_lock']
                and metadata['source_provenance']==case['provenance'] and metadata['execution_code']==plan['code']
                and metadata['data_contract']==case['data'], 'Output identity differs')
            base.require(metadata['model']['sha256']==lock['models_and_maps'][role]['checkpoint']['sha256']
                and metadata['model']['strict_state_match'] and not metadata['model']['missing_keys']
                and not metadata['model']['unexpected_keys'],'Strict model evidence differs')
            score_path=base.verify_record(metadata['raw_scores']);evaluated_path=base.verify_record(metadata['evaluated_mask'])
            scores[role]=np.load(score_path,allow_pickle=False);evaluated=np.load(evaluated_path,allow_pickle=False)
            base.require(scores[role].dtype==np.float32 and scores[role].shape==labels.shape
                and evaluated.dtype==np.bool_ and evaluated.shape==labels.shape,'Output array contract differs')
            finite=np.isfinite(scores[role]);base.require(np.all((scores[role][finite]>=0)&(scores[role][finite]<=1)),'Scores outside raw range')
            common &= evaluated & finite
            map_records.extend([base.record(score_path),base.record(evaluated_path),base.record(metadata_path)])
        domains=dict(primary=primary & common,secondary=secondary & common)
        counts={name:base.class_counts(labels,mask) for name,mask in domains.items()}
        for name,mask in domains.items():np.save(directory/f'{name}-common-mask.npy',mask,allow_pickle=False)
        base.write_json(directory/'common-support.json',dict(preflight=preflight,final=counts,
            primary_lost_pixels=int((primary&~common).sum()),secondary_lost_pixels=int((secondary&~common).sum())))
        base.require(base.sufficient(counts['primary']),'STOP: common primary class support inadequate; no relaxation')
        results={}
        for name,mask in domains.items():
            models={};occupancy=arrays['source_observed_nonzero_mask']
            for role in ARMS:
                models[role],curve=metric.binary_metrics(labels[mask],scores[role][mask])
                curve_path=directory/f'{name}-{role}-curves.npz';np.savez_compressed(curve_path,**curve)
                models[role]['curves']=base.record(curve_path)
                threshold=lock['thresholds'][role]['threshold']
                operating=operating_metrics(labels[mask],scores[role][mask],threshold)
                negative=operating['supplied_negative_pixels'];tn=operating['confusion']['tn']
                operating['reserved_specificity_meets95_percent']=(tn*20>=negative*19) if negative else None
                operating['scope']='Sensitivity at development-locked threshold; not matched95% reserved specificity if transfer fails.'
                models[role]['development_locked_operating_point']=operating
                models[role]['occupancy_strata']={}
                for stratum,subset in [('nonzero_profiles',mask&occupancy),('observed_zero_profiles',mask&~occupancy)]:
                    stratum_metric,_=metric.binary_metrics(labels[subset],scores[role][subset])
                    stratum_metric['development_locked_operating_point']=operating_metrics(labels[subset],scores[role][subset],threshold)
                    models[role]['occupancy_strata'][stratum]=stratum_metric
            dense,control=models['russo_ex01675k'],models['russo_control75k']
            paired=metric.paired_score_diagnostics(scores['russo_ex01675k'][mask],scores['russo_control75k'][mask],labels[mask])
            for key in ['roc_auc','average_precision']:
                paired['ex016_minus_control_'+key]=dense[key]-control[key]
            for key in ['sensitivity','specificity']:
                paired['ex016_minus_control_locked_'+key]=dense['development_locked_operating_point'][key]-control['development_locked_operating_point'][key]
            results[name]=dict(counts=counts[name],models=models,paired_ex016_control=paired)
        contexts[case_id]=dict(physical_group=case['physical_group'],data=case['data'],domains=results,
            masks={name:base.record(directory/f'{name}-common-mask.npy') for name in domains})
        base.write_json(directory/'metrics.json',contexts[case_id])
        plot_context(directory,case,arrays,labels,domains['primary'],domains['secondary'],scores)
    groups={}
    for group in sorted({c['physical_group'] for c in contexts.values()}):
        ids=[k for k,v in contexts.items() if v['physical_group']==group];models={}
        for role in ARMS:
            metrics=[contexts[k]['domains']['primary']['models'][role] for k in ids]
            models[role]={key:mean_range([m[key] for m in metrics]) for key in ['roc_auc','average_precision','supplied_positive_fraction']}
            for key in ['sensitivity','specificity']:
                models[role]['locked_'+key]=mean_range([m['development_locked_operating_point'][key] for m in metrics])
        groups[group]=dict(contexts=ids,context_count=len(ids),equal_context_macro=models,
            paired_ex016_control={key:mean_range([contexts[k]['domains']['primary']['paired_ex016_control'][key] for k in ids])
                for key in ['ex016_minus_control_roc_auc','ex016_minus_control_average_precision','ex016_minus_control_locked_sensitivity','ex016_minus_control_locked_specificity']})
    across={role:{key:float(np.mean([g['equal_context_macro'][role][key]['mean'] for g in groups.values()]))
        for key in ['roc_auc','average_precision','locked_sensitivity','locked_specificity']}
        for role in ARMS}
    report=dict(status='complete_descriptive_panel',plan=base.record(plan_path),threshold_lock=plan['threshold_lock'],
        contexts=contexts,physical_group_summaries=groups,equal_specimen_macro=across,
        interpretation='Two physical specimens reserved only from our current tuning; released-model training/validation lineage remains unresolved. Equal-context then equal-specimen summaries are descriptive, not independent pixel replication.',
        limits=['No pooled pixel metric/CI/significance test.','No reserved threshold tuning or causal acquisition-domain inference.',
            'No calibrated confidence, ink authenticity, geometry authentication or text reading.'])
    base.write_json(analysis/'metrics.json',report)
    lines=['# Fixed cross-scroll annotation panel','',
        'Thresholds were locked on used w029 development support. Reserved specificity is measured rather than forced to95%; sensitivity is at that locked operating point. All rows are retained.','',
        '| Context / specimen | Model | AUC | AP | Prevalence | Locked sensitivity | Locked specificity |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case_id,context in contexts.items():
        for role in ARMS:
            m=context['domains']['primary']['models'][role];o=m['development_locked_operating_point']
            lines.append(f"| {case_id} / {context['physical_group']} | {role} | {m['roc_auc']:.6f} | {m['average_precision']:.6f} | {m['supplied_positive_fraction']:.4f} | {o['sensitivity']:.6f} | {o['specificity']:.6f} |")
    lines +=['','Specimen summaries weight contexts equally; the across-specimen summary weights the two specimens equally. These are descriptive summaries without pixel-based uncertainty or independent-model-heldout claims. Exact paired differences, full-validation secondary results, midpoint diagnostics, occupancy strata and raw maps remain available.','',
        'This pooled-fine panel does not test native acquisition effects or authentic reading. Keep historical exposure, teacher ancestry and geometry limitations attached to every conclusion.','']
    (analysis/'RESULTS.md').write_text('\n'.join(lines))
    final,_,_=load_plan(plan_path,execute=True)
    base.require((output/'frozen-plan.json').read_bytes()==plan_path.read_bytes(),'Frozen plan changed during analysis')
    for case in final['cases']:load_case(case)
    helper=base.import_helper(final)
    for model in final['models']:helper.verify_artifact(model['path'],model)
    for item in map_records:base.verify_record(item)
    base.write_json(analysis/'final-integrity.json',dict(status='passed',plan=base.record(plan_path),verified_after_analysis=True,
        raw_output_records=map_records,threshold_lock=plan['threshold_lock']))
    return dict(metrics=base.record(analysis/'metrics.json'),report=base.record(analysis/'RESULTS.md'),integrity=base.record(analysis/'final-integrity.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(analyze(args.plan,args.output_dir)))


if __name__=='__main__':main()
