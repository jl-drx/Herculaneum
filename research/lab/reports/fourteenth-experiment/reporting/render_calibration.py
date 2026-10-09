#!/usr/bin/env python3
"""Render completed calibration JSON only; no score arrays, models or fitting."""
import argparse
import csv
import hashlib
import json
import math
import os
import socket
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
DESIGN_SHA='96ed37ff57dbcff80bb9cbd6e984a2bb27ee0f6bf2d7de079dac42adf0b0dc45'
AUDIT_SHA='abc19559a3550323ade1c48e7c373e16813aa9553751fb58116be4c05b78ab0d'
ROLES=['frozen_seed42','A_preserve','B_preserve']
RULES=['common_baseline','primary','secondary']
GROUPS=['PHerc0814','PHercParis1Fr39']
CONTEXTS={
    'pherc0814-46527-fresh-rank01':('PHerc0814','PHerc0814: context 1'),
    'pherc0814-46527-fresh-rank14':('PHerc0814','PHerc0814: context 2'),
    'frag4-context1':('PHercParis1Fr39','Frag4: context 1'),
    'frag4-context2':('PHercParis1Fr39','Frag4: context 2')}
COLORS={'frozen_seed42':'#687078','A_preserve':'#315caa','B_preserve':'#943e75'}


def offline(*args,**kwargs): raise RuntimeError('Reporting has no network access')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline


def record(path):
    path=Path(path).resolve(strict=True);b=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))


def percent(value): return f'{100*value:.3f}%'


def build_rows(metrics):
    if set(metrics['per_context'])!=set(CONTEXTS): raise ValueError('Four fixed contexts required')
    rows=[]
    for context,(group,_) in CONTEXTS.items():
        if metrics['per_context'][context]['physical_group']!=group:raise ValueError('Fixed specimen grouping changed')
        for role in ROLES:
            result=metrics['per_context'][context]['models'][role]
            for rule in RULES:
                op=result['operating'][rule];confusion=op['confusion']
                if not all(type(confusion[k]) is int and confusion[k]>=0 for k in ['tp','fp','tn','fn']):
                    raise ValueError('Exact nonnegative confusion counts required')
                if not all(math.isfinite(op[k]) and 0<=op[k]<=1 for k in ['sensitivity','specificity']):
                    raise ValueError('Finite achieved rates required')
                negative=confusion['fp']+confusion['tn']
                rows.append(dict(context=context,physical_group=group,role=role,rule=rule,
                    threshold=op['threshold'],sensitivity=op['sensitivity'],specificity=op['specificity'],precision=op['precision'],
                    tp=confusion['tp'],fp=confusion['fp'],tn=confusion['tn'],fn=confusion['fn'],
                    specificity_floor_pass=(20*confusion['fp']<=negative),auc=result['auc'],ap=result['ap']))
    return rows


def render(rows,directory):
    os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
    os.environ['XDG_CACHE_HOME']=str(HERE/'.cache/xdg')
    sys.dont_write_bytecode=True
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    shown=[r for r in rows if r['rule'] in ['common_baseline','primary']]
    limits={}
    for group in GROUPS:
        rr=[r for r in shown if r['physical_group']==group]
        limits[group]=(min(94.,min(r['specificity']*100 for r in rr)-1.),100.2,
            100.5 if group=='PHerc0814' else min(100.5,max(5.,max(r['sensitivity']*100 for r in rr)+1.)))
    fig=plt.figure(figsize=(13,11.5))
    grid=fig.add_gridspec(2,2,left=.075,right=.98,top=.875,bottom=.21,hspace=.58,wspace=.25)
    for context,(group,title) in CONTEXTS.items():
        ci=[c for c in CONTEXTS if CONTEXTS[c][0]==group].index(context);col=GROUPS.index(group)
        inner=grid[ci,col].subgridspec(2,1,height_ratios=[3.3,1.55],hspace=.31)
        ax=fig.add_subplot(inner[0]);table_ax=fig.add_subplot(inner[1]);table_ax.axis('off')
        def row(role,rule):return next(r for r in rows if r['context']==context and r['role']==role and r['rule']==rule)
        baseline=row('frozen_seed42','common_baseline');base_primary=row('frozen_seed42','primary')
        if any(baseline[k]!=base_primary[k] for k in ['threshold','tp','fp','tn','fn']):
            raise ValueError('Baseline original and primary operating points must coincide')
        ax.scatter(100*baseline['specificity'],100*baseline['sensitivity'],marker='D',s=65,color=COLORS['frozen_seed42'],zorder=4)
        table_rows=[['seed42 original',percent(baseline['specificity']),percent(baseline['sensitivity']),'pass' if baseline['specificity_floor_pass'] else 'fail']]
        row_colors=[COLORS['frozen_seed42']]
        for role in ROLES[1:]:
            common=row(role,'common_baseline');primary=row(role,'primary');color=COLORS[role]
            ax.plot([common['specificity']*100,primary['specificity']*100],
                    [common['sensitivity']*100,primary['sensitivity']*100],color=color,linewidth=1.,alpha=.65)
            ax.scatter(common['specificity']*100,common['sensitivity']*100,marker='o',s=85,facecolors='none',edgecolors=color,linewidth=1.5,zorder=5)
            ax.scatter(primary['specificity']*100,primary['sensitivity']*100,marker='^',s=75,color=color,zorder=6)
            for r,label in [(common,'common'),(primary,'primary')]:
                table_rows.append([f'{role[0]} {label}',percent(r['specificity']),percent(r['sensitivity']),'pass' if r['specificity_floor_pass'] else 'fail']);row_colors.append(color)
        lo,hi,top=limits[group]
        ax.axvspan(lo,95,color='#e4adad',alpha=.18);ax.axvline(95,color='#a24343',linestyle='--',linewidth=1.)
        ax.set_xlim(lo,hi);ax.set_ylim(-top*.025,top);ax.set_xlabel('Achieved specificity (%)',fontsize=10)
        ax.set_ylabel('Supplied-reference sensitivity (%)',fontsize=10);ax.set_title(title,fontsize=12)
        ax.grid(alpha=.15);ax.set_axisbelow(True)
        table=table_ax.table(cellText=table_rows,colLabels=['State / rule','Specificity','Sensitivity','95% floor'],
            loc='center',cellLoc='center',colWidths=[.36,.22,.22,.20]);table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.17)
        for i,color in enumerate(row_colors,1):
            table[(i,0)].get_text().set_color(color)
            if table_rows[i-1][3]=='fail':table[(i,3)].set_facecolor('#f9dada')
    legend=[Patch(color=COLORS['A_preserve'],label='A_preserve'),Patch(color=COLORS['B_preserve'],label='B_preserve'),
        Line2D([],[],marker='D',color=COLORS['frozen_seed42'],linestyle='none',label='Frozen seed42: original cutoff'),
        Line2D([],[],marker='o',markerfacecolor='none',color='#333333',linestyle='none',label='Common baseline cutoff'),
        Line2D([],[],marker='^',color='#333333',linestyle='none',label='Own development primary')]
    fig.legend(handles=legend,loc='lower center',ncol=3,bbox_to_anchor=(.5,.085),frameon=False,fontsize=10)
    fig.suptitle('Calibration transfer: sensitivity versus achieved specificity',fontsize=16,y=.975)
    fig.text(.5,.941,'PHerc0814 and Frag4 shown separately; dashed line is the predeclared 95% specificity floor.',ha='center',fontsize=11)
    fig.text(.5,.047,'Pairs connect two fixed cutoffs, not a fitted ROC curve. Domain axes differ; exact rates and floor flags are below each panel.',ha='center',fontsize=9,color='#444444')
    fig.text(.5,.020,'Already seen supplied-reference evidence; no fresh holdout, confidence intervals or automatic model promotion.',ha='center',fontsize=10,color='#444444')
    paths=[directory/'calibration-transfer.png',directory/'calibration-transfer.svg']
    for path in paths:fig.savefig(path,dpi=180)
    plt.close(fig);return paths,matplotlib.__version__


def write_report(metrics,rows,path):
    lines=['# Development-only calibration transfer','',
        'This calibration-only control changes operating cutoffs while retaining the literal thirteenth diagnostic score maps. Ranking is unchanged. Its primary comparisons use each model’s own development-only cutoff; common-baseline and secondary cutoffs remain separately reported. Root owns the scientific decision.','',
        'The specificity rule was fit only to the same three calibration regions’ supplied negatives, using the unchanged conservative float32 tie selector. It constrains empirical calibration-region specificity; achieved specificity on a diagnostic context must be checked separately. These are already seen contexts, with unresolved physical/teacher-history independence.','',
        '## Frozen operating cutoffs','',
        '| State | Common baseline | Own primary | Secondary PHerc1667 only |','| --- | ---: | ---: | ---: |']
    for role in ROLES:
        values={rule:next(r['threshold'] for r in rows if r['role']==role and r['rule']==rule) for rule in RULES}
        if any(r['threshold']!=values[r['rule']] for r in rows if r['role']==role):raise ValueError('One cutoff per model/rule required')
        lines.append(f"| {role} | {values['common_baseline']:.17g} | {values['primary']:.17g} | {values['secondary']:.17g} |")
    lines+=['','Primary is the maximum of all three regional cutoffs. Secondary is fixed PHerc1667-w029 only and is not selected instead of primary after viewing diagnostics. The canonical negative counts are 9,378, 40,400 and 14,644; Frag4 and PHerc0139 adaptation labels are excluded from cutoff fitting.','',
        '## Per-context achieved operating points','',
        '![Sensitivity and achieved specificity](render/calibration-transfer.png)','',
        '| Context | State | Common sensitivity / specificity | Primary sensitivity / specificity | Primary 95% floor |','| --- | --- | ---: | ---: | --- |']
    for context in CONTEXTS:
        for role in ROLES:
            common=next(r for r in rows if (r['context'],r['role'],r['rule'])==(context,role,'common_baseline'))
            primary=next(r for r in rows if (r['context'],r['role'],r['rule'])==(context,role,'primary'))
            lines.append(f"| {context} | {role} | {percent(common['sensitivity'])} / {percent(common['specificity'])} | {percent(primary['sensitivity'])} / {percent(primary['specificity'])} | {'pass' if primary['specificity_floor_pass'] else 'fail'} |")
    lines+=['','Floor flags use exact counts (`20*FP <= Nnegative`), not rounded percentages. The CSV retains all three operating rules, exact confusion counts, precision and unchanged AUC/AP.','',
        '## Predeclared primary recovery screens','',
        '| Arm | Specimen | All contexts meet 95% specificity | Mean sensitivity delta vs baseline (points) | No context loses more than 2 points | Full screen |','| --- | --- | --- | ---: | --- | --- |']
    baseline_failures=[case for group in metrics['baseline_specificity_floor'].values() for case,passed in group.items() if not passed]
    baseline_note='The frozen baseline fails the specificity floor in '+', '.join(baseline_failures)+'.' if baseline_failures else 'The frozen baseline meets the specificity floor in all four contexts.'
    for role in ROLES[1:]:
        for group in GROUPS:
            s=metrics['screens'][role][group]
            floor=all(v['passed'] for v in s['specificity_by_context'].values())
            context=all(v['passed'] for v in s['per_context_sensitivity_checks'].values())
            if s['automatic_promotion'] is not False:raise ValueError('No automatic promotion permitted')
            delta=100*s['mean_sensitivity_comparison']['delta']['value']
            lines.append(f"| {role} | {group} | {'pass' if floor else 'fail'} | {delta:+.4f} | {'pass' if context else 'fail'} | {'pass' if s['passed'] else 'fail'} |")
    lines+=['',baseline_note,'',
        'The screen also requires exact equal-context mean sensitivity to be at least the baseline mean. Its Fraction-based flags are taken from the saved analysis; sensitivity improvement versus the same model at the common cutoff does not replace a failed primary screen. Meeting a weak baseline does not establish a useful ink detector.','',
        '## Domain macros and fixed secondary analysis','',
        '| Specimen | State | Common sensitivity / specificity | Primary sensitivity / specificity | Secondary sensitivity / specificity |','| --- | --- | ---: | ---: | ---: |']
    for group in GROUPS:
        for role in ROLES:
            op=metrics['domain_macro'][group][role]['operating']
            pairs=[f"{percent(op[r]['sensitivity'])} / {percent(op[r]['specificity'])}" for r in RULES]
            lines.append(f'| {group} | {role} | '+ ' | '.join(pairs)+' |')
    lines+=['','Macros give the two contexts in each specimen equal weight. There is no pooled-pixel or cross-specimen headline, pixel-IID interval, diagnostic fit, score transformation or new ranking evidence. Public validation naming, disjoint 2D boxes and sequential computation do not create prospective validation. Optical registration, annotation transfer and upstream-exposure limits remain those of the earlier studies.','',
        'Independent numerical verification status and final input/output identities are recorded separately before this reporting package is sealed. This reporting code reads only completed JSON and does not load arrays, checkpoints or models.','']
    path.write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--metrics',required=True,type=Path)
    parser.add_argument('--metrics-sha256',required=True);parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();source=args.metrics.read_bytes()
    if hashlib.sha256(source).hexdigest()!=args.metrics_sha256:raise ValueError('Root-completed metric SHA differs')
    design=record(HERE.parent/'RESEARCH_DESIGN.md');audit=record(HERE.parent/'CONTRACT_AUDIT.md')
    if design['sha256']!=DESIGN_SHA or audit['sha256']!=AUDIT_SHA:raise ValueError('Frozen source contract differs')
    metrics=json.loads(source)
    if metrics['status']!='PASS':raise ValueError('Root completed PASS required')
    if args.output_dir.resolve().parent!=HERE:raise ValueError('Output must stay within owned reporting directory')
    args.output_dir.mkdir(parents=True,exist_ok=False);rows=build_rows(metrics)
    csv_path=args.output_dir/'operating-points.csv'
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures,version=render(rows,args.output_dir);report=HERE/'RESULTS.md'
    if report.exists():raise ValueError('Preserve earlier reporting stage')
    write_report(metrics,rows,report)
    if args.metrics.read_bytes()!=source:raise ValueError('Metrics changed during reporting')
    receipt=dict(status='saved_output_only',metrics=record(args.metrics),contracts=[design,audit],script=record(__file__),
        outputs=[record(csv_path),record(report),*[record(p) for p in figures]],matplotlib_version=version,
        raw_array_reads=0,checkpoint_reads=0,model_calls=0,threshold_fitting_calls=0,score_transforms=0,network_calls=0,
        input_bytes_unchanged=True,baseline_plotted_once=True,primary_and_common_plotted=True,secondary_retained_in_CSV=True)
    target=args.output_dir/'figure-receipt.json';target.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(record(target)))


if __name__=='__main__':main()
