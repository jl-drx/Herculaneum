"""Render completed retained metrics only; no model/data calls or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import re

MODELS=['frozen_seed42','A_smooth','B_smooth','A_preserve','B_preserve']
CASES=['pherc0814-46527-fresh-rank01','pherc0814-46527-fresh-rank14']
COUNTS={'pherc0814-46527-fresh-rank01':49033,'pherc0814-46527-fresh-rank14':21328}


def identity(path):
    data=path.read_bytes()
    return dict(path=str(path.resolve()),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--metrics-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}',args.metrics_sha256):raise ValueError('Exact expected metrics SHA256 required')
    source,raw=identity(args.metrics)
    if source['sha256']!=args.metrics_sha256:raise ValueError('Completed metrics hash differs')
    metrics=json.loads(raw)
    if metrics['status']!='PASS' or set(metrics['diagnostic'])!=set(CASES):raise ValueError('Completed fixed two-context metrics required')
    for case in CASES:
        item=metrics['diagnostic'][case]
        if set(item['models'])!=set(MODELS) or item['physical_group']!='PHerc0814' or item['common_primary_count']!=COUNTS[case]:
            raise ValueError('Fixed five-role diagnostic identity/domain differs')
    if args.output_dir.exists():raise ValueError('Fresh plot output directory required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=['Frozen seed42','A_smooth (reused)','B_smooth (reused)','A_preserve (new)','B_preserve (new)']
    colors=['#777777','#228833','#CCBB44','#4477AA','#66CCEE']
    x=np.arange(2);width=.145
    fig,axes=plt.subplots(1,3,figsize=(14.5,5.2),layout='constrained')
    plotted={}
    for ax,metric,title in zip(axes,['sensitivity','auc','ap'],['Sensitivity at primary cutoff','ROC AUC','Average precision']):
        plotted[metric]={}
        for index,(model,label,color) in enumerate(zip(MODELS,labels,colors)):
            values=[metrics['diagnostic'][case]['models'][model]['operating']['primary']['sensitivity'] if metric=='sensitivity'
                else metrics['diagnostic'][case]['models'][model][metric] for case in CASES]
            if any(v is None or not np.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('Finite reported metric in [0,1] required')
            plotted[metric][model]=dict(zip(CASES,values))
            bars=ax.bar(x+(index-2)*width,values,width=width,label=label,color=color,
                edgecolor='white',linewidth=.5,hatch='//' if model in ['A_smooth','B_smooth'] else None)
            ax.bar_label(bars,labels=[f'{value:.3f}' for value in values],fontsize=8,padding=3,rotation=90)
        ax.set(title=title,ylim=(0,1.12),xticks=x,xticklabels=[f'PHerc0814 {case.rsplit("-",1)[-1]}\n'
            f'n={metrics["diagnostic"][case]["common_primary_count"]:,}' for case in CASES])
        ax.set_yticks(np.linspace(0,1,6));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Supplied-annotation agreement metric')
    fig.suptitle('Reused supplied-supervision retention diagnostics\n'
        'Two PHerc0814 contexts; descriptive comparisons, one adaptation seed',fontsize=13)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=5,frameon=False,fontsize=9)
    args.output_dir.mkdir(parents=True,exist_ok=False);image_path=args.output_dir/'retention-comparison.png'
    fig.savefig(image_path,dpi=180);plt.close(fig)
    if identity(args.metrics)[0]!=source:raise ValueError('Metrics changed while rendering')
    receipt=dict(status='PASS',metrics=source,expected_metrics_sha256=args.metrics_sha256,
        script=identity(Path(__file__))[0],figure=identity(image_path)[0],models=MODELS,contexts=CASES,reported_values_plotted=plotted,
        primary_calibration='Maximum of three fixed development-region 95%-specificity cutoffs',
        uncertainty_intervals=False,model_calls=0,new_thresholds_selected=0,new_metrics_computed=False,
        scope='Reused supplied-supervision development retention diagnostics, one adaptation seed; no independent ink accuracy or fresh holdout claim.')
    target=args.output_dir/'figure-receipt.json';target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='PASS',figure=receipt['figure'],receipt=identity(target)[0])))


if __name__=='__main__':main()
