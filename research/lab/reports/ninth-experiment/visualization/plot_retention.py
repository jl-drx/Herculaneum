"""Render descriptive retention metrics; never loads data, models or weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path


def identity(path):
    data=path.read_bytes()
    return dict(path=str(path.resolve()),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); source,raw=identity(args.metrics); metrics=json.loads(raw)
    if metrics['status']!='PASS' or len(metrics['diagnostic'])!=2: raise ValueError('Completed two-context metrics required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    models=['frozen_seed42','A','B','A_smooth','B_smooth']
    labels=['Frozen seed42','A0: manual','B0: manual + pseudo','A0.5: manual','B0.5: manual + pseudo']
    colors=['#777777','#4477AA','#66CCEE','#228833','#CCBB44']
    cases=list(metrics['diagnostic']); x=np.arange(2); width=.145
    fig,axes=plt.subplots(1,3,figsize=(14.5,5.2),layout='constrained')
    for ax,metric,title in zip(axes,['sensitivity','auc','ap'],['Sensitivity at primary cutoff','ROC AUC','Average precision']):
        for index,(model,label,color) in enumerate(zip(models,labels,colors)):
            values=[metrics['diagnostic'][case]['models'][model]['operating']['primary']['sensitivity'] if metric=='sensitivity'
                else metrics['diagnostic'][case]['models'][model][metric] for case in cases]
            if any(v is None or not np.isfinite(v) or not 0<=v<=1 for v in values): raise ValueError('Finite metric in [0,1] required')
            bars=ax.bar(x+(index-2)*width,values,width=width,label=label,color=color,
                edgecolor='white',linewidth=.5,hatch='//' if model in ['A','B'] else None)
            ax.bar_label(bars,labels=[f'{value:.3f}' for value in values],fontsize=8,padding=3,rotation=90)
        ax.set(title=title,ylim=(0,1.12),xticks=x,xticklabels=[f'PHerc0814 {case.rsplit("-",1)[-1]}\n'
            f'n={metrics["diagnostic"][case]["common_primary_count"]:,}' for case in cases])
        ax.set_yticks(np.linspace(0,1,6)); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Supplied-annotation agreement metric')
    fig.suptitle('Reused supplied-supervision retention diagnostics\n'
        'Two PHerc0814 contexts; descriptive comparisons, one adaptation seed',fontsize=13)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=5,frameon=False,fontsize=9)
    args.output_dir.mkdir(parents=True,exist_ok=False); image_path=args.output_dir/'retention-comparison.png'
    fig.savefig(image_path,dpi=180); plt.close(fig)
    # Recheck the already-read metric payload after rendering; no resampling or fitting.
    if identity(args.metrics)[0]!=source: raise ValueError('Metrics changed while rendering')
    receipt=dict(status='PASS',metrics=source,script=identity(Path(__file__))[0],figure=identity(image_path)[0],
        models=models,contexts=cases,primary_calibration='Maximum of three fixed development-region 95%-specificity cutoffs',
        uncertainty_intervals=False,scope='Previously inspected supplied-supervision development diagnostics; not independent ink accuracy or authenticated reading.')
    (args.output_dir/'figure-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__': main()
