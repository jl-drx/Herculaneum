#!/usr/bin/env python3
"""Show saved Frag4 ranking and fixed-cutoff response together."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

EXPECTED='4f1d8df047dc034cfaed3ddce397191d152674921baca6c7e97acd1c692737c3'
ROLES=['frozen_seed42','A_preserve','B_preserve']


def record(path):
    path=Path(path).resolve();data=path.read_bytes()
    return dict(path=str(path),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--metrics',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path);args=parser.parse_args()
    source=args.metrics.read_bytes()
    if hashlib.sha256(source).hexdigest()!=EXPECTED:raise ValueError('Saved retention metric identity differs')
    metrics=json.loads(source)
    if metrics['status']!='PASS' or metrics['threshold_fitting_calls']!=0:raise ValueError('Completed frozen-cutoff result required')
    cutoff=metrics['baseline_cutoff']['value']
    if cutoff!=0.7558947205543518:raise ValueError('Baseline cutoff differs')
    domain=metrics['domain_macro']['PHercParis1Fr39']
    rows=[dict(role=role,domain='Frag4 / PHercParis1Fr39',contexts=2,**domain[role],cutoff=cutoff) for role in ROLES]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    csv_path=args.output_dir/'frag4-retention.csv'
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/herculaneum-thirteenth-matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/herculaneum-thirteenth-fontcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11.2,4.7));colors=['#687078','#315caa','#943e75']
    for ax,key,title,limit,scale in zip(axes,['auc','ap','sensitivity'],
            ['AUC','Average precision','Sensitivity at frozen cutoff'],[1.,1.,4.], [1.,1.,100.]):
        values=[row[key]*scale for row in rows]
        ax.bar(range(3),values,color=colors,width=.60)
        for x,y in enumerate(values):ax.text(x,y+limit*.025, f'{y:.4f}' if scale==1 else f'{y:.3f}%',ha='center',fontsize=10)
        ax.set_xticks(range(3),['Baseline\nseed42','A\n1e-6','B\n1e-6']);ax.set_ylim(0,limit)
        ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.20);ax.set_axisbelow(True)
        ax.set_ylabel('Equal-context mean (0–1)' if scale==1 else 'Supplied-reference sensitivity (%)')
    fig.suptitle('Frag4 retention: ranking and fixed-cutoff response',fontsize=14,y=.96)
    fig.text(.5,.065,f'Identical cutoff {cutoff:.6f}; no threshold fitting. Sensitivity panel uses a percentage axis.',ha='center',fontsize=9,color='#444444')
    fig.text(.5,.023,'Two reused contexts from one specimen; supplied-reference agreement. No fresh holdout or model promotion.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.065,right=.985,bottom=.24,top=.81,wspace=.40)
    figures=[args.output_dir/'frag4-ranking-and-response.png',args.output_dir/'frag4-ranking-and-response.svg']
    for path in figures:fig.savefig(path,dpi=180)
    plt.close(fig)
    if args.metrics.read_bytes()!=source:raise ValueError('Saved metrics changed during reporting')
    receipt=dict(status='saved_output_only',metrics=record(args.metrics),script=record(__file__),outputs=[record(csv_path),*[record(p) for p in figures]],
        matplotlib_version=matplotlib.__version__,threshold=cutoff,threshold_fitting_calls=0,
        raw_array_reads=0,model_calls=0,checkpoint_reads=0,network_calls=0,input_bytes_unchanged=True)
    target=args.output_dir/'figure-receipt.json';target.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(record(target)))


if __name__=='__main__':main()
