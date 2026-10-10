#!/usr/bin/env python3
"""Report saved fixed-batch risk only; no models, checkpoints or raw arrays."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

EXPECTED_SUMMARY='9b48a9f187d68dc9ebf57b88cef295547d553b5a5d0a1f181c218d9371ae7284'
EXPECTED_GATE='0e07252f2705e7905979a7c29dc90a7717133783d334ac3cd269711cf5351b7d'
ARMS=['A_preserve','B_preserve']; UPDATES=[0,1,8,16,32]


def record(path):
    path=Path(path).resolve();data=path.read_bytes()
    return dict(path=str(path),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True,type=Path);parser.add_argument('--gate',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path);args=parser.parse_args()
    summary_bytes=args.summary.read_bytes();gate_bytes=args.gate.read_bytes()
    if hashlib.sha256(summary_bytes).hexdigest()!=EXPECTED_SUMMARY or hashlib.sha256(gate_bytes).hexdigest()!=EXPECTED_GATE:
        raise ValueError('Saved root summary/gate identity mismatch')
    summary=json.loads(summary_bytes);gate=json.loads(gate_bytes)
    if summary['status']!='PASS' or gate['status']!='PASS' or summary['gate']!=record(args.gate):
        raise ValueError('Completed saved result required')
    args.output_dir.mkdir(parents=True,exist_ok=False);rows=[]
    for arm in ARMS:
        report=summary['arms'][arm];initial=report['initializer'];c=0 if arm=='A_preserve' else .25
        for update in UPDATES:
            values=initial if update==0 else report['all_fixed_checkpoints'][str(update)]['equal_batch_means']
            rows.append(dict(arm=arm,completed_updates=update,base_risk=values['base'],full_risk=values['full'],
                base_delta=values['base']-initial['base'],full_delta=values['full']-initial['full'],
                manual_smoothed_bce=values['loss_U'],pseudo_smoothed_bce=values['loss_P'],
                weighted_kl=values['loss_D_M']+c*values['loss_D_P'],
                hard_base_secondary=values['hard_base'],probability_drift_manual=values['probability_drift_manual'],
                probability_drift_pseudo=values['probability_drift_pseudo'],fixed_batch_exposures=32,
                cache_derived_initializer=(update==0)))
        final=rows[-1];contract=gate['arms'][arm]['fixed_evaluation']
        if final['base_delta']!=contract['base_delta'] or final['full_delta']!=contract['full_delta']:
            raise ValueError('Published gate and plot reduction disagree')
    csv_path=args.output_dir/'fixed-risk.csv'
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/herculaneum-thirteenth-matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/herculaneum-thirteenth-fontcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    for ax,arm,title in zip(axes,ARMS,['A: manual + preservation','B: manual + pseudo + preservation']):
        values=[r for r in rows if r['arm']==arm]
        ax.plot(UPDATES,[r['base_delta'] for r in values],'o-',color='#315caa',label='Smoothed base risk')
        ax.plot(UPDATES,[r['full_delta'] for r in values],'^-',color='#943e75',label='Optimized full risk (base + KL)')
        ax.axhline(0,color='#333333',linewidth=.9);ax.set_xticks(UPDATES)
        ax.set_xlim(-1.5,33.5);ax.set_ylim(-.0142,.0006)
        ax.set_xlabel('Completed optimizer updates');ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.20);ax.set_axisbelow(True)
    axes[0].set_ylabel('Fixed-batch risk change from cached initializer')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.09),frameon=False,fontsize=10)
    fig.suptitle('Fixed-batch adaptation risk falls at the measured checkpoints',fontsize=14,y=.96)
    fig.text(.5,.028,'Same 32 batch exposures; cache-derived baseline at 0; lower is better. No generalization or optimal-rate claim.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.10,right=.98,bottom=.27,top=.82,wspace=.12)
    figures=[args.output_dir/'fixed-risk-change.png',args.output_dir/'fixed-risk-change.svg']
    for path in figures:fig.savefig(path,dpi=180)
    plt.close(fig)
    if args.summary.read_bytes()!=summary_bytes or args.gate.read_bytes()!=gate_bytes:
        raise ValueError('Saved inputs changed while reporting')
    receipt=dict(status='saved_output_only',inputs=[record(args.summary),record(args.gate)],script=record(__file__),
        outputs=[record(csv_path),*[record(p) for p in figures]],matplotlib_version=matplotlib.__version__,
        raw_array_reads=0,model_calls=0,checkpoint_reads=0,network_calls=0,input_bytes_unchanged=True)
    target=args.output_dir/'figure-receipt.json';target.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(record(target)))


if __name__=='__main__':main()
