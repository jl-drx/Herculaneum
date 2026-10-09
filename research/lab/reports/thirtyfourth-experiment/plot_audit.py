"""All profile plots; fixed ranking axes and no selected depth."""
import os,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache/matplotlib');os.environ['XDG_CACHE_HOME']=str(HERE/'plot-cache/xdg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
j=json.loads((HERE/'run-v1/COMPLETE.json').read_text());cs=[json.loads(Path(r['result']['path']).read_text()) for r in j['contexts']]
old=json.loads((HERE.parent/'thirtythird-experiment/run-v1/COMPLETE.json').read_text());out=HERE/'figures';out.mkdir(exist_ok=False)
styles=[('single',1,'#0072B2','-','Single +'),('single',-1,'#D55E00','-','Single −'),('triplet',1,'#0072B2','--','Triplet +'),('triplet',-1,'#D55E00','--','Triplet −')]
for support in ('original','interior'):
    for metric,name in [('auc','AUROC'),('ap','Average precision')]:
        fig,axs=plt.subplots(4,2,figsize=(13,12));fig.subplots_adjust(left=.07,right=.99,bottom=.07,top=.89,hspace=.48,wspace=.18)
        for i,c in enumerate(cs):
            previous=next(p for p in old['contexts'] if p['id']==c['id'])
            oldm=previous['original']['baseline'] if support=='original' else previous['bands'][2]['metrics']['baseline']
            for k,feature in enumerate(('intensity','contrast')):
                ax=axs[i,k]
                for scale,pol,color,ls,label in styles:
                    rows=[r for r in c['metrics'] if (r['support'],r['feature'],r['scale'],r['polarity'])==(support,feature,scale,pol)]
                    ax.plot([r['depth'] for r in rows],[r[metric] for r in rows],color=color,ls=ls,lw=1.3,label=label)
                ax.axhline(.5 if metric=='auc' else oldm['prevalence'],color='black',ls=':',lw=1,label='0.5 / ink prevalence')
                ax.axhline(oldm[metric],color='#777777',ls='-.',lw=1,label='Retained CT baseline')
                ax.set_xlim(1,63);ax.set_ylim(0,1);ax.set_ylabel(name);ax.set_xlabel('Historical source layer / triplet center')
                ax.set_title(f"{c['id']} · {feature} · ink prevalence {oldm['prevalence']:.4f}",fontsize=10);ax.grid(alpha=.15)
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.94),ncol=3,frameon=False)
        fig.suptitle(f'{name} across every retained depth · {support} support\nFixed scalar CT features; all signs and scales shown; one development parent',fontsize=14,y=.99)
        fig.savefig(out/(support+'-'+metric+'-profiles.png'),dpi=140);plt.close(fig)
    # Distribution bands: same vertical limits across contexts for each feature.
    fig,axs=plt.subplots(4,2,figsize=(13,12));fig.subplots_adjust(left=.07,right=.99,bottom=.07,top=.89,hspace=.48,wspace=.20)
    ranges={}
    for feature in ('intensity','contrast'):
        vals=[v for c in cs for r in c['distributions'] if r['support']==support and r['feature']==feature for s in r['classes'].values() for v in (s['q25'],s['q75'])]
        lo,hi=min(vals),max(vals);pad=.08*(hi-lo or 1);ranges[feature]=(lo-pad,hi+pad)
    for i,c in enumerate(cs):
        for k,feature in enumerate(('intensity','contrast')):
            ax=axs[i,k]
            for scale,ls in [('single','-'),('triplet','--')]:
                rows=[r for r in c['distributions'] if (r['support'],r['feature'],r['scale'])==(support,feature,scale)];x=[r['depth'] for r in rows]
                for cls,color in [('ink','#CC3311'),('clear','#0077BB')]:
                    ax.plot(x,[r['classes'][cls]['median'] for r in rows],color=color,ls=ls,lw=1.3,label=scale+' '+cls)
                    ax.fill_between(x,[r['classes'][cls]['q25'] for r in rows],[r['classes'][cls]['q75'] for r in rows],color=color,alpha=.10 if scale=='single' else .05)
            ax.set_xlim(1,63);ax.set_ylim(*ranges[feature]);ax.set_title(c['id']+' · '+feature,fontsize=10);ax.set_xlabel('Historical source layer / triplet center');ax.set_ylabel('Retained TIFF units');ax.grid(alpha=.15)
            if feature=='contrast':ax.axhline(0,color='black',lw=.7,ls=':')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.94),ncol=4,frameon=False)
    fig.suptitle(f'Class medians and interquartile bands · {support} support\nShading is a distribution interval, not a confidence interval; equal feature axes across contexts',fontsize=13,y=.99)
    fig.savefig(out/(support+'-class-distributions.png'),dpi=140);plt.close(fig)
print(json.dumps(dict(status='PASS',figures=[str(p) for p in sorted(out.glob('*.png'))])))
