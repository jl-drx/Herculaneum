"""Scientific figures from all fixed cores and completed scores; no new inference."""
import os
from common import *
import numpy as np

def main():
    out=HERE/'figures-v1';out.mkdir(exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(out/'mpl-cache');os.environ['XDG_CACHE_HOME']=str(out/'xdg-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((HERE/'scores-v1/RESULTS.json').read_text())
    fig,axes=plt.subplots(4,4,figsize=(12,12),layout='constrained')
    cmap=matplotlib.colormaps['gray_r'].copy();cmap.set_bad('#ffd65c')
    for i,c in enumerate(result['contexts']):
        labels=np.load(verify(c['label_source']),allow_pickle=False)
        scores=np.load(verify(c['scores']),allow_pickle=False)
        baseline,candidate=scores[1];unknown=labels==2
        arrays=[np.ma.array(labels,mask=unknown),np.ma.array(baseline,mask=unknown),np.ma.array(candidate,mask=unknown),np.ma.array(candidate-baseline,mask=unknown)]
        for j,a in enumerate(arrays):
            axes[i,j].imshow(a,cmap=cmap if j<3 else 'coolwarm',vmin=0 if j<3 else -.5,vmax=1 if j<3 else .5,interpolation='nearest')
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        axes[i,0].set_ylabel(c['id']+'\nAP change '+f"{c['clean_AP_delta']:+.4f}")
    for a,title in zip(axes[0],['Supplied labels\nunknown excluded (yellow)','Unchanged seed42\nclean probability','Three-window mean\nclean probability','Candidate − baseline\nrange −0.5 to +0.5']):a.set_title(title)
    fig.suptitle('Frag6 supplied-reference development panel — one parent; no independent ink truth')
    fig.savefig(out/'fixed-core-comparison.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    x=np.arange(4);width=.35
    ax.bar(x-width/2,[c['mean_robust_AP_delta'] for c in result['contexts']],width,label='Mean over shifts −1, 0, +1')
    ax.bar(x+width/2,[c['clean_AP_delta'] for c in result['contexts']],width,label='Clean (shift 0)')
    ax.axhline(0,color='black',linewidth=.7);ax.axhline(.01,color='gray',linestyle='--',label='Mean robust screening margin')
    ax.set_xticks(x,[c['id'] for c in result['contexts']],rotation=12);ax.set_ylabel('Paired AP change (candidate − baseline)')
    ax.set_title('All four fixed contexts; no confidence intervals or independent-parent claim');ax.legend(fontsize=8)
    fig.savefig(out/'paired-AP-changes.png',dpi=150);plt.close(fig)
    save(out/'FIGURE_RECEIPT.json',dict(status='PASS',scores=pin(HERE/'scores-v1/RESULTS.json'),figures=[pin(out/n) for n in ['fixed-core-comparison.png','paired-AP-changes.png']],model_calls=0))

if __name__=='__main__':main()
