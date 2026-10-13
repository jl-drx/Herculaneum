"""Every fixed checking patch; fixed scales, no target-fitted display contrast."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=R/'panels';out.mkdir(exist_ok=False);manifest=[]
    for s in read(R/'ALTERNATIVES.json')['sites']:
        if not s['admitted_for_bounded_target_sensitivity']:continue
        sid=s['core_id']
        for op in OPS:
            fig,axs=plt.subplots(4,6,figsize=(15,10),layout='constrained');clipped=0
            for row,patch in enumerate(s['patches']):
                i=patch['index'];a=np.load(R/'results'/sid/f'p{i}-j14-{op}.npz');v=a['values'];ran=np.ptp(v,axis=0);diff=a['mean_target']-a['stationary_target']
                clipped+=int((ran>64).sum())+int((np.abs(diff)>1).sum())
                fields=[('Fitted target',a['fit'],'gray',0,255),('Zero residual',a['zero'],'gray',0,255),
                    ('Mean target',a['mean_target'],'gray',0,255),('Candidate score range',ran,'magma',0,64),
                    ('Fraction above threshold',(v>=s['threshold']).mean(0),'viridis',0,1),
                    ('Mean minus stationary',diff,'coolwarm',-1,1)]
                for col,(name,image,cmap,low,high) in enumerate(fields):
                    ax=axs[row,col];im=ax.imshow(image,cmap=cmap,vmin=low,vmax=high,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title(name)
                    if col==0:ax.set_ylabel(f'Check {i}\nN={len(v)}')
                    if row==3:fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.08,pad=.04)
            fig.suptitle(f'{sid} | plane14 | {op} | threshold ≥{s["threshold"]}\nConditional teacher sensitivity; no physical-ink or accuracy claim')
            p=out/f'{sid}-{op}.png';fig.savefig(p,dpi=120);plt.close(fig)
            manifest.append(dict(**receipt(p),all_four_fixed_patches=True,fixed_scales=True,range_above64_plus_abs_stationary_difference_above1_pixels=clipped))
    save(out/'MANIFEST.json',manifest)

if __name__=='__main__':main()
