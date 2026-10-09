import os
from common import *
os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache/mpl');os.environ['XDG_CACHE_HOME']=str(HERE/'plot-cache/xdg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=HERE/'figures';out.mkdir(exist_ok=False)
c=json.loads((HERE/'run-v2/COMPLETE.json').read_text());fig,axes=plt.subplots(4,4,figsize=(16,15),layout='constrained')
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
for row,r in enumerate(c['contexts']):
    with np.load(verify(r['arrays']),allow_pickle=False) as a:
        for col,(key,title) in enumerate([('baseline_depth','Accepted depth'),('smooth_depth','After 3-layer mean')]):
            im=axes[row,col].imshow(a[key],vmin=5.5,vmax=58.5,cmap=cmap);axes[row,col].set_title(f"{r['id']}\n{title}")
        status=a['baseline_accepted'].astype(int)+2*a['smooth_accepted'].astype(int)
        axes[row,2].imshow(status,vmin=0,vmax=3,cmap=plt.get_cmap('tab10',4));axes[row,2].set_title('Acceptance: neither / base / smooth / both')
        for key,label in [('baseline_depth','Baseline'),('smooth_depth','Smoothed')]:
            d=a[key];axes[row,3].hist(d[np.isfinite(d)],bins=np.arange(5,60),histtype='step',label=label)
        axes[row,3].set_title('All accepted depths; no selected window');axes[row,3].set_xlim(5,59);axes[row,3].legend(fontsize=8)
        for ax in axes[row,:3]:ax.set_xlabel('Valid coarse column');ax.set_ylabel('Valid coarse row')
fig.colorbar(im,ax=axes[:,:2],shrink=.65,label='Historical layer boundary; not authenticated ink depth')
fig.suptitle('Fixed CT texture-transition prototype — grey = abstention; previously inspected development data',fontsize=15)
fig.savefig(out/'all-contexts.png',dpi=130);plt.close(fig)
save(HERE/'FIGURES.json',dict(generator=pin(__file__),figures=[pin(p) for p in out.glob('*.png')]))
