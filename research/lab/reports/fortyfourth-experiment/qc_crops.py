"""Compute frozen common support and render optical/reference review without models."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/herculaneum-mpl')
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pixel_http import *
selection=json.loads((P/'CROP_SELECTION.json').read_text());results=[]
for c in selection:
 cid=c['case'];base=P/'crops';z=np.load(base/f'{cid}-height.npy');rgb=np.load(base/f'{cid}-rgb.npy');label=np.load(base/f'{cid}-label.npy')!=0;norm=np.load(base/f'{cid}-normalized.npy');finite=np.isfinite(z);coarse_valid=finite.reshape(384,4,384,4).all((1,3)).repeat(4,0).repeat(4,1)
 pos=binary_erosion(label,iterations=8,border_value=0);neg=binary_erosion(~label,iterations=8,border_value=0);support=coarse_valid&(pos|neg);np.save(base/f'{cid}-common-support.npy',support)
 vals=z[finite];lo,hi=np.quantile(vals,[.01,.99]);idx=np.flatnonzero(finite)[::37];a=z.ravel()[idx].astype(float);b=norm.ravel()[idx].astype(float);M=np.c_[a,np.ones(len(a))];slope,intercept=np.linalg.lstsq(M,b,rcond=None)[0];res=b-(slope*a+intercept);rank=float(spearmanr(a,b).statistic)
 q=dict(case=cid,parent=c['parent'],crop_xy=[c['x'],c['y']],pixels=int(label.size),finite_height_fraction=float(finite.mean()),missing_positive_fraction=float((~finite)[label].mean()),missing_negative_fraction=float((~finite)[~label].mean()),common_support_fraction=float(support.mean()),supported_positive=int((support&label).sum()),supported_annotation_negative=int((support&~label).sum()),height_quantiles=np.quantile(vals,[0,.01,.5,.99,1]).tolist(),raw_vs_normalized=dict(sample_pixels=len(a),spearman=rank,slope=float(slope),intercept=float(intercept),median_abs_residual=float(np.median(abs(res))),p99_abs_residual=float(np.quantile(abs(res),.99))),rgb_channel_quantiles=[np.quantile(rgb[:,:,i],[0,.01,.5,.99,1]).tolist() for i in range(3)],numeric_support_admitted=bool((support&label).sum()>=10000 and (support&~label).sum()>=10000))
 results.append(q)
 fig,axs=plt.subplots(2,3,figsize=(12,8),dpi=105)
 axs[0,0].imshow(rgb);axs[0,0].set_title('Brightfield: explicit RGB8 interpretation')
 axs[0,1].imshow(rgb);axs[0,1].contour(label,levels=[.5],colors=['cyan'],linewidths=.5);axs[0,1].set_title('Source annotation boundary (cyan)')
 axs[0,2].imshow(label,cmap='gray');axs[0,2].set_title('Source optical annotation')
 axs[1,0].imshow(z,cmap='viridis',vmin=lo,vmax=hi);axs[1,0].set_title('Raw measured height; holes blank')
 axs[1,1].imshow(norm,cmap='viridis');axs[1,1].set_title('Released normalized / inpainted height')
 axs[1,2].imshow(support,cmap='gray');axs[1,2].set_title('Common support: valid 4×4; boundary excluded')
 for ax in axs.ravel():ax.axis('off')
 fig.suptitle(f"{cid}: fixed1536² crop at x={c['x']}, y={c['y']}")
 fig.text(.5,.005,'Vesuvius Challenge (2026), doi:10.1038/s41598-026-58467-1 • CC BY-NC 4.0 • Diagnostic crop; relative pixel scale',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.03,1,.95]);fig.savefig(P/f'{cid}-review.png');plt.close(fig)
save(P/'PIXEL_QC.json',dict(created_at_utc=now(),cases=results,visual_review_pending=True));print(json.dumps(results,indent=2))
