"""Replot retained scientific arrays for a small portable review packet."""
import os
import numpy as np
from common import *
os.environ['MPLCONFIGDIR']=str(HERE/'plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
source=json.loads((HERE/'review/FIGURES.json').read_text());alignment=json.loads((HERE/'alignment/RESULTS.json').read_text());out=HERE/'review-compact';out.mkdir(exist_ok=False);rows=[]
for f in source['figures']:
 arrays=np.load(verify(f['views']));masks=np.load(HERE/'review'/(f['id']+'-coverage.npz'));fitted=next(r for r in alignment['rows'] if r['id']==f['id'])['fit'] is not None
 fig,axes=plt.subplots(4,5,figsize=(12,10));titles=['Parallel −76.8µm','Parallel 0µm','Parallel +76.8µm','U/normal','V/normal'];labels=['Native 2.4µm','Fixed blur','Coarse: public map','Coarse: fitted shift' if fitted else 'Coarse: no fit']
 for i in range(4):
  scale=f['display_percentiles']['new' if i<2 else 'old']
  for j in range(5):
   key=f'r{i}c{j}';a=axes[i,j];data=np.ma.array(arrays[key],mask=~masks[key]);a.imshow(data,cmap='gray',vmin=scale[0],vmax=scale[1],origin='upper',interpolation='nearest');a.axvline(64,color='#ee6655',alpha=.5,lw=.5);a.axhline(64,color='#ee6655',alpha=.5,lw=.5);a.set_xticks([0,64,127],['−154','0','+151']);a.set_yticks([0,64,127],['−154','0','+151']);a.tick_params(labelsize=6)
   if i==0:a.set_title(titles[j],fontsize=9)
   if j==0:a.set_ylabel(labels[i],fontsize=9)
 fig.suptitle(f['id']+' · acquisition review · material identity unverified',fontsize=13)
 fig.text(.02,.015,'Axes in µm; each panel spans 307µm. White gaps are missing coverage. Display ranges fixed within acquisition.\nBlur is a visualization, not a complete coarse-scanner simulation. Fitted shifts do not verify ink or wrap identity.',fontsize=9)
 fig.tight_layout(rect=(0,.055,1,.96));dest=out/(f['id']+'.png');fig.savefig(dest,dpi=110);plt.close(fig);need(dest.stat().st_size<=1_000_000,'Portable scientific figure bound');rows.append(dict(id=f['id'],source_views=f['views'],figure=pin(dest)))
save(out/'FIGURES.json',dict(status='PASS',created_at_utc=now(),scope='Replot of identical retained view arrays and coverage masks; no new scientific sampling',figures=rows))
