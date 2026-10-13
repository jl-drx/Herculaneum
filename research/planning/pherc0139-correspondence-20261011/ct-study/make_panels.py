"""Fixed whole-core CT displays; 0..255 only, no fitted contrast/candidate picking."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent

def main():
    results=json.loads((B/'results/RESULTS.json').read_text());plan=json.loads((B/'REQUEST_PLAN.json').read_text())
    out=B/'panels';out.mkdir(exist_ok=False)
    cmap=plt.get_cmap('gray').copy();cmap.set_bad('#ce78c5')
    for result,site in zip(results['sites'],plan['sites'],strict=True):
        sid=result['site'];sd=B/'results'/sid
        n=np.load(sd/'native_full.npy');f=np.load(sd/'fine_remapped.npy');full=np.load(sd/'fine_full.npy')
        delta=np.array(result.get('fit_translation_zyx',[0,0,0]))
        coords=np.indices(f.shape,dtype=float)+delta[:,None,None,None]
        fitted=map_coordinates(f,coords,order=0,mode='constant',cval=np.nan,prefilter=False)
        columns=[n[4:24],f,fitted]
        fig,axes=plt.subplots(5,3,figsize=(12,16),layout='constrained')
        for c,arr in enumerate(columns):
            for row,layer in enumerate([10,13,17]):
                axes[row,c].imshow(arr[layer-4,8:264,8:264],cmap=cmap,vmin=0,vmax=255,interpolation='nearest')
                axes[row,c].set_title(f'Native layer{layer} frame')
            axes[3,c].imshow(arr[:,136,8:264],cmap=cmap,vmin=0,vmax=255,interpolation='nearest',aspect='auto',extent=[0,256,23,4])
            axes[3,c].set_title('Across X at core Y128; layers4–23')
            axes[4,c].imshow(arr[:,8:264,136],cmap=cmap,vmin=0,vmax=255,interpolation='nearest',aspect='auto',extent=[0,256,23,4])
            axes[4,c].set_title('Across Y at core X128; layers4–23')
        labels=['Real native113','Fine78 geometry-mapped, zero residual CT shift',f'Fine78 at fit ZYX{delta.tolist()}']
        for c,label in enumerate(labels):axes[0,c].set_xlabel(label);axes[0,c].xaxis.set_label_position('top')
        fig.suptitle(f'{sid}: fixed CT comparison, all displays0–255. Magenta=outside retained mapped depth.\nRows3–4 normal axis enlarged for inspection; images are CT structure, not ink maps.',fontsize=12)
        fig.savefig(out/f'{sid}-comparison.png',dpi=130);plt.close(fig)
        q=np.load(B.parent/'geometry-review/pullback'/sid/'queries_uvk.npy')
        x0,y0,_,_=site['source_xy_half_open']['fine']
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for row,alongx in enumerate([True,False]):
            uv=q[9,136,8:264,:2] if alongx else q[9,8:264,136,:2]
            z=np.arange(109)[:,None]*np.ones((1,256))
            xx=np.broadcast_to(uv[:,0]-x0,z.shape);yy=np.broadcast_to(uv[:,1]-y0,z.shape)
            ray=map_coordinates(full,np.stack((z,yy,xx)),order=1,mode='constant',cval=np.nan,prefilter=False)
            native=n[:,136,8:264] if alongx else n[:,8:264,136]
            axes[row,0].imshow(native,cmap=cmap,vmin=0,vmax=255,aspect='auto',interpolation='nearest',extent=[0,256,126.387,-126.387])
            axes[row,1].imshow(ray,cmap=cmap,vmin=0,vmax=255,aspect='auto',interpolation='nearest',extent=[0,256,129.546,-129.546])
            axes[row,0].set_title('Native all28 source layers');axes[row,1].set_title('Fine all109 source layers; fixed UV from native j13 mapping')
            for ax in axes[row]:ax.set_ylabel('Signed nominal normal depth (µm)');ax.set_xlabel('Core X' if alongx else 'Core Y')
        fig.suptitle(f'{sid}: complete source depth context, fixed0–255.\nFine UV held constant through depth; this context view is not an exact paired normal-ray registration.',fontsize=12)
        fig.savefig(out/f'{sid}-full-depth-context.png',dpi=130);plt.close(fig)
    (out/'DISPLAY_RULES.json').write_text(json.dumps(dict(values=[0,255],layers=[10,13,17],normal_sections_core_coordinate=128,all_sites=True,full_depth_fine_uv_anchor_native_layer=13,no_fitted_contrast=True),indent=2)+'\n')
if __name__=='__main__':main()
