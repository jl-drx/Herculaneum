"""Every site, fixed source sections and intensity scales; no fitted contrast."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
def main():
    out=R/'panels';out.mkdir(exist_ok=False);plan=read(PLAN);geo={r['core_id']:r for r in read(R/'geometry/RESULTS.json')['sites']}
    cmap=plt.get_cmap('gray').copy();cmap.set_bad('#d0429e');diffmap=plt.get_cmap('coolwarm').copy();diffmap.set_bad('#d0429e')
    for site in plan['fixed_selected_sites']:
        sid=site['core_id'];d=R/'results'/sid;record=read(d/'RESULTS.json');g=geo[sid]
        n=np.load(d/'native_full28.npy');f=np.load(d/'fine_full109.npy');m=np.load(d/'fine_remapped.npy');valid=np.load(d/'remapped_support.npy')
        cx,cy,_,_=site['scored_core_xywh'];nx,ny,_,_=site['native_context_xywh'];native=n[:,cy-ny:cy-ny+128,cx-nx:cx-nx+128]
        safe=np.where(valid,m,np.nan);base=safe[6:14,8:136,8:136].mean(0);target=native[10:18].mean(0)
        if record['scored']:
            dz,dy,dx=record['fit_translation_zyx'];aligned=safe[6+dz:14+dz,8+dy:136+dy,8+dx:136+dx].mean(0)
        else:aligned=np.full_like(base,np.nan)
        fig,axes=plt.subplots(1,5,figsize=(14,3),layout='constrained')
        for ax,a,title in zip(axes,[target,base,aligned,base-target,aligned-target],['Native mean j10:18','Fine: geometry only','Fine: CT fit','Geometry minus native','Fit minus native']):
            difference='minus' in title;ax.imshow(a,cmap=diffmap if difference else cmap,vmin=-64 if difference else 0,vmax=64 if difference else 255,interpolation='nearest');ax.set_title(title,fontsize=9);ax.set_axis_off()
        fig.suptitle(sid+' | fixed scales: intensity 0–255; difference ±64 | magenta: unsupported');fig.savefig(out/(sid+'-xy.png'),dpi=135);plt.close(fig)
        q=np.load(R/'geometry'/sid/'queries_uvk.npy');centre=(q[9]+q[10])/2
        uvx=centre[72,8:136,:2];uvy=centre[8:136,72,:2];sections=[]
        for uv in [uvx,uvy]:
            u=np.broadcast_to(uv[:,0]-site['fine_group2_xyxy'][0],(109,128));v=np.broadcast_to(uv[:,1]-site['fine_group2_xyxy'][1],(109,128));z=np.broadcast_to(np.arange(109)[:,None],(109,128))
            sections.append(map_coordinates(f.astype(float),np.stack((z,v,u)),order=1,mode='constant',cval=np.nan,prefilter=False))
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for ax,a,title,spacing in zip(axes.flat,[native[:,64,:],native[:,:,64],*sections],['Native full28: center row','Native full28: center column','Fine full109: unshifted UV row','Fine full109: unshifted UV column'],[9.362,9.362,2.399,2.399]):
            extent=[0,128,-a.shape[0]*spacing/2,a.shape[0]*spacing/2]
            ax.imshow(a,cmap=cmap,vmin=0,vmax=255,origin='lower',extent=extent,aspect='auto',interpolation='nearest');ax.set_title(title);ax.set_ylabel('Nominal normal offset (µm)');ax.set_xlabel('Native core coordinate');ax.axhline(0,color='cyan',linewidth=.5)
        fig.suptitle(sid+' | fixed 0–255 | full source depth, no fitted CT shift');fig.savefig(out/(sid+'-depth.png'),dpi=135);plt.close(fig)
    save(out/'FILES.json',[receipt(p) for p in sorted(out.glob('*.png'))]);print('wrote',len(list(out.glob('*.png'))),'fixed panels')
if __name__=='__main__':main()
