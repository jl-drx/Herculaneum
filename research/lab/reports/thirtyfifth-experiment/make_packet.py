"""Fixed complete layer montages and geometric center cross-sections; no labels/model."""
import os,resource,signal,socket,time
from packet_common import *
os.environ['MPLCONFIGDIR']=str(HERE/'packet-cache/matplotlib');os.environ['XDG_CACHE_HOME']=str(HERE/'packet-cache/xdg')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def offline(*a,**k):raise RuntimeError('Packet rendering is offline')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('120-second packet cap')));signal.alarm(120)
go=json.loads((HERE/'PACKET_GO.json').read_text())
for p in go['files']:verify(p)
plan=json.loads((HERE/'PACKET_PLAN.json').read_text());out=HERE/'packet-v1';out.mkdir(exist_ok=False)
records=[];fig,axs=plt.subplots(4,3,figsize=(16,14));fig.subplots_adjust(top=.92,left=.065,right=.98,bottom=.055,hspace=.38,wspace=.25)
with Image.open(verify(plan['optical'])) as optical:
    for i,c in enumerate(plan['contexts']):
        raw=np.load(verify(c['raw']),allow_pickle=False);require(raw.shape==(63,384,384) and raw.dtype==np.uint16,'CT input contract')
        mean=raw.reshape(63,128,3,128,3).sum(axis=(2,4),dtype=np.uint32).astype(np.float64)/9
        y,x,h,w=c['source_bbox_yxhw'];rgba=np.array(optical.crop((x,y,x+w,y+h)));require(rgba.shape==(384,384,4),'Optical crop')
        require(np.array_equal(rgba[...,0],rgba[...,1]) and np.array_equal(rgba[...,0],rgba[...,2]),'Grayscale IR')
        sections=dict(row193=raw[:,193,:].copy(),column193=raw[:,:,193].copy())
        arr=out/(c['id']+'-display-arrays.npz');np.savez(arr,CT_mean3=mean,optical_RGBA=rgba,**sections)
        f,a=plt.subplots(7,9,figsize=(18,15));f.subplots_adjust(top=.93,bottom=.015,left=.01,right=.99,wspace=.05,hspace=.15)
        for z,ax in enumerate(a.flat):
            ax.imshow(mean[z],cmap='gray',vmin=0,vmax=65535,interpolation='nearest');ax.set_title('Layer '+str(z+1),fontsize=9);ax.set_xticks([]);ax.set_yticks([])
        f.suptitle(c['id']+' · every retained layer, fixed 0–65535 grayscale\n3×3 spatial mean; raster views, not authenticated physical ink depth',fontsize=14)
        montage=out/(c['id']+'-all-layers.png');f.savefig(montage,dpi=120);plt.close(f)
        ax=axs[i,0];shown=np.ma.array(rgba[...,0],mask=rgba[...,3]!=255);cmap=plt.get_cmap('gray').copy();cmap.set_bad('#d95fbe')
        ax.imshow(shown,cmap=cmap,vmin=0,vmax=255,interpolation='nearest');ax.axhline(193,color='#00bfff',lw=.7);ax.axvline(193,color='#ff8c00',lw=.7)
        ax.set_title(c['id']+' · optical');ax.set_xlabel('Local raw column');ax.set_ylabel('Local raw row')
        for k,(name,values) in enumerate(sections.items(),1):
            ax=axs[i,k];ax.imshow(values,cmap='gray',vmin=0,vmax=65535,aspect='auto',interpolation='nearest',extent=(-.5,383.5,63.5,.5))
            ax.set_title(c['id']+' · '+name);ax.set_ylabel('Historical layer');ax.set_xlabel('Local raw '+('column' if name=='row193' else 'row'))
        records.append(dict(id=c['id'],source=c,arrays=pin(arr),montage=pin(montage),opaque_optical_pixels=int(np.count_nonzero(rgba[...,3]==255)),total_optical_pixels=384*384))
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=1024**3,'Packet RSS cap')
fig.suptitle('Exact optical crops and predetermined CT cross-sections\nBlue: row193; orange: column193. CT axes are raster index and historical layer; aspect is not physical.\nMagenta optical pixels, if any, are nonopaque. No annotation or prediction overlays.',fontsize=14)
fig.savefig(out/'optical-and-cross-sections.png',dpi=140);plt.close(fig)
(HERE/'LANDMARK_REVIEW.csv').write_text('context,landmark_id,structure_type,optical_raw_y,optical_raw_x,CT_raw_y,CT_raw_x,CT_layer,visible_in_both,uncertainty_pixels,reviewer,evidence,fit_or_check\n')
for p in go['files']:verify(p)
require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<64*1024**2,'Packet disk cap')
receipt=dict(status='PASS',created_at_utc=now(),GO=pin(HERE/'PACKET_GO.json'),records=records,comparison=pin(out/'optical-and-cross-sections.png'),
    elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_source_requests=0,new_model_forwards=0)
save(out/'COMPLETE.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ['records']}))
