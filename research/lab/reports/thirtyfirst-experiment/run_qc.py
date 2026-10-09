"""Read retained PNGs, measure support, freeze panel and make analytical figures."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, resource, signal, sys, time
import numpy as np
from PIL import Image
from reference_qc import classify, tile_table, select_tiles
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'thirtieth-experiment'))
from run_correspondence import pin, save, require


def main():
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('QC deadline')))
    signal.alarm(240)
    run=HERE/'qc-v1';run.mkdir(exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(run/'mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    freeze=json.loads((HERE/'GO.json').read_text())
    for item in freeze['files']: require(pin(Path(item['path']))==item,'Frozen file changed')
    objects=json.loads((HERE/'acquisition-v1/PNG_OBJECTS.json').read_text())
    images, encodings={},{}
    for name,obj in objects.items():
        path=HERE/'acquisition-v1'/name
        require(pin(path)==obj['body'],'PNG source changed')
        with Image.open(path) as im: im.verify()
        with Image.open(path) as im:
            encodings[name]=dict(mode=im.mode, histogram=im.histogram(),
                                metadata_keys=sorted(im.info), dimensions=list(im.size))
            if name!='ir.png': images[name]=classify(im)
    labels,mask=images['inklabels.png'],images['mask.png']
    joint=np.zeros((3,3),dtype=np.int64)
    for y in range(0,len(labels),128):
        joint+=np.bincount((3*mask[y:y+128]+labels[y:y+128]).ravel(),minlength=9).reshape(3,3)
    table=tile_table(labels,mask);chosen=select_tiles(table)
    save(run/'TILE_TABLE.json',table)
    summary=dict(canvas_pixels=int(labels.size),code_meaning={'0':'black / supplied negative or mask excluded',
                '1':'white / supplied positive or mask valid','2':'unknown'},
                joint_counts_mask_row_label_column=joint.tolist(),
                label_counts=np.sum(joint,axis=0).tolist(),mask_counts=np.sum(joint,axis=1).tolist(),
                supported_positive=int(joint[1,1]),supported_negative=int(joint[1,0]),
                positive_outside_white_mask=int(joint[0,1]+joint[2,1]),
                label_unknown_inside_mask=int(joint[1,2]),
                grid_tiles=len(table),eligible_tiles=sum(t['eligible'] for t in table),
                selected_tiles=len(chosen),missing_vertical_bands=sorted(set(range(4))-{t['vertical_band'] for t in chosen}),
                interpretation='Supplied-reference convention; zero does not establish independently ink-free tissue')
    require(summary['supported_positive']>0 and summary['supported_negative']>0,'Convention lacks both classes')
    with Image.open(HERE/'acquisition-v1/ir.png') as ir:
        overview=ir.copy();overview.thumbnail((1000,1000))
        fig,ax=plt.subplots(1,2,figsize=(12,9),layout='constrained')
        for a in ax: a.imshow(overview,cmap='gray',extent=(0,6205,8853,0));a.set_xlabel('Raster column');a.set_ylabel('Raster row')
        labsmall=np.asarray(Image.fromarray(labels).resize(overview.size,Image.Resampling.NEAREST))
        overlay=np.zeros((*labsmall.shape,4),dtype=np.float32);overlay[labsmall==1]=[1,0,0,.8]
        ax[1].imshow(overlay,extent=(0,6205,8853,0),interpolation='nearest')
        for t in chosen:
            for a in ax:
                a.add_patch(Rectangle((t['x'],t['y']),512,512,fill=False,edgecolor='cyan',linewidth=1))
                a.text(t['x'],t['y'],str(t['vertical_band']),color='cyan',fontsize=9)
        ax[0].set_title('Published optical image; fixed selected tiles')
        ax[1].set_title('Supplied positives (red), nearest-neighbor overview')
        fig.savefig(run/'optical-overview.png',dpi=130);plt.close(fig)
        diagnostics=[]
        for t in chosen:
            y,x=t['y'],t['x'];lab=labels[y:y+512,x:x+512];msk=mask[y:y+512,x:x+512]
            valid=(msk==1)&(lab<2)
            selected=np.where(valid,lab,2).astype(np.uint8)
            np.save(run/(t['id']+'-labels.npy'),selected,allow_pickle=False)
            crop=ir.crop((x,y,x+512,y+512));gray=np.asarray(crop.convert('L'))
            stats={}
            for name,cls in [('supplied_positive',1),('supplied_negative',0)]:
                values=gray[valid&(lab==cls)]
                stats[name]=dict(n=int(values.size),mean=float(values.mean()),median=float(np.median(values)),
                                 histogram=np.bincount(values,minlength=256).tolist())
            diagnostics.append(dict(tile_id=t['id'],grayscale_rule='Pillow L conversion for descriptive intensity only',stats=stats))
            fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained')
            ax[0].imshow(crop,cmap='gray');ax[0].set_title('Published optical image')
            ax[1].imshow(selected,cmap=matplotlib.colors.ListedColormap(['white','black','#ffcc00']),vmin=0,vmax=2)
            ax[1].set_title('Supplied positives black; unknown yellow')
            ax[2].imshow(crop,cmap='gray');ax[2].contour(selected==1,levels=[.5],colors=['red'],linewidths=.6)
            ax[2].set_title('Supplied-positive contours on optical')
            for a in ax:a.set_xticks([0,255,511]);a.set_yticks([0,255,511])
            fig.suptitle(f"{t['id']} | {t['positive']} positives, {t['negative']} negatives | registration unqualified")
            fig.savefig(run/(t['id']+'-optical.png'),dpi=110);plt.close(fig)
    panel=dict(status='SUPPLIED_REFERENCE_PANEL_SELECTED' if len(chosen)==4 else 'PARTIAL_PANEL_SELECTED',
               candidate='Frag6 / PHerc51',parent_count=1,split='annotation-selected development panel',
               source_objects={k:v['body'] for k,v in objects.items()},protocol=pin(HERE/'PROTOCOL.md'),
               table=pin(run/'TILE_TABLE.json'),tiles=chosen,
               label_arrays=[pin(run/(t['id']+'-labels.npy')) for t in chosen],
               model_outputs_consulted=False,independent_optical_truth=False,optical_registration_qualified=False,
               negative_exhaustiveness_known=False,model_execution_released=False)
    save(run/'ENCODINGS.json',encodings);save(run/'QC_SUMMARY.json',summary)
    save(run/'OPTICAL_DIAGNOSTICS.json',diagnostics);save(run/'PANEL.json',panel)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(peak<=1024**3,'RSS bound');require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<=96*1024**2,'Disk bound')
    save(run/'RUN_RECEIPT.json',dict(status='PASS',started_from_freeze=pin(HERE/'GO.json'),
         completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
         peak_rss_bytes=peak,mask_pixels_decoded=int(labels.size),label_pixels_decoded=int(labels.size),
         optical_pixels_decoded=int(labels.size),scientific_model_forwards=0,GPU_spend_GBP=0))
    print(json.dumps(summary,indent=2)); print(json.dumps(chosen,indent=2))

if __name__=='__main__':main()
