"""Run the fixed d=+1 reading side on both original tile phases, sequential CPU."""
from pathlib import Path
import dataclasses,gc,hashlib,importlib.util,json,logging,os,sys,time
P=Path(__file__).resolve().parent;W=P/'work';OLD=P.parent/'fortyfifth-experiment';NAME='concat_w047-w048_R5B2_z9500-11000'
sys.path.insert(0,str(P/'runtime-source'))
import numpy as np,torch,tifffile
from vesuvius.ink_detection.inference import infer
def save(path,v):path.write_text(json.dumps(v,indent=2)+'\n')
def pin(path):
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path),sha256=sha,bytes=path.stat().st_size)
assert json.loads((P/'RENDER_VERIFICATION.json').read_text())['status']=='PASS'
qualification=json.loads((P/'INFERENCE_QUALIFICATION.json').read_text());assert qualification['status']=='PASS'
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
logging.basicConfig(level=logging.INFO)
infer.load_checkpoint=lambda p:torch.load(p,map_location='cpu',weights_only=True)
infer.load_flat_inference_state=lambda m,s:m.load_state_dict(s,strict=True)
class Count(torch.nn.Module):
    def __init__(self,model,label):super().__init__();self.model=model;self.label=label;self.tiles=0;self.start=time.monotonic()
    def forward(self,*args,**kwargs):
        out=self.model(*args,**kwargs);self.tiles+=args[0].shape[0]
        assert torch.isfinite(out).all()
        if self.tiles%400==0:print(json.dumps(dict(label=self.label,tiles=self.tiles,seconds=time.monotonic()-self.start)),flush=True)
        return out
for rec in qualification['models']:
    ck=Path(rec['checkpoint']['path']);assert pin(ck)==rec['checkpoint']
    name=rec['model']
    if name=='hecate':
        spec=importlib.util.spec_from_file_location('hecate',OLD/'source-evidence/hecate-hecate.py');hec=importlib.util.module_from_spec(spec);spec.loader.exec_module(hec)
        model=hec.load_model(ck,'cpu')
    else:
        initial=infer.parse_args(['unused.zarr',str(ck),'unused.tif','--no-compile']);conf=infer.configure_model(initial)
    for phase in ['plain','padded']:
        outdir=W/f'maps_{phase}';outdir.mkdir(exist_ok=True)
        render_name=NAME+'_66'+('_padded' if phase=='padded' else '')
        filename=f'{render_name}__{name}__L66'+('zc33.50_reverse.png' if name=='hecate' else 's26_reverse.tif')
        output=outdir/filename;receipt=outdir/(filename+'.receipt.json')
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['output']==pin(output);print('Verified completed map',filename,flush=True);continue
        assert not output.exists(),f'Unreceipted output requires inspection: {output}'
        start=time.monotonic()
        if name=='hecate':
            volume=hec.open_volume(W/f'hecate-{phase}.zarr');result=np.zeros(volume.shape[-2:],np.uint8)
            # The original Hecate implementation is used unchanged.
            hec.predict(model,volume,result,reverse=True,stride=None,batch_size=4,precision='fp32')
            hec.write_png(result,output)
            metadata=dict(sampling_um=9.6,reverse_z=True,precision='fp32',stride=32,shape_yx=list(result.shape))
            save(output.with_suffix('.json'),metadata);tiles=None
        else:
            args=infer.parse_args([str(W/f'{phase}-level0.zarr'),str(ck),str(output),'--overlap','0.5','--blend-mode','hann','--layer-start','26','--layer-end','43','--direction','reverse','--batch-size','4','--num-workers','0','--no-compile'])
            count=Count(conf.model,name+'/'+phase)
            infer.infer_single_zarr(args=args,input_zarr=args.input_zarr,configured_model=dataclasses.replace(conf,model=count),device=torch.device('cpu'),output_tiff=output,layer_direction='reverse')
            tiles=count.tiles
            result=tifffile.imread(output)
        assert result.ndim==2 and result.dtype==np.uint8
        save(receipt,dict(status='PASS',model=name,phase=phase,checkpoint=rec['checkpoint'],output=pin(output),decoded_sha256=hashlib.sha256(result.tobytes()).hexdigest(),shape=list(result.shape),min=int(result.min()),max=int(result.max()),tile_forwards=tiles,elapsed_seconds=time.monotonic()-start,device='cpu',precision='fp32',compile=False))
        print('Completed',filename,flush=True)
    if name=='hecate':del model
    else:del conf,count
    gc.collect()
save(P/'FIXED_DEPTH_INFERENCE.json',dict(status='PASS',maps=[json.loads(p.read_text()) for p in sorted(W.glob('maps_*/*.receipt.json'))],gpu_spend_GBP=0))
