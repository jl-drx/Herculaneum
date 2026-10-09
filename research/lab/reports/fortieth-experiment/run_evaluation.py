import argparse,gc,hashlib,time,socket
import numpy as np
from common import *
def offline(*a,**k):raise RuntimeError('Network forbidden')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline

class Capture:
    def __init__(self,model,expected):self.model=model;self.expected=expected;self.logits=[];self.hashes=[]
    def __call__(self,x):
        digest=hashlib.sha256(x.numpy().tobytes()).hexdigest();i=len(self.logits)
        require(i<25,'Tile budget')
        if self.expected is not None:require(digest==self.expected[i]['input_sha256'],'Exact archived normalized tile')
        result=self.model(x);z=result['ink'];z=z[0] if isinstance(z,(tuple,list)) else z
        require(z.shape==(1,1,128,128),'Output tile');self.logits.append(z.detach().numpy()[0,0].copy());self.hashes.append(digest)
        require(hashlib.sha256(x.numpy().tobytes()).hexdigest()==digest,'Input mutation');return result

def main(args):
    started=time.monotonic();limits();p=plan();old=legacy();base=json.loads(verify(p['training_plan']).read_text());out=HERE/'run-v1'/f'eval-{args.arm}-{args.step}';out.mkdir(exist_ok=False)
    require(json.loads((HERE/'TRAINING_VERIFICATION.json').read_text())['status']=='PASS','Independent numerical training gate')
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261008)
    helper=module(verify(base['helper']),'phase40_evaluation_model');runtime=helper.load_runtime();model,meta=helper.build_model(Path(base['initializer']['path']),base['initializer'],runtime)
    trained=json.loads((HERE/'run-v1'/args.arm/'COMPLETE.json').read_text());source=trained['checkpoints'][str(args.step)]
    state=torch.load(verify(source['file']),map_location='cpu',weights_only=True);require(old.state_hash(state)==source['state_sha256'],'Checkpoint semantic identity')
    helper.verify_storage_contract(state,model.state_dict(),torch);model.load_state_dict(state,strict=True);del state;gc.collect();model.eval();before=old.state_hash(model.state_dict());rows=[]
    for c in p['cases']:
        require(time.monotonic()<args.deadline and rss()<4*1024**3,'Evaluation budget')
        x,y,m=load_case(p,c);capture=Capture(model,c['baseline_metadata'].get('captured_tiles'))
        scores,metadata,evaluated=runtime['reference'].infer_surface(capture,x,c['layer_start'],'forward',runtime['normalize_robust'],None)
        require(metadata['tile_count']==metadata['scored_tile_count']==len(capture.logits)==25 and evaluated[m].all() and np.isfinite(scores[m]).all(),'Complete unchanged evaluation support')
        expected=[[xx,yy,128,128] for yy in range(0,257,64) for xx in range(0,257,64)]
        require([t['bbox_roi_xywh'] for t in metadata['tiles']]==expected,'Unchanged raster tiling')
        d=out/c['id'];d.mkdir();np.save(d/'scores.npy',scores,allow_pickle=False);np.save(d/'evaluated.npy',evaluated,allow_pickle=False);np.save(d/'logits.npy',np.stack(capture.logits),allow_pickle=False)
        rows.append(dict(id=c['id'],scores=pin(d/'scores.npy'),evaluated=pin(d/'evaluated.npy'),logits=pin(d/'logits.npy'),inference=metadata,input_hashes=capture.hashes,archived_input_hashes_checked=capture.expected is not None))
    require(before==old.state_hash(model.state_dict()) and all(v.grad is None for v in model.parameters()),'Evaluation preserves model')
    require(time.monotonic()<args.deadline and rss()<4*1024**3,'Final evaluation budgets')
    save(out/'COMPLETE.json',dict(status='PASS',arm=args.arm,step=args.step,checkpoint=source,contexts=rows,student_forwards=200,model_state_unchanged=True,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss()))
    print(json.dumps(dict(status='PASS',arm=args.arm,step=args.step,elapsed_seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--arm',choices=ARMS,required=True);a.add_argument('--step',type=int,choices=[512],required=True);a.add_argument('--deadline',type=float,required=True);main(a.parse_args())
