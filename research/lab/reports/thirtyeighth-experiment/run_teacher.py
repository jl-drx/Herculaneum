"""One frozen original-seed42 cache on the fixed mixed schedule; no learning."""
import argparse,time,socket,gc
import numpy as np
from common import *
from data_loader import batches
def offline(*a,**k):raise RuntimeError('Network forbidden')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
def main(args):
    start=time.monotonic();limits();p=plan();old=legacy();base=json.loads(verify(p['training_plan']).read_text());out=HERE/'run-v1/teacher';out.mkdir(exist_ok=False)
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261008)
    helper=module(verify(base['helper']),'phase38_teacher_helper');runtime=helper.load_runtime();model,meta=helper.build_model(Path(base['initializer']['path']),base['initializer'],runtime)
    old.mode_guard(model,meta,torch);model.train();before=old.state_hash(model.state_dict());require(before==base['historical_initial_state_sha256'],'Exact seed42')
    source=batches(p,'mixed',torch);historical=np.load(verify(base['historical']['A_preserve']['teacher']),allow_pickle=False);rows=[]
    with torch.no_grad():
        for i,b in enumerate(source):
            require(time.monotonic()<args.deadline and rss()<4*1024**3,'Teacher budgets')
            z=old.forward(model,b[0],torch).numpy();keep=1-i%2
            require(np.array_equal(z[keep].view(np.uint32),historical[i,keep].view(np.uint32)),'Unchanged patch teacher logits bitwise match')
            rows.append(z.copy())
    require(before==old.state_hash(model.state_dict()) and all(x.grad is None for x in model.parameters()),'Frozen teacher unchanged')
    np.save(out/'logits.npy',np.stack(rows),allow_pickle=False)
    save(out/'COMPLETE.json',dict(status='PASS',teacher_forwards=32,optimizer_updates=0,checkpoint=base['initializer'],state_sha256=before,retained_PHerc0139_logits_exact=32,logits=pin(out/'logits.npy'),elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=rss()))
    print(json.dumps(dict(status='PASS',teacher_forwards=32,elapsed_seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--deadline',type=float,required=True);main(a.parse_args())
