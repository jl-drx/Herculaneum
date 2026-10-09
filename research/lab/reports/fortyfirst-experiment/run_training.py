import argparse,gc,math,platform,time,random,socket
import numpy as np
from common import *
from data_loader import batches
from objective import terms
def offline(*a,**k):raise RuntimeError('Network forbidden')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
def main(args):
    started=time.monotonic();limits();p=plan();old=legacy();base=json.loads(verify(p['training_plan']).read_text());out=HERE/'run-v1/E_no_new_BCE';out.mkdir(exist_ok=False)
    import torch
    require(platform.python_version()=='3.12.14' and np.__version__=='2.5.3' and torch.__version__=='2.12.0+cpu','Pinned runtime')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261008)
    helper=module(verify(base['helper']),'phase41_training_helper');runtime=helper.load_runtime();model,meta=helper.build_model(Path(base['initializer']['path']),base['initializer'],runtime)
    old.mode_guard(model,meta,torch);model.train();initial=old.state_hash(model.state_dict());require(initial==base['historical_initial_state_sha256'],'Initializer identity')
    mixed=batches(p,'mixed',torch);original=batches(p,'original',torch)
    teacher_receipt=json.loads(verify(p['teacher_receipt']).read_text());require(teacher_receipt['status']=='PASS' and teacher_receipt['teacher_forwards']==32,'Reused frozen phase40 teacher complete')
    cache=np.load(verify(teacher_receipt['logits']),allow_pickle=False);teachers=[torch.from_numpy(q.copy()) for q in cache]
    original_cache=np.load(verify(base['historical']['A_preserve']['teacher']),allow_pickle=False);original_teachers=[torch.from_numpy(q.copy()) for q in original_cache]
    identities=[old.state_hash(dict(input=b[0],manual_mask=b[1],manual_hard_target=b[2])) for b in mixed]
    opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0)
    pre=np.lib.format.open_memmap(out/'preupdate-logits.npy',mode='w+',dtype=np.float32,shape=(512,2,1,128,128));fixed=np.lib.format.open_memmap(out/'fixed-risk-logits.npy',mode='w+',dtype=np.float32,shape=(5,32,2,1,128,128))
    with torch.no_grad():
        baseline=[terms(q,q,b,torch,i%2)[1] for i,(q,b) in enumerate(zip(teachers,mixed))]
        original_baseline=[terms(q,q,b,torch)[1] for q,b in zip(original_teachers,original)]
    measurements={'0':dict(rows=baseline,mean=old.average(baseline),scope='frozen cache-derived initializer')};checks={};forwards=0;checkpoints={}
    def measure(bs,qs,is_mixed=False):
        before=old.isolation_state(model,opt,torch);rng=old.tree_hash((torch.random.get_rng_state(),np.random.get_state(),random.getstate()),torch);rows=[];zs=[]
        with torch.no_grad():
            for i,(b,q) in enumerate(zip(bs,qs)):
                z=old.forward(model,b[0],torch);rows.append(terms(z,q,b,torch,i%2 if is_mixed else None)[1]);zs.append(z.numpy().copy())
        require(before==old.isolation_state(model,opt,torch) and rng==old.tree_hash((torch.random.get_rng_state(),np.random.get_state(),random.getstate()),torch),'Measurement state/optimizer/gradient/mode/RNG isolation')
        return np.stack(zs),dict(rows=rows,mean=old.average(rows),isolation=before,state_sha256=before['model'])
    with (out/'steps.jsonl').open('x') as log:
        for step in range(1,513):
            require(time.monotonic()<args.deadline and rss()<4*1024**3,'Training budgets');clock=time.monotonic();i=(step-1)%32;b=mixed[i];q=teachers[i]
            require(old.state_hash(dict(input=b[0],manual_mask=b[1],manual_hard_target=b[2]))==identities[i],'Fixed batch unchanged')
            opt.zero_grad(set_to_none=True);z=old.forward(model,b[0],torch);forwards+=1;pre[step-1]=z.detach().numpy();z.retain_grad();loss,values=terms(z,q,b,torch,i%2)
            require(all(math.isfinite(v) for v in values.values()),'Finite objective')
            if step==1:require(np.array_equal(z.detach().numpy().view(np.uint32),cache[0].view(np.uint32)),'First logits equal original-seed42 cache')
            loss.backward();require(bool((z.grad[~b[1]]==0).all()),'Zero unknown output gradient');norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);require(float(norm)>0,'Nonzero gradient');opt.step()
            require(all(bool(torch.isfinite(v).all()) for v in model.parameters()),'Finite parameters')
            if step==1:
                with torch.no_grad():post=old.forward(model,b[0],torch);forwards+=1
                np.save(out/'first-post-logits.npy',post.numpy(),allow_pickle=False);del post
            if step in UPDATES:
                fz,r=measure(mixed,teachers,True);forwards+=32;fixed[UPDATES.index(step)]=fz;measurements[str(step)]=r
                steps=[int(v['step'].item()) for v in opt.state.values()];require(len(steps)==264 and set(steps)=={step},'Uninterrupted optimizer state');r['optimizer_step_min']=min(steps);r['optimizer_step_max']=max(steps)
                if step in [32,512]:
                    ck=out/f'state-{step}.pth';torch.save(model.state_dict(),ck);loaded=torch.load(ck,map_location='cpu',weights_only=True);require(old.state_hash(loaded)==r['state_sha256'],'Saved state exact')
                    storage=helper.verify_storage_contract(loaded,model.state_dict(),torch);result=model.load_state_dict(loaded,strict=True);require(not result.missing_keys and not result.unexpected_keys,'Strict reload')
                    checkpoints[str(step)]=dict(file=pin(ck),state_sha256=r['state_sha256'],storage=storage);del loaded;gc.collect()
                print(json.dumps(dict(step=step,mean=r['mean'],elapsed_seconds=time.monotonic()-started)),flush=True)
            log.write(json.dumps(dict(step=step,schedule_index=i,input_target_sha256=identities[i],pre=values,gradient_norm=float(norm),unknown_gradient_max=0,wall_seconds=time.monotonic()-clock),allow_nan=False)+'\n');log.flush();del z,loss
    original_z,original_final=measure(original,original_teachers);forwards+=32;np.save(out/'original-final-logits.npy',original_z,allow_pickle=False)
    pre.flush();fixed.flush();del pre,fixed
    require(forwards==705 and time.monotonic()<args.deadline and rss()<4*1024**3,'Final budgets')
    plan()
    save(out/'COMPLETE.json',dict(status='PASS',arm='E_no_new_BCE',GO=pin(HERE/'GO.json'),initial_state_sha256=initial,checkpoints=checkpoints,measurements=measurements,original_baseline=dict(rows=original_baseline,mean=old.average(original_baseline)),original_final=original_final,original_final_logits=pin(out/'original-final-logits.npy'),batch_fingerprints=identities,teacher=teacher_receipt['logits'],pre_logits=pin(out/'preupdate-logits.npy'),fixed_logits=pin(out/'fixed-risk-logits.npy'),steps=pin(out/'steps.jsonl'),optimizer_updates=512,backwards=512,student_forwards=forwards,teacher_forwards_in_this_worker=0,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--deadline',type=float,required=True);main(a.parse_args())
