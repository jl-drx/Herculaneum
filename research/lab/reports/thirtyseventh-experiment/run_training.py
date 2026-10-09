import argparse,gc,json,math,platform,time,signal,socket
import numpy as np
from study_common import *
from objectives import terms
def offline(*a,**k):raise RuntimeError('Network forbidden')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline

def main(args):
    started=time.monotonic();limits();p=plan();old=legacy();base=json.loads(verify(p['training_plan']).read_text());out=HERE/'run-v1'/args.arm;out.mkdir(exist_ok=False)
    import torch
    require(platform.python_version()=='3.12.14' and np.__version__=='2.5.3' and torch.__version__=='2.12.0+cpu','Pinned runtime')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261008)
    helper=module(verify(base['helper']),'phase37_model_contract');preservation=module(verify(base['shared_sources']['preservation.py']),'phase37_preservation')
    runtime=helper.load_runtime();model,meta=helper.build_model(Path(base['initializer']['path']),base['initializer'],runtime)
    old.mode_guard(model,meta,torch);model.train();initial=old.state_hash(model.state_dict());require(initial==base['historical_initial_state_sha256'],'Initializer state')
    schedule=json.loads(verify(base['schedule']).read_text());regions=[old.training_region(c) for c in base['adaptation']]
    historical=[json.loads(l) for l in verify(base['historical']['A_preserve']['steps']).read_text().splitlines()]
    cache=np.load(verify(base['historical']['A_preserve']['teacher']),allow_pickle=False)
    batches=[];identities=[];teachers=[]
    for i,s in enumerate(schedule):
        b,h,e=old.batch(s,regions,runtime,torch);require(h==historical[i]['input_target_sha256'] and e==historical[i]['exposures'] and s['patches']==historical[i]['patches'],'Historical batch/exposure identity')
        batches.append(b);identities.append(h);teachers.append(torch.from_numpy(cache[i].copy()))
    np.savez_compressed(out/'SUPERVISION.npz',manual_mask=np.stack([b[1].numpy() for b in batches]),manual_target=np.stack([b[2].numpy() for b in batches]))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0)
    pre=np.lib.format.open_memmap(out/'preupdate-logits.npy',mode='w+',dtype=np.float32,shape=(512,2,1,128,128))
    fixed=np.lib.format.open_memmap(out/'fixed-risk-logits.npy',mode='w+',dtype=np.float32,shape=(5,32,2,1,128,128))
    with torch.no_grad():baseline=[terms(q,q,b,args.arm,old,preservation,torch)[1] for q,b in zip(teachers,batches)]
    measurements={'0':dict(rows=baseline,mean=old.average(baseline),scope='cache-derived initializer')};checkpoints={};first={};forwards=0
    def measure_all():
        before=old.isolation_state(model,opt,torch);rng=old.tree_hash((torch.random.get_rng_state(),np.random.get_state()),torch);rows=[];zs=[]
        with torch.no_grad():
            for b,q in zip(batches,teachers):
                z=old.forward(model,b[0],torch);rows.append(terms(z,q,b,args.arm,old,preservation,torch)[1]);zs.append(z.numpy().copy())
        after=old.isolation_state(model,opt,torch)
        require(before==after and rng==old.tree_hash((torch.random.get_rng_state(),np.random.get_state()),torch),'Risk evaluation isolation')
        return np.stack(zs),dict(rows=rows,mean=old.average(rows),isolation=before,state_sha256=before['model'])
    with (out/'steps.jsonl').open('x') as log:
        for step in range(1,513):
            clock=time.monotonic();require(clock<args.deadline and rss()<4*1024**3,'Training time/memory cap')
            i=(step-1)%32;b=batches[i];q=teachers[i];opt.zero_grad(set_to_none=True)
            z=old.forward(model,b[0],torch);forwards+=1;pre[step-1]=z.detach().numpy();z.retain_grad()
            loss,values=terms(z,q,b,args.arm,old,preservation,torch)
            if step==1:require(np.array_equal(z.detach().numpy().view(np.uint32),cache[0].view(np.uint32)),'Initial logits bitwise cache identity')
            loss.backward();require(bool((z.grad[~b[1]]==0).all()),'Nonmanual output gradient')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);require(float(norm)>0,'Nonzero finite gradient')
            opt.step();require(all(bool(torch.isfinite(v).all()) for v in model.parameters()),'Finite updated weights')
            if step==1:
                with torch.no_grad():post=old.forward(model,b[0],torch);forwards+=1
                np.save(out/'first-post-logits.npy',post.numpy(),allow_pickle=False);first=dict(state_sha256=old.state_hash(model.state_dict()),gradient_norm=float(norm),historical_match=None)
                if args.arm=='A_control':
                    proof=base['first_step_probe']['A_preserve'];expected=np.load(verify(proof['post_logits']),allow_pickle=False)
                    require(first['state_sha256']==proof['post_state_sha256'] and np.array_equal(post.numpy().view(np.uint32),expected.view(np.uint32)),'Control first-step historical state/logits')
                    first['historical_match']=True
                del post
            if step in UPDATES:
                fz,entry=measure_all();forwards+=32;fixed[UPDATES.index(step)]=fz;measurements[str(step)]=entry
                state_steps=[int(v['step'].item()) for v in opt.state.values()]
                require(len(state_steps)==264 and set(state_steps)=={step},'Uninterrupted optimizer steps')
                entry['optimizer_step_min']=min(state_steps);entry['optimizer_step_max']=max(state_steps)
                if args.arm=='A_control' and step==32:
                    prior=json.loads(verify(p['historical_control32']).read_text())
                    require(entry['state_sha256']==prior['final_state_sha256'],'Control update32 bitwise historical state')
                    oldfixed=np.load(verify(prior['raw_logits']['fixed_risk']),mmap_mode='r',allow_pickle=False)
                    require(np.array_equal(fz.view(np.uint32),oldfixed[3].view(np.uint32)),'Control update32 all fixed-risk logits')
                    entry['historical_update32_exact']=True
                if step in [32,512]:
                    ck=out/f'state-{step}.pth';torch.save(model.state_dict(),ck);loaded=torch.load(ck,map_location='cpu',weights_only=True)
                    require(old.state_hash(loaded)==entry['state_sha256'],'Saved state exact')
                    storage=helper.verify_storage_contract(loaded,model.state_dict(),torch);result=model.load_state_dict(loaded,strict=True)
                    require(not result.missing_keys and not result.unexpected_keys,'Checkpoint strict load')
                    checkpoints[str(step)]=dict(file=pin(ck),state_sha256=entry['state_sha256'],storage=storage);del loaded;gc.collect()
                print(json.dumps(dict(arm=args.arm,step=step,mean=entry['mean'],elapsed_seconds=time.monotonic()-started)),flush=True)
            log.write(json.dumps(dict(step=step,schedule_index=i,input_target_sha256=identities[i],pre=values,gradient_norm=float(norm),unknown_gradient_max=0,wall_seconds=time.monotonic()-clock),allow_nan=False)+'\n');log.flush();del z,loss
    pre.flush();fixed.flush();del pre,fixed
    require(forwards==673 and rss()<4*1024**3 and time.monotonic()<args.deadline,'Final budgets')
    for pfile in json.loads((HERE/'GO.json').read_text())['files']:verify(pfile)
    save(out/'COMPLETE.json',dict(status='PASS',arm=args.arm,GO=pin(HERE/'GO.json'),initial_state_sha256=initial,checkpoints=checkpoints,first_step=first,measurements=measurements,batch_fingerprints=identities,
        supervision=pin(out/'SUPERVISION.npz'),teacher=base['historical']['A_preserve']['teacher'],pre_logits=pin(out/'preupdate-logits.npy'),fixed_logits=pin(out/'fixed-risk-logits.npy'),steps=pin(out/'steps.jsonl'),
        optimizer_updates=512,backwards=512,student_forwards=forwards,teacher_forwards=0,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--arm',choices=ARMS,required=True);a.add_argument('--deadline',type=float,required=True);main(a.parse_args())
