"""One approved native trajectory, with exact predecessor replay and optimizer state."""
import sys, time, hashlib, json
from pathlib import Path
P = Path(__file__).resolve().parent
OLD = P.parent / 'fortyfourth-experiment'
sys.path.insert(0, str(OLD))
import learn_resolution as old
import numpy as np
import torch

def run():
    go = json.loads((P / 'RUN_GO.json').read_text())
    for item in go['pins']:
        old.need(old.pin(item['path']) == item, 'Frozen continuation dependency')
    out = P / 'training'
    out.mkdir(exist_ok=False)
    data = old.prepare()
    train_ids = [1, 2]
    torch.manual_seed(old.SEED)
    rng = np.random.default_rng(old.SEED)
    model = old.Net()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    truth = np.concatenate([data[i]['label'][data[i]['support']] for i in train_ids])
    prevalence = float(truth.mean())
    pos_weight = torch.tensor((1-prevalence)/prevalence)
    losses, fits = [], []
    start = time.monotonic()
    log = (out / 'progress.jsonl').open('x')
    def emit(row):
        log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
        print(json.dumps(row, allow_nan=False), flush=True)
    for step in range(1, 8193):
        model.train()
        x,y,m = old.draw_batch(data, train_ids, 'native', rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)[:,:,32:160,32:160]
        pixel_loss = old.F.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight,reduction='none')
        loss = pixel_loss[m].mean()
        old.need(torch.isfinite(loss).item(), 'Finite optimization')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 128 == 0:
            emit(dict(step=step, mean_loss=float(np.mean(losses[-128:])), elapsed_seconds=time.monotonic()-start))
        if step in [512,1024,2048,4096,8192]:
            if step == 2048:
                ref = torch.load(OLD/'training/fold0-native.pt', map_location='cpu', weights_only=True)
                actual = model.state_dict()
                old.need(actual.keys() == ref.keys(), 'Replay tensor names')
                equal = {k: bool(torch.equal(actual[k],ref[k])) for k in ref}
                old.save(out/'REPLAY_VERIFICATION.json', dict(step=step,tensors=equal,all_equal=all(equal.values())))
                old.need(all(equal.values()), 'Exact 2048-step replay before continuation')
            fit=[]
            for i in train_ids:
                score=old.predict(model,data[i]['images']['native'])
                if step in [2048,4096,8192]: np.save(out/f'step{step}-train-{data[i]["case"]}.npy',score)
                fit.append(dict(case=data[i]['case'],metrics=old.metrics(data[i]['label'],score,data[i]['support'])))
            fits.append(dict(step=step,cases=fit)); emit(dict(training_fit=fits[-1]))
            if step in [2048,4096,8192]:
                checkpoint=dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
                dest=out/f'step{step}-continuation.pt'; torch.save(checkpoint,dest)
                # Check serialization preserves all AdamW moments, counts and generator state.
                reload=torch.load(dest,map_location='cpu',weights_only=True)
                old.need(reload['numpy_rng']==checkpoint['numpy_rng'] and torch.equal(reload['torch_rng'],checkpoint['torch_rng']),'Saved RNG state')
                for key,state in checkpoint['optimizer']['state'].items():
                    for name,value in state.items(): old.need(torch.equal(value,reload['optimizer']['state'][key][name]),'Saved optimizer tensor')
                old.need(all(torch.equal(v,reload['model'][k]) for k,v in checkpoint['model'].items()),'Saved model tensors')
                old.save(out/f'step{step}-checkpoint.json',dict(pin=old.pin(dest),model_tensors=len(checkpoint['model']),optimizer_parameters=len(checkpoint['optimizer']['state']),optimizer_and_rng_readback=True))
    evaluation=[]
    for quality in old.QUALITIES:
        score=old.predict(model,data[0]['images'][quality])
        np.save(out/f'step8192-reserved-{quality}.npy',score)
        evaluation.append(dict(quality=quality,metrics=old.metrics(data[0]['label'],score,data[0]['support'])))
    old.save(P/'TRAINING_RESULTS.json',dict(created_at_utc=old.now(),seed=old.SEED,updates=8192,replayed_updates=2048,new_continuation_updates=6144,training_fit=fits,reserved_development_evaluation=evaluation,elapsed_seconds=time.monotonic()-start,prevalence=prevalence,pos_weight=float(pos_weight)))
    emit(dict(status='COMPLETE',elapsed_seconds=time.monotonic()-start))
    log.close()

if __name__ == '__main__': run()
