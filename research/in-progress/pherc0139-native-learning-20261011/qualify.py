"""Synthetic execution qualification; no real annotations/CT or held-out models."""
import argparse
import copy
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import native_learning as n


def main(out):
    n.setup()
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(name)
    plan = n.read(Path(__file__).with_name('PLAN.json'))
    # Asymmetric nonzero source origins and the final depth plane.
    yy, xx = np.indices((512,512))
    a = np.empty((28,512,512), np.uint8)
    for z in range(28):
        a[z] = (xx + 3*yy + 7*z) % 256
    chunks = {(y,x): a[:, y*128:(y+1)*128, x*128:(x+1)*128] for y in range(4) for x in range(4)}
    context = n.assemble_context(chunks, 128, 256)
    check('asymmetric full28 chunk assembly and final plane', np.array_equal(context, a[:,192:448,64:320]))
    logits = torch.tensor([[[1.,-2.],[8.,-5.]], [[0.,0.],[3.,4.]]])
    y = torch.tensor([[[1.,0.],[1.,0.]], [[1.,0.],[0.,0.]]])
    mask = torch.tensor([[[1.,1.],[0.,0.]], [[1.,0.],[0.,0.]]])
    one = n.masked_core_loss(logits,y,mask)
    changed = y.clone(); changed[mask == 0] = 1-changed[mask == 0]
    check('stored ink outside supervision is ignored', torch.equal(one,n.masked_core_loss(logits,changed,mask)))
    per_pixel = torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
    expected = ((per_pixel[0]*mask[0]).sum()+(per_pixel[1]*mask[1]).sum())/mask.sum()
    weighted_core = 2/3*n.masked_core_loss(logits[:1],y[:1],mask[:1]) + 1/3*n.masked_core_loss(logits[1:],y[1:],mask[1:])
    check('count proportional sampling gives uniform supervised pixel objective', torch.allclose(expected,weighted_core))
    try:
        n.masked_core_loss(logits,y,torch.zeros_like(mask))
        raise AssertionError('empty mask accepted')
    except ValueError:
        checks.append('empty training mask rejected')
    metric = n.ranking_metrics([1,0,1,0], [2,2,2,2])
    check('constant score ties yield prevalence AP and chance AUC', metric['ap']==.5 and metric['auc']==.5)
    metric = n.ranking_metrics([1,0,1,0], [4,1,3,2])
    check('perfect ranking AP/AUC', metric['ap']==1 and metric['auc']==1)
    check('blank core rank metrics are undefined', n.ranking_metrics([0,0],[1,3])['ap'] is None)
    rng = np.random.default_rng(12)
    x = rng.integers(0,256,(4,28,256,256),dtype=np.uint8)
    target = (x[:,27,64:192,64:192] > 127)
    m = np.zeros((4,128,128),bool)
    m[:,::4,::4] = True
    data = dict(x=x, y=target, mask=m, rows=[dict(core_id=f'c{i}', winding='a' if i<3 else 'b') for i in range(4)])
    norm = n.normalization(data, [0,1,2])
    data['x'][3] = 0
    check('held-out intensities cannot alter normalization', norm == n.normalization(data,[0,1,2]))
    fold = dict(retained_training_core_ids=['c0','c2'], original_evaluation_core_ids=['c3'], held_out='b', fold_geometry_admitted=True)
    check('fold loader uses exact exclusion IDs', n.fold_indices(data,fold) == ([0,2],[3]))
    fold['retained_training_core_ids'] += ['c3']
    try:
        n.fold_indices(data,fold)
        raise RuntimeError('leak accepted')
    except AssertionError:
        checks.append('overlapping train and held-out IDs rejected')
    torch.manual_seed(17)
    model = n.NativeNet()
    optimizer = torch.optim.Adam(model.parameters(),lr=plan['learning_rate'])
    sample_rng = np.random.default_rng(17)
    output = model(n.normalized(data['x'][:1], norm))
    check('model accepts28planes full256 and scores128', tuple(output.shape)==(1,128,128))
    # Check exact restart across actual model, augmentation sampler, Adam and RNG.
    initial = n.initial_state(17,dict(bce=.7,ap=.5,auc=.5),plan)
    with tempfile.TemporaryDirectory(prefix='native-learning-qualify-') as temporary:
        directory = Path(temporary)
        for _ in range(2):
            n.train_step(model,optimizer,data,[0,1,2],norm,sample_rng,4)
            initial['step'] += 1
        row = n.commit_checkpoint(directory,dict(test='synthetic'),model,optimizer,sample_rng,initial)
        continued_losses = []
        for _ in range(2):
            loss, picked, _ = n.train_step(model,optimizer,data,[0,1,2],norm,sample_rng,4)
            continued_losses.append((loss,picked))
        continued = copy.deepcopy(model.state_dict())
        continued_rng = copy.deepcopy(sample_rng.bit_generator.state)
        fresh = n.NativeNet()
        fresh_opt = torch.optim.Adam(fresh.parameters(), lr=plan['learning_rate'])
        loaded, fresh_rng = n.load_checkpoint(directory,directory/'step-000002.json',dict(test='synthetic'),fresh,fresh_opt)
        replay_losses = []
        for _ in range(2):
            loss, picked, _ = n.train_step(fresh,fresh_opt,data,[0,1,2],norm,fresh_rng,4)
            replay_losses.append((loss,picked))
        check('resume reproduces sampled cores and exact losses', continued_losses==replay_losses)
        check('resume reproduces model tensors bit for bit', all(torch.equal(v,fresh.state_dict()[k]) for k,v in continued.items()))
        check('resume preserves sampler RNG and optimizer moments', continued_rng==fresh_rng.bit_generator.state and all(
            torch.equal(v, fresh_opt.state_dict()['state'][i][k]) for i,row in optimizer.state_dict()['state'].items() for k,v in row.items()))
        # Missing publication marker means no committed checkpoint; damaged data
        # are rejected before any deserialization.
        (directory/'step-999999.pt').write_bytes(b'interrupted file')
        check('partial unreceipted checkpoint is not discoverable', len(list(directory.glob('step-*.json')))==1)
        path = directory/'step-000002.pt'
        with path.open('r+b') as f:
            f.seek(10); old=f.read(1); f.seek(10); f.write(bytes([old[0]^1]))
        try:
            n.load_checkpoint(directory,directory/'step-000002.json',dict(test='synthetic'),fresh,fresh_opt)
            raise RuntimeError('corruption accepted')
        except ValueError:
            checks.append('checkpoint hash mismatch rejected before load')
        try:
            n.evaluate(directory,Path('/unused'),{},[{'held_out':'x'}],plan,'test')
            raise RuntimeError('unsealed evaluation accepted')
        except AssertionError:
            checks.append('evaluation blocked until every training endpoint is complete')
    state = n.initial_state(17,dict(bce=.7),plan)
    for i in range(16):
        state['step'] += 256
        stop = n.update_plateau(state,.7,plan)
    check('training plateau reaches minimum lr without premature stop', state['lr']==plan['minimum_lr'] and not stop)
    for _ in range(8):
        state['step'] += 256
        stop = n.update_plateau(state,.7,plan)
    check('sustained floor plateau terminates after declared patience', stop)
    state = n.initial_state(17,dict(bce=.7),plan)
    for i in range(20):
        state['step']+=256
        stop=n.update_plateau(state,.7-.001*(i+1),plan)
    check('material continued improvement prevents plateau stop', not stop and state['lr']==plan['learning_rate'])
    # A known final-plane synthetic label validates the convex reference path.
    fit = n.fit_profile(data,[0,1,2],norm,plan)
    check('profile fits known synthetic depth27 signal and converges', fit['converged'] and fit['train']['ap']>.99)
    for _ in range(2):
        n.train_step(model,optimizer,data,[0,1,2],norm,sample_rng,4)
    begin=time.monotonic()
    steps=12
    for _ in range(steps):
        n.train_step(model,optimizer,data,[0,1,2],norm,sample_rng,4)
    elapsed=time.monotonic()-begin
    result=dict(created_utc=n.now(),all_passed=True,checks=checks,environment=n.environment(),
                parameter_count=sum(p.numel() for p in model.parameters()),
                benchmark=dict(updates=steps,batch_size=4,context=[28,256,256],seconds=elapsed,seconds_per_update=elapsed/steps,
                includes='sampling, normalization, augmentation, forward/backward, clipping, Adam',
                excludes='real data loading, full training checks, checkpoint serialization, backup and baseline fits'),
                real_CT_or_labels_read=False,held_out_predictions=0)
    n.atomic_json(out,result)
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
