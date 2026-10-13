"""Counterfactual old-source loss-weight test, with no model or optimization.

Identical original logits/targets/mask and batch slots; only the added stream's
supervision density changes. Uses the retained smoothed BCE + Dice objective.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    root=a.repo/'research/lab/reports/fortysixth-experiment/runtime-source'
    sys.path.insert(0,str(root))
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.training.losses import create_loss
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg=InkConfig.from_mapping(dict(mode='flat',model_type='vesuvius_unet_3d_stem_2d',in_channels=1,
        patch_size=[17,128,128],patch_overlap=.25,patch_min_labeled_coverage=0.,
        datasets=[dict(segments_path='synthetic',volume_scale=0)],
        targets=dict(ink=dict(activation='none',out_channels=1)),
        loss=dict(bce_label_smoothing=.5,dice_label_smoothing=0.),out_dir='unused'))
    criterion=create_loss(cfg);bce=create_loss(cfg)
    for term in bce.terms:term.module.weight_dice=0.
    targets=torch.zeros((2,1,16,16));targets[0,0,:4,:4]=1.
    sparse=torch.zeros_like(targets);sparse[:,:,:8,:8]=1.
    dense=sparse.clone();dense[1]=1.
    base=torch.linspace(-.7,.8,512).reshape(2,1,16,16)
    def compute(mask,mode,objective):
        logits=base.clone().requires_grad_(True);joined=torch.cat([targets,1-mask],dim=1)
        if mode=='pooled':loss=objective(logits,joined)
        else:loss=.5*objective(logits[:1],joined[:1])+.5*objective(logits[1:],joined[1:])
        loss.backward();return float(loss.detach()),logits.grad.detach().clone()
    a_loss,a_grad=compute(sparse,'pooled',criterion)
    b_loss,b_grad=compute(dense,'pooled',criterion)
    delta=float((a_grad[0]-b_grad[0]).abs().max());assert delta>1e-6
    _,a_bce=compute(sparse,'pooled',bce);_,b_bce=compute(dense,'pooled',bce)
    expected=float(sparse.sum()/dense.sum())
    valid=sparse[0].bool();observed=b_bce[0][valid]/a_bce[0][valid]
    assert torch.allclose(observed,torch.full_like(observed,expected),atol=1e-7,rtol=1e-6)
    fixed_a,grad_a=compute(sparse,'fixed_stream',criterion)
    fixed_b,grad_b=compute(dense,'fixed_stream',criterion)
    assert torch.equal(grad_a[0],grad_b[0])
    assert not torch.equal(grad_a[1],grad_b[1])
    result=dict(status='PASS_MECHANISM_IDENTIFIED',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        fixed=dict(original_batch_slots=1,additional_batch_slots=1,original_supervised_pixels=64,
                   original_logits_targets_and_mask=True,bce_label_smoothing=.5,dice_label_smoothing=0.),
        changed=dict(additional_supervised_pixels_A=64,additional_supervised_pixels_B=256),
        pooled_objective=dict(loss_A=a_loss,loss_B=b_loss,
            original_gradient_max_absolute_change=delta,original_bce_gradient_ratio_B_over_A=float(observed.mean()),
            analytic_ratio=expected,conclusion='Identical patch draws do not fix original-source BCE weight under a global mask denominator'),
        fixed_stream_objective=dict(formula='.5 * original_stream_loss + .5 * added_stream_loss',
            loss_A=fixed_a,loss_B=fixed_b,original_logit_gradients_bitwise_equal=True,
            added_logit_gradients_differ=True,
            conclusion='Separate denominators and fixed stream weights remove this specific dilution mechanism'),
        scope='Synthetic float32 logit gradients through the retained loss. Not a training effect, full-model gradient guarantee, or novel objective claim.',
        limits=['A changed added corpus can still change label amount, class balance and annotation quality',
                'Actual source quotas, coefficient and all objective/deep-supervision terms must be frozen for the final study',
                'Batch-dependent model normalization or stochastic state requires separate control',
                'This does not amend the completed historical experiments or running comparison'],
        trained_model_forwards=0,optimizer_updates=0,paid_compute=False,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_pins=[])
    for rel in ['vesuvius/ink_detection/training/losses.py','vesuvius/models/training/loss/nnunet_losses.py']:
        path=root/rel;result['source_pins'].append(dict(path=str(path.relative_to(a.repo)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
