"""Same manual smoothed BCE + detached seed42 KL, without pseudo diagnostics."""
def terms(z,q,b,torch):
    _,m,t=b
    if z.shape!=q.shape or z.shape!=m.shape or z.shape!=t.shape or m.dtype!=torch.bool:
        raise ValueError('Aligned logits and manual support required')
    if not bool(m.any()) or not bool(((t[m]==0)|(t[m]==1)).all()):
        raise ValueError('Nonempty explicit binary supervision required')
    u=torch.nn.functional.binary_cross_entropy_with_logits(z[m],.25+.5*t[m])
    d=z.double();teacher=q.detach().double();p=torch.sigmoid(teacher)
    divergence=torch.logaddexp(d,torch.zeros_like(d))-torch.logaddexp(teacher,torch.zeros_like(teacher))-p*(d-teacher)
    kl=divergence[m].mean();total=u+kl
    with torch.no_grad():
        hard=torch.nn.functional.binary_cross_entropy_with_logits(z[m],t[m])
        probability=torch.sigmoid(z);safe=torch.where(m,t,torch.zeros_like(t))
        dice=-((2*(probability*safe*m).sum((2,3))+1)/((probability*m).sum((2,3))+safe.sum((2,3))+1)).mean()
        values=dict(full=float(total),loss_U=float(u),loss_D_M=float(kl),hard_base=float(hard),dice_loss=float(dice),probability_drift_manual=float((torch.sigmoid(d)-p).abs()[m].mean()))
    return total,values
