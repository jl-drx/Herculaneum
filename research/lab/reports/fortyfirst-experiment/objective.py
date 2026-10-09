"""Remove only new-crop BCE at the fixed joint support denominator."""
def terms(z,q,b,torch,new_slot=None):
    _,m,t=b
    if z.shape!=q.shape or z.shape!=m.shape or m.dtype!=torch.bool or z.shape!=t.shape:
        raise ValueError('Aligned logits and explicit support required')
    if not bool(m.any()) or not bool(((t[m]==0)|(t[m]==1)).all()):
        raise ValueError('Nonempty binary support required')
    if new_slot not in (None,0,1):raise ValueError('Explicit replacement slot')
    u=torch.nn.functional.binary_cross_entropy_with_logits(z[m],.25+.5*t[m])
    if new_slot is None:
        retained=u;removed=u.new_zeros(())
    else:
        keep=m.clone();keep[new_slot]=False;drop=m.clone();drop[1-new_slot]=False
        retained=torch.nn.functional.binary_cross_entropy_with_logits(z[keep],.25+.5*t[keep],reduction='sum')/m.sum()
        removed=torch.nn.functional.binary_cross_entropy_with_logits(z[drop],.25+.5*t[drop],reduction='sum')/m.sum()
        # Mean-reduction backward differs by an FP32 ULP from sum()/N.
        # Keep its exact old-slot gradient, while reporting the direct retained sum/N.
        masked_logits=torch.where(drop,z.detach(),z)
        mean_path=torch.nn.functional.binary_cross_entropy_with_logits(masked_logits[m],.25+.5*t[m])
        retained=retained.detach()+(mean_path-mean_path.detach())
    d=z.double();teacher=q.detach().double();p=torch.sigmoid(teacher)
    divergence=torch.logaddexp(d,torch.zeros_like(d))-torch.logaddexp(teacher,torch.zeros_like(teacher))-p*(d-teacher)
    kl=divergence[m].mean();optimized=retained+kl
    with torch.no_grad():
        hard=torch.nn.functional.binary_cross_entropy_with_logits(z[m],t[m])
        probability=torch.sigmoid(z);safe=torch.where(m,t,torch.zeros_like(t))
        dice=-((2*(probability*safe*m).sum((2,3))+1)/((probability*m).sum((2,3))+safe.sum((2,3))+1)).mean()
        values=dict(full=float(u+kl),optimized=float(optimized),loss_U=float(u),loss_U_original=float(retained),loss_U_new=float(removed),loss_D_M=float(kl),hard_base=float(hard),dice_loss=float(dice),probability_drift_manual=float((torch.sigmoid(d)-p).abs()[m].mean()))
    return optimized,values
