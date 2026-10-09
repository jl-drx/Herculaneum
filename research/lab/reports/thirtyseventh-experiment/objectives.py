"""Original manual smoothed-BCE/KL plus one optionally enabled masked Dice term."""
def dice_loss(z,m,t):
    import torch
    if z.shape!=m.shape or z.shape!=t.shape or m.dtype!=torch.bool or z.ndim!=4:
        raise ValueError('Aligned BCHW manual support required')
    if not bool(m.flatten(1).any(1).all()):raise ValueError('Every patch needs manual support')
    if not bool(((t[m]==0)|(t[m]==1)).all()):raise ValueError('Binary manual targets required')
    p=torch.sigmoid(z);target=torch.where(m,t,torch.zeros_like(t));axes=(2,3)
    # Match upstream non-batch, unsmoothed target Dice: -mean((2I+1)/(P+Y+1)).
    intersection=(p*target*m).sum(axes);prediction=(p*m).sum(axes);truth=target.sum(axes)
    return -((2*intersection+1)/(prediction+truth+1).clamp_min(1e-8)).mean()

def terms(z,q,batch,arm,old,preservation,torch):
    base,values=old.terms(z,q,batch,'A_preserve',preservation,torch)
    dice=dice_loss(z,batch[1],batch[2]);total=base if arm=='A_control' else base+dice
    values.update(dice_loss=float(dice.detach()),full=float(total.detach()),control_objective=float(base.detach()))
    return total,values
