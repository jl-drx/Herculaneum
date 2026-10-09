import unittest,importlib.util
import numpy as np
import torch
from study_common import REPORTS,legacy,module
from objectives import dice_loss,terms

class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.z=torch.tensor([[[[-2.,.5,1.],[2.,-1.,0.]]],[[[.3,-.5,3.],[-3.,1.,2.]]]],dtype=torch.float64,requires_grad=True)
        self.m=torch.tensor([[[[1,1,0],[1,0,1]]],[[[1,0,1],[0,1,1]]]],dtype=torch.bool)
        self.t=torch.tensor([[[[0.,1.,-1.],[1.,-1.,0.]]],[[[1.,-1.,0.],[-1.,0.,1.]]]],dtype=torch.float64)
    def scalar(self,z):
        p=1/(1+np.exp(-z));m=self.m.numpy();t=self.t.numpy();return -sum((2*sum(p[i][m[i]]*t[i][m[i]])+1)/(sum(p[i][m[i]])+sum(t[i][m[i]])+1) for i in range(2))/2
    def test_independent_scalar_dice(self):self.assertAlmostEqual(float(dice_loss(self.z,self.m,self.t).detach()),self.scalar(self.z.detach().numpy()),places=14)
    def test_finite_difference_and_unknown_gradient(self):
        value=dice_loss(self.z,self.m,self.t);value.backward();a=self.z.detach().numpy().copy()
        for idx in np.ndindex(a.shape):
            plus=a.copy();minus=a.copy();plus[idx]+=1e-5;minus[idx]-=1e-5
            self.assertAlmostEqual(self.z.grad[idx].item(),(self.scalar(plus)-self.scalar(minus))/2e-5,places=9)
        self.assertTrue(bool((self.z.grad[~self.m]==0).all()))
    def test_masked_values_have_no_effect(self):
        z=self.z.detach().clone();t=self.t.clone();z[~self.m]=100;t[~self.m]=99
        self.assertEqual(float(dice_loss(z,self.m,t)),float(dice_loss(self.z,self.m,self.t).detach()))
    def test_patch_mean_not_pooled_dice(self):
        expected=(dice_loss(self.z[:1],self.m[:1],self.t[:1])+dice_loss(self.z[1:],self.m[1:],self.t[1:]))/2
        self.assertEqual(float(expected.detach()),float(dice_loss(self.z,self.m,self.t).detach()))
    def test_empty_patch_rejected(self):
        m=self.m.clone();m[1]=False
        with self.assertRaises(ValueError):dice_loss(self.z,m,self.t)
    def test_upstream_value_and_gradient_parity(self):
        u=module(REPORTS/'ninth-experiment/source-audit/upstream-nnunet-losses.py','phase37_upstream_dice')
        original=u.MemoryEfficientSoftDiceLoss(apply_nonlin=torch.sigmoid,batch_dice=False,do_bg=True,smooth=1.,ddp=False,label_smoothing=0.)
        safe=torch.where(self.m,self.t,torch.zeros_like(self.t));a=original(self.z,safe,loss_mask=self.m);b=dice_loss(self.z,self.m,self.t)
        self.assertEqual(a.item(),b.item());torch.testing.assert_close(torch.autograd.grad(a,self.z)[0],torch.autograd.grad(b,self.z)[0],rtol=0,atol=0)
    def test_control_identity_and_manual_only_full_gradient(self):
        old=legacy();p=module(REPORTS/'tenth-experiment/implementation/preservation.py','phase37_test_preservation')
        z=self.z.detach().float().requires_grad_();m=self.m;t=self.t.float();other=~m;pt=torch.where(other,torch.zeros_like(t),-torch.ones_like(t));batch=(None,m,t,other,pt);q=z.detach()+.25
        expected,_=old.terms(z,q,batch,'A_preserve',p,torch)
        a,_=terms(z,q,batch,'A_control',old,p,torch);self.assertTrue(torch.equal(a,expected))
        for arm in ['A_control','B_dice']:
            loss,_=terms(z,q,batch,arm,old,p,torch);g=torch.autograd.grad(loss,z,retain_graph=True)[0];self.assertTrue(bool((g[other]==0).all()))
if __name__=='__main__':unittest.main()
