import unittest
import numpy as np
import torch
from common import *
from objective import terms

class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.old=legacy();cls.prior=json.loads((REPORTS/'thirtyseventh-experiment/PLAN.json').read_text());cls.base=json.loads(verify(cls.prior['training_plan']).read_text())
        cls.preservation=module(verify(cls.base['shared_sources']['preservation.py']),'phase38_test_preservation')
    def tiny(self):
        z=torch.tensor([[[[-2.,1.],[.5,2.]]],[[[.7,-.8],[2.,-1.]]]],requires_grad=True)
        m=torch.tensor([[[[1,1],[0,0]]],[[[1,0],[0,1]]]],dtype=torch.bool)
        t=torch.tensor([[[[0.,1.],[-1.,-1.]]],[[[1.,-1.],[-1.,0.]]]])
        q=z.detach()+.3;return z,q,(None,m,t)
    def test_control_loss_and_gradient_exact_all_32_historical_batches(self):
        r=json.loads((REPORTS/'thirtyseventh-experiment/run-v1/A_control/COMPLETE.json').read_text())
        raw=np.load(verify(r['pre_logits']),mmap_mode='r',allow_pickle=False);teacher=np.load(verify(r['teacher']),allow_pickle=False)
        helper=module(verify(self.base['helper']),'phase38_test_helper');runtime=helper.load_runtime();regions=[self.old.training_region(c) for c in self.base['adaptation']]
        for i,step in enumerate(json.loads(verify(self.base['schedule']).read_text())):
            b,_,_=self.old.batch(step,regions,runtime,torch);z=torch.from_numpy(raw[i].copy()).requires_grad_();q=torch.from_numpy(teacher[i].copy())
            a,av=self.old.terms(z,q,b,'A_preserve',self.preservation,torch);v,values=terms(z,q,b[:3],torch)
            self.assertTrue(torch.equal(a,v));torch.testing.assert_close(torch.autograd.grad(a,z)[0],torch.autograd.grad(v,z)[0],rtol=0,atol=0)
            for key in ['full','loss_U','loss_D_M','hard_base','probability_drift_manual']:self.assertEqual(av[key],values[key])
    def test_unknown_gradient_and_teacher_detach(self):
        z,q,b=self.tiny();q=q.requires_grad_();v,_=terms(z,q,b,torch);v.backward();self.assertTrue(bool((z.grad[~b[1]]==0).all()));self.assertIsNone(q.grad)
    def test_unknown_values_do_not_contribute(self):
        z,q,b=self.tiny();a,_=terms(z,q,b,torch);zz=z.detach().clone();zz[~b[1]]=100;tt=b[2].clone();tt[~b[1]]=999;v,_=terms(zz,q,(None,b[1],tt),torch);self.assertTrue(torch.equal(a.detach(),v))
    def test_independent_scalar(self):
        z,q,b=self.tiny();v,values=terms(z,q,b,torch);a=z.detach().numpy().astype(float);t=q.numpy().astype(float);m=b[1].numpy();y=b[2].numpy()[m]
        expected=np.mean(np.logaddexp(0,a[m])-a[m]*(.25+.5*y))+np.mean(np.logaddexp(0,a[m])-np.logaddexp(0,t[m])-(a[m]-t[m])/(1+np.exp(-t[m])))
        self.assertAlmostEqual(float(v.detach()),float(expected),places=6)
    def test_empty_support_rejected(self):
        z,q,b=self.tiny()
        with self.assertRaises(ValueError):terms(z,q,(None,torch.zeros_like(b[1]),b[2]),torch)
    def test_per_update_class_exposure_and_fixed_original_patch(self):
        m=np.load(HERE/'data/mixed-mask.npy',mmap_mode='r');t=np.load(HERE/'data/mixed-target.npy',mmap_mode='r');om=np.load(HERE/'data/original-mask.npy',mmap_mode='r');ot=np.load(HERE/'data/original-target.npy',mmap_mode='r');x=np.load(HERE/'data/mixed-inputs.npy',mmap_mode='r');ox=np.load(HERE/'data/original-inputs.npy',mmap_mode='r')
        for i in range(32):
            for value in [0,1]:self.assertEqual(int((m[i]&(t[i]==value)).sum()),int((om[i]&(ot[i]==value)).sum()))
            keep=1-i%2
            for a,b in [(m,om),(t,ot),(x,ox)]:self.assertTrue(np.array_equal(a[i,keep],b[i,keep]))
    def test_masks_never_promote_unknown_original_targets(self):
        m=np.load(HERE/'data/original-mask.npy');t=np.load(HERE/'data/original-target.npy');self.assertTrue(np.all(t[~m]==-1));self.assertEqual(set(np.unique(t[m])),{0.,1.})
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(StudyTests);r=unittest.TextTestRunner(verbosity=2).run(suite)
    save(HERE/'TEST_RECEIPT.json',dict(tests_run=r.testsRun,passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),new_model_forwards=0,new_optimizer_updates=0))
    raise SystemExit(not r.wasSuccessful())
