import unittest
import numpy as np
import torch
from common import *
from objective import terms

class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.prior=json.loads((REPORTS/'fortieth-experiment/PLAN.json').read_text())
        cls.full=module(REPORTS/'fortieth-experiment/objective.py','phase40_frozen_objective')
        cls.teacher=json.loads((REPORTS/'fortieth-experiment/run-v1/teacher/COMPLETE.json').read_text())
        cls.q=np.load(verify(cls.teacher['logits']))
        cls.m=np.load(verify(cls.prior['batch_arrays']['mixed']['mask']))
        cls.t=np.load(verify(cls.prior['batch_arrays']['mixed']['target']))
        cls.z=np.load(REPORTS/'fortieth-experiment/run-v1/D_replication/preupdate-logits.npy',mmap_mode='r')
    def values(self,i=0):
        return torch.from_numpy(self.z[i].copy()).requires_grad_(),torch.from_numpy(self.q[i].copy()),(None,torch.from_numpy(self.m[i].copy()),torch.from_numpy(self.t[i].copy()))
    def test_32_full_objective_parity_and_gradient_decomposition(self):
        for i in range(32):
            z,q,b=self.values(i);full,fv=self.full.terms(z,q,b,torch);control,cv=terms(z,q,b,torch,i%2)
            for k in fv:self.assertEqual(fv[k],cv[k])
            drop=b[1].clone();drop[1-i%2]=False
            removed=torch.nn.functional.binary_cross_entropy_with_logits(z[drop],.25+.5*b[2][drop],reduction='sum')/b[1].sum()
            fg=torch.autograd.grad(full,z,retain_graph=True)[0];cg=torch.autograd.grad(control,z,retain_graph=True)[0];rg=torch.autograd.grad(removed,z)[0]
            torch.testing.assert_close(fg-cg,rg,rtol=1e-6,atol=2e-11)
            torch.testing.assert_close(fg[1-i%2],cg[1-i%2],rtol=0,atol=0)
            self.assertAlmostEqual(float(full.detach()-control.detach()),float(removed.detach()),places=6)
    def test_original_diagnostic_parity(self):
        z,q,b=self.values();a,av=self.full.terms(z,q,b,torch);v,values=terms(z,q,b,torch)
        self.assertTrue(torch.equal(a,v));torch.testing.assert_close(torch.autograd.grad(a,z)[0],torch.autograd.grad(v,z)[0],rtol=0,atol=0)
        for key in av:self.assertEqual(av[key],values[key])
    def test_new_labels_no_optimized_effect_at_fixed_support(self):
        z,q,b=self.values(3);a,_=terms(z,q,b,torch,1);tt=b[2].clone();tt[1][b[1][1]]=1-tt[1][b[1][1]];v,_=terms(z,q,(None,b[1],tt),torch,1)
        self.assertTrue(torch.equal(a,v));torch.testing.assert_close(torch.autograd.grad(a,z)[0],torch.autograd.grad(v,z)[0],rtol=0,atol=0)
    def test_joint_teacher_KL_retained_on_new_slot(self):
        z,q,b=self.values(1);q=(q+.4).requires_grad_();v,_=terms(z,q,b,torch,1);g=torch.autograd.grad(v,z)[0]
        expected=(torch.sigmoid(z.double())-torch.sigmoid(q.detach().double()))/b[1].sum()
        torch.testing.assert_close(g[1][b[1][1]],expected.float()[1][b[1][1]],rtol=0,atol=0)
        self.assertTrue(bool((g[1][b[1][1]]!=0).any()))
    def test_unknown_gradient_and_teacher_detach(self):
        z,q,b=self.values();q.requires_grad_();v,_=terms(z,q,b,torch,0);v.backward();self.assertTrue(bool((z.grad[~b[1]]==0).all()));self.assertIsNone(q.grad)
    def test_original_scaling_uses_joint_denominator(self):
        z,q,b=self.values();q=z.detach();v,values=terms(z,q,b,torch,0);keep=b[1].clone();keep[0]=False
        expected=torch.nn.functional.binary_cross_entropy_with_logits(z[keep],.25+.5*b[2][keep],reduction='sum')/b[1].sum()
        self.assertEqual(values['optimized'],float(expected.detach()));self.assertLess(values['optimized'],values['full'])
        # The unchanged FP64 KL expression has cancellation roundoff at q=z.
        torch.testing.assert_close(torch.autograd.grad(v,z)[0],torch.autograd.grad(expected,z)[0],rtol=2e-7,atol=1e-18)
    def test_empty_support_rejected(self):
        z,q,b=self.values()
        with self.assertRaises(ValueError):terms(z,q,(None,torch.zeros_like(b[1]),b[2]),torch,0)
    def test_schedule_inputs_and_retained_teacher_exact(self):
        schedule=json.loads(verify(self.prior['schedule40']).read_text())['steps']
        original={k:np.load(verify(v),mmap_mode='r') for k,v in self.prior['batch_arrays']['original'].items()}
        mixed={k:np.load(verify(v),mmap_mode='r') for k,v in self.prior['batch_arrays']['mixed'].items()}
        base=json.loads(verify(self.prior['training_plan']).read_text());oldq=np.load(verify(base['historical']['A_preserve']['teacher']))
        for i,s in enumerate(schedule):
            self.assertEqual(s['replaced_slot'],i%2);keep=1-i%2
            for k in original:self.assertTrue(np.array_equal(original[k][i,keep],mixed[k][i,keep]))
            self.assertTrue(np.array_equal(oldq[i,keep],self.q[i,keep]))
            for cls in [0,1]:self.assertEqual(int((self.m[i]&(self.t[i]==cls)).sum()),int((original['mask'][i]&(original['target'][i]==cls)).sum()))
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StudyTests))
    save(HERE/'TEST_RECEIPT.json',dict(tests_run=r.testsRun,passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),new_model_forwards=0,new_optimizer_updates=0))
    raise SystemExit(not r.wasSuccessful())
