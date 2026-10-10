"""Focused synthetic checks for changed allocation semantics; no actual data/model."""
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'allocation-implementation'
sys.path.insert(0,str(SOURCE))
import torch
import numpy as np
from depth_adapter import manual_terms
from depth_data import load_support_array


def pin(path):
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checks=[]
    def check(name,value):
        assert bool(value),name
        checks.append(name)
    def rejects(name,callback):
        try:callback()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    shape=(1,1,3,4)
    z0=torch.tensor([-.7,.2,1.1,-1.3,.4,.9,-.2,1.7,-.8,2.1,.6,-1.4],dtype=torch.float32).reshape(shape)
    t0=torch.tensor([.3,-.5,.8,-.4,.6,-.3,.9,1.4,-.1,1.7,-.8,.2],dtype=torch.float32).reshape(shape)
    manual=torch.zeros(shape,dtype=torch.bool);manual.flatten()[:2]=True
    retired=torch.zeros(shape,dtype=torch.bool);retired.flatten()[2:9]=True
    hard=torch.full(shape,float('nan'),dtype=torch.float32);hard.flatten()[:2]=torch.tensor([0.,1.])
    rows={}
    for arm,cm,cv in [('manual_double',2.,0.),('manual_retired',1.,1.)]:
        z=z0.clone().requires_grad_();teacher=t0.clone().requires_grad_()
        loss,terms=manual_terms(z,teacher,manual,hard,retired,arm)
        loss.backward()
        expected=torch.zeros_like(z)
        a=z.detach()[manual].double();b=teacher.detach()[manual].double()
        expected[manual]=((a.sigmoid()-(.25+.5*hard[manual].double())+cm*(a.sigmoid()-b.sigmoid()))/int(manual.sum())).float()
        a=z.detach()[retired].double();b=teacher.detach()[retired].double()
        expected[retired]=(cv*(a.sigmoid()-b.sigmoid())/int(retired.sum())).float()
        check(arm+': independently derived group-normalized derivative',torch.allclose(z.grad,expected,rtol=0,atol=3e-8))
        check(arm+': zero unknown-output derivative',bool((z.grad[~(manual|retired)]==0).all()))
        check(arm+': detached teacher',teacher.grad is None)
        check(arm+': own full objective formula',terms['full']==terms['objective_A' if arm=='manual_double' else 'objective_B'])
        check(arm+': both common formulas retained',abs(terms['objective_A']-terms['base']-2*terms['kl_manual'])<1e-12 and abs(terms['objective_B']-terms['base']-terms['kl_manual']-terms['kl_retired'])<1e-12)
        check(arm+': retired hard labels are not required',torch.isfinite(loss))
        if arm=='manual_double':check('manual_double: retired output gradients exactly zero',bool((z.grad[retired]==0).all()))
        else:check('manual_retired: active retired teacher gradient',bool((z.grad[retired]!=0).any()))
        poison=hard.clone();poison[~manual]=torch.arange(int((~manual).sum()),dtype=torch.float32)*999
        same,same_terms=manual_terms(z0,t0,manual,poison,retired,arm)
        check(arm+': changing nonmanual label bytes does not affect objective',float(same)==float(loss.detach()) and same_terms==terms)
        rows[arm]=terms
    check('common diagnostics agree for common inputs',all(rows['manual_double'][k]==rows['manual_retired'][k] for k in rows['manual_double'] if k!='full'))
    union=(2*rows['manual_double']['kl_manual']+7*rows['manual_double']['kl_retired'])/9
    check('group KL means are distinct from a union mean',abs((rows['manual_double']['kl_manual']+rows['manual_double']['kl_retired'])/2-union)>1e-5)
    for arm in ('manual_double','manual_retired'):
        z=t0.clone().requires_grad_()
        _,terms=manual_terms(z,t0,manual,hard,retired,arm)
        check(arm+': initial KL values exactly zero',terms['kl_manual']==terms['kl_retired']==0)
    empty=torch.zeros_like(manual)
    rejects('empty manual mask rejected',lambda:manual_terms(z0,t0,empty,hard,retired,'manual_double'))
    rejects('empty retired mask rejected even in manual_double',lambda:manual_terms(z0,t0,manual,hard,empty,'manual_double'))
    rejects('overlapping roles rejected',lambda:manual_terms(z0,t0,manual,hard,manual|retired,'manual_retired'))
    rejects('unknown arm rejected',lambda:manual_terms(z0,t0,manual,hard,retired,'explicit_depth'))
    rejects('nonboolean role rejected',lambda:manual_terms(z0,t0,manual.float(),hard,retired,'manual_double'))
    rejects('non-FP32 execution rejected',lambda:manual_terms(z0.double(),t0.double(),manual,hard,retired,'manual_double'))
    bad=hard.clone();bad[manual]=2
    rejects('nonbinary manual target rejected',lambda:manual_terms(z0,t0,manual,bad,retired,'manual_retired'))
    badz=z0.clone();badz[retired]=float('nan')
    rejects('nonfinite measured retired logits rejected',lambda:manual_terms(badz,t0,manual,hard,retired,'manual_double'))
    synthetic=manual.numpy().copy()
    array_path=HERE/'synthetic-support.npy'
    assert not array_path.exists()
    np.save(array_path,synthetic,allow_pickle=False)
    descriptor=dict(pin(array_path),shape=list(synthetic.shape),dtype=str(synthetic.dtype),
                    raw_sha256=hashlib.sha256(synthetic.tobytes()).hexdigest())
    loaded=load_support_array(descriptor,synthetic.shape,np.bool_)
    check('extended support descriptor loads with strict canonical pin',np.array_equal(loaded,synthetic))
    rejects('support descriptor raw-hash corruption rejected',lambda:load_support_array(dict(descriptor,raw_sha256='0'*64),synthetic.shape,np.bool_))
    rejects('support descriptor shape corruption rejected',lambda:load_support_array(dict(descriptor,shape=[12]),synthetic.shape,np.bool_))
    rejects('support descriptor dtype corruption rejected',lambda:load_support_array(dict(descriptor,dtype='int32'),synthetic.shape,np.bool_))
    rejects('support file-hash corruption rejected',lambda:load_support_array(dict(descriptor,sha256='0'*64),synthetic.shape,np.bool_))
    fusion=SOURCE/'fusion_adapter.py'
    check('architecture source unchanged from reviewed phase20',pin(fusion)['sha256']=='a7004b8d06f163b8a20bfad87cd00dfb3d015d36c6be0833362255ee68a53d24')
    for filename,function in [('depth_data.py','features'),('frozen_backbone.py','FrozenFusionModel')]:
        tree=ast.parse((SOURCE/filename).read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id==function]
        check(filename+': all feature paths fixed to explicit_depth',bool(calls) and all(isinstance(node.args[-1],ast.Constant) and node.args[-1].value=='explicit_depth' for node in calls))
    receipt=dict(status='PASS',checks=checks,check_count=len(checks),
                 source_pins=[pin(SOURCE/n) for n in ('depth_adapter.py','depth_data.py','frozen_backbone.py','fusion_adapter.py','depth_common.py')],
                 fixture_source=pin(Path(__file__).resolve()),scientific_array_reads=0,actual_model_forwards=0,
                 synthetic_support_file=pin(array_path),
                 checkpoint_reads=0,optimizer_steps=0,network_calls=0,
                 scope='Focused finite synthetic objective and feature-wiring controls, not actual-data numerical verification.')
    with (HERE/'allocation-controls-v1.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='PASS',check_count=len(checks))))


if __name__=='__main__':main()
