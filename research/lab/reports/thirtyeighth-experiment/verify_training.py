"""Independent float64 arithmetic on all retained training logits; no model calls."""
import time,math
import numpy as np
from scipy.special import expit
from common import *
def main():
    started=time.monotonic();p=plan();r=json.loads((HERE/'run-v1/C_coverage/COMPLETE.json').read_text());base=json.loads(verify(p['training_plan']).read_text());maximum={};rows=checks=0
    def check(z,q,m,t,record):
        nonlocal rows,checks
        z=z.astype(float);q=q.astype(float);prob=expit(z);teacher=expit(q);y=t[m];v=z[m]
        u=float((np.logaddexp(0,v)-v*(.25+.5*y)).mean());kl=float((np.logaddexp(0,v)-np.logaddexp(0,q[m])-teacher[m]*(v-q[m])).mean());hard=float((np.logaddexp(0,v)-v*y).mean())
        dice=-math.fsum((2*float((prob[i][m[i]]*t[i][m[i]]).sum())+1)/(float(prob[i][m[i]].sum())+float(t[i][m[i]].sum())+1) for i in range(2))/2
        expected=dict(full=u+kl,loss_U=u,loss_D_M=kl,hard_base=hard,dice_loss=dice,probability_drift_manual=float(abs(prob[m]-teacher[m]).mean()))
        for key,value in expected.items():
            d=abs(value-record[key]);maximum[key]=max(maximum.get(key,0),d);require(d<=1e-6,'Independent loss '+key);checks+=1
        rows+=1
    require(r['optimizer_updates']==r['backwards']==512 and r['student_forwards']==705,'Exact training budget')
    m=np.load(verify(p['batch_arrays']['mixed']['mask']),allow_pickle=False);t=np.load(verify(p['batch_arrays']['mixed']['target']),allow_pickle=False);q=np.load(verify(r['teacher']),allow_pickle=False)
    pre=np.load(verify(r['pre_logits']),mmap_mode='r',allow_pickle=False);fixed=np.load(verify(r['fixed_logits']),mmap_mode='r',allow_pickle=False);log=[json.loads(s) for s in verify(r['steps']).read_text().splitlines()]
    require(len(log)==512 and pre.shape==(512,2,1,128,128) and fixed.shape==(5,32,2,1,128,128),'Complete raw training outputs')
    for i,entry in enumerate(log):
        j=i%32;require(entry['step']==i+1 and entry['schedule_index']==j and entry['input_target_sha256']==r['batch_fingerprints'][j] and entry['unknown_gradient_max']==0,'Exact schedule/gradient receipt');check(pre[i],q[j],m[j],t[j],entry['pre'])
    for step,entry in r['measurements'].items():
        for j,row in enumerate(entry['rows']):check(q[j] if step=='0' else fixed[UPDATES.index(int(step)),j],q[j],m[j],t[j],row)
        for key,value in entry['mean'].items():require(math.fsum(row[key] for row in entry['rows'])/32==value,'Equal-batch risk mean')
        if step!='0':require(entry['optimizer_step_min']==entry['optimizer_step_max']==int(step),'Optimizer continuity receipt')
    om=np.load(verify(p['batch_arrays']['original']['mask']));ot=np.load(verify(p['batch_arrays']['original']['target']));oq=np.load(verify(base['historical']['A_preserve']['teacher']));final=np.load(verify(r['original_final_logits']))
    for name,array in [('original_baseline',oq),('original_final',final)]:
        entry=r[name]
        for j,row in enumerate(entry['rows']):check(array[j],oq[j],om[j],ot[j],row)
        for key,value in entry['mean'].items():require(math.fsum(row[key] for row in entry['rows'])/32==value,'Original-support mean')
    control=json.loads(verify(p['control_training']).read_text())
    for key,value in r['original_baseline']['mean'].items():require(value==control['measurements']['0']['mean'][key],'Common original baseline exact')
    for c in r['checkpoints'].values():verify(c['file'])
    require(time.monotonic()-started<120 and rss()<1024**3,'Verification bounds')
    save(HERE/'TRAINING_VERIFICATION.json',dict(status='PASS',created_at_utc=now(),loss_rows_checked=rows,scalar_values_checked=checks,maximum_differences=maximum,optimizer_autograd_replayed=False,new_model_forwards=0,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss(),verifier=pin(__file__)))
    print(json.dumps(dict(status='PASS',loss_rows=rows,scalar_values=checks,maximum_differences=maximum)),flush=True)
if __name__=='__main__':main()
