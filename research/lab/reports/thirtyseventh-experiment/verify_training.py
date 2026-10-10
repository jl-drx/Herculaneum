"""Independent NumPy loss reconstruction; no model or optimizer replay."""
import math,time
import numpy as np
from scipy.special import expit
from study_common import *

def main():
    started=time.monotonic();p=plan();require(not (HERE/'TRAINING_VERIFICATION.json').exists(),'Write once');maximum={};checks=0;rows_checked=0
    def check(z,q,m,t,row,arm):
        nonlocal checks,rows_checked
        z=z.astype(np.float64);q=q.astype(np.float64);prob=expit(z);teacher=expit(q);s=.25+.5*t[m]
        bce=float((np.logaddexp(0,z[m])-z[m]*s).mean());hard=float((np.logaddexp(0,z[m])-z[m]*t[m]).mean())
        kl=float((np.logaddexp(0,z[m])-np.logaddexp(0,q[m])-teacher[m]*(z[m]-q[m])).mean())
        dice=-math.fsum((2*float(np.sum(prob[i][m[i]]*t[i][m[i]]))+1)/(float(prob[i][m[i]].sum())+float(t[i][m[i]].sum())+1) for i in range(2))/2
        expected=dict(loss_U=bce,base=bce,loss_D_M=kl,hard_base=hard,dice_loss=dice,control_objective=bce+kl,full=bce+kl+(dice if arm=='B_dice' else 0),probability_drift_manual=float(abs(prob[m]-teacher[m]).mean()))
        for k,v in expected.items():
            d=abs(v-row[k]);maximum[k]=max(maximum.get(k,0),d);require(d<=1e-6,'Independent loss '+k);checks+=1
        rows_checked+=1
    receipts={}
    for arm in ARMS:
        r=json.loads((HERE/'run-v1'/arm/'COMPLETE.json').read_text());require(r['status']=='PASS' and r['optimizer_updates']==r['backwards']==512 and r['student_forwards']==673 and r['teacher_forwards']==0,'Exact training counts')
        a=np.load(verify(r['supervision']),allow_pickle=False);m=a['manual_mask'];t=a['manual_target'];teacher=np.load(verify(r['teacher']),mmap_mode='r',allow_pickle=False)
        pre=np.load(verify(r['pre_logits']),mmap_mode='r',allow_pickle=False);fixed=np.load(verify(r['fixed_logits']),mmap_mode='r',allow_pickle=False)
        require(pre.shape==(512,2,1,128,128) and fixed.shape==(5,32,2,1,128,128) and m.dtype==np.bool_,'Raw arrays')
        logs=[json.loads(s) for s in verify(r['steps']).read_text().splitlines()];require(len(logs)==512,'512 step receipts')
        for i,row in enumerate(logs):
            j=i%32;require(row['step']==i+1 and row['schedule_index']==j and row['input_target_sha256']==r['batch_fingerprints'][j] and row['unknown_gradient_max']==0 and math.isfinite(row['gradient_norm']) and row['gradient_norm']>0,'Update schedule/gradient receipts')
            check(pre[i],teacher[j],m[j],t[j],row['pre'],arm)
        for step,entry in r['measurements'].items():
            require(len(entry['rows'])==32,'Fixed risk support')
            for j,row in enumerate(entry['rows']):check(teacher[j] if step=='0' else fixed[UPDATES.index(int(step)),j],teacher[j],m[j],t[j],row,arm)
            for key,val in entry['mean'].items():require(math.fsum(row[key] for row in entry['rows'])/32==val,'Exact equal-batch scalar reduction')
            if step!='0':require(entry['optimizer_step_min']==entry['optimizer_step_max']==int(step),'Optimizer state continuity receipt')
        for ck in r['checkpoints'].values():verify(ck['file'])
        receipts[arm]=pin(HERE/'run-v1'/arm/'COMPLETE.json')
    a=json.loads(Path(receipts['A_control']['path']).read_text());old=json.loads(verify(p['historical_control32']).read_text())
    require(a['checkpoints']['32']['state_sha256']==old['final_state_sha256'] and a['measurements']['32']['historical_update32_exact'],'Historical control reproduction')
    require(time.monotonic()-started<120 and rss()<1024**3,'Verification resource cap')
    receipt=dict(status='PASS',created_at_utc=now(),verifier=pin(__file__),arms=receipts,loss_rows_checked=rows_checked,scalar_values_checked=checks,maximum_differences=maximum,
        optimizer_autograd_replayed=False,checkpoint_tensor_decode_repeated=False,historical_control32_verified_by_worker=True,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss(),new_model_forwards=0)
    save(HERE/'TRAINING_VERIFICATION.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
