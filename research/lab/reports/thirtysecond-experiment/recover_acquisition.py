"""Explicit second attempt: verified cache reuse and four bounded HTTP workers."""
import argparse,os,signal,subprocess,time
from common import *
from transport import Client

def worker(index):
    check_freeze('RECOVERY_GO.json')
    plan=json.loads((HERE/'PLAN.json').read_text());recovery=json.loads((HERE/'RECOVERY_PLAN.json').read_text())
    job=recovery['missing'][index];root=HERE/'recovery-v2'/job['name']
    client=Client(root,plan)
    path,entry=client.get(job['url'],job['ranges'],job['name']+'.bin',job['identity'])
    save(root/'COMPLETE.json',dict(status='PASS',job=job,entry=entry,ledger=pin(root/'LEDGER.json')))

def parent():
    frozen=check_freeze('RECOVERY_GO.json');plan=json.loads((HERE/'PLAN.json').read_text())
    recovery=json.loads((HERE/'RECOVERY_PLAN.json').read_text())
    root=HERE/'recovery-v2';root.mkdir(exist_ok=False)
    started=time.monotonic();active={};done=[];launched=[];pending=list(enumerate(recovery['missing']))
    reserved=recovery['prior_reserved_body_bytes'];total_requests=recovery['prior_requests'];peak_aggregate=0
    receipt=dict(status='RUNNING',created_at_utc=now(),GO=pin(HERE/'RECOVERY_GO.json'),jobs_launched=[],jobs_completed=[])
    save(root/'RUN.json',receipt)
    try:
        while pending or active:
            require(time.monotonic()-started<600,'Second attempt600second deadline')
            while pending and len(active)<4:
                index,job=pending.pop(0)
                cap=sum(b-a+1 for a,b in job['ranges'])+768*256+1024+1
                require(reserved+cap<=192*1024**2 and total_requests+1<=800,'Cumulative both-attempt budget')
                reserved+=cap;total_requests+=1
                logpath=root/(job['name']+'.log');log=logpath.open('xb')
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                child=subprocess.Popen([sys.executable,'-B',str(HERE/'recover_acquisition.py'),'--worker',str(index)],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[index]=dict(child=child,log=log,job=job,started=time.monotonic())
                launched.append(job['name']);receipt['jobs_launched']=launched.copy()
                receipt['reserved_body_bytes_both_attempts']=reserved;receipt['requests_both_attempts']=total_requests
                save(root/'RUN.json',receipt)
            aggregate=0
            for index,state in list(active.items()):
                child=state['child'];code=child.poll()
                if code is None:
                    require(time.monotonic()-state['started']<45,'Worker45second deadline')
                    try:
                        fields=Path(f'/proc/{child.pid}/status').read_text().splitlines()
                        aggregate+=max((int(line.split()[1])*1024 for line in fields if line.startswith(('VmRSS:','VmHWM:'))),default=0)
                    except FileNotFoundError:pass
                    continue
                state['log'].close();del active[index]
                require(code==0,'Recovery HTTP worker failed; no further retry')
                item=json.loads((root/state['job']['name']/'COMPLETE.json').read_text())
                require(item['status']=='PASS' and item['job']==state['job'],'Worker job/completion changed')
                verify(item['entry']['file']);done.append(item)
                receipt['jobs_completed']=[v['job']['name'] for v in done]
            peak_aggregate=max(peak_aggregate,aggregate)
            require(peak_aggregate<=2*1024**3,'Concurrent HTTP worker RSS cap')
            require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<1024**3,'Phase disk cap')
            save(root/'RUN.json',receipt)
            if len(done)%12==0 and done:print(json.dumps(dict(recovery_complete=len(done),remaining=len(pending)+len(active))),flush=True)
            time.sleep(.1)
        require(len(done)==len(recovery['missing']),'Missing recovery response')
        # A source ledger contains exactly one successful body per scientific range batch.
        # The original failed attempt and all charged attempts remain separately pinned.
        prior=json.loads(verify(recovery['prior_ledger']).read_text())
        sources=[e for e in prior['requests'] if e['success']]
        sources += [v['entry'] for v in done]
        source_root=HERE/'pixels-v2';source_root.mkdir(exist_ok=False)
        save(source_root/'SOURCE_LEDGER.json',dict(requests=sources,prior_all_attempts=recovery['prior_ledger'],
             recovery_run=pin(root/'RUN.json'),failed_prior_attempts=[e for e in prior['requests'] if not e['success']]))
        import numpy as np
        out=HERE/'raw-v2';out.mkdir(exist_ok=False)
        arrays={c['id']:np.lib.format.open_memmap(out/(c['id']+'.npy'),mode='w+',dtype='<u2',shape=(63,384,384)) for c in plan['contexts']}
        objects=json.loads((HERE/'metadata-v1/OBJECTS.json').read_text())['objects'];byurl={o['url']:o for o in objects}
        seen=np.zeros((63,4,384),bool);bodies=0
        for entry in sources:
            if len(entry['ranges'])==1:continue
            obj=byurl[entry['url']];body=verify(entry['file']).read_bytes()
            parts=parse_multipart(body,entry['response_headers']['Content-Type'],entry['ranges'],obj['identity']['total_bytes'])
            require(parts==entry['parts'],'Multipart cache changed');bodies+=1
            for part in parts:
                start=part['range_inclusive'][0];y,x=divmod((start-obj['layout']['strip_offset'])//2,6205)
                choices=[(i,c) for i,c in enumerate(plan['contexts']) if c['source_bbox_yxhw'][0]<=y<c['source_bbox_yxhw'][0]+384 and x==c['source_bbox_yxhw'][1]]
                require(len(choices)==1,'Physical crop membership')
                ci,c=choices[0];rr=y-c['source_bbox_yxhw'][0];z=obj['layer']-1
                require(not seen[z,ci,rr] and part['payload_bytes']==768,'Duplicate/wrong crop row');seen[z,ci,rr]=True
                arrays[c['id']][z,rr]=np.frombuffer(body,dtype=obj['layout']['dtype'],offset=part['payload_offset'],count=384)
        require(bodies==378 and seen.all(),'Incomplete source support')
        for a in arrays.values():a.flush()
        elapsed=time.monotonic()-started;require(elapsed<600,'Second attempt total deadline')
        returned=prior['returned_body_bytes']+sum(v['entry']['returned_body_bytes'] for v in done)
        receipt.update(status='PASS',completed_utc=now(),elapsed_seconds=elapsed,observed_peak_worker_RSS_bytes=peak_aggregate,
                       returned_body_bytes_both_attempts=returned,prior_acquisition_seconds=recovery['prior_acquisition_seconds'],
                       cumulative_active_seconds=recovery['prior_acquisition_seconds']+elapsed)
        save(root/'RUN.json',receipt)
        # Rebind the successful final run identity after its terminal status is saved.
        source_ledger=json.loads((source_root/'SOURCE_LEDGER.json').read_text());source_ledger['recovery_run']=pin(root/'RUN.json');save(source_root/'SOURCE_LEDGER.json',source_ledger)
        save(source_root/'COMPLETE.json',dict(status='PASS',attempt='explicit_recovery_v2',original_attempt_status='STOP_TIME_LIMIT',
            raw_arrays={k:pin(out/(k+'.npy')) for k in arrays},rows_verified=int(seen.sum()),useful_CT_bytes=74317824,
            requests=total_requests,returned_body_bytes=returned,reserved_body_bytes=reserved,
            active_acquisition_seconds=receipt['cumulative_active_seconds'],ledger=pin(source_root/'SOURCE_LEDGER.json'),
            original_all_attempts=recovery['prior_ledger'],recovery_run=pin(root/'RUN.json'),GO=pin(HERE/'RECOVERY_GO.json')))
        print(json.dumps(receipt,indent=2))
    except Exception as e:
        receipt.update(status='STOP',error=type(e).__name__+': '+str(e),elapsed_seconds=time.monotonic()-started)
        save(root/'RUN.json',receipt)
        raise
    finally:
        for state in active.values():
            child=state['child']
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            state['log'].close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);a=parser.parse_args()
    parent() if a.worker is None else worker(a.worker)
