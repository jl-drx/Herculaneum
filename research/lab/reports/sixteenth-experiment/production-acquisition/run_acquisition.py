"""Root-only metadata or pixel controller: durable reservations, four isolated requests maximum."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time
sys.dont_write_bytecode=True
from production_common import *
from prepare_plan import load_plan


def single_job(obj,phase,span,pin=None):
    return dict(request_id=f"m-{obj['object_id']}-{phase}",object_id=obj['object_id'],phase=phase,
                url=obj['url'],mode='single',ranges=[span],pin=pin,
                reserved_cap_bytes=span[1]-span[0]+2)


def batch_record(request,terminal,entry):
    parts=entry['parts'];by_range={tuple(p['range_inclusive']):p for p in parts}
    require(len(parts)==request['row_count'] and set(by_range)==set(map(tuple,request['ranges'])),
            'Successful batch parts differ from fixed row schedule')
    rows=[]
    for i,span in enumerate(request['ranges']):
        part=by_range[tuple(span)]
        rows.append(dict(native_y=request['native_row_start']+i,range_inclusive=span,
                         payload_offset=part['payload_offset'],payload_bytes=part['payload_bytes'],
                         payload_sha256=part['payload_sha256']))
    return dict(request_id=request['request_id'],case_id=request['case_id'],fragment=request['fragment'],
                plane_index=request['plane_index'],native_row_start=request['native_row_start'],
                row_count=request['row_count'],native_x_start=request['native_x_start'],
                native_crop_width=request['native_crop_width'],source_url=request['url'],
                etag=terminal['etag'],object_total_bytes=terminal['object_total_bytes'],status_code=206,
                content_type=terminal['content_type'],multipart=terminal['body'],parts=parts,rows=rows)


class Controller:
    def __init__(self,root,plan,plan_record,spawn=None):
        self.root,self.plan,self.stage=Path(root),plan,plan['stage']
        require(not self.root.exists(),'Fresh acquisition directory required')
        self.root.mkdir(parents=True);(self.root/'requests').mkdir();(self.root/'batches').mkdir()
        self.spawn=spawn or subprocess.Popen
        self.active={};self.stopped=False;self.failure=None
        self.started=time.monotonic();self.deadline=self.started+LIMITS[self.stage]['wall_seconds']-LIMITS['cleanup_reserve_seconds']
        self.objects={o['object_id']:o for o in plan['fixed_inputs']['objects']}
        self.states={};self.contracts={c['object_id']:c for c in plan['fixed_inputs']['carried_contracts']}
        self.batches={};self.terminals=[]
        self.logical={r['request_id']:r for r in plan['requests']}
        self.pending=deque(single_job(o,'header',[0,7])for o in plan['fixed_inputs']['objects']if o['plane_index']!=7)if self.stage=='metadata'else deque(plan['requests'])
        self.ledger=dict(schema_version=1,stage=self.stage,plan=plan_record,limits=LIMITS[self.stage],
                         maximum_concurrent_requests=LIMITS['concurrency'],requests=[],charged_response_bytes=0,
                         retained_body_bytes=0,automatic_retries=0,redirects_followed=0,verified_tls=True,
                         accounting='Durably reserve exact bounded cap before child launch. Successful validated HTTP bodies charge actual bytes; failed/unconfirmed requests retain full reservation. Stored partial bytes separately counted; not complete wire traffic.')
        self.persist()

    def persist(self):
        save(self.root/'request-ledger.json',self.ledger)

    def launch(self,request):
        require(not self.stopped and len(self.active)<LIMITS['concurrency'],'Scheduling after stop or concurrency overflow')
        identity=self.logical.get(request['request_id'])
        require(identity is not None and (self.stage=='metadata'and identity['object_id']==request['object_id']
                and identity['phase']==request['phase']or self.stage=='pixels'and identity==request),'Unplanned request')
        cap=request['reserved_cap_bytes'];limits=LIMITS[self.stage]
        require(cap<=limits['per_response']and len(self.ledger['requests'])<limits['attempts']
                and self.ledger['charged_response_bytes']+cap<=limits['body_bytes'],'Attempt/body reservation cap exhausted')
        folder=self.root/'requests'/request['request_id'];folder.mkdir()
        config=dict(schema_version=1,python='3.12.14',stage=self.stage,limits=LIMITS,
                    code=self.plan['code'],request=request,plan=self.ledger['plan'])
        save(folder/'request.json',config);r=record(folder/'request.json')
        entry=dict(request_id=request['request_id'],request=r,url=request['url'],mode=request['mode'],
                   reserved_cap_bytes=cap,charged_response_bytes=cap,retained_body_bytes=0,status='RESERVED',
                   reserved_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        self.ledger['requests'].append(entry);self.ledger['charged_response_bytes']+=cap;self.persist()
        out=(folder/'stdout.log').open('wb');err=(folder/'stderr.log').open('wb')
        try:
            command=[sys.executable,str(ROOT/'fetch_worker.py'),'--request',r['path'],'--sha256',r['sha256']]
            process=self.spawn(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
                               env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            self.active[request['request_id']]=dict(process=process,request=request,entry=entry,folder=folder,
                                                   out=out,err=err,started=time.monotonic())
        except Exception as error:
            out.close();err.close()
            terminal=dict(schema_version=1,status='STOP',request_id=request['request_id'],request=r,
                          failure_type=type(error).__name__,worker_started=False,
                          charged_response_bytes=cap,retained_body_bytes=0,partial_files=[])
            save(folder/'terminal.json',terminal)
            entry.update(status='STOP',terminal=record(folder/'terminal.json'),failure_type=type(error).__name__)
            self.terminals.append(dict(request_id=request['request_id'],terminal=entry['terminal']))
            self.persist();raise

    def settle(self,job,advance=True):
        try:
            return self._settle(job,advance)
        except Exception as error:
            entry,folder=job['entry'],job['folder']
            if entry.get('status')!='PASS':
                # Parent rejection never discounts an unconfirmed request reservation.
                entry.update(status='STOP',failure_type=type(error).__name__)
                failure=dict(schema_version=1,status='STOP',request_id=job['request']['request_id'],
                             failure_type=type(error).__name__,charged_response_bytes=entry['reserved_cap_bytes'])
                if isinstance(error,ContractError):
                    failure['stop_reason']=str(error);entry['stop_reason']=str(error)
                if (folder/'terminal.json').exists():
                    failure['child_terminal']=record(folder/'terminal.json')
                failure['retained_files']=[record(p)for p in folder.rglob('*')
                                           if p.is_file()and(p.name=='body.bin'or p.name.endswith('.partial'))]
                retained=sum(r['bytes']for r in failure['retained_files'])
                self.ledger['retained_body_bytes']+=max(0,retained-entry.get('retained_body_bytes',0))
                entry['retained_body_bytes']=max(retained,entry.get('retained_body_bytes',0))
                save(folder/'parent-stop.json',failure)
                entry['parent_failure']=record(folder/'parent-stop.json')
                self.persist()
            raise

    def _settle(self,job,advance=True):
        process,folder,request,entry=job['process'],job['folder'],job['request'],job['entry']
        job['out'].close();job['err'].close()
        terminal_path=folder/'terminal.json'
        if not terminal_path.exists():
            partials=[record(p)for p in folder.rglob('*.partial')]
            # A terminated worker may have completed a body but not a receipt; keep it unconfirmed.
            completed=[record(p)for p in folder.rglob('body.bin')]
            terminal=dict(schema_version=1,status='STOP',request_id=request['request_id'],request=entry['request'],
                          failure_type='MissingWorkerTerminal',returncode=process.returncode,
                          charged_response_bytes=entry['reserved_cap_bytes'],
                          retained_body_bytes=sum(p['bytes']for p in partials+completed),partial_files=partials,
                          unconfirmed_completed_files=completed)
            save(terminal_path,terminal)
        else:
            terminal=json.loads(terminal_path.read_text())
        tref=record(terminal_path)
        entry.update(terminal=tref,returncode=process.returncode,stdout=record(folder/'stdout.log'),
                     stderr=record(folder/'stderr.log'),elapsed_seconds=time.monotonic()-job['started'])
        self.terminals.append(dict(request_id=request['request_id'],terminal=tref))
        require(type(terminal.get('retained_body_bytes'))is int and 0<=terminal['retained_body_bytes']<=entry['reserved_cap_bytes'],
                'Terminal retained byte count outside reservation')
        entry['retained_body_bytes']=terminal['retained_body_bytes']
        self.ledger['retained_body_bytes']+=entry['retained_body_bytes']
        if process.returncode!=0 or terminal.get('status')!='PASS':
            entry.update(status='STOP',failure_type=terminal.get('failure_type','WorkerFailure'))
            if 'stop_reason'in terminal:
                entry['stop_reason']=terminal['stop_reason']
            self.persist()
            raise ContractError(entry.get('stop_reason','Scheduled HTTP request failed; terminal receipt retained'))
        require(terminal['request_id']==request['request_id']and terminal['request']==entry['request'],
                'Worker terminal request identity changed')
        verify(terminal['body']);transport=json.loads(verify(terminal['transport_ledger']).read_text())
        require(len(transport['requests'])==1 and transport['automatic_retries']==transport['redirects_followed']==0
                and transport['verified_tls']is True,'Worker transport scope changed')
        http=transport['requests'][0]
        require(http['success']is True and http['url']==request['url']and http['ranges_inclusive']==request['ranges']
                and http['mode']==request['mode']and http['http_status']==terminal['status_code']==206
                and http['file']==terminal['body']and http['charged_response_bytes']==terminal['charged_response_bytes']
                and http['retained_body_bytes']==terminal['retained_body_bytes']and
                http['reserved_cap_bytes']==entry['reserved_cap_bytes'],'Terminal transport binding changed')
        if request.get('pin'):
            require(http['etag']==request['pin']['etag']and http['object_total_bytes']==request['pin']['object_total_bytes']
                    and http['request_headers']['If-Match']==request['pin']['etag'],'Audited object validator changed')
        charged=terminal['charged_response_bytes']
        require(0<charged<=entry['reserved_cap_bytes'],'Successful response charge invalid')
        self.ledger['charged_response_bytes']-=entry['reserved_cap_bytes']-charged
        entry.update(status='PASS',charged_response_bytes=charged,transport_ledger=terminal['transport_ledger'])
        self.persist()
        if advance:
            self.complete(request,terminal,http,tref)

    def complete(self,request,terminal,http,tref):
        if self.stage=='pixels':
            batch=batch_record(request,terminal,http);batch['terminal_receipt']=tref
            path=self.root/'batches'/(request['request_id']+'.json');save(path,batch)
            self.batches[request['request_id']]={k:request[k]for k in ('request_id','case_id','fragment','plane_index','native_row_start','row_count')}
            self.batches[request['request_id']]['batch']=record(path)
            return
        obj=self.objects[request['object_id']]
        state=self.states.setdefault(obj['object_id'],{})
        phase=request['phase'];state[phase]=terminal['body']
        raw=verify(terminal['body']).read_bytes()
        pin={'etag':terminal['etag'],'object_total_bytes':terminal['object_total_bytes']}
        if phase=='header':
            layout=tiff_header(raw);offset=layout['ifd_offset']
            require(8<=offset and offset+2<=pin['object_total_bytes'],'IFD offset outside object')
            state.update(layout=layout,pin=pin)
            self.pending.appendleft(single_job(obj,'count',[offset,offset+1],pin))
        elif phase=='count':
            count=struct.unpack(state['layout']['struct_endian']+'H',raw)[0]
            require(1<=count<=128,'IFD count outside frozen adaptive rule')
            offset=state['layout']['ifd_offset'];size=2+12*count+4
            require(offset+size<=pin['object_total_bytes'],'IFD table outside object')
            state['count_value']=count
            self.pending.appendleft(single_job(obj,'ifd',[offset,offset+size-1],pin))
        else:
            require(struct.unpack_from(state['layout']['struct_endian']+'H',raw)[0]==state['count_value'],'IFD count changed')
            contract=parse_tiff(verify(state['header']).read_bytes(),raw,pin['object_total_bytes'],obj['shape_yx'])
            self.contracts[obj['object_id']]=dict(contract,fragment=obj['fragment'],plane_index=obj['plane_index'],
                object_id=obj['object_id'],url=obj['url'],etag=pin['etag'],object_total_bytes=pin['object_total_bytes'],
                prefix_reused=False,raw_metadata={k:state[k]for k in ('header','count','ifd')})
            # Canonical prior naming is ifd_count.
            self.contracts[obj['object_id']]['raw_metadata']['ifd_count']=self.contracts[obj['object_id']]['raw_metadata'].pop('count')

    def cleanup(self):
        self.stopped=True;self.pending.clear()
        # Reap all already-complete children before signalling the remainder.
        for identity,job in list(self.active.items()):
            if job['process'].poll()is not None:
                self.active.pop(identity)
                try:
                    self.settle(job,advance=False)
                except Exception:
                    pass
        for job in self.active.values():
            try:
                os.killpg(job['process'].pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        grace=time.monotonic()+2
        while self.active and time.monotonic()<grace and any(j['process'].poll()is None for j in self.active.values()):
            time.sleep(.02)
        for identity,job in list(self.active.items()):
            process=job['process']
            if process.poll()is None:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=2)
            self.active.pop(identity)
            try:
                self.settle(job,advance=False)
            except Exception:
                pass
        self.persist()

    def run(self):
        try:
            while self.pending or self.active:
                require(time.monotonic()<self.deadline,'Stage wall deadline reached; stop and cleanup')
                # Examine every completed child before scheduling any replacement.
                done=[(k,j)for k,j in self.active.items()if j['process'].poll()is not None]
                for identity,job in done:
                    self.active.pop(identity)
                    try:
                        self.settle(job,advance=self.failure is None)
                    except Exception as error:
                        self.failure=self.failure or error;self.stopped=True
                if self.failure:
                    raise self.failure
                for job in self.active.values():
                    require(time.monotonic()-job['started']<=LIMITS['child_timeout_seconds']+1,
                            'Child completion timeout; stop and cleanup')
                while self.pending and len(self.active)<LIMITS['concurrency']:
                    self.launch(self.pending.popleft())
                if self.active:
                    time.sleep(.02)
            required=450 if self.stage=='metadata'else 1530
            require(len(self.ledger['requests'])==required and all(r['status']=='PASS'for r in self.ledger['requests']),
                    'Complete exact request inventory required')
            if self.stage=='metadata':
                require(len(self.contracts)==153,'All153 TIFF layouts required')
            else:
                require(len(self.batches)==1530,'All1530 batches required')
        finally:
            self.cleanup()


def size_of_outputs(root,plan):
    folders={Path(root).resolve(),Path(plan['plan_path']).resolve().parent}
    if plan['stage']=='pixels':
        folders.add(Path(plan['metadata_result']['path']).resolve().parent)
        folders.add(Path(plan['metadata_plan']['path']).resolve().parent)
    paths={p.resolve()for folder in folders for p in folder.rglob('*')if p.is_file()}
    # Reserve128MiB for six pooled arrays, pooling receipts/plans and source metadata.
    return sum(p.stat().st_size for p in paths)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();started=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
    plan=load_plan(args.plan);root=Path(args.output_dir).resolve()
    production_start=plan.get('production_started_epoch_seconds',time.time())
    require(type(production_start)in (int,float)and 0<production_start<=time.time()+1
            and time.time()-production_start<1500,'Shared M/P/C1500second window exhausted')
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Controller terminated')))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Controller stage wall cap')))
    signal.alarm(LIMITS[plan['stage']]['wall_seconds']-LIMITS['cleanup_reserve_seconds'])
    controller=Controller(root,plan,record(args.plan))
    result=dict(schema_version=1,status='STARTED',stage=plan['stage'],plan=record(args.plan),
                fixed_inputs=plan['fixed_inputs'],raw_schema=plan['fixed_inputs']['references']['schema'],
                numeric_pixel_decodes=0,model_calls=0,maximum_concurrent_requests=LIMITS['concurrency'],
                production_started_epoch_seconds=production_start,production_total_wall_cap_seconds=1500)
    try:
        controller.deadline=min(controller.deadline,time.monotonic()+1500-(time.time()-production_start)-LIMITS['cleanup_reserve_seconds'])
        controller.run();load_plan(args.plan)
        require(time.time()-production_start<=1500,'Shared production wall cap exceeded')
        result['status']='PASS'
    except Exception as error:
        result.update(status='STOP',failure_type=type(error).__name__)
        if isinstance(error,ContractError):
            result['stop_reason']=str(error)
        raise
    finally:
        signal.alarm(0);controller.cleanup()
        ordered=plan['fixed_inputs']['objects']
        result['tiff_contracts']=[controller.contracts[o['object_id']]for o in ordered if o['object_id']in controller.contracts]if plan['stage']=='metadata'else plan['tiff_contracts']
        if plan['stage']=='pixels':
            result['batches']=[controller.batches[r['request_id']]for r in plan['requests']if r['request_id']in controller.batches]
        save(root/'terminal-receipts.json',{'schema_version':1,'terminals':controller.terminals})
        result.update(request_ledger=record(root/'request-ledger.json'),terminal_receipts=record(root/'terminal-receipts.json'),
                      wall_seconds=time.monotonic()-started,http_attempts=len(controller.ledger['requests']),
                      charged_response_bytes=controller.ledger['charged_response_bytes'],
                      retained_body_bytes=controller.ledger['retained_body_bytes'],
                      combined_metadata_pixel_retained_bytes=controller.ledger['retained_body_bytes']+(
                          json.loads(verify(plan['metadata_result']).read_text())['retained_body_bytes']if plan['stage']=='pixels'else 0),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      production_elapsed_wall_seconds=time.time()-production_start)
        save(root/'result.json',result)
        if plan['stage']=='pixels'and result['status']=='PASS':
            save(root/'raw-provenance.json',result)
        check_plan=dict(plan,plan_path=str(Path(args.plan).resolve()))
        actual=size_of_outputs(root,check_plan)
        if actual>LIMITS['output_bytes']-134217728:
            result.update(status='STOP',failure_type='ContractError',stop_reason='Combined M/P outputs exceed cap minus128MiB reserved pooling margin')
            save(root/'result.json',result)
            if (root/'raw-provenance.json').exists():
                save(root/'raw-provenance.json',result)
            raise ContractError(result['stop_reason'])
    print(json.dumps({'status':'PASS','result':record(root/'result.json'),
                      **({'raw_provenance':record(root/'raw-provenance.json')}if plan['stage']=='pixels'else{})}))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'STOP','failure_type':type(error).__name__,
                          **({'stop_reason':str(error)}if isinstance(error,ContractError)else{})}))
        raise SystemExit(1)
