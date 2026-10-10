"""Source-only controller fixtures using fake processes, never real subprocesses or HTTP."""
from pathlib import Path
import hashlib
import json
import socket
import sys
import tempfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
IMPL=HERE.parent.parent/'production-acquisition'
sys.path.insert(0,str(IMPL))
def blocked(*a,**k):raise AssertionError('Real operation forbidden in synthetic review')
socket.socket.connect=socket.socket.connect_ex=socket.create_connection=blocked
import production_common as c
import run_acquisition as r
r.subprocess.Popen=blocked
CASES=[]
def reject(f):
 try:f()
 except (ValueError,OSError):return
 raise AssertionError('Expected rejection')
def plan(n):
 requests=[dict(request_id=f'p-synthetic-{i}',url='https://dl.ash2txt.org/synthetic/07.tif',mode='multipart',reserved_cap_bytes=400000)for i in range(n)]
 return dict(stage='pixels',code=[],requests=requests,fixed_inputs=dict(objects=[],carried_contracts=[]))
class Process:
 returncode=0
 pid=999999999
 def poll(self):return self.returncode
 def wait(self,timeout=None):return self.returncode

def main():
 out=HERE/'scheduler-fixture-result.json';assert not out.exists()
 with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic-')as tmp:
  tmp=Path(tmp)
  # Parent reservation must exist on disk before the fake Popen sees the request.
  launched=[]
  def spawn(command,**kwargs):
   request=Path(command[command.index('--request')+1]);root=request.parents[2]
   ledger=json.loads((root/'request-ledger.json').read_text())
   config=json.loads(request.read_text());identity=config['request']['request_id']
   match=[e for e in ledger['requests']if e['request_id']==identity]
   assert len(match)==1 and match[0]['status']=='RESERVED'and match[0]['charged_response_bytes']==400000
   assert kwargs['start_new_session']is True and kwargs['stdout'].name.endswith('stdout.log')
   launched.append(identity)
   # A successful exit with a terminal bound to the wrong deterministic request.
   c.save(request.parent/'terminal.json',dict(status='PASS',request_id='wrong',request=c.record(request),retained_body_bytes=0))
   return Process()
  controller=r.Controller(tmp/'failure',plan(7),{},spawn=spawn)
  reject(controller.run)
  assert len(launched)==4 and not controller.active and not controller.pending
  assert len(controller.ledger['requests'])==4 and all(e['status']=='STOP'for e in controller.ledger['requests'])
  assert controller.ledger['charged_response_bytes']==4*400000
  assert len(controller.terminals)==4
  assert all(Path(e['parent_failure']['path']).exists()for e in controller.ledger['requests'])
  CASES.extend(['reservation-before-launch','at-most-four-before-failure','drain-all-completed-before-replacement','no-queued-work-after-failure','bad-binding-terminal-becomes-parent-STOP','full-reservation-kept-for-unconfirmed-results','all-child-terminal-evidence-preserved'])
  spawned=[]
  def fail_spawn(*args,**kwargs):spawned.append(1);raise OSError('Synthetic launch failure')
  failed=r.Controller(tmp/'spawn',plan(2),{},spawn=fail_spawn)
  reject(failed.run)
  assert len(spawned)==1 and len(failed.ledger['requests'])==1
  assert failed.ledger['requests'][0]['status']=='STOP'and failed.ledger['charged_response_bytes']==400000
  assert len(failed.terminals)==1 and not failed.active and not failed.pending
  CASES.append('spawn-failure-has-terminal-full-charge-and-no-second-launch')
  # A completed process killed before writing terminal retains its partial bytes.
  def no_terminal(command,**kwargs):
   request=Path(command[command.index('--request')+1]);p=request.parent/'response';p.mkdir();(p/'body.bin.partial').write_bytes(b'partial')
   q=Process();q.returncode=-9;return q
  absent=r.Controller(tmp/'missing',plan(1),{},spawn=no_terminal)
  reject(absent.run)
  assert absent.ledger['requests'][0]['status']=='STOP'
  assert absent.ledger['retained_body_bytes']==7 and absent.ledger['charged_response_bytes']==400000
  terminal=json.loads(Path(absent.terminals[0]['terminal']['path']).read_text())
  assert terminal['failure_type']=='MissingWorkerTerminal'and terminal['partial_files'][0]['bytes']==7
  CASES.append('missing-terminal-preserves-partial-bytes-and-reservation')
  expired=r.Controller(tmp/'expired',plan(2),{},spawn=blocked);expired.deadline=0
  reject(expired.run)
  assert not expired.ledger['requests']and not expired.active and not expired.pending
  CASES.append('expired-deadline-starts-no-request')
 result=dict(status='PASS',count=len(CASES),cases=CASES,source=[c.record(IMPL/n)for n in c.CODE_NAMES],reviewer=c.record(__file__),actual_subprocesses=0,real_network_requests=0,scientific_arrays_read=0,model_calls=0)
 c.save(out,result);print(json.dumps(c.record(out)))
if __name__=='__main__':main()
