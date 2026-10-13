"""Relocated-packet Python audit test, not OS filesystem isolation."""
from pathlib import Path
import contextlib, hashlib, json, os, runpy, shutil, sys, datetime
# Import the scientific dependencies before installing the scoped audit hook.
import numpy
from PIL import Image

SRC=Path('/workspace/herculaneum-lab/reports/fiftysecond-experiment/completion-review')
PROOF=Path(__file__).resolve().parent
PACKET=PROOF/'relocated-packet'
assert not PACKET.exists()
PACKET.mkdir()
outputs=json.loads((SRC/'OUTPUT_MANIFEST.json').read_text())['files']
saved=json.loads((SRC/'SAVED_INPUTS_MANIFEST.json').read_text())['files']
files={r['path'] for r in outputs+saved}|{'OUTPUT_MANIFEST.json','SAVED_INPUTS_MANIFEST.json','verify_saved_maps.py'}
for rel in sorted(files):
    source=SRC/rel; target=PACKET/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
assert len(list(PACKET.rglob('*.npy')))==48
assert len(list(PACKET.rglob('*.npz')))==1
source_root=SRC.parent.parent.parent.parent
allowed_read_roots=tuple({PACKET.resolve(),Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()})
state={'enabled':False,'open_events':0,'allowed_packet_open_events':0,'allowed_runtime_open_events':0,'denied_events':[]}
write_flags=os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND
mutation_events={'os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.chown','os.truncate','os.utime'}

def guard(event,args):
    if not state['enabled']: return
    if event in mutation_events:
        state['denied_events'].append({'event':event,'arguments':str(args)})
        raise PermissionError('Verifier attempted filesystem mutation: '+event)
    if event!='open': return
    state['open_events']+=1
    path,mode,flags=args
    if flags is not None and flags & write_flags:
        state['denied_events'].append({'event':'open-write','path':str(path)})
        raise PermissionError('Verifier attempted writable open')
    if isinstance(path,int):
        if path in (0,1,2): return
        raise PermissionError('Unresolved descriptor open')
    resolved=Path(os.fsdecode(path)).resolve()
    if resolved.is_relative_to(PACKET.resolve()):
        state['allowed_packet_open_events']+=1
        return
    if any(resolved.is_relative_to(root) for root in allowed_read_roots[0:] if root!=PACKET.resolve()):
        state['allowed_runtime_open_events']+=1
        return
    state['denied_events'].append({'event':'open-outside-allowlist','path':str(resolved)})
    raise PermissionError('Read outside relocated packet/runtime: '+str(resolved))

sys.addaudithook(guard)
with (PROOF/'RELOCATED_RESULT.json').open('w') as capture:
    sys.argv=[str(PACKET/'verify_saved_maps.py'),'--packet',str(PACKET)]
    state['enabled']=True
    try:
        with contextlib.redirect_stdout(capture):
            runpy.run_path(str(PACKET/'verify_saved_maps.py'),run_name='__main__')
    finally:
        state['enabled']=False
result=json.loads((PROOF/'RELOCATED_RESULT.json').read_text())
assert result['status']=='PASS' and not state['denied_events']
proof={'status':'PASS','checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Python audit-hook filesystem-read/write guard during relocated saved-map verifier execution; this is not full OS filesystem isolation and does not intercept arbitrary direct C syscalls.','packet':str(PACKET),'copied_files':len(files),'copied_total_bytes':sum((PACKET/f).stat().st_size for f in files),'npy_files':48,'npz_files':1,'original_checkpoint_present':False,'model_or_ct_files_present':False,'read_allowlist_roots':sorted(map(str,allowed_read_roots)),'write_policy':'Reject writable open flags and listed os mutation audit events; stdout captured by harness using a file opened before guard activation.','mutation_events_rejected':sorted(mutation_events),'unresolved_fd_policy':'Only fd 0/1/2 allowed if open event supplies an integer','audit':{k:v for k,v in state.items() if k!='enabled'},'verifier_sha256':hashlib.sha256((PACKET/'verify_saved_maps.py').read_bytes()).hexdigest(),'assessment_plan_sha256':hashlib.sha256((PACKET/'ASSESSMENT_PLAN.json').read_bytes()).hexdigest(),'saved_inputs_manifest_sha256':hashlib.sha256((PACKET/'SAVED_INPUTS_MANIFEST.json').read_bytes()).hexdigest()}
(PROOF/'RELOCATION_GUARD_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
