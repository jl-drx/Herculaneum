"""Run the original renderer, retaining failures and repairing transient 5xx fetches."""
from pathlib import Path
import os,subprocess,re,json,hashlib,time,requests
P=Path(__file__).resolve().parent;W=P/'work';OLD=P.parent/'fortyfifth-experiment'
BASE='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr'
NAME='concat_w047-w048_R5B2_z9500-11000'
cache=next(p for p in (W/'renderer-cache').glob('*.zarr-*') if p.is_dir())
assert cache.is_dir()
env=dict(os.environ,QT_QPA_PLATFORM='offscreen',VC3D_CONFIG_DIR=str(W/'vc3d-config'),OMP_NUM_THREADS='4')
cmd=[str(W/'squashfs-root/AppRun'),'vc_render_tifxyz','--volume',str(W/'ct-cache'),'--remote-url',BASE,'--group-idx','0','--scale','1','--segmentation',str(OLD/'ph343-mesh-replay'/f'{NAME}.tifxyz'),'--num-slices','66','--slice-step','1','--cache-gb','2','--zarr-output',str(W/f'{NAME}_66.zarr'),'--resume']
events=json.loads((P/'RENDER_EXECUTION.json').read_text()) if (P/'RENDER_EXECUTION.json').exists() else []
start=max(int(p.stem.removeprefix('render-v')) for p in W.glob('render-v*.log'))+1
for attempt in range(start,start+4):
    prior=W/f'render-v{attempt-1}.log'
    error=re.search(r'HTTP (5\d\d) fetching ([^: ]+):',prior.read_text())
    if error:
        key=error.group(2);assert key in ['.zattrs','.zgroup'] or re.fullmatch(r'[0-5]/\.z(?:attrs|array)',key) or re.fullmatch(r'0/\d+/\d+/\d+',key)
        t=time.monotonic();r=requests.get(BASE+'/'+key,timeout=90);r.raise_for_status()
        if re.fullmatch(r'0/\d+/\d+/\d+',key):
            assert len(r.content)==128**3
            dest=cache/key;assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
            dest.with_suffix('.partial').write_bytes(r.content);dest.with_suffix('.partial').rename(dest)
        else:
            assert r.json()==json.loads((cache/key).read_text())
        events.append(dict(repaired_key=key,status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),seconds=time.monotonic()-t))
    else:raise RuntimeError('Nontransient renderer failure; inspect retained log')
    t=time.monotonic()
    with (W/f'render-v{attempt}.log').open('xb') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    events.append(dict(attempt=attempt,command=cmd,exit_code=result.returncode,elapsed_seconds=time.monotonic()-t))
    (P/'RENDER_EXECUTION.json').write_text(json.dumps(events,indent=2)+'\n')
    if result.returncode==0:
        print('Render complete',flush=True);break
else:raise RuntimeError('Transient retry budget exhausted; retained partial render')
subprocess.run(['/workspace/.venvs/pherc343/bin/python','-B',str(P/'verify_render.py')],check=True,env=env)
