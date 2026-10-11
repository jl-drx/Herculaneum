"""Offline, read-only-input replay of the bounded PHerc0139 CT comparison."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def compare(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,'different keys')
        return max([compare(v,b[k],path+'/'+k) for k,v in a.items()] or [0])
    if isinstance(a,list):
        assert len(a)==len(b),(path,'length')
        return max([compare(x,y,path+f'/{i}') for i,(x,y) in enumerate(zip(a,b))] or [0])
    if isinstance(a,float):
        d=abs(a-b);assert d<=1e-8,(path,a,b);return d
    assert a==b,(path,a,b);return 0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve()
    assert not out.exists(),'Choose a new output directory'
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    mf=json.loads((B/'PACKET_FILES.json').read_text())
    for e in mf['files']:
        p=B/e['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==e['bytes'] and digest(p)==e['sha256'],e['path']
    subprocess.run([sys.executable,str(B/'ct-study/analyse.py'),'--output',str(out)],check=True)
    import numpy as np
    saved=json.loads((B/'ct-study/results/RESULTS.json').read_text());new=json.loads((out/'RESULTS.json').read_text())
    error=compare(saved['sites'],new['sites'])
    maps=[]
    for s in saved['sites']:
        sid=s['site'];x=np.load(B/'ct-study/results'/sid/'landscapes.npy');y=np.load(out/sid/'landscapes.npy')
        diff=float(np.max(np.abs(x-y)));assert diff<=1e-8;maps.append(dict(site=sid,max_score_error=diff))
    # Execute the independently authored direct-spatial checker on regenerated
    # arrays. Rebind only input/output directories; its calculations are unchanged.
    independent_path=B/'method-review/score-check/replay_real_scores.py'
    spec=importlib.util.spec_from_file_location('independent_ct_replay',independent_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audit=out/'independent';audit.mkdir();module.B=B;module.D=out;module.R=audit;module.main()
    originals=json.loads((B/'ct-study/results/OUTPUT_FILES.json').read_text())
    arr=[e for e in originals if e['path'].endswith('.npy')]
    same=sum(digest(out/e['path'])==e['sha256'] for e in arr)
    import importlib.metadata as md
    result=dict(status='PASS',packet_files_verified=len(mf['files']),source_input_bytes=sum(e['bytes'] for e in mf['files']),
                all_frozen_decisions_match=True,max_summary_numeric_error=error,landscapes=maps,
                derived_arrays_byte_identical=same,derived_arrays_compared=len(arr),
                independently_recomputed_scores=3*8*13**3,
                runtime=dict(python=sys.version,numpy=md.version('numpy'),scipy=md.version('scipy')),
                scope='Offline chunk assembly, fixed resampling, scoring and independent score replay. No network, models, inference, fitting beyond frozen finite translation, or paid resources.')
    (out/'REPLAY_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
