"""Relocate the unchanged frozen geometry solver; no CT, target or network reads."""
import argparse,copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
B=Path(__file__).resolve().parent

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve()
    assert not out.exists(),'Choose a new output directory'
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    original=B/'geometry-review/pullback';code=original/'pullback.py';plan=json.loads((original/'PLAN.json').read_text())
    assert sha(code)==plan['code_sha256']
    relocation=copy.deepcopy(plan)
    for entry in relocation['inputs']:
        name=Path(entry['path']).name
        p=B/'provenance'/name if name=='PHerc0139-retained-catalog.json' else B/'geometry/responses'/name
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'];entry['path']=str(p)
    spec=importlib.util.spec_from_file_location('frozen_geometry_pullback',code);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    out.mkdir(parents=True);(out/'PLAN.json').write_bytes((original/'PLAN.json').read_bytes())
    mod.ROOT=out;mod.GEO=B/'geometry/responses';mod.CAT=B/'provenance/PHerc0139-retained-catalog.json'
    # Only replace the path-dependent plan preparation; all physical parameters,
    # verified source bytes and solver calculations remain those in the frozen plan.
    mod.prepare=lambda:relocation
    old=sys.argv;sys.argv=[str(code)]
    try:mod.main()
    finally:sys.argv=old
    import numpy as np
    results=[]
    for sid in ['site1','site2','site3']:
        saved=json.loads((original/sid/'RESULTS.json').read_text());new=json.loads((out/sid/'RESULTS.json').read_text())
        assert saved['samples']==new['samples'] and saved['accepted']==new['accepted'] and saved['failure_codes']==new['failure_codes']
        support=np.load(original/sid/'support.npy');assert np.array_equal(support,np.load(out/sid/'support.npy'))
        q=np.load(original/sid/'queries_uvk.npy');r=np.load(out/sid/'queries_uvk.npy');difference=float(np.max(np.abs(q-r)))
        assert difference<=1e-4 and new['residual_max_accepted']<=.01
        same=sum(sha(out/sid/e['name'])==e['sha256'] for e in saved['output_files'])
        results.append(dict(site=sid,max_query_component_error=difference,support_identical=True,byte_identical_arrays=same,total_arrays=len(saved['output_files'])))
    report=dict(status='PASS',sites=results,scope='Frozen geometry solve with paths rebound to verified local bytes; no CT registration, inference or new data acquisition.')
    (out/'GEOMETRY_REPLAY_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
