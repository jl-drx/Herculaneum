"""Rebuild a temporary complete reference view and replay saved qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--current',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    work=a.current/'work';q=work/'added-qualification-20261012'
    expected=json.loads((q/'RESULTS.json').read_text())
    freeze=json.loads((q/'FREEZE.json').read_text())
    assert sha(q/'FREEZE.json')==expected['freeze_sha256']
    assert sha(q/'qualify_added.py')==freeze['implementation_sha256']
    for name,digest in freeze['reference_sha256'].items():assert sha(q/'reference'/name)==digest
    with tempfile.TemporaryDirectory(prefix='plinius-added-replay-',dir='/tmp') as folder:
        root=Path(folder);base=root/'base';shutil.copytree(a.current/'base',base)
        for source in (q/'reference').rglob('*'):
            if not source.is_file() or source.name=='CANDIDATES.csv':continue
            dest=base/source.relative_to(q/'reference');dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==sha(source)
            else:shutil.copyfile(source,dest)
        assert sha(base/'support-reservation-v2/CANDIDATES.csv')==sha(q/'reference/CANDIDATES.csv')
        target=root/'qualification'
        subprocess.run([sys.executable,str(q/'qualify_added.py'),'--base',str(base),'--work',str(work),'--output',str(target)],check=True)
        actual=json.loads((target/'RESULTS.json').read_text())
        # Freeze timestamp changes; scientific outputs and all case decisions must not.
        for key in expected:
            if key!='freeze_sha256':assert actual[key]==expected[key],key
        files=[]
        for case in expected['cases']:
            for name,digest in case.get('outputs',{}).items():
                assert sha(target/name)==digest;files.append(name)
    result=dict(status='PASS',production_case_results_equal=True,derived_files_byte_identical=files,
        qualification_results_sha256=sha(q/'RESULTS.json'),implementation_sha256=sha(Path(__file__)),
        scope='Fresh local production replay from restored raw source cache; not another-machine review or physical accuracy.',
        new_model_forwards=0,paid_compute=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
