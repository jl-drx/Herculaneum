"""Verify and replay the saved PHerc0800 qualification; no inference or network."""
import argparse,json,subprocess,sys
from pathlib import Path
from workspace import checked,restore_bundle,sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]/'workspace'
    index=json.loads((root/'PHASE49_BUNDLE.json').read_text());archive=root/'bundles'/index['bundle']['name']
    if not checked(archive,index['bundle']):raise ValueError('Download only the named LFS object first; checksum mismatch or pointer present')
    dest=args.destination.resolve()
    if dest.exists():raise ValueError('Use a fresh restore destination')
    restore=restore_bundle(archive,dest,index['members'])
    study=dest/'reports/fortyninth-experiment'
    if sha(study/'final-verification.json')!=index['seal_sha256']:raise ValueError('Study seal mismatch')
    done=subprocess.run([sys.executable,'-B',str(study/'verify_saved.py')],check=True,capture_output=True,text=True)
    replay=json.loads(done.stdout);assert replay['status']=='PASS'
    print(json.dumps(dict(status='PASS',archive_and_all_members_verified=True,files=len(index['members']),restore=restore,replay=replay,inference_run=False,scope='Relocated saved-output replay on this host, not independent physical-device inference')))
if __name__=='__main__':main()
