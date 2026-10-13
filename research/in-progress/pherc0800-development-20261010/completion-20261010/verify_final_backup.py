"""Verify the final remote generation and local coverage of the published index.

No inference or extraction. Other generations are not freshly downloaded here.
"""
import hashlib,json,subprocess,tarfile
from pathlib import Path
from datetime import datetime,timezone

P=Path('/workspace/herculaneum-lab/reports/fiftyfirst-experiment')
LAB=P.parent.parent
REPO=Path('/workspace/Herculaneum')
COMMIT='dfb4fa95e0328730757d38a1711e2ebbe6a56850'
REF='refs/heads/codex/plinius-pherc0800-phase51-checkpoints'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

remote=subprocess.check_output(['git','ls-remote','origin',REF],cwd=REPO,text=True).split()
assert remote==[COMMIT,REF]
raw=subprocess.check_output(['git','show',COMMIT+':INDEX.json'],cwd=REPO)
index=json.loads(raw); records={}
assert index['go_sha256']==sha(P/'INFERENCE_GO.json')
for generation in index['generations']:
    for r in generation['members']:
        assert r['path'] not in records
        path=LAB/r['path'];assert path.is_relative_to(LAB) and '..' not in path.parts
        assert path.is_file() and path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],r['path']
        records[r['path']]=r
g=index['generations'][-1]
archive=Path('/workspace/scratch/phase51-final-backup-lfs/objects')/g['sha256'][:2]/g['sha256'][2:4]/g['sha256']
assert archive.stat().st_size==g['bytes'] and sha(archive)==g['sha256']
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers();assert len(members)==len(g['members'])
    for member,r in zip(members,g['members']):
        assert member.isfile() and member.name==r['path'] and member.size==r['bytes']
        with tar.extractfile(member) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
receipts=list((P/'work/inference').glob('*/batch-*.json'))
assert len(receipts)==3454
for receipt in receipts:
    assert str(receipt.relative_to(LAB)) in records
    assert str(Path(json.loads(receipt.read_text())['logits']['path']).relative_to(LAB)) in records
completed=list((P/'work/inference').glob('*/COMPLETE.json'));assert len(completed)==16
for path in completed+[P/'INFERENCE_RESULTS.json']:
    assert str(path.relative_to(LAB)) in records
result=dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),remote_ref=REF,remote_commit=COMMIT,index_sha256=hashlib.sha256(raw).hexdigest(),generations=len(index['generations']),indexed_local_files_verified=len(records),committed_batches_covered=len(receipts),completed_maps_covered=len(completed),final_results_covered=True,fresh_remote_archive={k:v for k,v in g.items() if k!='members'},fresh_remote_members_verified=len(g['members']),scope='Fresh remote download of final generation only; all indexed files checked against retained local bytes. Generations 0 and 1 were separately downloaded/restored earlier. No fresh full-chain restore or independent model replay in this check.',inference_run=False)
out=P/'FINAL_BACKUP_VERIFICATION.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
