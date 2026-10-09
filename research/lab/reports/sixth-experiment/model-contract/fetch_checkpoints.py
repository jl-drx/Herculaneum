"""Two immutable public checkpoint objects, one attempt each, bounded verified reuse."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urljoin, urlsplit
import requests
from model_contract import HERE, ContractError, load_plan, require, verify_artifact


class Anonymous(requests.auth.AuthBase):
    def __call__(self, request):
        request.headers.pop('Authorization', None)
        return request  # Explicit auth prevents implicit netrc credential loading.


def allowed_url(url):
    parsed = urlsplit(url)
    host = parsed.hostname or ''
    return parsed.scheme == 'https' and not parsed.username and not parsed.password and (
        host == 'huggingface.co' or host.endswith('.huggingface.co') or
        host == 'hf.co' or host.endswith('.hf.co'))


def bounded_error_body(raw, cap):
    """Never read cap+1 bytes; a full cap without EOF stops rather than assuming complete."""
    count = 0
    digest = hashlib.sha256()
    while count < cap:
        block = raw.read(min(65536, cap-count), decode_content=False)
        if not block:
            return count,digest.hexdigest(),True
        digest.update(block)
        count += len(block)
    return count,digest.hexdigest(),False


def acquire(artifact, session, budget):
    path = Path(artifact['local_path'])
    require(path.parent == HERE/'checkpoints', 'New cache must stay in owned checkpoint directory')
    if path.exists():
        verify_artifact(path,artifact)
        return dict(label=artifact['label'],path=str(path),status='verified_reuse',
                    sha256=artifact['sha256'],size_bytes=artifact['size_bytes'],requests=0,new_payload_bytes=0)
    partial = path.with_suffix(path.suffix+'.partial')
    require(not partial.exists(), 'Partial transfer exists; stop for diagnosis instead of repeated full download')
    require(shutil.disk_usage(path.parent).free >= artifact['size_bytes']+512*1024*1024,
            'Minimum free disk reserve would be violated')
    url = artifact['source_url']
    transfer = dict(label=artifact['label'],path=str(path),status='started',requests=0,
                    new_payload_bytes=0,redirects=[],sha256=artifact['sha256'],size_bytes=artifact['size_bytes'])
    budget['transfers'].append(transfer)
    started = time.monotonic()
    for hop in range(6):
        require(allowed_url(url), 'Unexpected redirect host/scheme; URL suppressed')
        require(budget['requests'] < 12 and time.monotonic()-started < 300, 'Transfer request/time limit')
        budget['requests'] += 1
        transfer['requests'] += 1
        with session.get(url,stream=True,allow_redirects=False,timeout=(15,30),
                         headers={'Accept-Encoding':'identity'}) as response:
            require(response.headers.get('Content-Encoding','identity') == 'identity', 'Encoded HTTP response rejected')
            if response.status_code != 200:
                remaining = 128*1024-budget['overhead_bytes']
                require(remaining > 0, 'HTTP response overhead exhausted')
                size,digest,eof = bounded_error_body(response.raw,min(65536,remaining))
                budget['overhead_bytes'] += size
                transfer['redirects'].append(dict(status=response.status_code,host=urlsplit(url).hostname,
                    body_bytes=size,body_sha256=digest,complete=eof))
                require(eof, 'HTTP redirect/error body reached bounded cap')
                if response.status_code in (301,302,303,307,308):
                    require(hop < 5 and response.headers.get('Location'), 'Redirect limit or missing target')
                    url = urljoin(url,response.headers['Location'])
                    continue
                raise ContractError('Checkpoint HTTP status '+str(response.status_code)+'; no fill or retry')
            expected = artifact['size_bytes']
            require(response.headers.get('Content-Length') == str(expected), 'Checkpoint Content-Length differs from pin')
            digest = hashlib.sha256()
            count = 0
            with partial.open('xb') as output:
                while count < expected:
                    require(time.monotonic()-started < 300, 'Checkpoint transfer time limit')
                    block = response.raw.read(min(1024*1024,expected-count),decode_content=False)
                    require(bool(block), 'Incomplete checkpoint body; partial retained')
                    require(budget['payload_bytes']+len(block) <= 276734604, 'Checkpoint payload budget exceeded')
                    output.write(block)
                    digest.update(block)
                    count += len(block)
                    transfer['new_payload_bytes'] += len(block)
                    budget['payload_bytes'] += len(block)
                output.flush()
                os.fsync(output.fileno())
            require(count == expected and digest.hexdigest() == artifact['sha256'], 'Downloaded checkpoint SHA mismatch; partial retained')
            # Publish only a complete verified file and never replace an existing object.
            os.link(partial,path)
            partial.unlink()
            directory_fd = os.open(path.parent,os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            verify_artifact(path,artifact)
            transfer.update(status='downloaded_verified',final_host=urlsplit(url).hostname)
            return transfer
    raise ContractError('Redirect limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan,args.plan_sha256)
    output = args.output_dir.resolve()
    require(output.parent == HERE, 'Report must stay in owned directory')
    output.mkdir(exist_ok=False)
    cache = HERE/'checkpoints'
    cache.mkdir(exist_ok=True)
    budget = dict(schema_version=1,plan_path=str(args.plan.resolve()),plan_sha256=args.plan_sha256,
                  requests=0,payload_bytes=0,overhead_bytes=0,transfers=[],completed=False,
                  automatic_retries=0,paid_compute=False)
    try:
        with (cache/'.download.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            session = requests.Session()
            session.auth = Anonymous()
            session.mount('https://',requests.adapters.HTTPAdapter(max_retries=0))
            try:
                for artifact in plan['artifacts']:
                    if artifact['acquisition'] == 'new_public_object':
                        record = acquire(artifact,session,budget)
                        if record not in budget['transfers']:
                            budget['transfers'].append(record)
                load_plan(args.plan,args.plan_sha256)
                budget['completed'] = True
            finally:
                session.close()
    except Exception as error:
        # Exception text can include signed redirect URLs. Keep type plus our bounded messages only.
        budget['stop'] = dict(type=type(error).__name__,reason=str(error) if isinstance(error,ContractError) else 'Transfer or I/O failed; credentials/URLs suppressed')
    (output/'download-report.json').write_text(json.dumps(budget,indent=2)+'\n')
    print(json.dumps({k:budget[k] for k in ('completed','requests','payload_bytes','overhead_bytes')}))
    return 0 if budget['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
