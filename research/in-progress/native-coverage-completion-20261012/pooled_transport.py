"""Bounded persistent-connection intake of unchanged frozen source requests."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import requests


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ['base', 'work', 'attempt']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--limit', type=int, required=True)
    a = parser.parse_args()
    if a.limit < 1:
        raise ValueError('Positive bounded request count required')
    a.attempt.mkdir(parents=True, exist_ok=False)
    plan = json.loads((a.work / 'REQUEST_PLAN.json').read_text())
    for rel, digest in plan['base_pins'].items():
        assert sha(a.base / rel) == digest
    pending = []
    prior_bytes = 0
    for row in plan['requests']:
        receipt = a.work / 'receipts' / (row['id'] + '.json')
        if not receipt.exists():
            pending.append(row)
            continue
        saved = json.loads(receipt.read_text())
        assert saved['request'] == row
        if saved['status'] == 'PRESENT':
            source = a.work / row['path']
            assert source.stat().st_size == row['bytes'] and sha(source) == saved['sha256']
        else:
            assert saved['status'] == 'MISSING'
        prior_bytes += sum(r.get('body_bytes', 0) for r in saved['attempts'])
    chosen = pending[:a.limit]
    assert prior_bytes + sum(r['bytes'] for r in chosen) <= plan['maximum_body_bytes_with_retries']
    write(a.attempt / 'PLAN.json', dict(utc=stamp(), transport='requests.Session with inherited proxy and TLS trust',
        workers=1, retries=0, request_plan_sha256=sha(a.work / 'REQUEST_PLAN.json'),
        implementation_sha256=sha(Path(__file__)), plan_sha256=sha(Path(__file__).with_name('PLAN.md')),
        request_ids=[r['id'] for r in chosen], pending_before=len(pending),
        maximum_new_source_bytes=sum(r['bytes'] for r in chosen), prior_receipted_body_bytes=prior_bytes,
        eligibility_or_urls_changed=False, paid_compute=False))
    started = time.monotonic()
    completed = missing = body_bytes = 0
    sockets = []
    connection_objects = []
    failure = None
    session = requests.Session()
    # Default HTTPAdapter retries=0. Keep trust_env, proxy and TLS verification.
    assert session.trust_env and session.verify
    try:
        for row in chosen:
            request_start = time.monotonic()
            evidence = dict(request_id=row['id'], utc=stamp(), source_body_bytes=0)
            headers = {'Accept-Encoding': 'identity'}
            if 'etag' in row:
                headers['If-Match'] = row['etag']
            if row['kind'] == 'teacher':
                headers['Range'] = f"bytes={row['offset']}-{row['offset'] + row['bytes'] - 1}"
            try:
                with session.get(row['url'], headers=headers, stream=True, timeout=(20, 45), allow_redirects=False) as response:
                    evidence.update(http_status=response.status_code, server=response.headers.get('Server'))
                    connection = getattr(response.raw, '_connection', None)
                    sock = getattr(connection, 'sock', None)
                    evidence['same_connection_object_seen_before'] = connection is not None and any(connection is c for c in connection_objects)
                    evidence['same_socket_seen_before'] = sock is not None and any(sock is s for s in sockets)
                    if connection is not None:
                        connection_objects.append(connection)
                    if sock is not None:
                        sockets.append(sock)
                    expected = 206 if row['kind'] == 'teacher' else 200
                    if response.status_code != expected:
                        error = response.raw.read(2048, decode_content=False)
                        evidence['error_body'] = error.decode(errors='replace')
                        code = None
                        if response.status_code == 404:
                            try:
                                code = ET.fromstring(error).findtext('Code')
                            except ET.ParseError:
                                pass
                        if code in ['NoSuchKey', 'NoSuchObject']:
                            saved = dict(status='MISSING', request=row,
                                attempts=[dict(attempt=1, http_status=404, body_bytes=0)], utc=stamp(), transport='requests-session')
                            write(a.work / 'receipts' / (row['id'] + '.json'), saved)
                            evidence['status'] = 'CONFIRMED_SOURCE_MISSING'
                            missing += 1
                            write(a.attempt / 'requests' / (row['id'] + '.json'), evidence)
                            continue
                        raise RuntimeError(f'Unexpected HTTP {response.status_code}; source remains unresolved')
                    etag = response.headers['ETag']
                    assert etag and ('etag' not in row or etag == row['etag']), 'ETag changed'
                    assert int(response.headers['Content-Length']) == row['bytes']
                    assert response.headers.get('Content-Encoding', 'identity') == 'identity'
                    if row['kind'] == 'teacher':
                        assert response.headers['Content-Range'] == f"bytes {row['offset']}-{row['offset']+row['bytes']-1}/{row['object_bytes']}"
                    payload = response.raw.read(row['bytes'] + 1, decode_content=False)
                    evidence['source_body_bytes'] = len(payload)
                    body_bytes += len(payload)
                    assert len(payload) == row['bytes']
                    assert prior_bytes + body_bytes <= plan['maximum_body_bytes_with_retries']
                dest = a.work / row['path']
                receipt = a.work / 'receipts' / (row['id'] + '.json')
                assert not dest.exists() and not receipt.exists(), 'Concurrent source writer or unreceipted artifact'
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_suffix('.partial')
                temp.write_bytes(payload)
                temp.replace(dest)
                digest = hashlib.sha256(payload).hexdigest()
                saved = dict(status='PRESENT', request=row, sha256=digest, etag=etag, bytes=len(payload),
                    attempts=[dict(attempt=1, status='SUCCESS', body_bytes=len(payload))], utc=stamp(), transport='requests-session')
                write(receipt, saved)
                evidence.update(status='PRESENT', sha256=digest, etag=etag)
                completed += 1
            except Exception as error:
                evidence.update(status='FAILED_UNRESOLVED', error_type=type(error).__name__, reason=str(error))
                failure = evidence
            evidence['seconds'] = time.monotonic() - request_start
            write(a.attempt / 'requests' / (row['id'] + '.json'), evidence)
            if failure:
                break
            if completed % 16 == 0:
                print(json.dumps(dict(completed=completed, chosen=len(chosen), source_body_bytes=body_bytes, utc=stamp())), flush=True)
    finally:
        session.close()
    result = dict(status='FAILED_UNRESOLVED' if failure else 'BOUNDED_REQUEST_LIST_COMPLETE',
        utc=stamp(), chosen=len(chosen), completed=completed, confirmed_missing=missing,
        source_body_bytes=body_bytes, unattempted=len(chosen)-completed-missing-int(failure is not None),
        failure=failure, distinct_connection_objects=len({id(c) for c in connection_objects}),
        distinct_socket_objects=len({id(s) for s in sockets}), seconds=time.monotonic()-started,
        full_population_admitted=False, new_model_forwards=0, paid_compute=False)
    write(a.attempt / 'RESULTS.json', result)
    print(json.dumps(result), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
