"""One root-authorized bounded source GET; no array or model operations."""
from pathlib import Path
import datetime
import hashlib
import json
import time

import requests

ROOT = Path(__file__).resolve().parent
DESIGN = ROOT.parent / 'RESEARCH_DESIGN.md'
DESIGN_SHA = '08bc48f203d6722216c70d54b961d0d29ce59204359da49d2b6110661a9a9cc1'
URL = 'https://raw.githubusercontent.com/educelab/ink-id/58b1519551b73554168371a31b1ca27767ef6e33/inkid/data/ppm.py'
EXPECTED_LENGTH = 7494
EXPECTED_BLOB = '05ef3b591fc593df6fbc5531a429c9ebd870d62c'
CAP = 32768


def record(path):
    body = path.read_bytes()
    return dict(path=str(path), size_bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    design = record(DESIGN)
    assert design['sha256'] == DESIGN_SHA, 'Frozen research design differs'
    plan_path = ROOT / 'PPM_SOURCE_REQUEST_PLAN.json'
    receipt_path = ROOT / 'PPM_SOURCE_RECEIPT.json'
    source_path = ROOT / 'historical-inkid-ppm.py'
    assert not receipt_path.exists() and not source_path.exists(), 'No overwrite or retry'
    plan = dict(schema_version=1, kind='bounded-authoritative-source-retrieval',
                design=design, script=record(Path(__file__).resolve()),
                created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                method='GET', url=URL, maximum_attempts=1, reserved_body_bytes=CAP,
                redirects=False, retries=0, tls_verification=True,
                headers={'Accept-Encoding': 'identity'}, timeout_seconds=[10, 20],
                required_http_status=200, required_length_bytes=EXPECTED_LENGTH,
                required_git_blob_sha1=EXPECTED_BLOB,
                output_source=str(source_path), output_receipt=str(receipt_path))
    write_new(plan_path, plan)
    started = time.monotonic()
    receipt = dict(schema_version=1, plan=record(plan_path), status='STOP',
                   attempted_requests=0, retained_body_bytes=0,
                   reserved_response_body_bytes=CAP, automatic_retries=0,
                   redirects_followed=0, tls_verification=True)
    body = b''
    try:
        with requests.Session() as session:
            # Requests adapters default to no retries; enforce and record that contract.
            assert session.get_adapter(URL).max_retries.total == 0
            receipt['attempted_requests'] = 1
            with session.get(URL, headers=plan['headers'], stream=True,
                             timeout=(10, 20), allow_redirects=False, verify=True) as response:
                receipt['http_status'] = response.status_code
                receipt['response_headers'] = {key: response.headers[key] for key in
                    ('Content-Length', 'Content-Type', 'Content-Encoding', 'ETag', 'Last-Modified')
                    if key in response.headers}
                assert response.status_code == 200, 'Expected HTTP 200; no redirect/retry'
                assert response.headers.get('Content-Encoding', 'identity').lower() in ('', 'identity'), 'Encoded response not allowed'
                length = response.headers.get('Content-Length')
                if length is not None:
                    assert int(length) == EXPECTED_LENGTH, 'Advertised source length differs'
                # Never consume more than the authorized cap, even for absent/bad length.
                body = response.raw.read(CAP, decode_content=False)
                receipt['retained_body_bytes'] = len(body)
                assert len(body) == EXPECTED_LENGTH, 'Actual source length differs'
                blob = hashlib.sha1(b'blob ' + str(len(body)).encode('ascii') + b'\0' + body).hexdigest()
                receipt['git_blob_sha1'] = blob
                assert blob == EXPECTED_BLOB, 'Historical Git blob identity differs'
                body.decode('utf-8')
                with source_path.open('xb') as output:
                    output.write(body)
                receipt['source'] = record(source_path)
                receipt['status'] = 'PASS'
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        if body and not source_path.exists():
            failed_path = ROOT / 'historical-inkid-ppm.failed-response.bin'
            with failed_path.open('xb') as output:
                output.write(body)
            receipt['failed_response'] = record(failed_path)
        raise
    finally:
        receipt['duration_seconds'] = time.monotonic() - started
        receipt['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_new(receipt_path, receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
