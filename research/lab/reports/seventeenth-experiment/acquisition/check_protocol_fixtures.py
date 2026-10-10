"""Exercise the strict transport with synthetic bytes; network and decoding forbidden."""
import argparse
from email.message import Message
import hashlib
import http.client
import json
from pathlib import Path
import socket

from acquisition_common import ContractError, parse_multipart, record, require, save
from regional_http import AcquisitionClient


BASE = ('https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/'
        'working/54keV_exposed_surface/')
TOTAL = 2000000
ETAG = '"fixture-stable"'
BOUNDARY = 'regional-fixture'
RANGES = [[4096 + i * 512, 4096 + i * 512 + 143] for i in range(243)]
PREFIX = bytes(range(256)) * 4
IR = b'Synthetic opaque IR transport body; never decoded.\n'


def payload(span, boundary=BOUNDARY):
    """Embed both regular and terminal delimiter lookalikes inside binary payloads."""
    marker = b'\r\n--' + boundary.encode('ascii')
    sample = bytes([span[0] % 251]) + marker + b'\r\n' + marker + b'--\r\n\x00\xff'
    size = span[1] - span[0] + 1
    return (sample * (size // len(sample) + 1))[:size]


def multipart(ranges=RANGES, total=TOTAL, boundary=BOUNDARY, reverse=False, prefix=b''):
    selected = list(reversed(ranges)) if reverse else ranges
    chunks = []
    for span in selected:
        a, b = span
        chunks.append(b'--' + boundary.encode('ascii') + b'\r\n'
                      b'Content-Type: application/octet-stream\r\n'
                      b'Content-Range: bytes ' + f'{a}-{b}/{total}'.encode('ascii') + b'\r\n'
                      b'Content-Length: ' + str(b - a + 1).encode('ascii') + b'\r\n\r\n'
                      + payload(span, boundary) + b'\r\n')
    return prefix + b''.join(chunks) + b'--' + boundary.encode('ascii') + b'--\r\n'


class Response:
    def __init__(self, body, status=206, headers=(), incomplete=None, overread=False):
        self.body, self.status, self.position = body, status, 0
        self.incomplete, self.overread = incomplete, overread
        self.read_calls, self.closed = 0, False
        self.headers = Message()
        for name, value in (headers.items() if isinstance(headers, dict) else headers):
            self.headers[name] = str(value)

    def read(self, size):
        self.read_calls += 1
        require(size > 0, 'Fixture received an unbounded read')
        if self.incomplete is not None:
            partial, self.incomplete = self.incomplete, None
            raise http.client.IncompleteRead(partial, len(self.body))
        if self.overread:
            return b'x' * (size + 1)
        end = self.position + min(size, 317)
        block = self.body[self.position:end]
        self.position += len(block)
        return block

    def close(self):
        self.closed = True


class Opener:
    def __init__(self, response, root, frozen):
        self.response, self.root, self.frozen = response, root, frozen
        self.calls = []

    def open(self, request, timeout):
        ledger = json.loads((self.root / 'request-ledger.json').read_text())
        require(len(ledger['requests']) == 1 and ledger['requests'][0]['success'] is False
                and ledger['charged_response_bytes'] == self.frozen['reserved_cap_bytes'],
                'Full reservation must be durable before opening response')
        require(timeout == 25 and request.get_method() == 'GET', 'Expected bounded GET')
        self.calls.append(request)
        return self.response


def request_for(mode):
    ranges = [] if mode == 'full' else ([[0, 1023]] if mode == 'single' else RANGES)
    cap = (48 * 1024**2 if mode == 'full' else 1025 if mode == 'single'
           else sum(b - a + 1 for a, b in ranges) + 768 * 243 + 1024)
    request = dict(request_id='synthetic-' + mode, url=BASE + ('ir.png' if mode == 'full' else 'result.ppm'),
                   mode=mode, ranges=ranges, reserved_cap_bytes=cap)
    if mode == 'multipart':
        request['pin'] = dict(etag=ETAG, object_total_bytes=TOTAL)
    return request


def headers_for(mode, body):
    headers = {'ETag': ETAG, 'Content-Length': str(len(body))}
    if mode == 'single':
        headers['Content-Range'] = f'bytes 0-1023/{TOTAL}'
    elif mode == 'multipart':
        headers['Content-Type'] = 'multipart/byteranges; boundary=' + BOUNDARY
    return headers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    require(not root.exists(), 'Fresh fixture output required')
    root.mkdir(parents=True)
    results, clients = [], []
    attempted_network = []

    def forbidden_network(*args, **kwargs):
        attempted_network.append(True)
        raise AssertionError('Real network forbidden in synthetic protocol fixtures')

    original_network = (socket.create_connection, socket.socket.connect, socket.socket.connect_ex)
    socket.create_connection = socket.socket.connect = socket.socket.connect_ex = forbidden_network

    def passed(name, condition=True):
        require(condition, 'Synthetic fixture failed: ' + name)
        results.append(dict(name=name, status='PASS'))

    def rejected(name, callback):
        try:
            callback()
        except ContractError as error:
            results.append(dict(name=name, status='PASS', expected_rejection=str(error)))
            return
        raise AssertionError('Expected rejection: ' + name)

    def client_for(response, mode, frozen=None, limits=None):
        request = request_for(mode) if frozen is None else frozen
        folder = root / ('http-' + str(len(clients) + 1).zfill(3))
        opener = Opener(response, folder, request)
        budget = dict(attempts=1, body_bytes=request['reserved_cap_bytes'],
                      per_response=request['reserved_cap_bytes']) if limits is None else limits
        client = AcquisitionClient(folder, [request], budget, {'synthetic_fixture': True}, opener=opener)
        clients.append(client)
        return client, opener, request

    def fail_http(name, mode, response, before_body=True, retained=None):
        client, opener, request = client_for(response, mode)
        rejected(name, lambda: client.get(request))
        entry = client.ledger['requests'][0]
        passed(name + ': closed and full reservation retained', response.closed and len(opener.calls) == 1
               and not entry['success'] and client.ledger['charged_response_bytes'] == request['reserved_cap_bytes'])
        if before_body:
            passed(name + ': rejected before body read', response.read_calls == 0
                   and client.ledger['retained_body_bytes'] == 0 and not list((client.root / 'raw').iterdir()))
        if retained is not None:
            partial = client.root / 'raw' / (request['request_id'] + '.bin.partial')
            passed(name + ': exact partial evidence', partial.read_bytes() == retained
                   and entry['retained_body_bytes'] == len(retained)
                   and client.ledger['retained_body_bytes'] == len(retained)
                   and entry['body_sha256'] == hashlib.sha256(retained).hexdigest()
                   and not (client.root / 'raw' / (request['request_id'] + '.bin')).exists())

    try:
        body = multipart()
        content_type = 'multipart/byteranges; boundary=' + BOUNDARY
        parsed = parse_multipart(body, content_type, RANGES, TOTAL)
        passed('243 binary payloads preserve embedded delimiter and terminal delimiter tokens',
               len(parsed) == 243 and all(
                   body[p['payload_offset']:p['payload_offset'] + p['payload_bytes']] == payload(p['range_inclusive'])
                   and p['payload_sha256'] == hashlib.sha256(payload(p['range_inclusive'])).hexdigest()
                   for p in parsed))
        reverse_body = multipart(reverse=True, prefix=b'\r\n')
        reverse_parts = parse_multipart(reverse_body, content_type, RANGES, TOTAL)
        passed('243 reordered parts and one leading CRLF preserve exact addressing',
               [p['range_inclusive'] for p in reverse_parts] == list(reversed(RANGES)))
        quoted = multipart(boundary='fixture=a')
        passed('quoted special-character boundary accepted',
               len(parse_multipart(quoted, 'multipart/byteranges; boundary="fixture=a"', RANGES, TOTAL)) == 243)
        rejected('unquoted special-character boundary',
                 lambda: parse_multipart(quoted, 'multipart/byteranges; boundary=fixture=a', RANGES, TOTAL))
        parser_cases = [
            ('242 requested parts', body, RANGES[:-1], TOTAL),
            ('244 requested parts', body, RANGES + [[500000, 500143]], TOTAL),
            ('duplicate requested span', body, RANGES[:-1] + [RANGES[0]], TOTAL),
            ('overlapping requested spans', body, [RANGES[0], [RANGES[0][0] + 1, RANGES[0][1] + 1]] + RANGES[2:], TOTAL),
            ('missing returned part', multipart(RANGES[:-1]), RANGES, TOTAL),
            ('duplicate returned part', multipart(RANGES[:-1] + [RANGES[0]]), RANGES, TOTAL),
            ('coalesced returned part', multipart([[RANGES[0][0], RANGES[1][1]]] + RANGES[2:]), RANGES, TOTAL),
            ('unrequested returned part', multipart([[500000, 500143]] + RANGES[1:]), RANGES, TOTAL),
            ('wrong object total', multipart(total=TOTAL + 1), RANGES, TOTAL),
            ('truncated payload or final delimiter', body[:-20], RANGES, TOTAL),
            ('trailing byte', body + b'x', RANGES, TOTAL),
            ('arbitrary preamble', b'preamble' + body, RANGES, TOTAL),
            ('two leading CRLF', b'\r\n\r\n' + body, RANGES, TOTAL),
            ('oversized MIME header', body.replace(b'application/octet-stream', b'x' * 513, 1), RANGES, TOTAL),
            ('duplicate MIME header', body.replace(b'Content-Length: 144', b'Content-Length: 144\r\ncontent-length: 144', 1), RANGES, TOTAL),
            ('MIME length mismatch', body.replace(b'Content-Length: 144', b'Content-Length: 143', 1), RANGES, TOTAL),
            ('MIME transfer encoding', body.replace(b'Content-Type:', b'Content-Transfer-Encoding: base64\r\nContent-Type:', 1), RANGES, TOTAL),
            ('non-ASCII MIME header', body.replace(b'application/octet-stream', b'\xff', 1), RANGES, TOTAL),
            ('MIME control character', body.replace(b'application/octet-stream', b'app\x00type', 1), RANGES, TOTAL),
            ('framing allowance overflow', body + b'x' * (768 * 243 + 1024), RANGES, TOTAL),
        ]
        for name, data, ranges, total in parser_cases:
            rejected('multipart parser: ' + name,
                     lambda data=data, ranges=ranges, total=total: parse_multipart(data, content_type, ranges, total))

        for mode, data in [('full', IR), ('single', PREFIX), ('multipart', body)]:
            status = 200 if mode == 'full' else 206
            headers = headers_for(mode, data)
            response = Response(data, status, headers)
            client, opener, request = client_for(response, mode)
            path, entry = client.get(request)
            sent = opener.calls[0]
            passed(mode + ': complete body and actual-byte accounting', path.read_bytes() == data
                   and entry['success'] and response.closed and client.ledger['charged_response_bytes'] == len(data)
                   and client.ledger['retained_body_bytes'] == len(data)
                   and entry['body_sha256'] == hashlib.sha256(data).hexdigest())
            passed(mode + ': exact GET request headers', sent.get_header('Accept-encoding') == 'identity'
                   and sent.get_header('Range') == (None if mode == 'full' else 'bytes=' + ','.join(f'{a}-{b}' for a, b in request['ranges']))
                   and sent.get_header('If-match') == (ETAG if mode == 'multipart' else None))
            if mode == 'full':
                passed('full IR: independent IR retained-byte accounting', client.ledger['ir_retained_body_bytes'] == len(IR))
            elif mode == 'multipart':
                passed('multipart HTTP: exactly 243 verified parts', len(entry['parts']) == 243)
            rejected(mode + ': retry of completed frozen request forbidden', lambda: client.get(request))
            passed(mode + ': retry consumes no additional attempt', len(opener.calls) == len(client.ledger['requests']) == 1)

            bad_headers = [
                ('missing ETag', {k: v for k, v in headers.items() if k != 'ETag'}),
                ('weak ETag', {**headers, 'ETag': 'W/' + ETAG}),
                ('unquoted ETag', {**headers, 'ETag': 'stable'}),
                ('missing Content-Length', {k: v for k, v in headers.items() if k != 'Content-Length'}),
                ('negative Content-Length', {**headers, 'Content-Length': '-1'}),
                ('zero Content-Length', {**headers, 'Content-Length': '0'}),
                ('nondecimal Content-Length', {**headers, 'Content-Length': '1.5'}),
                ('declared cap overflow', {**headers, 'Content-Length': str(request['reserved_cap_bytes'] + 1)}),
                ('content encoding', {**headers, 'Content-Encoding': 'gzip'}),
                ('transfer encoding', {**headers, 'Transfer-Encoding': 'chunked'}),
            ]
            if mode == 'single':
                bad_headers += [('wrong prefix range', {**headers, 'Content-Range': f'bytes 1-1024/{TOTAL}'}),
                                ('invalid prefix total', {**headers, 'Content-Range': 'bytes 0-1023/1023'}),
                                ('wrong prefix length', {**headers, 'Content-Length': '1023'}),
                                ('missing prefix range', {k: v for k, v in headers.items() if k != 'Content-Range'})]
            else:
                bad_headers.append(('unexpected top-level range', {**headers, 'Content-Range': f'bytes 0-1023/{TOTAL}'}))
            if mode == 'multipart':
                bad_headers.append(('changed pinned ETag', {**headers, 'ETag': '"changed"'}))
            for name, modified in bad_headers:
                fail_http(mode + ': ' + name, mode, Response(data, status, modified))
            for key in ('ETag', 'Content-Length', 'Content-Range', 'Content-Type', 'Content-Encoding', 'Transfer-Encoding'):
                duplicate = list(headers.items())
                if key not in headers:
                    duplicate.append((key, 'identity'))
                duplicate.append((key.lower(), headers.get(key, 'identity')))
                fail_http(mode + ': duplicate ' + key, mode, Response(data, status, duplicate))
            for bad_status in ([206, 301, 404] if mode == 'full' else [200, 302, 416]):
                fail_http(mode + ': HTTP ' + str(bad_status), mode, Response(data, bad_status, headers))

        prefix_headers = headers_for('single', PREFIX)
        fail_http('prefix: clean EOF truncation', 'single', Response(PREFIX[:333], headers=prefix_headers),
                  before_body=False, retained=PREFIX[:333])
        fail_http('prefix: IncompleteRead preserves partial bytes', 'single',
                  Response(PREFIX, headers=prefix_headers, incomplete=PREFIX[:137]), before_body=False, retained=PREFIX[:137])
        fail_http('prefix: reader exceeding requested count rejected', 'single',
                  Response(PREFIX, headers=prefix_headers, overread=True), before_body=False, retained=b'')
        fail_http('full IR: clean EOF truncation', 'full', Response(IR[:-1], 200, headers_for('full', IR)),
                  before_body=False, retained=IR[:-1])
        for name, data in [('changed object total', multipart(total=TOTAL + 1)),
                           ('missing returned part', multipart(RANGES[:-1])),
                           ('duplicate returned part', multipart(RANGES[:-1] + [RANGES[0]])),
                           ('coalesced returned part', multipart([[RANGES[0][0], RANGES[1][1]]] + RANGES[2:]))]:
            fail_http('multipart HTTP: ' + name, 'multipart', Response(data, headers=headers_for('multipart', data)),
                      before_body=False, retained=data)
        frozen = request_for('single')
        response = Response(PREFIX, headers=prefix_headers)
        client, opener, _ = client_for(response, 'single', frozen=frozen)
        rejected('modified frozen request forbidden before reservation', lambda: client.get({**frozen, 'request_id': 'changed'}))
        passed('modified request spends no attempt or body budget', not opener.calls and not client.ledger['requests']
               and client.ledger['charged_response_bytes'] == 0)
        budget = dict(attempts=1, body_bytes=1024, per_response=1025)
        client, opener, frozen = client_for(Response(PREFIX, headers=prefix_headers), 'single', limits=budget)
        rejected('insufficient reservation budget forbidden before open', lambda: client.get(frozen))
        passed('exhausted budget spends no attempt', not opener.calls and not client.ledger['requests'])
        passed('no real network attempted', not attempted_network)
        passed('all HTTP ledgers retain zero redirects and automatic retries', all(
            c.ledger['redirects_followed'] == c.ledger['automatic_retries'] == 0 for c in clients))
        summary = dict(schema_version=1, status='PASS', synthetic_only=True, network_attempts=len(attempted_network),
                       scientific_decoding=False, source_image_or_ppm_content_read=False,
                       checked_code=[record(Path(__file__).resolve().parent / name) for name in
                                     ('check_protocol_fixtures.py', 'acquisition_common.py', 'regional_http.py')],
                       checks=len(results), http_fixture_directories=len(clients), results=results,
                       ledgers=[record(c.root / 'request-ledger.json') for c in clients])
        save(root / 'fixture-results.json', summary)
        print(json.dumps(dict(status='PASS', checks=len(results), http_fixture_directories=len(clients),
                              network_attempts=0, summary=str(root / 'fixture-results.json'))))
    finally:
        socket.create_connection, socket.socket.connect, socket.socket.connect_ex = original_network


if __name__ == '__main__':
    main()
