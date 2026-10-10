"""Acquire frozen post hoc 768 context, with verified conditional-HEAD cache reuse."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import requests
from urllib3.exceptions import HTTPError as StreamHTTPError

BASE = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/'
        '20251028220955-auto_grown_20251028220955262/surface-volumes/'
        '8.64um-1.2m-116keV-volume-20250521135224.zarr')
CHUNK_BYTES = 31 * 128 * 128
PAYLOAD_LIMIT = 27 * CHUNK_BYTES
OVERHEAD_LIMIT = 128 * 1024
TOTAL_LIMIT = PAYLOAD_LIMIT + OVERHEAD_LIMIT
CONTRACT = dict(new_chunk_gets=27, metadata_gets=2, cache_validation_heads=9,
                maximum_total_requests=38, max_new_payload_bytes=PAYLOAD_LIMIT,
                max_response_body_bytes=TOTAL_LIMIT)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_inputs(records):
    for record in records:
        path = Path(record['path'])
        require(path.stat().st_size == record['size_bytes'] and sha(path) == record['sha256'],
                'Frozen input SHA/size mismatch: ' + path.name)


def validate_plan(path, expected_sha):
    require(sha(path) == expected_sha, 'Acquisition plan SHA mismatch')
    plan = read_json(path)
    verify_inputs(plan['verified_inputs'])
    require(plan['schema_version'] == 2 and plan['sample_id'] == 'PHerc0800'
            and plan['target_volume_id'] == '20250521135224'
            and plan['segment_id'] == '20251028220955', 'Wrong frozen source identity')
    require(plan['surface_url'] == BASE and plan['source_array_url'] == BASE + '/0', 'Source prefix differs')
    require(plan['bbox_source_xywh'] == [768, 256, 768, 768]
            and plan['shape_zyx'] == [31, 768, 768], 'Frozen crop differs')
    require(plan['maximum_chunk_payload_bytes'] == 36 * CHUNK_BYTES
            and plan['maximum_new_chunk_payload_bytes'] == PAYLOAD_LIMIT
            and plan['maximum_all_remote_response_bytes_including_metadata'] == TOTAL_LIMIT
            and plan['maximum_remote_request_count'] == 29 and plan['automatic_retries'] == 0,
            'Frozen GET/payload budget differs')
    meta, attrs = read_json(plan['array_metadata_path']), read_json(plan['attrs_path'])
    require(meta['zarr_format'] == 2 and meta['shape'] == [31, 2100, 2100]
            and meta['chunks'] == [31, 128, 128] and meta['dtype'] == '|u1'
            and meta['order'] == 'C' and meta.get('compressor') is None
            and not meta.get('filters') and meta['fill_value'] == 0
            and meta.get('dimension_separator') == '/', 'Unexpected source representation')
    x, y, w, h = plan['bbox_source_xywh']
    mask = np.load(plan['geometry_mask_path'], mmap_mode='r', allow_pickle=False)
    xyz = np.load(plan['coordinate_map_path'], mmap_mode='r', allow_pickle=False)
    require(mask.dtype == np.bool_ and mask.shape == (2100, 2100), 'Wrong global geometry mask')
    require(xyz.dtype == np.float32 and xyz.shape == (2100, 2100, 3), 'Wrong global CT map')
    roi_mask, roi_xyz = np.asarray(mask[y:y+h, x:x+w]), np.asarray(xyz[y:y+h, x:x+w])
    require(np.isfinite(roi_xyz[roi_mask]).all(), 'Valid geometry lacks finite coordinates')
    selected_mask = np.load(plan['selected_geometry_mask']['path'], allow_pickle=False)
    selected_xyz = np.load(plan['selected_base_ct_coordinates']['path'], allow_pickle=False)
    require(np.array_equal(roi_mask, selected_mask)
            and np.array_equal(roi_xyz, selected_xyz, equal_nan=True), 'Frozen selected geometry differs')
    require(float(roi_mask.mean()) >= .95, 'Whole crop geometry fraction below frozen policy')
    previous_path = Path(plan['reuse_source_provenance_path'])
    require(sha(previous_path) == plan['reuse_source_provenance_sha256'], 'Prior source provenance changed')
    previous = read_json(previous_path)
    require(previous['surface_url'] == BASE and previous['bbox_source_xywh'] == [1024, 384, 384, 384],
            'Wrong prior source crop')
    previous_chunks = {tuple(r['chunk_index_zyx']): r for r in previous['source_chunks']}
    indices = [[0, r, c] for r in range(2, 8) for c in range(6, 12)]
    require([r['chunk_index_zyx'] for r in plan['source_chunks']] == indices, 'Wrong chunk schedule')
    cached = {}
    for record in plan['source_chunks']:
        index = tuple(record['chunk_index_zyx'])
        require(record['index_zyx'] == list(index) and record['url'] == BASE + '/0/' + '/'.join(map(str, index))
                and record['expected_payload_bytes'] == CHUNK_BYTES
                and record['expected_uncompressed_payload_bytes'] == CHUNK_BYTES, 'Chunk source contract differs')
        if record['acquisition_action'] == 'reuse_verified_observed_source':
            reuse = record['reuse']
            original = previous_chunks[index]
            original_path = (previous_path.parent / original['local_file']).resolve()
            require(original['http_status'] == reuse['original_http_status'] == 200
                    and original['url'] == reuse['original_url'] == record['url']
                    and original['etag'] == reuse['original_etag'] and bool(original['etag'])
                    and original['sha256'] == reuse['sha256']
                    and original['size_bytes'] == reuse['size_bytes'] == CHUNK_BYTES
                    and Path(reuse['path']).resolve() == original_path,
                    'Reuse record differs from prior observed source')
            require(original_path.stat().st_size == CHUNK_BYTES and sha(original_path) == reuse['sha256'],
                    'Prior cached payload SHA/size mismatch')
            cached[index] = (original_path, original)
        else:
            require(record['acquisition_action'] == 'download_new_observed_or_explicit_NoSuchKey_fill'
                    and 'reuse' not in record, 'Unknown acquisition action')
    require(len(cached) == 9 and len(plan['source_chunks']) == 36, 'Cache/new object counts differ')
    return plan, meta, attrs, roi_mask, roi_xyz, cached, previous


def validate_decision(path, expected_sha, plan_path, plan_sha):
    require(sha(path) == expected_sha, 'Parent context decision SHA mismatch')
    decision = read_json(path)
    require(decision['schema_version'] == 1 and decision['allow_context_acquisition'] is True
            and decision['allow_paid_compute'] is False, 'Context acquisition gate has not passed')
    require(Path(decision['context_plan']['path']).resolve() == plan_path.resolve()
            and decision['context_plan']['sha256'] == plan_sha, 'Decision references another context plan')
    require(decision['technical_mapping']['passed'] is True
            and decision['visual_review']['passed_for_diagnostic_context'] is True, 'Mapping/visual gate missing')
    for role in ['technical_mapping', 'visual_review']:
        require(sha(decision[role]['path']) == decision[role]['sha256'], 'Decision evidence SHA mismatch')
    require(decision['acquisition_contract'] == CONTRACT, 'Decision must explicitly supersede 29 GET with 38 GET/HEAD requests')
    return decision


class BoundedHTTP:
    def __init__(self, directory):
        self.directory = directory
        self.session = requests.Session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
        self.payload_bytes = self.overhead_bytes = self.requests = 0
        self.records = []

    def begin(self, url, method):
        require(url == BASE + '/.zattrs' or url == BASE + '/0/.zarray'
                or url.startswith(BASE + '/0/0/'), 'Unexpected public source URL')
        self.requests += 1
        require(self.requests <= 38, 'Maximum remote request count exceeded')
        record = dict(method=method, url=url, retrieved_at_utc=datetime.now(timezone.utc).isoformat())
        self.records.append(record)
        return record

    def head(self, url, etag, filename):
        record = self.begin(url, 'HEAD')
        with self.session.head(url, headers={'If-Match': etag, 'Accept-Encoding': 'identity'},
                               stream=True, allow_redirects=False, timeout=(10, 45)) as response:
            record.update(http_status=response.status_code, request_if_match=etag,
                          etag=response.headers.get('ETag'), content_length=response.headers.get('Content-Length'),
                          response_body_bytes=0)
            require(response.status_code == 200, 'Conditional cache HEAD failed; no cache fallback')
            require(record['etag'] == etag and record['content_length'] == str(CHUNK_BYTES),
                    'Cached source ETag or Content-Length changed')
            record['content_length'] = int(record['content_length'])
            write_json(self.directory / filename, record)
        return record

    def get(self, url, filename, allow_missing=False):
        record = self.begin(url, 'GET')
        with self.session.get(url, headers={'Accept-Encoding': 'identity'}, stream=True,
                              allow_redirects=False, timeout=(10, 45)) as response:
            record.update(http_status=response.status_code, etag=response.headers.get('ETag'),
                          last_modified=response.headers.get('Last-Modified'), local_file=filename)
            require(response.headers.get('Content-Encoding', 'identity') in ['', 'identity'], 'Unsupported response encoding')
            payload = allow_missing and response.status_code == 200
            left = PAYLOAD_LIMIT - self.payload_bytes if payload else OVERHEAD_LIMIT - self.overhead_bytes
            cap = min(CHUNK_BYTES if payload else 65536, left)
            length_header = response.headers.get('Content-Length')
            known_length = length_header is not None
            if known_length:
                require(length_header.isdecimal() and int(length_header) <= cap,
                        'Response Content-Length exceeds frozen byte bound')
                length = int(length_header)
            else:
                require(response.status_code != 200, 'Observed source lacks Content-Length')
                length = cap
            body = bytearray()
            complete_eof = False
            while len(body) < length:
                block = response.raw.read(min(16384, length - len(body)), decode_content=False)
                if not block:
                    complete_eof = True
                    break
                body.extend(block)
                if payload:
                    self.payload_bytes += len(block)
                else:
                    self.overhead_bytes += len(block)
            data = bytes(body)
            record.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                          response_body_bytes=len(data), downloaded_payload_bytes=len(data) if payload else 0,
                          reused_payload_bytes=0)
            if response.status_code != 200:
                error_file = filename + '.http-error.xml'
                (self.directory / error_file).write_bytes(data)
                record['error_record_file'] = error_file
                require((known_length and len(body) == length) or (not known_length and complete_eof),
                        'Error body truncated or reached byte cap before EOF; no fill')
                require(response.status_code == 404 and allow_missing, 'HTTP failure; no generic error-to-fill fallback')
                require(b'<!DOCTYPE' not in data.upper() and b'<!ENTITY' not in data.upper(), 'Unsupported XML declaration')
                root = ET.fromstring(data)
                require(root.tag == 'Error' and root.findtext('Code') == 'NoSuchKey', '404 was not explicit S3 NoSuchKey')
                record.update(source_absence='Explicit S3 NoSuchKey', reconstruction='Declared Zarr v2 fill_value=0',
                              local_payload_role='reconstructed fill', observed_payload_bytes=0,
                              acquisition_action='new_explicit_NoSuchKey_fill')
                data = bytes(CHUNK_BYTES)
                record['reconstructed_sha256'] = hashlib.sha256(data).hexdigest()
            else:
                require(len(body) == length, 'Truncated observed response; no retry')
                record.update(local_payload_role='observed source', observed_payload_bytes=len(data),
                              acquisition_action='new_chunk_get' if payload else 'new_metadata_revalidation')
                etag = (record['etag'] or '').strip('"')
                if len(etag) == 32 and all(c in '0123456789abcdefABCDEF' for c in etag):
                    require(hashlib.md5(data).hexdigest() == etag.lower(), 'ETag MD5 and source payload differ')
                    record['etag_matches_payload_md5'] = True
            (self.directory / filename).write_bytes(data)
            return data, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--inference-plan-sha256')
    parser.add_argument('--decision', type=Path)
    parser.add_argument('--decision-sha256')
    args = parser.parse_args()
    plan, meta, attrs, roi_mask, roi_xyz, cached, previous = validate_plan(args.plan, args.plan_sha256)
    if args.check_only:
        print(json.dumps(dict(offline_verified=True, geometry_valid_pixels=int(roi_mask.sum()),
            cache_chunks=len(cached), new_chunks=27, decoded_bytes=36*CHUNK_BYTES,
            acquisition_contract=CONTRACT)))
        return
    require(args.decision is not None and args.decision_sha256 is not None
            and args.inference_plan_sha256 is not None, 'Frozen parent decision and inference-plan hashes required')
    decision = validate_decision(args.decision, args.decision_sha256, args.plan, args.plan_sha256)
    inference_path = Path(__file__).with_name('inference-plan.json')
    require(sha(inference_path) == args.inference_plan_sha256, 'Inference plan SHA mismatch')
    inference = read_json(inference_path)
    require(inference['acquisition_plan']['sha256'] == args.plan_sha256
            and Path(inference['acquisition_plan']['path']).resolve() == args.plan.resolve(), 'Inference source plan differs')
    verify_inputs(inference['verified_inputs'])
    require(not args.output_dir.exists(), 'Preserve existing or partial acquisition; choose a fresh output')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'frozen-plan.json').write_bytes(args.plan.read_bytes())
    (args.output_dir / 'frozen-inference-plan.json').write_bytes(inference_path.read_bytes())
    http = BoundedHTTP(args.output_dir)
    records, metadata_records, retained_cache = [], [], {}
    try:
        prior_metadata = {r['url']: r for r in previous['metadata_sources']}
        for suffix, path, expected, filename in [('/0/.zarray', plan['array_metadata_path'], meta, 'source.zarray.json'),
                                                  ('/.zattrs', plan['attrs_path'], attrs, 'source.zattrs.json')]:
            data, record = http.get(BASE + suffix, filename)
            require(hashlib.sha256(data).hexdigest() == sha(path) and json.loads(data) == expected,
                    'Current source metadata bytes changed after freeze')
            record['original_cached_metadata_record'] = prior_metadata[record['url']]
            metadata_records.append(record)
        # Validate every cached object before downloading any new source chunk.
        for index, (path, original) in sorted(cached.items()):
            filename = 'source-chunk-' + '-'.join(map(str, index)) + '.bin'
            head_name = 'head-' + filename + '.json'
            head = http.head(original['url'], original['etag'], head_name)
            require(sha(path) == original['sha256'], 'Cache bytes changed after local validation')
            data = path.read_bytes()
            require(len(data) == CHUNK_BYTES, 'Cache payload length changed')
            (args.output_dir / filename).write_bytes(data)
            retained_cache[index] = dict(original, local_file=filename, acquisition_action='reuse_verified_observed_source',
                downloaded_payload_bytes=0, reused_payload_bytes=CHUNK_BYTES, response_body_bytes=0,
                original_source_record=original, cache_source_path=str(path), conditional_head=head,
                head_record_file=head_name, head_record_sha256=sha(args.output_dir / head_name))
        surface = np.empty((31, 768, 768), dtype=np.uint8)
        observed = np.zeros((768, 768), dtype=bool)
        x, y, w, h = plan['bbox_source_xywh']
        for item in plan['source_chunks']:
            index = tuple(item['chunk_index_zyx'])
            filename = 'source-chunk-' + '-'.join(map(str, index)) + '.bin'
            if index in retained_cache:
                record = retained_cache[index]
                data = (args.output_dir / filename).read_bytes()
            else:
                data, record = http.get(item['url'], filename, allow_missing=True)
            require(len(data) == CHUNK_BYTES, 'Source chunk length differs from metadata')
            _, row, col = index
            dy, dx = row * 128 - y, col * 128 - x
            surface[:, dy:dy+128, dx:dx+128] = np.frombuffer(data, dtype=np.uint8).reshape(31, 128, 128)
            observed[dy:dy+128, dx:dx+128] = record['http_status'] == 200
            record['chunk_index_zyx'] = list(index)
            records.append(record)
        outputs = {'surface.npy': surface, 'geometry-mask.npy': roi_mask,
                   'ct-coordinates.xyz.npy': roi_xyz, 'observed-mask.npy': observed}
        for filename, values in outputs.items():
            np.save(args.output_dir / filename, values, allow_pickle=False)
        verify_inputs(plan['verified_inputs'])
        verify_inputs(inference['verified_inputs'])
        require(sha(args.plan) == args.plan_sha256 and sha(inference_path) == args.inference_plan_sha256
                and sha(args.decision) == args.decision_sha256, 'Frozen plan or decision changed during acquisition')
        validate_decision(args.decision, args.decision_sha256, args.plan, args.plan_sha256)
        require(http.requests == 38 and http.payload_bytes <= PAYLOAD_LIMIT
                and http.overhead_bytes <= OVERHEAD_LIMIT, 'Acquisition request/body accounting differs')
        provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
            sample_id='PHerc0800', target_volume_id='20250521135224', segment_id=plan['segment_id'],
            surface_url=BASE, pyramid_level=0, bbox_source_xywh=[x,y,w,h], shape_zyx=list(surface.shape),
            dtype='uint8', voxel_spacing_zyx_um=[8.64]*3, nominal_canvas_size_mm_xy=[768*.00864]*2,
            array_sha256=sha(args.output_dir/'surface.npy'), frozen_plan_path=str(args.plan.resolve()),
            frozen_plan_sha256=args.plan_sha256, verified_inputs=plan['verified_inputs'],
            render_provenance=plan['render_provenance'], metadata_sources=metadata_records, source_chunks=records,
            maximum_chunk_payload_bytes=36*CHUNK_BYTES, maximum_new_chunk_payload_bytes=PAYLOAD_LIMIT,
            downloaded_chunk_payload_bytes=http.payload_bytes, reused_chunk_payload_bytes=9*CHUNK_BYTES,
            decoded_chunk_payload_bytes=36*CHUNK_BYTES, total_response_body_bytes=http.payload_bytes+http.overhead_bytes,
            maximum_response_body_bytes=TOTAL_LIMIT, overhead_response_body_bytes=http.overhead_bytes,
            prior_reused_payload_response_bytes=9*CHUNK_BYTES,
            prior_cached_metadata_response_bytes=sum(r['size_bytes'] for r in previous['metadata_sources']),
            prior_original_response_body_bytes=previous['total_response_body_bytes'],
            automatic_retries=0, request_count=http.requests, get_request_count=29, head_request_count=9,
            head_response_body_bytes=0, acquisition_contract=CONTRACT,
            contract_supersession='Frozen v2 plan retains 29 GET requests; parent decision explicitly adds 9 conditional cache HEAD requests, for 38 total with unchanged payload/body budgets.',
            context_decision_path=str(args.decision.resolve()), context_decision_sha256=args.decision_sha256,
            inference_plan_path=str(inference_path.resolve()), inference_plan_sha256=args.inference_plan_sha256,
            reuse_source_provenance_path=plan['reuse_source_provenance_path'],
            reuse_source_provenance_sha256=plan['reuse_source_provenance_sha256'],
            missing_chunks_reconstructed_as_declared_fill=sum(r['http_status']==404 for r in records),
            geometry_valid_pixels=int(roi_mask.sum()), observed_source_pixels=int(observed.sum()),
            derived_outputs={name:dict(sha256=sha(args.output_dir/name),shape=list(a.shape),dtype=str(a.dtype))
                             for name,a in outputs.items()},
            acquisition_code_path=str(Path(__file__).resolve()), acquisition_code_sha256=sha(__file__),
            status='Post hoc context acquisition; no ink/text accuracy claim', physical_independence=plan['physical_independence'],
            letter=None, transcription=None, translation=None)
        write_json(args.output_dir/'provenance.json', provenance)
        print(json.dumps({k:provenance[k] for k in ['bbox_source_xywh','downloaded_chunk_payload_bytes',
            'reused_chunk_payload_bytes','total_response_body_bytes','request_count','missing_chunks_reconstructed_as_declared_fill']}))
    except (ValueError, OSError, requests.RequestException, StreamHTTPError, ET.ParseError) as exc:
        write_json(args.output_dir/'failure.json', dict(status='stopped_no_retry',
            exception_type=type(exc).__name__, requests=http.requests,
            payload_response_bytes=http.payload_bytes, overhead_response_bytes=http.overhead_bytes, request_records=http.records))
        raise RuntimeError('Acquisition stopped; retained failure record, no retry.') from None
    finally:
        http.session.close()


if __name__ == '__main__':
    main()
