"""Bounded public metadata acquisition only; no volume/image chunks."""
from __future__ import annotations
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).parent
REPORTS = Path('/workspace/herculaneum-lab/reports/native-reference-preflight-20261008')
BASE = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
ARCHIVE = Path('/workspace/herculaneum-lab/data/experiments/pherc0800-native-ct-consistency/metadata.full.json.gz')
BUDGET = 15 * 1024 * 1024  # Reserve separate allowances for the other researchers.
records = []
wire_bytes = 0


def retain(name: str, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    target = ROOT / ('source-' + digest + '-' + name)
    if target.exists():
        if target.read_bytes() != data:
            raise RuntimeError('Immutable evidence collision')
    else:
        target.write_bytes(data)
        target.chmod(0o444)
    return str(target)


def request(name: str, url: str, *, method: str = 'GET', extra_headers=None):
    global wire_bytes
    headers = {'User-Agent': 'HerculaneumResearch/1.0'}
    headers.update(extra_headers or {})
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(req, timeout=25)
    except urllib.error.HTTPError as exc:
        if exc.code != 304:
            record = {'url': url, 'method': method, 'http_status': exc.code,
                      'etag': exc.headers.get('ETag'), 'payload_bytes': 0}
            records.append(record)
            return None, record
        response = exc
    with response:
        if method == 'HEAD' or response.code == 304:
            raw = b''
        else:
            limit = min(BUDGET - wire_bytes, 2 * 1024 * 1024)
            raw = response.read(limit + 1)
            wire_bytes += len(raw)
            if len(raw) > limit:
                raise RuntimeError('Metadata response exceeded the bounded budget')
        record = {'url': url, 'method': method, 'http_status': response.code,
                  'etag': response.headers.get('ETag'),
                  'last_modified': response.headers.get('Last-Modified'),
                  'content_length': response.headers.get('Content-Length'),
                  'content_encoding': response.headers.get('Content-Encoding'),
                  'payload_bytes': len(raw)}
    if raw:
        record['sha256'] = hashlib.sha256(raw).hexdigest()
        record['retained_path'] = retain(name, raw)
    records.append(record)
    if not raw:
        return None, record
    decoded = gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw
    record['decoded_sha256'] = hashlib.sha256(decoded).hexdigest()
    return json.loads(decoded), record


ROOT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
archive_raw = ARCHIVE.read_bytes()
archive = json.loads(gzip.decompress(archive_raw))
_, full_head = request('metadata-head', BASE + 'metadata.json', method='HEAD')
updated, conditional = request('metadata-full.json.gz', BASE + 'metadata.json',
                              extra_headers={'If-None-Match': '"b4788ae14c64e13e57346c07c3f0b8a5"'})
if updated is not None:
    metadata = updated
elif conditional['http_status'] == 304:
    metadata = archive
else:
    raise RuntimeError('Could not revalidate or replace archived public metadata')
sample = metadata['samples']['PHerc1667']
sample_path = retain('pherc1667-metadata-extract.json', (json.dumps(sample, indent=2) + '\n').encode())

surface_rows = []
for sid, segment in sample['segments'].items():
    for data in segment['data']:
        if data['type'] != 'layers-zarr' or data.get('parameters', {}).get('target_volume') != '20251217075048':
            continue
        path = data['origins'][0]['path']
        url = BASE + path
        group_attrs, attrs_record = request(sid + '-native-zattrs.json', url + '.zattrs')
        level0, array_record = request(sid + '-native-zarray.json', url + '0/.zarray')
        mesh_meta = []
        for mesh in segment['data']:
            if mesh['type'] != 'tifxyz':
                continue
            if mesh.get('parameters', {}).get('target_volume') != '20251217075048':
                continue
            mesh_path = mesh['origins'][0]['path']
            meta, meta_record = request(sid + '-mesh-meta.json', BASE + mesh_path + 'meta.json')
            mesh_meta.append({'metadata': meta, 'source': meta_record})
        row = {'segment_id': sid, 'long_id': segment['long_id'], 'public_surface_url': url,
               'surface_parameters': data.get('parameters'),
               'surface_creation': data.get('creation_info'),
               'surface_origins': data['origins'], 'level0_array': level0,
               'group_attrs': group_attrs, 'metadata_sources': [attrs_record, array_record],
               'mesh_metadata': mesh_meta,
               'paper_alias': None,
               'paper_alias_identity_verified': False}
        surface_rows.append(row)

volume = sample['volumes']['20251217075048']
volume_path = next(o['path'] for d in volume['data'] for o in d.get('origins', [])
                   if d['type'] == 'ome-zarr')
volume_attrs, vattrs_record = request('volume-zattrs.json', BASE + volume_path + '.zattrs')
volume_array, varray_record = request('volume-zarray.json', BASE + volume_path + '0/.zarray')
result = {'schema_version': 1, 'kind': 'source-metadata-only-preflight',
          'retrieved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'experiment_id': 'pherc1667-native-reference-v1',
          'network_payload_bytes': wire_bytes, 'network_payload_cap_bytes': BUDGET,
          'archive_path': str(ARCHIVE), 'archive_sha256': hashlib.sha256(archive_raw).hexdigest(),
          'public_metadata_revalidation': [full_head, conditional],
          'sample_extract_path': sample_path,
          'native_volume': {'metadata': volume, 'zattrs': volume_attrs, 'level0_array': volume_array,
                            'metadata_sources': [vattrs_record, varray_record]},
          'surface_inventory': surface_rows, 'network_records': records,
          'source_array_chunks_acquired': 0, 'weights_acquired': 0,
          'frozen_regions': None, 'paper_alias_mapping': {'l_2': None, 'l_5': None}}
encoded = (json.dumps(result, indent=2) + '\n').encode()
manifest_path = retain('metadata-manifest.json', encoded)
(REPORTS / 'source-metadata-manifest-location.json').write_text(json.dumps({
    'path': manifest_path, 'sha256': hashlib.sha256(encoded).hexdigest(),
    'network_payload_bytes': wire_bytes}, indent=2) + '\n')
print(json.dumps({'manifest_path': manifest_path, 'network_payload_bytes': wire_bytes,
                  'native_surface_count': len(surface_rows)}, indent=2))
for row in surface_rows:
    print(row['segment_id'], row['long_id'].split('-')[1].split('_')[0], row['level0_array'])
