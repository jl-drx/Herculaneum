"""Synthetic installed-Zarr parity and malformed-shard checks; no external I/O."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import struct

import google_crc32c
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BytesCodec
from zarr.core.buffer import default_buffer_prototype

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'indexed_shard.py'
RECEIPT = HERE / 'synthetic-fixture-receipt.json'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


async def shard_bytes(store):
    keys = sorted([key async for key in store.list()])
    require(keys == ['c/0/0', 'zarr.json'], 'Exactly one installed-Zarr shard')
    blob = (await store.get('c/0/0', prototype=default_buffer_prototype())).to_bytes()
    metadata = (await store.get('zarr.json', prototype=default_buffer_prototype())).to_bytes()
    return blob, json.loads(metadata)


def execute(decoder, receipt):
    cases = receipt['cases']
    store = zarr.storage.MemoryStore()
    array = zarr.create_array(store=store, shape=(7, 6), dtype=np.uint8, chunks=(4, 4),
        shards=(8, 8), serializer=BytesCodec(endian='little'), fill_value=0,
        compressors=[BloscCodec(typesize=1, cname='zstd', clevel=3,
                                shuffle='bitshuffle', blocksize=0)], dimension_names=['y', 'x'])
    values = (np.arange(7)[:, None] * 29 + np.arange(6)[None, :] * 7 + 3).astype(np.uint8)
    values[:4, 4:] = 0; values[1, 2] = 0
    array[:] = values
    blob, metadata = asyncio.run(shard_bytes(store))
    layout, entries, index_receipt = decoder.validate_shard_index(blob, metadata,
        maximum_encoded_bytes=len(blob), expected_sha256=sha(blob), expected_byte_length=len(blob))
    require(layout['grid'] == (2, 2) and entries[1] == (decoder.UINT64_MAX,) * 2,
            'C-order omitted top-right chunk')
    require(index_receipt['present_inner_chunks'] == 3, 'Exactly three encoded chunks')
    receipt['synthetic_source'] = dict(writer='installed zarr.create_array / MemoryStore',
        metadata=metadata, metadata_source='installed-Zarr-written zarr.json',
        byte_length=len(blob), sha256=sha(blob), index=index_receipt,
        values_sha256=sha(values.tobytes()), shard_not_written_to_disk=True)

    def decode(data=blob, meta=metadata, bbox=(0, 0, 6, 7), **options):
        parameters = dict(maximum_encoded_bytes=1024, maximum_expanded_bytes=1024)
        parameters.update(options)
        return decoder.decode_shard_roi(data, meta, bbox, **parameters)

    def reject(name, operation, reason):
        try:
            operation()
        except ValueError as error:
            require(reason in str(error), name + ': wrong rejection guard: ' + str(error))
            cases.append(dict(name=name, status='PASS', outcome='rejected', error=str(error)))
        else:
            raise AssertionError(name + ': malformed input accepted')

    expected_presence = np.ones((7, 6), dtype=bool); expected_presence[:4, 4:] = False
    for name, bbox in [('full_asymmetric_partial_edge', (0, 0, 6, 7)),
                       ('cross_inner_boundaries', (2, 2, 4, 5)),
                       ('omitted_fill_only', (4, 0, 2, 4)), ('bottom_right_edge', (4, 5, 2, 2))]:
        x, y, w, h = bbox; out, present, result = decode(bbox=bbox)
        standard = np.asarray(array[y:y+h, x:x+w])
        require(out.dtype == np.uint8 and present.dtype == np.bool_, 'Exact output types')
        require(np.array_equal(out, standard) and np.array_equal(standard, values[y:y+h, x:x+w]),
                name + ': installed-Zarr ROI equality')
        require(np.array_equal(present, expected_presence[y:y+h, x:x+w]), name + ': payload presence')
        require(result['output_sha256'] == sha(standard.tobytes()), 'ROI hash receipt')
        require(result['payload_presence_is_participation_or_geometry'] is False
                and result['coordinate_or_depth_transform_applied'] is False, 'Storage-only diagnostic')
        cases.append(dict(name=name, status='PASS', outcome='exact_zarr_equality', receipt=result))
    out, present, _ = decode()
    require(out[1, 2] == 0 and present[1, 2] and not present[1, 4], 'Encoded zero differs from omitted fill')
    cases.append(dict(name='zero_is_not_a_participation_mask', status='PASS'))
    decode(maximum_encoded_bytes=len(blob), maximum_expanded_bytes=148,
           expected_sha256=sha(blob), expected_byte_length=len(blob))
    cases.append(dict(name='exact_encoded_and_expansion_caps', status='PASS'))

    data_end = len(blob) - layout['index_bytes']
    def indexed(replacements, payload=blob[:data_end]):
        changed = list(entries)
        for position, value in replacements.items():
            changed[position] = value
        index = b''.join(struct.pack('<QQ', *entry) for entry in changed)
        return payload + index + struct.pack('<I', google_crc32c.value(index))

    fill_blob = indexed({i: (decoder.UINT64_MAX,) * 2 for i in range(4)}, payload=b'')
    out, present, result = decode(data=fill_blob)
    require(not out.any() and not present.any() and result['canonical_fill_pixels'] == 42,
            'CRC-valid all-missing index means declared zero fill, without payload presence')
    cases.append(dict(name='canonical_all_missing_index', status='PASS', receipt=result))
    reject('empty_object_is_not_canonical_fill', lambda: decode(data=b''), 'Truncated shard index')
    reject('corrupt_crc', lambda: decode(data=blob[:-1] + bytes([blob[-1] ^ 1])), 'CRC32C mismatch')
    reject('truncated_index', lambda: decode(data=blob[:layout['index_bytes'] - 1]), 'Truncated shard index')
    for name, changed, reason in [
        ('half_missing_offset', {1: (decoder.UINT64_MAX, 1)}, 'Half-missing'),
        ('half_missing_length', {1: (0, decoder.UINT64_MAX)}, 'Half-missing'),
        ('aliased_intervals', {2: entries[0]}, 'Overlapping'),
        ('overlapping_intervals', {2: (entries[0][0] + 1, entries[0][1])}, 'Overlapping'),
        ('payload_in_index_region', {2: (data_end, 16)}, 'outside bounded'),
        ('uint64_offset_overflow', {2: (decoder.UINT64_MAX - 1, 16)}, 'outside bounded'),
        ('zero_payload_length', {0: (0, 0)}, 'outside bounded'),
        ('payload_length_cap', {0: (0, layout['inner_bytes'] + 17)}, 'outside bounded'),
        ('truncated_blosc_header', {0: (0, 15)}, 'Truncated Blosc header')]:
        reject(name, lambda changed=changed: decode(data=indexed(changed), bbox=(0, 0, 1, 1)), reason)
    for name, offset, encoding, value in [
        ('blosc_wrong_typesize', 3, '<B', 2), ('blosc_wrong_flags', 2, '<B', 0),
        ('blosc_wrong_uncompressed_length', 4, '<I', 15), ('blosc_oversized_uncompressed_length', 4, '<I', 1 << 30),
        ('blosc_wrong_blocksize', 8, '<I', 0), ('blosc_wrong_compressed_length', 12, '<I', 31)]:
        changed = bytearray(blob); struct.pack_into(encoding, changed, offset, value)
        reject(name, lambda changed=changed: decode(data=bytes(changed)), 'Blosc header differs')
    for name, options, reason in [
        ('encoded_cap', dict(maximum_encoded_bytes=len(blob)-1), 'Whole object exceeds'),
        ('expansion_cap', dict(maximum_expanded_bytes=147), 'expansion exceeds'),
        ('bool_encoded_cap', dict(maximum_encoded_bytes=True), 'positive encoded'),
        ('bool_expansion_cap', dict(maximum_expanded_bytes=True), 'expansion exceeds'),
        ('wrong_hash', dict(expected_sha256='0' * 64), 'SHA256 mismatch'),
        ('wrong_length', dict(expected_byte_length=len(blob)+1), 'length mismatch')]:
        reject(name, lambda options=options: decode(**options), reason)
    for name, bbox in [('negative_bbox', (-1, 0, 1, 1)), ('zero_bbox', (0, 0, 0, 1)),
                       ('edge_overrun_bbox', (5, 6, 2, 1)), ('transposed_bbox', (0, 0, 7, 6)),
                       ('float_bbox', (0., 0, 1, 1)), ('bool_bbox', (False, 0, 1, 1))]:
        reject(name, lambda bbox=bbox: decode(bbox=bbox), 'BBox')
    for name, mutate in [('bool_fill', lambda m: m.update(fill_value=False)),
                         ('nonzero_fill', lambda m: m.update(fill_value=1)),
                         ('extra_sharding_configuration', lambda m: m['codecs'][0]['configuration'].update(extra=True)),
                         ('unsupported_start_index', lambda m: m['codecs'][0]['configuration'].update(index_location='start'))]:
        altered = copy.deepcopy(metadata); mutate(altered)
        reject(name, lambda altered=altered: decode(meta=altered), 'Unsupported')


def main():
    before = sha(SOURCE.read_bytes())
    receipt = dict(status='RUNNING', scope=__doc__, cases=[], decoder_source_sha256_before=before,
        fixture_source_sha256=sha(Path(__file__).read_bytes()), network_calls=0,
        real_candidate_arrays_read=0, model_calls=0,
        versions={package: importlib.metadata.version(package) for package in ['zarr', 'numcodecs', 'google-crc32c', 'numpy']})
    try:
        spec = importlib.util.spec_from_file_location('synthetic_checked_indexed_shard', SOURCE)
        decoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(decoder)
        execute(decoder, receipt)
        require(sha(SOURCE.read_bytes()) == before, 'Decoder source changed during fixtures')
        receipt['status'] = 'PASS'
    except Exception as error:
        receipt.update(status='FAIL', failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        receipt['decoder_source_sha256_after'] = sha(SOURCE.read_bytes())
        receipt['case_count'] = len(receipt['cases'])
        RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(status=receipt['status'], case_count=receipt['case_count'],
        decoder_source_sha256=before, receipt_sha256=sha(RECEIPT.read_bytes()),
        fixture_sha256=receipt['fixture_source_sha256']), sort_keys=True))


if __name__ == '__main__':
    main()
