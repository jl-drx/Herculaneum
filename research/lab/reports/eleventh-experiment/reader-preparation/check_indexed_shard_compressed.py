"""One synthetic installed-Zarr compressed-path run; no network, real arrays, or models."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import struct

import google_crc32c
import numpy as np
from numcodecs.blosc import cbuffer_complib
import zarr
from zarr.codecs import BloscCodec, BytesCodec
from zarr.core.buffer import default_buffer_prototype

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'indexed_shard.py'
RECEIPT = HERE / 'compressed-fixture-receipt.json'
PRESERVED = [SOURCE, HERE / 'check_indexed_shard.py', HERE / 'synthetic-fixture-receipt.json']


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


async def read_memory_store(store):
    require(sorted([key async for key in store.list()]) == ['c/0/0', 'zarr.json'], 'Exactly one shard')
    blob = (await store.get('c/0/0', prototype=default_buffer_prototype())).to_bytes()
    metadata = (await store.get('zarr.json', prototype=default_buffer_prototype())).to_bytes()
    return blob, json.loads(metadata)


def execute(decoder, receipt):
    store = zarr.storage.MemoryStore()
    array = zarr.create_array(store=store, shape=(100, 90), dtype=np.uint8,
        chunks=(64, 64), shards=(128, 128), serializer=BytesCodec(endian='little'),
        compressors=[BloscCodec(typesize=1, cname='zstd', clevel=3, shuffle='bitshuffle', blocksize=0)],
        fill_value=0, dimension_names=['y', 'x'])
    values = ((np.arange(100)[:, None] // 8 * 5 + np.arange(90)[None, :] // 7 * 3) % 29).astype(np.uint8)
    array[:] = values
    blob, metadata = asyncio.run(read_memory_store(store))
    layout, entries, index = decoder.validate_shard_index(blob, metadata,
        maximum_encoded_bytes=len(blob), expected_sha256=sha(blob), expected_byte_length=len(blob))
    require(index['present_inner_chunks'] == 4, 'All four synthetic chunks encoded')
    require(index['metadata_canonical_json_sha256'] == sha(json.dumps(metadata,
        sort_keys=True, separators=(',', ':')).encode()), 'Canonical metadata hash')
    headers = []
    for position, (offset, length) in enumerate(entries):
        payload = blob[offset:offset+length]
        version, codec_version, flags, typesize, raw_bytes, blocksize, encoded_bytes = struct.unpack('<BBBBIII', payload[:16])
        require(flags & 4 and not flags & 2, 'Actual bitshuffle compression, without memcpy fallback')
        require(typesize == 1 and raw_bytes == 4096 and encoded_bytes == length < raw_bytes,
                'Compressed uint8 64-square payload')
        require(cbuffer_complib(payload).lower() == 'zstd', 'Actual zstd header')
        headers.append(dict(inner_yx=list(divmod(position, 2)), absolute_offset=offset,
            encoded_byte_length=length, decoded_byte_length=raw_bytes, flags=flags,
            bitshuffle=True, memcpy=False, compression_library='zstd', blocksize=blocksize,
            version=version, codec_version=codec_version, encoded_sha256=sha(payload)))
    receipt['synthetic_source'] = dict(writer='installed Zarr / MemoryStore', metadata=metadata,
        metadata_source='installed-Zarr-written zarr.json', sha256=sha(blob), byte_length=len(blob),
        values_sha256=sha(values.tobytes()), index=index, payload_headers=headers)
    receipt['cases'].append(dict(name='four_actual_compressed_zstd_bitshuffle_payloads', status='PASS'))

    def decode(data=blob, bbox=(0, 0, 90, 100), verify_source=False):
        options = dict(expected_sha256=sha(blob), expected_byte_length=len(blob)) if verify_source else {}
        return decoder.decode_shard_roi(data, metadata, bbox,
            maximum_encoded_bytes=65536, maximum_expanded_bytes=65536, **options)

    for name, bbox in [('full_asymmetric_partial_edges', (0, 0, 90, 100)),
                       ('cross_inner_boundary_and_partial_edges', (57, 61, 33, 39)),
                       ('bottom_right_partial_edge', (80, 93, 10, 7))]:
        x, y, w, h = bbox; output, present, result = decode(bbox=bbox, verify_source=True)
        standard = np.asarray(array[y:y+h, x:x+w])
        require(output.dtype == np.uint8 and present.dtype == np.bool_ and bool(present.all()), 'Exact types and presence')
        require(np.array_equal(output, standard) and np.array_equal(standard, values[y:y+h, x:x+w]), 'Exact installed-Zarr ROI parity')
        require(result['output_sha256'] == sha(standard.tobytes()), 'Output hash')
        for selected in result['selected_inner_chunks']:
            cy, cx = selected['inner_yx']; y0, x0 = cy * 64, cx * 64
            padded = np.zeros((64, 64), dtype=np.uint8)
            padded[:min(64, 100-y0), :min(64, 90-x0)] = values[y0:y0+64, x0:x0+64]
            require(selected['decoded_byte_length'] == 4096 and selected['decoded_sha256'] == sha(padded.tobytes()),
                    'Independent padded inner-chunk hash')
        receipt['cases'].append(dict(name=name, status='PASS', outcome='exact_zarr_equality', receipt=result))

    def reject(name, data):
        try:
            decode(data=data, bbox=(0, 0, 1, 1))
        except ValueError as error:
            require('Blosc header differs' in str(error), name + ': wrong guard: ' + str(error))
            receipt['cases'].append(dict(name=name, status='PASS', outcome='rejected_before_decompression', error=str(error)))
        else:
            raise AssertionError(name + ': malformed compressed payload accepted')

    # Physically remove one byte from the first compressed payload and rebase
    # later absolute offsets; a fresh valid CRC prevents an index-only rejection.
    first_offset, first_length = entries[0]; cut = first_offset + first_length - 1
    data_end = len(blob) - layout['index_bytes']; body = blob[:cut] + blob[cut+1:data_end]
    changed = [(offset - int(offset > cut), length - int(position == 0))
               for position, (offset, length) in enumerate(entries)]
    encoded_index = b''.join(struct.pack('<QQ', *entry) for entry in changed)
    truncated = body + encoded_index + struct.pack('<I', google_crc32c.value(encoded_index))
    decoder.validate_shard_index(truncated, metadata, maximum_encoded_bytes=65536)
    reject('actual_compressed_payload_truncation_with_valid_rebased_index', truncated)
    for name, relative_offset, value in [('compressed_header_oversized_expansion', 4, 1 << 30),
                                        ('compressed_header_wrong_encoded_length', 12, first_length + 1)]:
        changed = bytearray(blob); struct.pack_into('<I', changed, first_offset + relative_offset, value)
        reject(name, bytes(changed))
    changed = bytearray(blob); changed[first_offset + 2] &= ~4
    reject('compressed_header_missing_bitshuffle_and_memcpy', bytes(changed))


def main():
    require(not RECEIPT.exists(), 'This additive fixture is authorized for one execution')
    before = {path.name: sha(path.read_bytes()) for path in PRESERVED}
    receipt = dict(status='RUNNING', scope=__doc__, cases=[], preserved_source_sha256_before=before,
        decoder_source_sha256_before=before[SOURCE.name], fixture_source_sha256=sha(Path(__file__).read_bytes()),
        execution_attempts=1, network_calls=0, real_candidate_arrays_read=0, model_calls=0,
        versions={name: importlib.metadata.version(name) for name in ['zarr', 'numcodecs', 'google-crc32c', 'numpy']})
    try:
        spec = importlib.util.spec_from_file_location('synthetic_compressed_indexed_shard', SOURCE)
        decoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(decoder)
        execute(decoder, receipt)
        require({path.name: sha(path.read_bytes()) for path in PRESERVED} == before, 'Preserved sources changed during execution')
        receipt['status'] = 'PASS'
    except Exception as error:
        receipt.update(status='FAIL', failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        after = {path.name: sha(path.read_bytes()) for path in PRESERVED}
        receipt.update(preserved_source_sha256_after=after, decoder_source_sha256_after=after[SOURCE.name],
                       case_count=len(receipt['cases']))
        RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(status=receipt['status'], case_count=receipt['case_count'],
        payloads=receipt['synthetic_source']['payload_headers'], decoder_source_sha256=before[SOURCE.name],
        fixture_sha256=receipt['fixture_source_sha256'], receipt_sha256=sha(RECEIPT.read_bytes())), sort_keys=True))


if __name__ == '__main__':
    main()
