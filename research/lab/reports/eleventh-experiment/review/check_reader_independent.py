"""Independent synthetic-only reader checks; no network or candidate data."""
from pathlib import Path
import asyncio
import hashlib
import importlib.util
import json
import struct

import google_crc32c
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BytesCodec

HERE = Path(__file__).resolve().parent
IMPL = HERE.parent / 'reader-preparation'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = IMPL / 'indexed_shard.py'
    fixture_source = IMPL / 'check_indexed_shard.py'
    frozen = {'decoder': sha(source), 'owner_fixture': sha(fixture_source)}
    decoder = load('independent_decoder', source)
    fixtures = load('owner_fixtures_read_only', fixture_source)
    receipt = {'status': 'RUNNING', 'cases': [], 'sources': frozen,
               'network_calls': 0, 'candidate_arrays_read': 0, 'model_calls': 0}
    try:
        fixtures.execute(decoder, receipt)
        assert google_crc32c.value(b'123456789') == 0xe3069283
        receipt['cases'].append({'name': 'known_castagnoli_check_vector', 'status': 'PASS'})

        # Different synthetic shape and real compressed chunks, including edges.
        store = zarr.storage.MemoryStore()
        array = zarr.create_array(store=store, shape=(111, 99), dtype=np.uint8,
            chunks=(64, 64), shards=(128, 128), fill_value=0,
            serializer=BytesCodec(endian='little'),
            compressors=[BloscCodec(typesize=1, cname='zstd', clevel=3,
                shuffle='bitshuffle', blocksize=0)], dimension_names=['y', 'x'])
        yy, xx = np.indices(array.shape)
        values = ((yy // 5 + 3 * (xx // 7)) % 19).astype(np.uint8)
        array[:] = values
        blob, metadata = asyncio.run(fixtures.shard_bytes(store))
        layout, entries, _ = decoder.validate_shard_index(blob, metadata,
            maximum_encoded_bytes=len(blob))
        assert all(length < 4096 + 16 and not (blob[offset + 2] & 2)
                   for offset, length in entries)
        bbox = (41, 37, 58, 74)
        output, present, detail = decoder.decode_shard_roi(blob, metadata, bbox,
            maximum_encoded_bytes=len(blob), maximum_expanded_bytes=25000)
        assert np.array_equal(output, values[37:111, 41:99])
        assert np.array_equal(output, np.asarray(array[37:111, 41:99]))
        assert present.all()
        receipt['cases'].append({'name': 'actual_zstd_bitshuffle_cross_edges',
            'status': 'PASS', 'receipt': detail})

        # Reverse physical payload order while preserving each logical entry.
        # Offsets are absolute byte locations, not implicit raster ordering.
        payloads = [blob[offset:offset + length] for offset, length in entries]
        reversed_payload = b''.join(reversed(payloads))
        reordered = [None] * len(entries)
        cursor = 0
        for logical in reversed(range(len(entries))):
            reordered[logical] = (cursor, len(payloads[logical]))
            cursor += len(payloads[logical])
        index = b''.join(struct.pack('<QQ', *item) for item in reordered)
        shuffled = reversed_payload + index + struct.pack('<I', google_crc32c.value(index))
        output, present, detail = decoder.decode_shard_roi(shuffled, metadata, bbox,
            maximum_encoded_bytes=len(shuffled), maximum_expanded_bytes=25000)
        assert np.array_equal(output, values[37:111, 41:99]) and present.all()
        receipt['cases'].append({'name': 'nonmonotonic_absolute_payload_offsets',
            'status': 'PASS', 'receipt': detail})
        assert frozen == {'decoder': sha(source), 'owner_fixture': sha(fixture_source)}
        receipt['status'] = 'PASS'
    except Exception as error:
        receipt.update(status='FAIL', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['case_count'] = len(receipt['cases'])
        receipt['reviewer_source_sha256'] = sha(Path(__file__))
        target = HERE / 'reader-independent-receipt.json'
        target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'cases': receipt['case_count'],
                      'receipt_sha256': sha(target), 'sources': frozen}))


if __name__ == '__main__':
    main()
