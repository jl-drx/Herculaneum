"""Bounded IFD reads with actual object-size checks; no CT strips requested."""
import json
import struct
from acquire import ROOT, PLAN, fetch

header = (ROOT / 'slice32-header.bin').read_bytes()
assert header[:4] == b'II*\x00'
offset = struct.unpack_from('<I', header, 4)[0]
ledger = json.loads((ROOT / 'request-ledger.json').read_text())
receipt = next(r for r in ledger['requests'] if r['name'] == 'slice32-header.bin')
total = receipt['object_total_bytes']
assert 8 <= offset and offset + 2 <= total
count_raw = fetch('slice32-ifd-count.bin', PLAN['header_assets']['slice32'], 2, (offset, offset + 1))
count = struct.unpack('<H', count_raw)[0]
length = 2 + 12 * count + 4
assert 0 < count <= 128 and offset + length <= total
data = fetch('slice32-ifd.bin', PLAN['header_assets']['slice32'], length, (offset, offset + length - 1))
tags = []
for i in range(count):
    tag, kind, n, value = struct.unpack_from('<HHII', data, 2 + i * 12)
    tags.append({'tag': tag, 'type': kind, 'count': n, 'inline_value_or_offset': value})
result = {'ifd_offset': offset, 'object_total_bytes': total, 'tags': tags,
          'next_ifd': struct.unpack_from('<I', data, 2 + count * 12)[0],
          'initial_probe_deviation': 'Executed initial v1 probe read16 bytes rather than the8 needed for a classic TIFF header. Bytes8..15 are eight zero bytes before IFD and may be incidental CT payload. Retained verbatim; no CT pixel decode or later strip request in this stage.'}
(ROOT / 'tiff-structural-observations.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
