"""Fixed source, binary PNG and selection contracts for the fifteenth label-only stage."""
from pathlib import Path
import hashlib
import json
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parent
DESIGN_SHA = '9bf490e33e039ccacfdfbf99ab89b207d373ad1a08f0975afc9c0893f87d42e4'
SOURCE_SHA = '53ff9fd5cccb2604ebcd8b116b69b1f3c4ba25ed79fa624aaedf21bdaa709890'
RUNTIME = {'python': '3.12.14', 'numpy': '2.5.3', 'Pillow': '12.3.0', 'scipy': '1.18.1'}
LIMITS = dict(http_attempts=6, maximum_response_bytes=524288, total_response_bytes=2097152,
              maximum_pixels=150000000, worker_wall_seconds=120, worker_cpu_seconds=360,
              threads=4, worker_peak_rss_bytes=3221225472, worker_address_space_bytes=12884901888,
              selection_wall_seconds=420, acquisition_wall_seconds=180,
              total_output_bytes=8388608, selection_output_bytes=6291456)
SELECTION = dict(native_size=1152, stride=1152, square_size=33, xy_factor=3,
                 center_offset=1, coarse_size=384, border=32, positive_min=512,
                 negative_min=8192, separated_y_min=2304,
                 surface_gate='10*inside_count >= 9*1152*1152')
EXPECTED = [('Frag1', 'PHercParis2Fr47', 'Paris2', 'evaluation', [8181, 6330]),
            ('Frag2', 'PHercParis2Fr143', 'Paris2', 'evaluation', [14830, 9506]),
            ('Frag3', 'PHercParis1Fr34', 'Paris1', 'calibration', [7606, 5249])]
CODE_NAMES = ['label_common.py', 'prepare_plan.py', 'acquire_labels.py',
              'select_labels.py', 'run_selection.py', 'check_fixtures.py']


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(item):
    path = Path(item['path']).resolve(strict=True)
    require(path.stat().st_size == item.get('bytes', item.get('size_bytes')), 'Input size changed')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Input hash changed')
    return path


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def runtime_check():
    from importlib.metadata import version
    observed = {'python': sys.version.split()[0], **{k: version(k) for k in ('numpy', 'Pillow', 'scipy')}}
    require(observed == RUNTIME, 'Runtime differs from frozen versions')
    return observed


def load_plan(path, ready=True):
    path = Path(path).resolve(strict=True)
    plan = json.loads(path.read_text())
    require(plan['schema_version'] == 1 and (not ready or plan['ready'] is True), 'Plan is not ready')
    require(plan['limits'] == LIMITS and plan['selection'] == SELECTION and plan['runtime'] == RUNTIME,
            'Frozen method or limits changed')
    require(plan['design']['sha256'] == DESIGN_SHA and plan['source_contract']['sha256'] == SOURCE_SHA,
            'Scientific design/source identity changed')
    for pin in plan['input_pins']:
        verify(pin)
    require(plan['code'] == [record(ROOT / n) for n in CODE_NAMES], 'Executed source differs from plan')
    source = json.loads(verify(plan['source_contract']).read_text())
    require(plan['fragments'] == source['eligible_candidates'], 'Source fragments changed')
    require([(c['legacy_alias'], c['id'], c['parent_papyrus'], c['role'], c['shape_yx'])
             for c in plan['fragments']] == EXPECTED, 'Role/shape/identity changed')
    require(len(plan['objects']) == 6 and plan['objects'] == [
        {'fragment': c['legacy_alias'].lower(), 'kind': kind, 'url': c[key], 'shape_yx': c['shape_yx']}
        for c in plan['fragments'] for kind, key in [('labels', 'labels_url'), ('mask', 'mask_url')]],
        'Exact six-object order changed')
    if ready:
        require(plan['technical_reviews'], 'Ready plan needs reviewed source')
    runtime_check()
    return plan


def png_structure(body, expected_shape, maximum_pixels=150000000):
    """Validate every chunk CRC and IHDR before allocating any decoded image."""
    require(body[:8] == b'\x89PNG\r\n\x1a\n', 'Not PNG')
    offset, chunks, facts = 8, [], None
    while offset < len(body):
        require(offset + 12 <= len(body), 'Truncated PNG chunk')
        length = struct.unpack_from('>I', body, offset)[0]
        end = offset + 12 + length
        require(end <= len(body), 'Truncated PNG payload')
        kind = body[offset+4:offset+8]
        payload = body[offset+8:offset+8+length]
        require(zlib.crc32(kind + payload) & 0xffffffff == struct.unpack_from('>I', body, end-4)[0],
                'PNG CRC mismatch')
        if not chunks:
            require(kind == b'IHDR' and length == 13, 'PNG must begin with IHDR')
            w, h, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', payload)
            require([h, w] == list(expected_shape) and 0 < w*h <= maximum_pixels, 'PNG canvas exceeds contract')
            require(color in (0, 3) and compression == filtering == 0 and interlace in (0, 1),
                    'Unsupported binary PNG format')
            facts = {'shape_yx': [h, w], 'png_bit_depth': depth, 'png_color_type': color}
        require(kind != b'tRNS', 'Transparency is not allowed')
        chunks.append(kind)
        offset = end
        if kind == b'IEND':
            require(length == 0 and offset == len(body), 'Trailing PNG data')
            break
    require(chunks.count(b'IHDR') == chunks.count(b'IEND') == 1 and b'IDAT' in chunks,
            'Incomplete PNG structure')
    return facts


def decode_binary(path, shape):
    import numpy as np
    from PIL import Image
    body = Path(path).read_bytes()
    facts = png_structure(body, shape)
    Image.MAX_IMAGE_PIXELS = LIMITS['maximum_pixels']
    with Image.open(path) as image:
        require(image.mode in ('1', 'L', 'P') and 'transparency' not in image.info, 'Unsupported mode/transparency')
        mode = image.mode
        image.verify()
    with Image.open(path) as image:
        image.load()
        palette = image.getpalette('RGB') if mode == 'P' else None
        raw = np.asarray(image)
        require(list(raw.shape) == list(shape) and raw.dtype in (np.dtype(bool), np.dtype('uint8')),
                'Decoded shape/dtype changed')
        counts = np.zeros(256, dtype=np.int64)
        for y in range(0, shape[0], 256):
            counts += np.bincount(raw[y:y+256].reshape(-1).astype(np.uint8), minlength=256)
        used = np.flatnonzero(counts).tolist()
        require(set(used) <= {0, 1} or set(used) <= {0, 255}, 'Nonbinary raw values')
        palette_facts = {}
        if mode == 'P':
            require(palette is not None, 'Palette missing')
            for index in used:
                rgb = palette[index*3:index*3+3]
                require(rgb == ([0, 0, 0] if index == 0 else [255, 255, 255]), 'Ambiguous/inverted palette')
                palette_facts[str(index)] = rgb
        canonical = (raw != 0).astype(np.uint8)
    facts.update({'mode': mode, 'raw_dtype': str(raw.dtype), 'raw_value_counts':
                  {str(i): int(counts[i]) for i in used}, 'used_palette_rgb': palette_facts,
                  'canonicalization': 'raw nonzero -> uint8 1; no polarity inversion',
                  'encoded_bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()})
    return canonical, facts


def eligible(inside, area, positive, negative):
    return 10 * inside >= 9 * area and positive >= 512 and negative >= 8192


def chosen(candidates):
    valid = [c for c in candidates if c['eligible']]
    require(valid, 'No eligible first context; no relaxation')
    separated = [c for c in valid if c['y'] >= valid[0]['y'] + 2304]
    require(separated, 'No eligible separated second context; no relaxation')
    return [valid[0], separated[-1]]


def known_mask(labels, mask):
    from scipy.ndimage import minimum_filter, maximum_filter
    inside = minimum_filter(mask, size=33, mode='constant', cval=0).astype(bool)
    lo = minimum_filter(labels, size=33, mode='constant', cval=0)
    hi = maximum_filter(labels, size=33, mode='constant', cval=0)
    inside &= lo == hi
    return inside
