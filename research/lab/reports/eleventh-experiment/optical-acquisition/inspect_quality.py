"""Decode only the retained PNGs and inspect structural TIFF tags."""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def inspect_png(name):
    path = ROOT / name
    body = path.read_bytes()
    assert body[:8] == b'\x89PNG\r\n\x1a\n'
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack('>IIBBBBB', body[16:29])
    assert (height, width) == (9791, 6064) and width * height <= 60000000
    with Image.open(path) as check:
        check.verify()
    with Image.open(path) as im:
        assert len(im.getbands()) * width * height <= 240000000
        im.load()
        array = np.asarray(im)
        entry = {'sha256': hashlib.sha256(body).hexdigest(), 'encoded_bytes': len(body),
                 'shape': list(array.shape), 'mode': im.mode, 'dtype': str(array.dtype),
                 'png_bit_depth': bit_depth, 'png_color_type': color_type,
                 'decoded_bytes': array.nbytes, 'min': int(array.min()), 'max': int(array.max())}
        if name != 'registered-ir.png':
            values, counts = np.unique(array, return_counts=True)
            entry['value_counts'] = {str(int(v)): int(c) for v, c in zip(values, counts)}
        return array, entry


if __name__ == '__main__':
    output = {'scope': 'Private format/participation observations; no model, CT crop, inferred registration or translation', 'assets': {}}
    labels, output['assets']['inklabels.png'] = inspect_png('inklabels.png')
    mask, output['assets']['mask.png'] = inspect_png('mask.png')
    ir, output['assets']['registered-ir.png'] = inspect_png('registered-ir.png')
    assert labels.ndim == mask.ndim == 2
    output['support'] = {'mask_nonzero': int(np.count_nonzero(mask)),
                         'ink_nonzero': int(np.count_nonzero(labels)),
                         'ink_nonzero_inside_mask': int(np.count_nonzero((labels != 0) & (mask != 0))),
                         'ink_nonzero_outside_mask': int(np.count_nonzero((labels != 0) & (mask == 0))),
                         'unmarked_inside_mask': int(np.count_nonzero((labels == 0) & (mask != 0)))}
    output['registration_limit'] = 'Equal dimensions and official region linkage do not quantify optical registration error.'
    (ROOT / 'quality-observations.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))
