"""Independent synthetic-only checks; does not read scientific PNGs or use network."""
from pathlib import Path
import hashlib
import io
import json
import socket
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
IMPL = ROOT.parent / 'implementation'
sys.path.insert(0, str(IMPL))

def blocked(*args, **kwargs):
    raise AssertionError('Network forbidden in reviewer fixtures')

socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked

import numpy as np
from PIL import Image
import label_common as common

cases = []

def check(name, callback):
    callback()
    cases.append(name)

def equal(actual, expected):
    assert np.array_equal(actual, expected)

def rejects(callback):
    try:
        callback()
    except (ValueError, OSError):
        return
    raise AssertionError('Invalid fixture was accepted')

def main():
    assert not (ROOT / 'synthetic-fixture-result.json').exists(), 'Preserve earlier receipt'
    with tempfile.TemporaryDirectory(dir=ROOT, prefix='synthetic-') as tmp:
        tmp = Path(tmp)
        expected = np.zeros((48, 49), dtype=np.uint8)
        expected[20:35, 18:33] = 1
        for mode, high in [('1', 255), ('L', 1), ('L', 255), ('P', 1), ('P', 255)]:
            image = Image.fromarray(expected * high)
            if mode == '1':
                image = image.convert('1')
            if mode == 'P':
                image = Image.fromarray(expected * high, mode='P')
                palette = [0] * 768
                palette[high*3:high*3+3] = [255] * 3
                image.putpalette(palette)
            path = tmp / f'{mode}-{high}.png'
            image.save(path)
            check(f'canonical-{mode}-{high}', lambda p=path: equal(common.decode_binary(p, expected.shape)[0], expected))
        p = tmp / 'P-1.png'
        good = p.read_bytes()
        bad = bytearray(good); bad[-5] ^= 1
        check('bad-crc', lambda: rejects(lambda: common.png_structure(bytes(bad), expected.shape)))
        check('wrong-canvas', lambda: rejects(lambda: common.png_structure(good, (49, 48))))
        check('pixel-cap-before-decode', lambda: rejects(lambda: common.png_structure(good, expected.shape, 100)))
        check('trailing-bytes', lambda: rejects(lambda: common.png_structure(good+b'bad', expected.shape)))
        check('truncated-png', lambda: rejects(lambda: common.png_structure(good[:-1], expected.shape)))
        for name, transform in [
            ('inverted-palette', lambda im: im.putpalette([255]*3+[0]*765)),
            ('ambiguous-palette', lambda im: im.putpalette([0]*3+[200]*3+[0]*762)),
            ('transparent', lambda im: im.info.update(transparency=0)),
        ]:
            with Image.open(p) as im:
                im.load(); transform(im); out=tmp/(name+'.png'); im.save(out)
            check(name, lambda p=out: rejects(lambda: common.decode_binary(p, expected.shape)))
        im = Image.fromarray(expected*2); out=tmp/'nonbinary.png'; im.save(out)
        check('nonbinary-L', lambda: rejects(lambda: common.decode_binary(out, expected.shape)))

        # Independently enumerate every 33-square, including zero-padded exterior.
        labels = np.zeros((89, 101), dtype=np.uint8); labels[17:80, 54:98] = 1
        mask = np.ones_like(labels); mask[40, 45] = 0
        lp = np.pad(labels, 16); mp = np.pad(mask, 16)
        manual = np.zeros_like(labels, dtype=bool)
        for y in range(labels.shape[0]):
            for x in range(labels.shape[1]):
                l = lp[y:y+33, x:x+33]; m = mp[y:y+33, x:x+33]
                manual[y,x] = bool(np.all(m) and np.all(l == l[0,0]))
        check('full-known-square-independent-enumeration', lambda: equal(common.known_mask(labels, mask), manual))
        check('known-has-positive-and-negative-interiors', lambda: equal(np.unique(labels[manual]), np.array([0,1])))
        area=1152**2; cutoff=(9*area+9)//10
        check('support-90percent-ceiling', lambda: equal(common.eligible(cutoff,area,512,8192), True))
        check('support-one-below', lambda: equal(common.eligible(cutoff-1,area,512,8192), False))
        check('positive-one-below', lambda: equal(common.eligible(area,area,511,8192), False))
        check('negative-one-below', lambda: equal(common.eligible(area,area,512,8191), False))
        candidates=[dict(x=x,y=y,eligible=e) for x,y,e in [(0,0,False),(1152,0,True),(0,1152,True),(0,2304,True),(1152,2304,True),(0,3456,False)]]
        check('first-last-separated-tie', lambda: equal(common.chosen(candidates), [candidates[1],candidates[4]]))
        check('no-second-no-fallback', lambda: rejects(lambda: common.chosen(candidates[:3])))
        check('no-eligible-no-fallback', lambda: rejects(lambda: common.chosen(candidates[:1])))
        import select_labels
        shape=(3466,2321)
        labels=np.broadcast_to((np.arange(shape[1])%1152>=576).astype(np.uint8),shape)
        mask=np.ones(shape,dtype=np.uint8); known=np.ones(shape,dtype=bool)
        known[103,103]=False
        found,border=select_labels.candidates_for(labels,mask,known)
        check('grid-drops-partial-contexts',lambda: equal([(c['y'],c['x']) for c in found],[(y,x) for y in (0,1152,2304) for x in (0,1152)]))
        check('exact-border-area',lambda: equal(np.count_nonzero(border),320**2))
        check('center-sampling-and-primary-counts',lambda: equal([(c['primary_positive'],c['primary_negative']) for c in found],[(51200,51199)]+[(51200,51200)]*5))
        check('selected-last-row-last-column',lambda: equal([(c['y'],c['x']) for c in common.chosen(found)],[(0,0),(2304,1152)]))
    result = dict(status='PASS', cases=cases, count=len(cases),
                  source=[common.record(IMPL/n) for n in common.CODE_NAMES], reviewer=common.record(__file__),
                  scientific_arrays_read=0, model_calls=0, network_calls=0,
                  method='Synthetic PNGs and independent brute-force 33-square enumeration')
    common.save(ROOT/'synthetic-fixture-result.json',result)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
