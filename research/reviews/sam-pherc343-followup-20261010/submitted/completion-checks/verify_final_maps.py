#!/usr/bin/env python3
"""Read-only gate for copied FINAL PHerc343 outputs. Never launches inference/composition.
Prints JSON to stdout; exits nonzero on missing, partial or inconsistent artifacts.
A successful integrity gate is not scientific validation. Original recipe is authoritative.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
NAME = 'concat_w047-w048_R5B2_z9500-11000'
MODELS = ['dnative', 'hecate', 'hybrid_3d2d-seed42', 'hybrid_3d2d-seed43']

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def readjson(p):
    return json.loads(Path(p).read_text())

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def decoded(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def suffix(m):
    return m + ('__L66zc33.50_reverse.png' if m == 'hecate' else '__L66s26_reverse.tif')

def validate_completion(completion):
    require(completion.get('status') == 'PASS', 'No terminal inference PASS receipt')
    terminal = completion.get('maps', [])
    require(len(terminal) == 8, 'Terminal receipt must enumerate exactly eight full maps')
    entries = {(r.get('model'), r.get('phase')): r for r in terminal}
    require(set(entries) == {(m,p) for m in MODELS for p in ['plain','padded']}, 'Duplicate/missing model-phase pair')
    require(all(r.get('status') == 'PASS' for r in terminal), 'Nonfinal individual receipt')
    return entries


def validate_final_file(f, r, model, phase, expected, checkpoint):
    """Actual shared gate, also exercised by small synthetic tests. No resampling."""
    require(r.get('status') == 'PASS' and r.get('model') == model and r.get('phase') == phase, f'{f}: wrong receipt')
    with Image.open(f) as image:
        require(getattr(image, 'n_frames', 1) == 1, f'{f}: multi-frame file')
        require(image.mode == 'L', f'{f}: expected grayscale uint8 encoding')
        im = np.asarray(image).copy()
    require(im.ndim==2 and im.dtype==np.uint8 and im.shape==expected, f'{f}: not full expected uint8 map')
    require(r['shape']==list(expected) and r['output']['bytes']==f.stat().st_size, f'{f}: size/shape receipt mismatch')
    encoded, dec = sha(f), decoded(im)
    require(r['output']['sha256']==encoded and r['decoded_sha256']==dec, f'{f}: hash mismatch')
    require(Path(r['output']['path']).name==f.name, f'{f}: wrong receipt target')
    require(r['checkpoint']['sha256']==checkpoint['sha256'] and r['checkpoint']['bytes']==checkpoint['bytes'], f'{f}: changed weights')
    return im, encoded, dec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plain', type=Path, required=True, help='Copied final maps_plain directory')
    ap.add_argument('--padded', type=Path, required=True, help='Copied final maps_padded directory')
    ap.add_argument('--completion', type=Path, required=True, help='Final FIXED_DEPTH_INFERENCE.json')
    ap.add_argument('--node-source', type=Path, required=True)
    ap.add_argument('--valid', type=Path, required=True)
    ap.add_argument('--node-source-sha256', required=True, help='Trusted pin from original archived manifest, not derived from incoming file')
    ap.add_argument('--merged', type=Path, help='Optional copied original-composer merged map directory')
    a = ap.parse_args()
    if a.merged:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('--merged requires OpenCV for the exact source nearest-neighbor convention; core checks do not') from exc
    completion = readjson(a.completion)
    entries = validate_completion(completion)
    qualified = {r['model']:r['checkpoint'] for r in readjson(HERE/'INFERENCE_QUALIFICATION.json')['models']}
    retained = {(r['model'],r['phase']):r for r in readjson(HERE/'SAVED_MAPS_VERIFIED.json')['maps']}
    valid = np.load(a.valid, allow_pickle=False)
    require(valid.shape == (2560,6720) and valid.dtype == bool, 'Unexpected valid geometry/type')
    require(sha(a.valid) == '1905c0d05656b4472a533e00b1c140592c7b33ed7a8bb7780873517f747e41fc', 'Valid file changed')
    require(decoded(np.packbits(valid[105:2560,380:6021])) == 'f60549218edb9871e5acb8930e176a09fe1aaafef28c10a2261014f70a58dcbc', 'Published valid mask changed')
    require(sha(a.node_source) == a.node_source_sha256, 'Node-source hash differs from original trusted pin')
    source = np.load(a.node_source, allow_pickle=False)['source']
    require(source.shape == (128,336), 'Unexpected node-source geometry')
    owner = np.isin(np.repeat(np.repeat(source,20,0),20,1), [1,4])
    results = []
    for m in MODELS:
        maps = {}
        for phase, directory in [('plain',a.plain),('padded',a.padded)]:
            prefix = NAME+'_66'+('_padded' if phase == 'padded' else '')
            f = directory/(prefix+'__'+suffix(m))
            r = readjson(str(f)+'.receipt.json')
            require(r.get('status') == 'PASS' and r.get('model') == m and r.get('phase') == phase, f'{f}: wrong receipt')
            require(r == entries[m,phase], f'{f}: terminal and individual receipts differ')
            expected = ((2304,6048) if phase=='plain' else (2340,6228)) if m=='hecate' else ((2560,6720) if phase=='plain' else (2600,6920))
            im, encoded, dec = validate_final_file(f, r, m, phase, expected, qualified[m])
            if (m,phase) in retained:
                old = retained[m,phase]
                require(encoded==old['encoded_sha256'] and dec==old['decoded_sha256'], f'{f}: retained completed map changed')
            if m=='hecate':
                meta = readjson(f.with_suffix('.json'))
                require(meta['shape_yx']==list(expected) and meta['sampling_um']==9.6 and meta['reverse_z'] is True and meta['stride']==32 and meta['precision']=='fp32', f'{f}: Hecate metadata mismatch')
            maps[phase] = im
            results.append(dict(model=m,phase=phase,shape=list(im.shape),dtype=str(im.dtype),encoded_sha256=encoded,decoded_sha256=dec))
        if a.merged:
            plain, padded = maps['plain'], maps['padded']
            if m=='hecate':
                padded = padded[36:,180:]
                own = cv2.resize(owner.astype(np.uint8),(6048,2304),interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                padded = padded[40:2600,200:6920]
                own = owner
            merged = np.where(own,padded,plain)
            target = a.merged/(NAME+'__'+suffix(m))
            actual = cv2.imread(str(target),cv2.IMREAD_UNCHANGED)
            require(actual is not None and actual.dtype==np.uint8 and actual.shape==merged.shape and np.array_equal(actual,merged), f'{target}: ownership merge mismatch')
    print(json.dumps(dict(status='FINAL_MAP_INTEGRITY_PASS', scientific_accuracy_validated=False, original_composition_still_required=True, node_source_sha256=sha(a.node_source), completion_sha256=sha(a.completion), ownership_merge_checked=bool(a.merged), maps=results),indent=2))

if __name__=='__main__':
    main()
