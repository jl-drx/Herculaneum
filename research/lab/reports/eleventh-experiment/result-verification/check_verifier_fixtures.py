"""Synthetic artifact/tile checks for the independent verifier; no scientific data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import verify_reference as verifier

HERE = Path(__file__).resolve().parent


def rejected(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Malformed synthetic artifact was accepted')


def main():
    receipt_path = HERE/'verifier-fixture-receipt.json'
    if receipt_path.exists():
        raise ValueError('Preserve the existing fixture receipt')
    with tempfile.TemporaryDirectory(prefix='synthetic-verifier-', dir=HERE) as directory:
        root = Path(directory)
        file = root/'synthetic.bin'
        file.write_bytes(b'Synthetic content only')
        record = verifier.file_record(file)
        pins = verifier.Pins()
        assert pins.check(record) == file
        assert pins.check(record) == file and len(pins.records) == 1
        rejected(lambda: pins.check(dict(record, size_bytes=record['size_bytes']+1)))
        rejected(lambda: verifier.identity(dict(record, bytes=record['size_bytes']+1)))
        file.write_bytes(b'Changed synthetic content')
        rejected(pins.unchanged)
        np.save(root/'synthetic-logits.npy', np.zeros((25,128,128), dtype=np.float32))
        grid = [[x,y,128,128] for y in range(0,257,64) for x in range(0,257,64)]
        item = dict(inference=dict(tiles=[dict(bbox_roi_xywh=bbox, status='scored') for bbox in grid],
            tile_count=25,scored_tile_count=25),
            captured_tiles=[dict(bbox_roi_xywh=bbox,input_sha256='0'*64) for bbox in grid],
            logits=verifier.file_record(root/'synthetic-logits.npy'))
        _, coverage = verifier.check_capture(verifier.Pins(),item)
        assert coverage.shape==(384,384) and coverage.all()
        wrong = copy.deepcopy(item)
        wrong['captured_tiles'][0]['bbox_roi_xywh']=[64,0,128,128]
        rejected(lambda: verifier.check_capture(verifier.Pins(),wrong))
        wrong = copy.deepcopy(item)
        wrong['captured_tiles'][0]['input_sha256']='not-a-digest'
        rejected(lambda: verifier.check_capture(verifier.Pins(),wrong))
        wrong = copy.deepcopy(item)
        wrong['inference']['tiles'][0]['status']='unknown'
        rejected(lambda: verifier.check_capture(verifier.Pins(),wrong))
        np.save(root/'nonfinite-logits.npy',np.full((25,128,128),np.nan,dtype=np.float32))
        wrong = copy.deepcopy(item)
        wrong['logits']=verifier.file_record(root/'nonfinite-logits.npy')
        rejected(lambda: verifier.check_capture(verifier.Pins(),wrong))
        # Missing last tile leaves the bottom-right 64x64 uncovered, rather
        # than being mistaken for a real low prediction or complete coverage.
        np.save(root/'partial-logits.npy',np.zeros((24,128,128),dtype=np.float32))
        partial=copy.deepcopy(item)
        partial['inference']['tiles'][-1]['status']='skipped'
        partial['inference']['scored_tile_count']=24
        partial['captured_tiles']=partial['captured_tiles'][:-1]
        partial['logits']=verifier.file_record(root/'partial-logits.npy')
        _, coverage=verifier.check_capture(verifier.Pins(),partial)
        assert int((~coverage).sum())==64*64 and not coverage[383,383]
        rejected(lambda: verifier.compare_tree({'ap':.5},{'ap':.6},'synthetic mismatch'))
        rejected(lambda: verifier.compare_tree({'ap':.5},{'ap':.5,'auc':1},'synthetic key mismatch'))
    assert 'torch' not in sys.modules
    receipt=dict(status='PASS', synthetic_checks=13, scientific_arrays_read=0,
        model_calls=0, checkpoint_deserializations=0, network_calls=0,
        code=[verifier.file_record(HERE/'verify_reference.py'),verifier.file_record(__file__)])
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
