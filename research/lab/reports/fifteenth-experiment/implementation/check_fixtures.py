"""Synthetic PNG, known-support, selection and failed-stream fixtures; no real data/network."""
from pathlib import Path
import argparse
import hashlib
import http.client
import io
import json
import tempfile
import numpy as np
from PIL import Image
import label_common as common
import acquire_labels as acquisition
from select_labels import candidates_for


def rejected(call):
    try:
        call()
    except (ValueError, RuntimeError, OSError):
        return
    raise AssertionError('Expected rejection')


def image_bytes(mode, values=(0, 1), inverted=False, transparency=False):
    a = np.zeros((19, 27), dtype=np.uint8)
    a[:, 13:] = values[1]
    if mode == '1':
        image = Image.fromarray(a != 0)
    else:
        image = Image.fromarray(a)
        if mode == 'P':
            image = image.convert('P')
            palette = [0]*768
            palette[values[1]*3:values[1]*3+3] = [255]*3
            if inverted:
                palette[:3] = [255]*3
                palette[values[1]*3:values[1]*3+3] = [0]*3
            image.putpalette(palette)
    target = io.BytesIO()
    image.save(target, format='PNG', **({'transparency': 0} if transparency else {}))
    return target.getvalue()


class FakeResponse:
    def __init__(self, body=b'', status=200, declared=True, failure=None):
        self.status = status
        self.headers = {'Content-Type': 'image/png', 'ETag': '"synthetic-only"'}
        if declared:
            self.headers['Content-Length'] = str(len(body))
        self.body, self.failure, self.reads = io.BytesIO(body), failure, 0
    def read(self, size):
        self.reads += 1
        if self.failure:
            raise self.failure
        return self.body.read(size)
    def close(self):
        pass


class FakeOpener:
    def __init__(self, response):
        self.response, self.calls = response, 0
    def open(self, *args, **kwargs):
        self.calls += 1
        return self.response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    common.require(not output.exists(), 'Fresh fixture directory required')
    checks = []
    with tempfile.TemporaryDirectory(prefix='synthetic-labels-') as temporary:
        root = Path(temporary)
        for mode, values in [('1', (0, 1)), ('L', (0, 1)), ('L', (0, 255)),
                             ('P', (0, 1)), ('P', (0, 255))]:
            path = root/'image.png'
            path.write_bytes(image_bytes(mode, values))
            a, facts = common.decode_binary(path, (19, 27))
            assert a.dtype == np.uint8 and np.all(a[:, :13] == 0) and np.all(a[:, 13:] == 1)
            checks.append('canonical '+mode+str(values))
        path.write_bytes(image_bytes('P', inverted=True))
        rejected(lambda: common.decode_binary(path, (19, 27)))
        checks.append('inverted palette rejected')
        path.write_bytes(image_bytes('P', transparency=True))
        rejected(lambda: common.decode_binary(path, (19, 27)))
        checks.append('transparency rejected')
        path.write_bytes(image_bytes('L', (0, 2)))
        rejected(lambda: common.decode_binary(path, (19, 27)))
        checks.append('nonbinary grayscale rejected')
        body = image_bytes('L')
        corrupt = bytearray(body); corrupt[30] ^= 1
        rejected(lambda: common.png_structure(corrupt, (19, 27)))
        rejected(lambda: common.png_structure(body, (19, 26)))
        rejected(lambda: common.png_structure(body, (19, 27), maximum_pixels=512))
        checks += ['CRC rejected', 'canvas rejected', 'pixel limit rejected']
        labels = np.zeros((65, 71), dtype=np.uint8); labels[20:48, 35:55] = 1
        mask = np.ones_like(labels); mask[37, 27] = 0
        actual = common.known_mask(labels, mask)
        expected = np.zeros_like(labels, dtype=bool)
        for y in range(16, labels.shape[0]-16):
            for x in range(16, labels.shape[1]-16):
                m, l = mask[y-16:y+17,x-16:x+17], labels[y-16:y+17,x-16:x+17]
                expected[y,x] = bool(m.all() and np.all(l == l[0,0]))
        assert np.array_equal(actual, expected)
        checks.append('known square matches independent bounded neighborhoods including canvas and hole')
        area = 1152*1152
        first_inside = (9*area+9)//10
        assert common.eligible(first_inside, area, 512, 8192)
        assert not common.eligible(first_inside-1, area, 512, 8192)
        assert not common.eligible(area, area, 511, 8192)
        assert not common.eligible(area, area, 512, 8191)
        checks.append('exact90% and inclusive class boundaries')
        labels = np.zeros((3456, 1152), dtype=np.uint8); labels[:, :576] = 1
        mask = np.ones_like(labels)
        known = common.known_mask(labels, mask)
        candidates, border = candidates_for(labels, mask, known)
        assert [(c['y'],c['x']) for c in candidates] == [(0,0),(1152,0),(2304,0)]
        selected = common.chosen(candidates)
        assert [c['y'] for c in selected] == [0,2304]
        assert all(c['primary_positive']+c['primary_negative'] == c['primary_count'] for c in candidates)
        assert border.sum() == 320*320
        rejected(lambda: common.chosen(candidates[:2]))
        rejected(lambda: common.chosen([]))
        checks.append('row-major complete grid and first-last separation; failed gates stop')
        obj = dict(fragment='frag1',kind='labels',shape_yx=[19,27],
                   url='https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/inklabels.png')
        for name, response, succeed, expected_retained in [
            ('complete', FakeResponse(body), True, len(body)),
            ('unknown-eof', FakeResponse(body, declared=False), True, len(body)),
            ('truncated', FakeResponse(body, failure=http.client.IncompleteRead(b'abcde', 100)), False, 5),
            ('opaque-failure', FakeResponse(body, failure=OSError('synthetic-only')), False, 0),
            ('redirect', FakeResponse(body, status=307), False, len(body)),
            ('bad-crc', FakeResponse(bytes(corrupt)), False, len(corrupt))]:
            d = root/name; d.mkdir()
            ledger = dict(requests=[],response_body_bytes=0,charged_response_bytes=0)
            opener = FakeOpener(response)
            if succeed:
                file = acquisition.fetch(obj, d, ledger, opener)
                assert file['bytes'] == len(body)
                assert ledger['charged_response_bytes'] == len(body)
            else:
                rejected(lambda: acquisition.fetch(obj,d,ledger,opener))
                assert ledger['charged_response_bytes'] == common.LIMITS['maximum_response_bytes']
            assert ledger['response_body_bytes'] == expected_retained and opener.calls == 1
            retained_file = next(d.glob('*.png*'), None)
            assert retained_file is not None and retained_file.stat().st_size == expected_retained
            checks.append('bounded fake HTTP '+name)
    output.mkdir(parents=True)
    receipt = dict(status='PASS', checks=checks, check_count=len(checks),
                   runtime=common.runtime_check(), code=[common.record(common.ROOT/n) for n in common.CODE_NAMES],
                   actual_network_requests=0, scientific_images_decoded=0, model_calls=0,
                   scope='Synthetic images/logical arrays and in-memory HTTP stubs only')
    common.save(output/'fixture-receipt.json',receipt)
    print(json.dumps({'status':'PASS','checks':len(checks),'receipt':common.record(output/'fixture-receipt.json')}))


if __name__ == '__main__':
    main()
