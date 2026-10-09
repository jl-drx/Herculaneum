"""Local-only pinning and scalar assertions for the independent reviewer."""
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import numpy as np
from numerical_kernels import require


def offline(*args, **kwargs):
    raise RuntimeError("Network is forbidden during independent numeric reconstruction")


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline


def record(path):
    path = Path(path).resolve(strict=True)
    require(path.suffix.lower() not in {'.pth', '.pt', '.ckpt', '.safetensors', '.bin'},
            "Checkpoint/model-weight reads are forbidden")
    require(str(path).startswith('/workspace/') or str(path).startswith('/tmp/'), "Local workspace input required")
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest}


class Review:
    def __init__(self):
        self.pins = {}
        self.comparisons = 0
        self.maximum_absolute_errors = {}

    def verify(self, item):
        require(record(item['path']) == item, 'Pinned source differs: ' + item['path'])
        self.pins[item['path']] = item
        return Path(item['path'])

    def read(self, item):
        return json.loads(self.verify(item).read_text())

    def array(self, item, *, shape=None, dtype=None, finite=False):
        array = np.load(self.verify(item), mmap_mode='r', allow_pickle=False)
        if shape is not None:
            require(array.shape == shape, 'Array shape differs: ' + item['path'])
        if dtype is not None:
            require(array.dtype == dtype, 'Array dtype differs: ' + item['path'])
        if finite:
            require(np.isfinite(array).all(), 'Nonfinite array: ' + item['path'])
        return array

    def close(self, actual, expected, field, tolerance=1e-12):
        require(math.isfinite(actual) and math.isfinite(expected), 'Nonfinite scalar: ' + field)
        error = abs(float(actual) - float(expected))
        require(error <= tolerance, f'Independent mismatch {field}: absolute error {error}')
        self.maximum_absolute_errors[field] = max(self.maximum_absolute_errors.get(field, 0.), error)
        self.comparisons += 1

    def loss_record(self, reported, expected, prefix, tolerance=1e-12):
        require(set(reported) == set(expected), 'Loss fields changed: ' + prefix)
        for name, value in expected.items():
            self.close(reported[name], value, prefix + '/' + name, tolerance)

    def finish(self):
        for item in self.pins.values():
            require(record(item['path']) == item, 'Source changed during review: ' + item['path'])
        require('torch' not in sys.modules, 'Torch must not be imported')
