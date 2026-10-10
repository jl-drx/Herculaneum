"""Pinned local reads for independent reconstruction; no checkpoint decodes."""
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import tempfile
import numpy as np
from review_math import require


def forbidden_network(*args, **kwargs):
    raise RuntimeError('Network forbidden in independent review')


socket.create_connection = forbidden_network
socket.socket.connect = forbidden_network
socket.socket.connect_ex = forbidden_network

CHECKPOINT_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin'}


def record(path):
    path = Path(path).resolve(strict=True)
    require(path.suffix.lower() not in CHECKPOINT_SUFFIXES, 'Checkpoint reads forbidden')
    require(any(path.is_relative_to(root) for root in (Path('/workspace'), Path('/tmp'))),
            'Input path must be local to workspace or temporary storage')
    require(path.is_file(), 'Regular file required')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


class Review:
    def __init__(self):
        self.pins = {}
        self.scalar_comparisons = 0
        self.maximum_absolute_error = {}
        self.array_reads = 0

    def verify(self, pin):
        require(record(pin['path']) == pin, 'Pinned input changed: ' + pin['path'])
        self.pins[pin['path']] = pin
        return Path(pin['path'])

    def read_json(self, pin):
        return json.loads(self.verify(pin).read_text())

    def read_jsonl(self, pin):
        return [json.loads(line) for line in self.verify(pin).read_text().splitlines()]

    def array(self, pin, *, shape=None, dtype=None, finite=False):
        path = self.verify(pin)
        require(path.suffix == '.npy', 'Only uncompressed NumPy arrays accepted')
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        self.array_reads += 1
        require(isinstance(value, np.ndarray) and not value.dtype.hasobject, 'Plain array required')
        if shape is not None:
            require(value.shape == tuple(shape), 'Array shape changed: ' + pin['path'])
        if dtype is not None:
            require(value.dtype == np.dtype(dtype), 'Array dtype changed: ' + pin['path'])
        if finite:
            require(np.isfinite(value).all(), 'Nonfinite array: ' + pin['path'])
        return value

    def close(self, actual, expected, label, tolerance=1e-12):
        if expected is None:
            require(actual is None, 'Undefined metric differs: ' + label)
            return
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and math.isfinite(expected),
                'Finite scalar required: ' + label)
        error = abs(float(actual) - float(expected))
        require(error <= tolerance, f'Independent mismatch {label}: error {error} > {tolerance}')
        self.scalar_comparisons += 1
        self.maximum_absolute_error[label] = max(error, self.maximum_absolute_error.get(label, 0.))

    def finish(self):
        for pin in self.pins.values():
            require(record(pin['path']) == pin, 'Input changed during review: ' + pin['path'])
        require('torch' not in sys.modules, 'Torch/model imports forbidden')
        return dict(input_pins=list(self.pins.values()), array_reads=self.array_reads,
                    scalar_comparisons=self.scalar_comparisons,
                    maximum_absolute_error=self.maximum_absolute_error,
                    torch_imports=0, checkpoint_reads=0, network_calls=0)


def fixture_checks():
    with tempfile.TemporaryDirectory(prefix='phase20-independent-io-', dir='/tmp') as temporary:
        root = Path(temporary)
        path = root/'fixture.json'
        path.write_text('{"synthetic":true}\n')
        pin = record(path)
        require(Review().read_json(pin) == {'synthetic': True}, 'Pinned JSON fixture failed')
        path.write_text('{"synthetic":false}\n')
        try:
            Review().verify(pin)
        except ValueError:
            pass
        else:
            raise ValueError('Changed pin accepted')
        forbidden = root/'fake.pth'
        forbidden.write_bytes(b'not a checkpoint; synthetic path refusal fixture')
        try:
            record(forbidden)
        except ValueError:
            pass
        else:
            raise ValueError('Checkpoint-suffix read accepted')
    require('torch' not in sys.modules, 'Torch imported in independent fixture process')
    return dict(status='PASS', pin_identity_cases=1, changed_pin_rejection_cases=1,
                forbidden_checkpoint_rejection_cases=1, torch_imports=0, scientific_array_reads=0)
