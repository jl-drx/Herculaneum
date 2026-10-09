"""Offline byte pinning and assertions; model weight files are never read."""
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import numpy as np
from integral_support import require


def offline(*args, **kwargs):
    raise RuntimeError('Network is forbidden in independent numeric review')


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline
WEIGHT_SUFFIXES = {'.pth','.pt','.ckpt','.safetensors'}


def identity(item):
    sizes = [item[k] for k in ('size_bytes','byte_length','bytes') if k in item]
    require(sizes and all(type(v) is int and v >= 0 and v == sizes[0] for v in sizes), 'Invalid size pin')
    path = Path(item['path'])
    require(path.is_absolute() and (str(path).startswith('/workspace/') or str(path).startswith('/tmp/')),
            'Absolute local workspace pin required')
    digest = item['sha256']
    require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
            'Invalid SHA256 pin')
    return {'path':str(path.resolve()),'size_bytes':sizes[0],'sha256':digest}


def record(path):
    path = Path(path).resolve(strict=True)
    require(path.suffix.lower() not in WEIGHT_SUFFIXES, 'Weight/checkpoint byte reads forbidden')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'size_bytes':path.stat().st_size,'sha256':digest}


class Review:
    def __init__(self):
        self.files = {}
        self.weight_metadata = {}
        self.scalar_comparisons = 0
        self.maximum_errors = {}

    def pin(self, item):
        item = identity(item)
        path = Path(item['path'])
        table = self.weight_metadata if path.suffix.lower() in WEIGHT_SUFFIXES else self.files
        if item['path'] in table:
            require(table[item['path']] == item, 'Conflicting source pin')
        else:
            if table is self.files:
                require(record(path) == item, 'Byte pin changed: ' + path.name)
            table[item['path']] = item
        return path

    def collect(self, value):
        if isinstance(value,dict):
            if {'path','sha256'} <= value.keys() and any(k in value for k in ('size_bytes','byte_length','bytes')):
                self.pin(value)
            for child in value.values():
                self.collect(child)
        elif isinstance(value,list):
            for child in value:
                self.collect(child)

    def json(self, item):
        value = json.loads(self.pin(item).read_text())
        self.collect(value)
        return value

    def array(self, item, *, shape=None, dtype=None, finite=False):
        require(Path(item['path']).suffix.lower() == '.npy', 'NPY arrays only')
        value = np.load(self.pin(item),mmap_mode='r',allow_pickle=False)
        if shape is not None: require(value.shape == shape, 'Array shape changed')
        if dtype is not None: require(value.dtype == np.dtype(dtype), 'Array dtype changed')
        if finite: require(np.isfinite(value).all(), 'Nonfinite retained array')
        return value

    def same(self, first, second, label):
        require(identity(first) == identity(second), 'Pinned identity mismatch: ' + label)

    def close(self, actual, expected, label, tolerance=1e-12):
        require((actual is None) == (expected is None), 'Undefined scalar mismatch: ' + label)
        if expected is None: return
        require(type(actual) in (int,float) and type(expected) in (int,float) and
                math.isfinite(actual) and math.isfinite(expected), 'Invalid scalar: ' + label)
        error = abs(actual-expected)
        require(error <= tolerance, f'Scalar mismatch {label}: {error}')
        self.scalar_comparisons += 1
        self.maximum_errors[label] = max(self.maximum_errors.get(label,0.),error)

    def tree(self, actual, expected, label):
        if isinstance(expected,dict):
            require(isinstance(actual,dict) and set(actual) == set(expected), 'Dictionary keys mismatch: ' + label)
            for k,v in expected.items(): self.tree(actual[k],v,label+'/'+k)
        elif isinstance(expected,list):
            require(isinstance(actual,list) and len(actual)==len(expected), 'List shape mismatch: ' + label)
            for i,(a,b) in enumerate(zip(actual,expected)): self.tree(a,b,label+'/'+str(i))
        elif isinstance(expected,bool):
            require(actual is expected, 'Boolean mismatch: ' + label)
        elif isinstance(expected,int):
            require(type(actual) is int and actual==expected, 'Exact integer mismatch: ' + label)
        elif isinstance(expected,str):
            require(actual==expected, 'Literal mismatch: ' + label)
        else:
            self.close(actual,expected,label)

    def unchanged(self):
        for item in self.files.values():
            require(record(item['path'])==item,'Source changed during review')
        require('torch' not in sys.modules,'Torch must not be imported')
