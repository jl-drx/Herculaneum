"""Source-bound isolated factories; importing this module does not build a model."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
import types
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
LAB = PHASE.parent.parent
INPUT = PHASE / 'native-input-contract'
STUB_COUNTS = {'export_reads': {}, 'runtime_calls': {}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'size_bytes': path.stat().st_size}


def verify(record):
    require(pin(record['path']) == {k: record[k] for k in ('path', 'sha256', 'size_bytes')},
            'Source pin mismatch: ' + record['path'])
    return Path(record['path'])


def read_json(record):
    return json.loads(verify(record).read_text())


def load_module(name, record):
    path = verify(record)
    if 'git_blob_sha1' in record:
        body = path.read_bytes()
        require(hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==record['git_blob_sha1'],
                'Historical Git blob mismatch: '+name)
    require(name not in sys.modules, 'Namespace collision: ' + name)
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def namespace(name):
    require(name not in sys.modules, 'Namespace collision: ' + name)
    module = types.ModuleType(name)
    module.__path__ = []
    module.__package__ = name
    sys.modules[name] = module


def inactive_stub(name, exports):
    """Permit named import lookup, fail every runtime constructor/function use."""
    require(name not in sys.modules, 'Stub namespace collision')
    module = types.ModuleType(name)
    module.__file__ = '<phase22-fail-closed-inactive-stub>'
    export_values = {}
    for export in exports:
        key = name + '.' + export
        STUB_COUNTS['export_reads'][key] = 0
        STUB_COUNTS['runtime_calls'][key] = 0
        def fail(*args, _key=key, **kwargs):
            STUB_COUNTS['runtime_calls'][_key] += 1
            raise RuntimeError('Inactive architecture invoked: ' + _key)
        if export[0].isupper():
            meta = type('InactiveMeta', (type,), {
                '__call__': lambda cls, *a, _fail=fail, **kw: _fail(*a, **kw),
                '__instancecheck__': lambda cls, obj, _fail=fail: _fail(obj),
                '__subclasscheck__': lambda cls, obj, _fail=fail: _fail(obj),
            })
            export_values[export] = meta(export, (), {})
        else:
            export_values[export] = fail
    def getattribute(attr):
        if attr == '__path__':
            raise AttributeError(attr)
        require(attr in export_values, 'Unexpected inactive module export: ' + name + '.' + attr)
        STUB_COUNTS['export_reads'][name + '.' + attr] += 1
        return export_values[attr]
    module.__getattr__ = getattribute
    sys.modules[name] = module


EXTRACT_NAMES = [
    'Block', 'ChunkKey', 'TargetHeadWrapper', 'disable_z_projection_for_normal_pooled_3d',
    'normalize_training_config_for_inference', 'compute_importance_map_2d',
    'resolve_patch_stride', 'center_crop_layer_indices', '_sliding_positions_1d',
    'iter_blocks', 'logits_to_probabilities', 'OmeZarrPatchReader', 'convert_volume_dtype',
    'is_url_like_path', 'iter_overlapping_chunks', 'compute_chunk_contribution_counts',
    'ChunkAccumulator', 'OmeZarrBlockDataset', 'run_block_inference', 'iter_probability_tiles',
]


def extraction_records(source_record):
    path = verify(source_record)
    text = path.read_text()
    nodes = [node for node in ast.parse(text).body if getattr(node, 'name', None) in EXTRACT_NAMES]
    require(sorted(n.name for n in nodes) == sorted(EXTRACT_NAMES), 'Extraction selection mismatch')
    lines = text.splitlines(keepends=True)
    return [{'name': node.name, 'first_line': min([node.lineno] + [d.lineno for d in node.decorator_list]),
             'last_line': node.end_lineno,
             'sha256': hashlib.sha256(''.join(lines[min([node.lineno] + [d.lineno for d in node.decorator_list])-1:node.end_lineno]).encode()).hexdigest()}
            for node in nodes]


def historical_inference(source_record):
    """Compile unchanged selected AST definitions only; no CLI/import/main execution."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    path = verify(source_record)
    parsed = ast.parse(path.read_text())
    nodes = [n for n in parsed.body if getattr(n, 'name', None) in EXTRACT_NAMES]
    require(sorted(n.name for n in nodes) == sorted(EXTRACT_NAMES), 'Extraction definitions mismatch')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    name = 'phase22_historical_inference_extraction'
    require(name not in sys.modules, 'Inference extraction already loaded')
    module = types.ModuleType(name)
    sys.modules[name] = module
    def forbidden(*args, **kwargs):
        raise RuntimeError('Excluded historical inference path invoked')
    module.__dict__.update(dict(np=np, torch=torch, nn=torch.nn, F=F, dataclass=dataclass,
        math=math, copy=copy, Path=Path, urlparse=urlparse, nullcontext=nullcontext, Dataset=Dataset,
        DataLoader=DataLoader, LOGGER=logging.getLogger(name),
        tqdm=lambda iterable, **kwargs: iterable, open_zarr_readonly=forbidden,
        predict_with_mirror_tta=forbidden, iter_mirror_axes=forbidden))
    exec(compile(tree, str(path) + ':selected-definitions', 'exec'), module.__dict__)
    return module


def config_guards(config):
    require(config['model_type'] == 'vesuvius_unet_3d_stem_2d' and config['mode'] == 'flat', 'Factory family differs')
    require(config['patch_size'] == [17, 128, 128] and config['in_channels'] == 1, 'Input config differs')
    require(config['model_config'] == {'autoconfigure': True, 'basic_decoder_block': 'ConvBlock',
        'basic_encoder_block': 'BasicBlockD', 'stem_channels': 16, 'z_projection_mode': 'none'}, 'Active settings differ')
    require(config['targets'] == {'ink': {'activation': 'none', 'out_channels': 1, 'z_projection_mode': 'none'}}, 'Target config differs')
    require(not config.get('enable_deep_supervision', False) and not config.get('pretrained_backbone'), 'Excluded config active')


def build_factory(which, sources):
    """Called only by separately authorized worker, after qualification receipts pass."""
    require(which in ('historical', 'current'), 'Unknown factory')
    closure = read_json(sources['historical_closure'])
    config = read_json(closure['authored_configuration'])
    config_guards(config)
    if which == 'historical':
        for name in closure['namespace_only_parents']:
            namespace(name)
        for name in ('vesuvius.image_proc', 'vesuvius.image_proc.intensity'):
            namespace(name)
        for item in closure['fail_closed_inactive_imports']:
            inactive_stub(item['module'], item['exports'])
        indexed = {item['module_name']: item for item in closure['modules']}
        for name in closure['suggested_load_order']:
            load_module(name, indexed[name])
        norm = load_module('vesuvius.image_proc.intensity.normalization', closure['normalization_module_for_separate_input_parity'])
        load_module('koine_machines.models.input_padding', closure['historical_padding_module_for_separate_input_parity'])
        inference = historical_inference(closure['historical_inference_source_for_reviewed_pure_extraction'])
        normalized_config = inference.normalize_training_config_for_inference(config)
        model = sys.modules['koine_machines.models.make_model'].make_model(normalized_config)
        require(not any(STUB_COUNTS['runtime_calls'].values()), 'Inactive path used')
        return model, norm.normalize_robust, inference, normalized_config
    baseline = load_module('phase22_current_baseline', sources['baseline_adapter'])
    InkConfig, make_model, normalizer, manifest_hash = baseline.load_focused_sources()
    require(manifest_hash == sources['current_vendor_manifest']['sha256'], 'Vendor manifest mismatch')
    typed = InkConfig.from_mapping(config)
    require(typed.model.input_pad_depth_to is None and typed.model.pretrained_backbone is None, 'Unexpected current options')
    model = make_model(typed)
    return model, normalizer, baseline, config
