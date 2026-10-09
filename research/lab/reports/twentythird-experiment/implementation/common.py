"""Explicit source pins and offline helpers for one native development comparison."""
from __future__ import annotations
import ast
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
LAB=PHASE.parent.parent
FORBIDDEN={'network':0,'torch_load':0}


def offline(*args,**kwargs):
    FORBIDDEN['network']+=1
    raise RuntimeError('Network forbidden in phase23')


for name in ('create_connection','getaddrinfo'):
    setattr(socket,name,offline)
socket.socket.connect=offline
socket.socket.connect_ex=offline


def require(value,message):
    if not value: raise ValueError(message)


def pin(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),sha256=digest,size_bytes=path.stat().st_size)


def verify(record):
    require(pin(record['path'])=={k:record[k] for k in ('path','sha256','size_bytes')},'File pin changed: '+record['path'])
    return Path(record['path'])


def read_json(record): return json.loads(verify(record).read_text())


def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def module(name,record):
    require(name not in sys.modules,'Module namespace already used: '+name)
    loader=importlib.machinery.SourceFileLoader(name,str(verify(record)))
    spec=importlib.util.spec_from_loader(name,loader)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;loader.exec_module(result)
    return result


def qualified_helpers(plan):
    """Unchanged phase22 factory helper and one unchanged state-inspection AST node."""
    import torch
    factory=module('phase23_qualified_factory',plan['sources']['phase22_factory_helper'])
    path=verify(plan['sources']['phase22_state_source'])
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='state_receipt']
    require(len(nodes)==1,'Qualified state function selection differs')
    tree=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    scope=dict(torch=torch,hashlib=hashlib,json=json,require=require)
    exec(compile(tree,str(path)+':state_receipt','exec'),scope)
    return factory,scope['state_receipt']


def load_array(record):
    import numpy as np
    result=np.load(verify(record),allow_pickle=False,mmap_mode='r')
    if 'shape' in record: require(list(result.shape)==record['shape'],'Array shape differs')
    if 'dtype' in record: require(result.dtype.str==record['dtype'],'Array dtype differs')
    require(np.isfinite(result).all(),'Nonfinite source array')
    if 'raw_c_order_sha256' in record:
        require(hashlib.sha256(result.tobytes(order='C')).hexdigest()==record['raw_c_order_sha256'],'Array raw bytes differ')
    return result


def load_plan(path,release):
    plan=json.loads(Path(path).read_text())
    require(plan['kind']=='phase23-native-reused-development-comparison-v1','Wrong plan kind')
    for record in plan['sources'].values(): verify(record)
    for name,record in plan['code'].items(): require(verify(record)==HERE/name,'Code location differs')
    require(set(plan['code'])=={p.name for p in HERE.glob('*.py')},'Executable set differs')
    freeze=read_json(plan['code_freeze'])
    require(freeze['code']==plan['code'] and freeze['protocol']==plan['sources']['protocol'],'Code freeze differs')
    go=json.loads(Path(release).read_text())
    require(go['status']=='GO' and go['plan']==pin(path) and go['code_freeze']==plan['code_freeze'],'Bound root GO required')
    contract=read_json(plan['sources']['input_contract'])
    require(contract['status']=='PASS_RETAINED_SEED42_OUTPUTS_REUSABLE_UNDER_THIS_EXACT_INPUT_CONTRACT'
        and contract['total_contexts']==6 and contract['total_captured_tiles']==150,'Input audit not ready')
    require(read_json(plan['sources']['input_verification'])['status']=='PASS','Input verification not PASS')
    require(read_json(plan['sources']['phase22_independent_verification'])['status']=='PASS','Checkpoint parity unqualified')
    require([c['id'] for c in contract['cases']]==plan['case_ids'],'Case order differs')
    return plan,contract


def verify_runtime_sources(plan):
    for item in plan['sources'].values(): verify(item)
    sources=read_json(plan['sources']['phase22_plan'])['sources']
    closure=read_json(sources['historical_closure'])
    verify(closure['authored_configuration'])
    vendor=read_json(sources['current_vendor_manifest'])
    root=Path(sources['current_vendor_manifest']['path']).parent
    for item in vendor['files']:
        require(pin(root/item['path'])['sha256']==item['sha256'],'Qualified vendor source changed')
    verify(sources['baseline_adapter'])


def check_coverage(scores,evaluated,mask):
    import numpy as np
    require(scores.shape==evaluated.shape==mask.shape,'Coverage shape differs')
    require(evaluated.dtype==mask.dtype==np.bool_,'Coverage mask type differs')
    require(mask.any() and evaluated[mask].all(),'Missing primary coverage; no mask intersection allowed')
    require(np.isfinite(scores[mask]).all() and ((scores[mask]>=0)&(scores[mask]<=1)).all(),'Invalid primary probabilities')


def encode(raw):
    import numpy as np
    raw=np.asarray(raw)
    require(raw.dtype==np.float32 and np.isfinite(raw).all(),'Encoding requires finite FP32')
    byte=(np.clip(raw,0,1)*np.float32(255)).astype(np.uint8)
    decoded=np.clip((byte.astype(np.float32)-64)/128,0,1)
    return byte,decoded
