"""Independent Stage1 verification: safetensors library, no author-code imports."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time

PROTOCOL = Path('/workspace/herculaneum-lab/reports/twentysecond-experiment/CHECKPOINT_QUALIFICATION_PROTOCOL.md')
PROTOCOL_SHA = '07ceebb353ec771d228e6e71c82a2edd9528b99cb9e4b1beaf15c0fc3b085078'
SOURCE = Path('/workspace/herculaneum-lab/reports/twentyfirst-experiment/native-checkpoint-contract')
REFERENCE_SHA = 'ae6665633364e584a41b4729c6751ca9a880242e97f87a63925c1d162ad45c1f'
EXPECTED = dict(size_bytes=272767904,
 sha256='d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d',
 prefix_sha256='b94f491fdf06f1bf3f8592ca9c5c3f575fa9ca8464f76296559b94a752f1271f',
 header_sha256='b58a17c22694c0083156de30f02cb58838777eb4716fdb56e76a95d5b2962a24',
 header_bytes=66192, named_count=508, named_numel=68175426,
 data_bytes=272701704, alias_pair_count=244, group_count=264,
 unique_numel=34546498, unique_bytes=138185992)

class Reject(ValueError): pass

def require(condition, code):
    if not condition: raise Reject(code)

def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while part:=f.read(1024*1024): h.update(part)
    return h.hexdigest()

def pin(path):
    path=Path(path)
    return dict(path=str(path.resolve()),size_bytes=path.stat().st_size,sha256=sha_file(path))

def unique_pairs(pairs):
    result={}
    for k,v in pairs:
        require(k not in result,'duplicate_json_key')
        result[k]=v
    return result

def parse_header(raw):
    try: return json.loads(raw.decode('utf-8'),object_pairs_hook=unique_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(Reject('nonstandard_json_number')))
    except (UnicodeError,json.JSONDecodeError) as exc: raise Reject('malformed_header_json') from exc

def integer(x): return type(x) is int and x>=0

def verify_schema(header,reference,contract):
    require(type(header) is dict,'root_not_object')
    require('__metadata__' not in header,'unexpected_metadata')
    rows=reference['tensor_schema']
    require(len(rows)==contract['named_count'],'reference_count')
    expected={r['name']:r for r in rows}
    require(len(expected)==len(rows),'reference_duplicate_name')
    require(set(header)==set(expected),'tensor_names')
    intervals=[]; total=0
    for name,row in expected.items():
        item=header[name]
        require(type(item) is dict and set(item)=={'dtype','shape','data_offsets'},'tensor_fields')
        shape=item['shape']; offsets=item['data_offsets']
        require(item['dtype']=='F32' and row['dtype']=='F32','tensor_dtype')
        require(type(shape) is list and all(integer(x) for x in shape),'shape_integer')
        require(shape==row['shape'],'tensor_shape')
        count=math.prod(shape)
        require(count==row['numel'] and 4*count==row['data_bytes'],'reference_numel')
        require(type(offsets) is list and len(offsets)==2 and all(integer(x) for x in offsets),'offset_integer')
        lo,hi=offsets
        require(hi>=lo and hi-lo==4*count and hi<=contract['data_bytes'],'tensor_extent')
        intervals.append((lo,hi,name)); total+=count
    cursor=0
    for lo,hi,name in sorted(intervals):
        require(lo==cursor,'noncontiguous_or_overlapping_intervals'); cursor=hi
    require(cursor==contract['data_bytes'] and total==contract['named_numel'],'data_totals')
    aliases=reference['expected_factory_alias_groups']
    require(len(aliases)==contract['alias_pair_count'],'alias_pair_count')
    grouped=set(); groups=[]
    for pair in aliases:
        names=pair['names']
        require(type(names) is list and len(names)==2 and len(set(names))==2,'alias_pair_definition')
        require(all(n in expected and n not in grouped for n in names),'alias_partition')
        require(all(expected[n]['shape']==pair['shape'] for n in names),'alias_shape')
        grouped.update(names); groups.append(names)
    groups.extend([[n] for n in sorted(set(expected)-grouped)])
    unique_numel=sum(expected[names[0]]['numel'] for names in groups)
    require(len(groups)==contract['group_count'],'group_count')
    require(unique_numel==contract['unique_numel'] and 4*unique_numel==contract['unique_bytes'],'unique_totals')
    return expected,groups

def verify_file(path,reference,contract,library_open=None):
    """Complete integrity and metadata checks precede the first get_tensor call."""
    path=Path(path); identity=pin(path)
    require(identity['size_bytes']==contract['size_bytes'],'whole_size')
    require(identity['sha256']==contract['sha256'],'whole_sha256')
    with path.open('rb') as f:
        prefix=f.read(8)
        require(len(prefix)==8 and hashlib.sha256(prefix).hexdigest()==contract['prefix_sha256'],'prefix_sha256')
        h=struct.unpack('<Q',prefix)[0]
        require(h==contract['header_bytes'] and h%8==0,'header_length')
        raw=f.read(h)
        require(len(raw)==h and hashlib.sha256(raw).hexdigest()==contract['header_sha256'],'header_sha256')
    require(8+h+contract['data_bytes']==identity['size_bytes'],'file_extent')
    header=parse_header(raw)
    expected,groups=verify_schema(header,reference,contract)
    # Only now import and open the distinct format implementation.
    import numpy as np
    import safetensors
    if library_open is None:
        from safetensors import safe_open
        library_open=safe_open
    finite=[]; pairs=[]; actual_schema=[]
    with library_open(str(path),framework='numpy') as tensors:
        require(set(tensors.keys())==set(expected),'library_tensor_names')
        require(tensors.metadata() is None,'library_metadata')
        for name,row in sorted(expected.items()):
            value=tensors.get_tensor(name)
            require(value.dtype==np.dtype('float32') and list(value.shape)==row['shape'],'library_dtype_shape')
            require(value.size==row['numel'] and value.nbytes==row['data_bytes'],'library_tensor_size')
            require(bool(np.isfinite(value).all()),'nonfinite_tensor:'+name)
            finite.append(name)
            actual_schema.append(dict(name=name,shape=list(value.shape),dtype='F32',numel=value.size,data_bytes=value.nbytes))
            del value
        for pair in reference['expected_factory_alias_groups']:
            left,right=pair['names']; a=tensors.get_tensor(left); b=tensors.get_tensor(right)
            # Byte-view exactness includes signed zero and all finite float encodings.
            same=bool(np.array_equal(a.view(np.uint8),b.view(np.uint8)))
            require(same,'alias_bit_mismatch:'+left)
            pairs.append(dict(names=[left,right],bitwise_equal=True,bytes=a.nbytes,
                left_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest(),
                right_sha256=hashlib.sha256(b.tobytes(order='C')).hexdigest()))
            del a,b
    # Check the source file has not changed during the offline scan.
    require(pin(path)==identity,'artifact_changed_during_verification')
    return dict(status='PASS',method='safetensors.safe_open NumPy; independent source and byte comparisons',
        artifact=identity,safetensors_version=safetensors.__version__,numpy_version=np.__version__,
        named_tensor_count=len(finite),named_numel=contract['named_numel'],
        serialized_data_bytes=contract['data_bytes'],all_named_tensors_finite=True,
        finite_checked_names=finite,actual_schema=actual_schema,
        alias_pairs=pairs,alias_pair_count=len(pairs),conditional_factory_group_count=len(groups),
        conditional_factory_unique_parameters=contract['unique_numel'],
        conditional_factory_unique_bytes=contract['unique_bytes'],
        serialized_storage='508 distinct contiguous nonoverlapping intervals; alias equality does not make them shared storage',
        model_instantiations=0,forwards=0,optimizer_updates=0,network_calls=0,
        strict_loading_verified=False,forward_parity_verified=False,accuracy_claim=False)

def deny_network(event,args):
    if event.startswith('socket.'):
        raise Reject('network_forbidden')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact',required=True)
    parser.add_argument('--root-release',required=True,help='Root-created release after verified download; binds artifact path and whole SHA')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    started=time.monotonic(); cpu=time.process_time()
    # Process-level hard ceilings; external launcher must enforce 180 seconds wall.
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    sys.addaudithook(deny_network)
    result={}
    try:
        require(sha_file(PROTOCOL)==PROTOCOL_SHA,'protocol_pin')
        ref_path=SOURCE/'REFERENCE_SCHEMA.json'
        require(sha_file(ref_path)==REFERENCE_SHA,'reference_pin')
        require(sha_file(SOURCE/'query-run/prefix.bin')==EXPECTED['prefix_sha256'],'retained_prefix_pin')
        require(sha_file(SOURCE/'query-run/header.json')==EXPECTED['header_sha256'],'retained_header_pin')
        release_path=Path(args.root_release)
        release=json.loads(release_path.read_text())
        require(release.get('status')=='GO_INDEPENDENT_OFFLINE_VERIFICATION','root_release_status')
        require(Path(release['artifact']['path']).resolve()==Path(args.artifact).resolve(),'root_release_path')
        require(release['artifact']['sha256']==EXPECTED['sha256'] and release['artifact']['size_bytes']==EXPECTED['size_bytes'],'root_release_identity')
        result=verify_file(args.artifact,json.loads(ref_path.read_text()),EXPECTED)
        result['root_release']=pin(release_path)
        result['protocol']=pin(PROTOCOL)
        result['reference']=pin(ref_path)
    except Exception as exc:
        result=dict(status='STOPPED',failure_type=type(exc).__name__,failure_code=str(exc) if isinstance(exc,Reject) else 'sanitized_unexpected_exception')
    result['wall_seconds']=time.monotonic()-started
    result['cpu_seconds']=time.process_time()-cpu
    result['max_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if result['max_rss_bytes']>4*1024**3:
        result['status']='STOPPED'; result['failure_code']='rss_limit'
    if result['wall_seconds']>180:
        result['status']='STOPPED'; result['failure_code']='wall_limit'
    out=Path(args.output)
    with out.open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(status=result['status'],output=str(out),size_bytes=out.stat().st_size,sha256=sha_file(out))))
    return 0 if result['status']=='PASS' else 2

if __name__=='__main__': raise SystemExit(main())
