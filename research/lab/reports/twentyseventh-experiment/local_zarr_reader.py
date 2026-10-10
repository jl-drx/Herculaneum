"""Strict single-writer, local synthetic Zarr2 adapter. No HTTP or cloud client."""
import hashlib
import json
import math
from pathlib import Path
import os
import struct
import sys

import numpy as np
import numcodecs

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'twentysixth-experiment'))
from reference_mechanics import VolumeSpec, require, plan_samples, sample_trilinear
from fetch_metadata import decode_json

COMPRESSOR=dict(id='blosc',cname='zstd',clevel=3,shuffle=0,blocksize=0)


def sample_checked(points, reader, **limits):
    plan=plan_samples(points,reader.spec,**limits)
    require(len(plan.groups)*reader.physical_bytes<=limits.get('max_bytes',64*1024**2),
            'Physical storage chunk budget before loading')
    return sample_trilinear(points,reader.spec,reader,**limits)


def digest(data):return hashlib.sha256(data).hexdigest()


def bounded_read(path, limit):
    path=Path(path)
    require(not path.is_symlink() and path.is_file(),'Missing or symlink object')
    require(path.stat().st_size<=limit,'Object byte limit before read')
    with path.open('rb') as stream:data=stream.read(limit+1)
    require(len(data)<=limit,'Object byte limit after read')
    return data


def schema(attrs, group, array):
    require(group=={'zarr_format':2},'Exact v2 group')
    m=attrs.get('multiscales')
    require(isinstance(m,list) and len(m)==1,'One multiscale image')
    require(m[0].get('version')=='0.4','NGFF version')
    axes=m[0].get('axes')
    require(axes==[{'name':a,'type':'space'} for a in 'zyx'],'Explicit unitless ZYX axes')
    require('coordinateTransformations' not in m[0],'Unexpected root transform')
    require(m[0]['datasets'][0]=={'path':'0','coordinateTransformations':[{'type':'scale','scale':[1,1,1]}]},'Exact native dataset transform')
    require(set(array)=={'shape','chunks','dtype','fill_value','order','filters','dimension_separator','compressor','zarr_format'},'Unsupported array schema')
    require(array['zarr_format']==2 and array['dtype']=='<u2' and array['order']=='C','Format/dtype/order')
    require(array['dimension_separator']=='/' and array['compressor']==COMPRESSOR,'Encoding mismatch')
    require(array['filters'] is None and type(array['fill_value']) is int and array['fill_value']==0,'Filters/fill mismatch')
    spec=VolumeSpec(tuple(array['shape']),tuple(array['chunks']),array['dtype'])
    require(math.prod(spec.chunks_zyx)*2<=4*1024**2,'Decoded physical chunk cap')
    return spec


class LocalReader:
    def __init__(self, root, manifest, *, cache=None, cache_limit=8*1024**2, interrupt_before_publish=False):
        self.root=Path(root).resolve(strict=True)
        # Detach caller-owned state. Manifest is a trusted, separately retained fixture identity.
        self.manifest=json.loads(json.dumps(manifest))
        require(set(self.manifest)=={'metadata','objects'},'Manifest schema')
        require(set(self.manifest['metadata'])=={'.zattrs','.zgroup','0/.zarray'},'Bound metadata set')
        self.spec=self._check_metadata()
        self.physical_bytes=math.prod(self.spec.chunks_zyx)*2
        self.codec=numcodecs.Blosc(**{k:v for k,v in COMPRESSOR.items() if k!='id'})
        self.cache=Path(cache) if cache is not None else None
        if self.cache is not None:
            self.cache.mkdir(parents=True,exist_ok=True)
            require(not self.cache.is_symlink(),'Symlink cache')
        require(type(cache_limit) is int and cache_limit>=0,'Cache budget')
        self.cache_limit=cache_limit;self.interrupt_before_publish=interrupt_before_publish
        self.stats=dict(source_reads=0,cache_hits=0,missing=0,decoded=0,source_bytes=0)

    def _path(self, name):
        path=self.root/name
        require(path.resolve(strict=False).is_relative_to(self.root),'Object escapes root')
        require(not any(p.is_symlink() for p in [path,*list(path.parents)[:len(path.relative_to(self.root).parts)-1]]),'Symlink path')
        return path

    def _check_metadata(self):
        data={}
        for name,item in self.manifest['metadata'].items():
            body=bounded_read(self._path(name),65536)
            require(len(body)==item['size_bytes'] and digest(body)==item['sha256'],'Changed metadata')
            data[name]=decode_json(body)
        return schema(data['.zattrs'],data['.zgroup'],data['0/.zarray'])

    def _decode(self, body):
        require(len(body)>=16,'Truncated Blosc header')
        version,codecversion,flags,typesize,nbytes,blocksize,cbytes=struct.unpack('<BBBBIII',body[:16])
        require(version==2 and codecversion==1,'Unsupported Blosc header version')
        require(typesize==2 and nbytes==self.physical_bytes and cbytes==len(body),'Blosc size/type mismatch')
        require(0<blocksize<=nbytes and blocksize%2==0,'Invalid Blosc block size')
        require(flags & 5 == 0,'Unexpected byte/bit shuffle')
        require(numcodecs.blosc.cbuffer_complib(body)=='Zstd','Wrong compressor header')
        if flags & 2:require(len(body)==nbytes+16,'Invalid memcpy payload')
        output=np.empty(self.spec.chunks_zyx,dtype='<u2',order='C')
        self.codec.decode(body,out=output)
        self.stats['decoded']+=1
        return output

    def __call__(self,key):
        self._check_metadata()
        edge=self.spec.chunk_shape(key)
        name='0/'+'/'.join(map(str,key))
        item=self.manifest['objects'].get(name)
        if item is None:self.stats['missing']+=1;return None
        require(type(item['size_bytes']) is int and 16<=item['size_bytes']<=self.physical_bytes+16,'Encoded object bound')
        sha=item['sha256']
        require(isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha),'Digest syntax')
        cached=self.cache/sha if self.cache is not None else None
        if cached is not None and cached.exists():
            body=bounded_read(cached,item['size_bytes']);self.stats['cache_hits']+=1
        else:
            body=bounded_read(self._path(name),item['size_bytes'])
            self.stats['source_reads']+=1;self.stats['source_bytes']+=len(body)
        require(len(body)==item['size_bytes'] and digest(body)==sha,'Changed encoded object')
        decoded=self._decode(body)
        if cached is not None and not cached.exists():
            partial=cached.with_suffix('.partial')
            # Retained interrupted files count against the disk cap. A matching partial is overwritten, never read as complete.
            files=list(self.cache.iterdir())
            require(all(p.is_file() and not p.is_symlink() for p in files),'Unexpected cache entry')
            usage=sum(p.stat().st_size for p in files if p!=partial)
            require(usage+len(body)<=self.cache_limit,'Cache disk budget exceeded')
            with partial.open('wb') as stream:
                stream.write(body);stream.flush();os.fsync(stream.fileno())
            if self.interrupt_before_publish:raise InterruptedError('Injected before atomic cache rename')
            partial.replace(cached)
        # Copy releases the padded physical buffer after this function returns.
        return decoded[tuple(slice(0,s) for s in edge)].copy(order='C')
