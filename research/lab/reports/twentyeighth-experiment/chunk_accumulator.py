"""One-pass local chunk accumulation. No persistent state or resume API."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

PRIOR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PRIOR/'twentysixth-experiment'))
from reference_mechanics import VolumeSpec,plan_samples,require


def frozen_array(value,dtype=None):
    a=np.array(value,dtype=dtype,copy=True,order='C');a.flags.writeable=False;return a


@dataclass(frozen=True)
class Group:
    key: tuple
    ids: np.ndarray
    local: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class Plan:
    spec: VolumeSpec
    points: np.ndarray
    mapped: np.ndarray
    geometric_valid: np.ndarray
    expected: np.ndarray
    groups: tuple
    batches: tuple
    batch_bytes: int
    batch_chunks: int
    plan_bytes: int
    sha256: str

    @property
    def physical_chunk_bytes(self):return math.prod(self.spec.chunks_zyx)*np.dtype(self.spec.dtype).itemsize

    def summary(self):
        return dict(points=len(self.points),geometrically_valid=int(self.geometric_valid.sum()),
            positive_contributions=int(self.expected.sum()),unique_chunks=len(self.groups),
            physical_chunk_bytes=self.physical_chunk_bytes,total_physical_bytes=len(self.groups)*self.physical_chunk_bytes,
            batch_count=len(self.batches),batch_chunks=[len(b) for b in self.batches],
            batch_physical_bytes=[len(b)*self.physical_chunk_bytes for b in self.batches],
            keys_zyx=[list(g.key) for g in self.groups],plan_array_bytes=self.plan_bytes,sha256=self.sha256)


def fingerprint(plan):
    h=hashlib.sha256()
    h.update(json.dumps(dict(shape=plan.spec.shape_zyx,chunks=plan.spec.chunks_zyx,dtype=plan.spec.dtype,
        keys=[g.key for g in plan.groups],batches=plan.batches,batch_bytes=plan.batch_bytes,
        batch_chunks=plan.batch_chunks,plan_bytes=plan.plan_bytes),sort_keys=True).encode())
    arrays=[plan.points,plan.mapped,plan.geometric_valid,plan.expected]
    for group in plan.groups:arrays.extend([group.ids,group.local,group.weights])
    for a in arrays:
        h.update(json.dumps([a.shape,a.dtype.str]).encode());h.update(a.tobytes(order='C'))
    return h.hexdigest()


def build_plan(points_xyz,spec,*,mapped=None,max_points=10000,max_total_chunks=128,
               max_total_bytes=512*1024**2,max_plan_bytes=16*1024**2,
               max_batch_chunks=128,max_batch_bytes=64*1024**2):
    for v in (max_points,max_total_chunks,max_total_bytes,max_plan_bytes,max_batch_chunks,max_batch_bytes):
        require(type(v) is int and v>=0,'Nonnegative integer limits')
    points=frozen_array(points_xyz,np.float64)
    require(points.ndim==2 and points.shape[1]==3 and len(points)<=max_points,'Point shape/budget')
    require(max(spec.shape_zyx)<2**31 and max(spec.chunks_zyx)<2**31,'Index range')
    if mapped is None:known=frozen_array(np.ones(len(points),dtype=bool))
    else:
        known=frozen_array(mapped);require(known.dtype==np.bool_ and known.shape==(len(points),),'Mapped mask')
    physical=math.prod(spec.chunks_zyx)*np.dtype(spec.dtype).itemsize
    valid=np.zeros(len(points),dtype=bool);expected=np.zeros(len(points),dtype=np.uint8)
    # Positive corner storage: int64 ID + three int32 indices + float64 weight.
    plan_bytes=points.nbytes+known.nbytes+valid.nbytes+expected.nbytes
    require(plan_bytes<=max_plan_bytes,'Plan buffer budget before source access')
    grouped={}
    for i,p in enumerate(points):
        one=plan_samples(p[None,:],spec,mapped=known[i:i+1],max_points=1,max_chunks=8,max_bytes=8*physical)
        valid[i]=one.geometric_valid[0]
        for key,entries in one.groups.items():
            if key not in grouped:
                require(physical<=max_batch_bytes and max_batch_chunks>0,'Physical chunk cannot fit batch')
                require(len(grouped)<max_total_chunks and (len(grouped)+1)*physical<=max_total_bytes,'Total support budget')
                grouped[key]=[]
            for ids,local,weights in entries:
                require(plan_bytes+len(ids)*28<=max_plan_bytes,'Plan contribution buffer budget')
                plan_bytes+=len(ids)*28;expected[i]+=len(ids)
                grouped[key].append((np.full(len(ids),i,dtype=np.int64),local.astype(np.int32),weights))
    groups=[]
    for key in sorted(grouped):
        rows=grouped[key]
        groups.append(Group(key,frozen_array(np.concatenate([r[0] for r in rows])),
            frozen_array(np.concatenate([r[1] for r in rows])),frozen_array(np.concatenate([r[2] for r in rows]))))
    batches=[];batch=[]
    for i in range(len(groups)):
        if batch and (len(batch)==max_batch_chunks or (len(batch)+1)*physical>max_batch_bytes):
            batches.append(tuple(batch));batch=[]
        batch.append(i)
    if batch:batches.append(tuple(batch))
    valid=frozen_array(valid);expected=frozen_array(expected)
    assert np.all(expected[valid]>0) and np.all(expected[~valid]==0)
    provisional=Plan(spec,points,known,valid,expected,tuple(groups),tuple(batches),max_batch_bytes,max_batch_chunks,plan_bytes,'')
    return Plan(**{**provisional.__dict__,'sha256':fingerprint(provisional)})


def execute(plan,loader,*,interrupt_after_chunks=None):
    require(fingerprint(plan)==plan.sha256,'Changed plan before source access')
    flat=[i for b in plan.batches for i in b]
    require(flat==list(range(len(plan.groups))),'Schedule must cover each group once')
    require(all(len(b)<=plan.batch_chunks and len(b)*plan.physical_chunk_bytes<=plan.batch_bytes for b in plan.batches),'Batch budget')
    if interrupt_after_chunks is not None:
        require(type(interrupt_after_chunks) is int and 0<=interrupt_after_chunks<=len(plan.groups),'Interrupt position')
    n=len(plan.points);values=np.zeros(n,dtype=np.float64);processed=np.zeros(n,dtype=np.uint8)
    missing=np.zeros(n,dtype=bool);nonfinite=np.zeros(n,dtype=bool)
    calls=[];missing_chunks=0;max_returned_bytes=0
    # No intermediate values are returned or persisted. Failure makes the invocation terminal.
    for batch in plan.batches:
        for index in batch:
            if interrupt_after_chunks is not None and len(calls)==interrupt_after_chunks:
                raise InterruptedError('Injected interruption; accumulator discarded, fresh run required')
            g=plan.groups[index];chunk=loader(g.key);calls.append(list(g.key))
            try:
                if chunk is None:
                    missing_chunks+=1;np.logical_or.at(missing,g.ids,True)
                else:
                    require(isinstance(chunk,np.ndarray),'Loader type')
                    require(chunk.shape==plan.spec.chunk_shape(g.key),'Logical edge shape mismatch')
                    require(chunk.dtype==np.dtype(plan.spec.dtype),'Source dtype/endianness mismatch')
                    max_returned_bytes=max(max_returned_bytes,chunk.nbytes)
                    raw=chunk[tuple(g.local.T)].astype(np.float64)
                    good=np.isfinite(raw)
                    np.logical_or.at(nonfinite,g.ids,~good)
                    np.add.at(values,g.ids,np.where(good,raw,0)*g.weights)
                    del raw,good
                np.add.at(processed,g.ids,np.uint8(1))
            finally:
                del chunk  # Release the returned buffer before the next loader call.
    if interrupt_after_chunks is not None and len(calls)==interrupt_after_chunks:
        raise InterruptedError('Injected after all chunks, before result publication')
    require(np.array_equal(processed,plan.expected),'Incomplete or duplicate positive contributions')
    valid=plan.geometric_valid & ~missing & ~nonfinite
    values[~valid]=np.nan
    # Flags remain separate even when a point has both missing and nonfinite support.
    return dict(values=values,valid=valid,geometric_valid=plan.geometric_valid.copy(),
        missing_source=missing,nonfinite_source=nonfinite,expected=plan.expected.copy(),processed=processed,
        stats=dict(loader_calls=calls,unique_chunks=len(calls),batch_count=len(plan.batches),missing_chunks=missing_chunks,
            max_scheduler_returned_chunks=1 if calls and max_returned_bytes else 0,
            max_scheduler_returned_bytes=max_returned_bytes,plan_sha256=plan.sha256,
            accumulator_array_bytes=sum(a.nbytes for a in [values,processed,missing,nonfinite,valid]),
            complete=True,resume_supported=False))
