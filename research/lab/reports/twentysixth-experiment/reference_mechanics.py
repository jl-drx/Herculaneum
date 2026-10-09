"""Offline reference-construction mechanics; no network or image/model dependencies."""
from dataclasses import dataclass
import itertools
import math
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class VolumeSpec:
    shape_zyx: tuple
    chunks_zyx: tuple
    dtype: str

    def __post_init__(self):
        for name in ('shape_zyx','chunks_zyx'):
            values=getattr(self,name)
            require(len(values)==3 and all(type(v) is int and v>0 for v in values),name)
        dtype=np.dtype(self.dtype)
        require((dtype.kind in 'ui' and dtype.itemsize<=4) or
                (dtype.kind=='f' and dtype.itemsize in (4,8)),
                'Supported sources: up to32-bit integers or FP32/FP64')

    def chunk_shape(self,key):
        require(len(key)==3 and all(type(v) is int and v>=0 for v in key),'Chunk key')
        result=tuple(min(c,s-k*c) for s,c,k in zip(self.shape_zyx,self.chunks_zyx,key))
        require(all(v>0 for v in result),'Chunk outside volume')
        return result


def ray_points(origins_xyz,directions_xyz,offsets_um,spacing_xyz_um,*,direction_space,mapped):
    origins=np.asarray(origins_xyz,dtype=np.float64)
    vectors=np.asarray(directions_xyz,dtype=np.float64)
    valid=np.asarray(mapped)
    offsets=np.asarray(offsets_um,dtype=np.float64)
    spacing=np.asarray(spacing_xyz_um,dtype=np.float64)
    require(origins.ndim==2 and origins.shape[1]==3 and vectors.shape==origins.shape,'Ray shapes')
    require(valid.dtype==np.bool_ and valid.shape==(len(origins),),'Explicit boolean mapped mask')
    require(offsets.ndim==1 and len(offsets)>0 and np.isfinite(offsets).all(),'Finite offsets')
    require(spacing.shape==(3,) and np.isfinite(spacing).all() and (spacing>0).all(),'Positive XYZ spacing')
    require(direction_space in ('physical_unit','voxel_unit_vector'),'Explicit vector convention required')
    require(np.isfinite(origins[valid]).all() and np.isfinite(vectors[valid]).all(),'Nonfinite mapped ray')
    lengths=np.linalg.norm(vectors[valid],axis=1)
    require(np.all(np.abs(lengths-1)<=1e-5),'Mapped vector not unit; no silent normalization')
    directions=np.zeros_like(vectors)
    if direction_space=='physical_unit':
        directions[valid]=vectors[valid]/spacing
    else:
        physical_lengths=np.linalg.norm(vectors[valid]*spacing,axis=1)
        directions[valid]=vectors[valid]/physical_lengths[:,None]
    points=np.full((len(offsets),len(origins),3),np.nan,dtype=np.float64)
    points[:,valid,:]=origins[None,valid,:]+offsets[:,None,None]*directions[None,valid,:]
    return points,np.broadcast_to(valid,(len(offsets),len(origins))).copy()


@dataclass
class SamplingPlan:
    spec: VolumeSpec
    point_count: int
    geometric_valid: np.ndarray
    groups: dict
    uncompressed_bytes: int
    positive_weight_corners: int

    def summary(self):
        return dict(point_count=self.point_count,geometric_valid_count=int(self.geometric_valid.sum()),
            unique_chunks=len(self.groups),uncompressed_source_bytes=self.uncompressed_bytes,
            positive_weight_corners=self.positive_weight_corners,
            chunks=[dict(key_zyx=list(key),shape_zyx=list(self.spec.chunk_shape(key)),
                         uncompressed_bytes=math.prod(self.spec.chunk_shape(key))*np.dtype(self.spec.dtype).itemsize)
                    for key in sorted(self.groups)])


def plan_samples(points_xyz,spec,*,mapped=None,max_points=100000,max_chunks=128,max_bytes=64*1024**2):
    points=np.asarray(points_xyz,dtype=np.float64)
    require(points.ndim==2 and points.shape[1]==3,'Points must be N by XYZ')
    require(len(points)<=max_points,'Point budget exceeded before loading')
    require(type(max_chunks) is int and max_chunks>=0 and type(max_bytes) is int and max_bytes>=0,'Budgets')
    if mapped is None:known=np.ones(len(points),dtype=bool)
    else:
        known=np.asarray(mapped)
        require(known.dtype==np.bool_ and known.shape==(len(points),),'Mapped shape/dtype')
    upper=np.array(spec.shape_zyx[::-1],dtype=np.float64)-1
    valid=known & np.isfinite(points).all(axis=1) & (points>=0).all(axis=1) & (points<=upper).all(axis=1)
    clean=np.where(valid[:,None],points,0)
    lower=np.floor(clean).astype(np.int64)
    fraction=clean-lower
    groups={};corners=0;byte_count=0
    for delta in itertools.product((0,1),repeat=3):
        delta=np.array(delta)
        weights=np.prod(np.where(delta,fraction,1-fraction),axis=1)
        ids=np.flatnonzero(valid & (weights>0))
        if not len(ids):continue
        indices=(lower[ids]+delta)[:,::-1]
        require((indices>=0).all() and (indices<np.array(spec.shape_zyx)).all(),'Contributing corner out of bounds')
        keys=indices//np.array(spec.chunks_zyx)
        for key_arr in np.unique(keys,axis=0):
            key=tuple(int(x) for x in key_arr)
            use=np.all(keys==key_arr,axis=1)
            if key not in groups:
                require(len(groups)<max_chunks,'Chunk budget exceeded before loading')
                byte_count+=math.prod(spec.chunk_shape(key))*np.dtype(spec.dtype).itemsize
                require(byte_count<=max_bytes,'Source byte budget exceeded before loading')
                groups[key]=[]
            local=indices[use]-key_arr*np.array(spec.chunks_zyx)
            groups[key].append((ids[use],local,weights[ids[use]]))
        corners+=len(ids)
    return SamplingPlan(spec,len(points),valid,groups,byte_count,corners)


def sample_trilinear(points_xyz,spec,loader,**limits):
    plan=plan_samples(points_xyz,spec,**limits)  # No loader access until the full plan fits.
    valid=plan.geometric_valid.copy()
    values=np.zeros(plan.point_count,dtype=np.float64)
    fetched=missing=0;calls=[]
    for key in sorted(plan.groups):
        calls.append(list(key))
        chunk=loader(key)
        if chunk is None:
            missing+=1
            for ids,_,_ in plan.groups[key]:valid[ids]=False
            continue
        require(isinstance(chunk,np.ndarray),'Loader must return ndarray or None')
        require(chunk.shape==plan.spec.chunk_shape(key),'Wrong edge-chunk shape')
        require(chunk.dtype==np.dtype(spec.dtype),'Wrong source dtype; no silent conversion')
        fetched+=chunk.nbytes
        for ids,local,weights in plan.groups[key]:
            raw=chunk[tuple(local.T)].astype(np.float64)
            good=np.isfinite(raw)
            np.logical_and.at(valid,ids,good)
            np.add.at(values,ids,np.where(good,raw,0)*weights)
    values[~valid]=np.nan
    return values,valid,dict(**plan.summary(),loader_calls=calls,missing_chunks=missing,
                             fetched_uncompressed_bytes=fetched)


def affine_landmark_check(source_xy,target_xy,ids,*,fit_ids,check_ids):
    source=np.asarray(source_xy,dtype=np.float64);target=np.asarray(target_xy,dtype=np.float64)
    require(source.ndim==2 and source.shape[1]==2 and target.shape==source.shape,'Landmark shape')
    require(np.isfinite(source).all() and np.isfinite(target).all(),'Finite landmarks required')
    require(len(ids)==len(source) and all(isinstance(x,str) and x for x in ids) and len(set(ids))==len(ids),'Unique named landmarks')
    require(len(fit_ids)>=3 and len(check_ids)>=1,'At least three fit and one check landmark')
    require(len(set(fit_ids))==len(fit_ids) and len(set(check_ids))==len(check_ids),'Duplicate partition ID')
    require(not set(fit_ids)&set(check_ids),'Fitting and check landmarks overlap')
    require(set(fit_ids)|set(check_ids)<=set(ids),'Unknown landmark ID')
    lookup={name:i for i,name in enumerate(ids)}
    fi=np.array([lookup[x] for x in fit_ids]);ci=np.array([lookup[x] for x in check_ids])
    center=source[fi].mean(axis=0);scale=source[fi].std(axis=0)
    require((scale>0).all(),'Degenerate fitting geometry')
    design=np.column_stack(((source[fi]-center)/scale,np.ones(len(fi))))
    require(np.linalg.matrix_rank(design)==3,'Collinear fitting landmarks')
    condition=float(np.linalg.cond(design));require(condition<=1e8,'Ill-conditioned fitting geometry')
    coefficient=np.linalg.lstsq(design,target[fi],rcond=None)[0]
    linear=coefficient[:2].T/scale
    translation=coefficient[2]-linear@center
    matrix=np.column_stack((linear,translation))
    prediction=source@linear.T+translation
    def residuals(indices,names):
        delta=prediction[indices]-target[indices]
        distance=np.linalg.norm(delta,axis=1)
        return dict(ids=list(names),vectors_xy=delta.tolist(),distances_pixels=distance.tolist(),
            rms_pixels=float(np.sqrt(np.mean(distance**2))),maximum_pixels=float(distance.max()))
    return dict(affine_source_xy_to_target_xy=matrix.tolist(),normalized_design_condition=condition,
                fit=residuals(fi,fit_ids),check=residuals(ci,check_ids),
                scope='Affine landmark mechanics; not authenticated image registration or physical ink accuracy.')
