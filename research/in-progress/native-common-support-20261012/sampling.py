"""Pure global-grid footprints and deterministic mask-only core selection."""
from __future__ import annotations
import math
import numpy as np


def starts(length, patch, stride):
    if length < patch:
        raise ValueError('No padded model patches in this pilot')
    positions = list(range(0, length-patch+1, stride))
    if positions[-1] != length-patch:
        positions.append(length-patch)
    return positions


def linear_interval(first, stop, source_size, target_size):
    """Inclusive/exclusive integer source support of resize destination pixels."""
    lo = ((2*first+1)*source_size-target_size)//(2*target_size)
    hi = ((2*(stop-1)+1)*source_size-target_size)//(2*target_size)+2
    return lo, hi


def tiles(shape, bbox, patch, stride):
    h,w = shape
    x0,y0,x1,y1 = bbox
    return [[x,y,patch,patch] for y in starts(h,patch,stride)
            if y < y1 and y+patch > y0
            for x in starts(w,patch,stride) if x < x1 and x+patch > x0]


def bounds(patches):
    return [min(p[0] for p in patches),min(p[1] for p in patches),
            max(p[0]+p[2] for p in patches),max(p[1]+p[3] for p in patches)]


def context(shape, spacing, x, y, size=128):
    h,w=map(int,shape)
    if min(x,y)<0 or x+size>w or y+size>h:
        return None
    core=[x,y,x+size,y+size]
    hybrid=tiles(shape,core,128,64)
    hh,ww=round(h*spacing/9.6),round(w*spacing/9.6)
    hx0,hx1=linear_interval(x,x+size,ww,w)
    hy0,hy1=linear_interval(y,y+size,hh,h)
    # Require unclamped support; edge/padding behavior is not this pilot's target.
    if min(hx0,hy0)<0 or hx1>ww or hy1>hh:
        return None
    hecate=tiles((hh,ww),[hx0,hy0,hx1,hy1],64,32)
    ax0,ay0,ax1,ay1=bounds(hecate)
    if spacing/9.6 < .95:
        source=[ax0*w//ww,ay0*h//hh,
                (ax1*w+ww-1)//ww,(ay1*h+hh-1)//hh]
        interpolation='INTER_AREA'
    else:
        sx0,sx1=linear_interval(ax0,ax1,w,ww)
        sy0,sy1=linear_interval(ay0,ay1,h,hh)
        source=[sx0,sy0,sx1,sy1]
        interpolation='INTER_LINEAR'
    if min(source[:2])<0 or source[2]>w or source[3]>h:
        return None
    hb=bounds(hybrid)
    union=[min(hb[0],source[0]),min(hb[1],source[1]),
           max(hb[2],source[2]),max(hb[3],source[3])]
    chunks=sorted({(0,cy,cx) for box in [hb,source]
                   for cy in range(box[1]//128,(box[3]-1)//128+1)
                   for cx in range(box[0]//128,(box[2]-1)//128+1)})
    return dict(core_xywh=[x,y,size,size],source_bbox_xyxy=union,
                hybrid_tiles_xywh=hybrid,hecate_tiles_xywh=hecate,
                hecate_grid_yx=[hh,ww],hecate_output_support_xyxy=[hx0,hy0,hx1,hy1],
                hecate_source_bbox_xyxy=source,hecate_interpolation=interpolation,
                native_chunks_zyx=[list(c) for c in chunks])


def spread(candidates,count=6):
    """Farthest-point selection with rational centroid comparisons, no labels."""
    if len(candidates)<count:
        raise ValueError(f'Only {len(candidates)} eligible cores; need {count}')
    n=len(candidates);sx=sum(r['x'] for r in candidates);sy=sum(r['y'] for r in candidates)
    first=min(candidates,key=lambda r:((n*r['x']-sx)**2+(n*r['y']-sy)**2,r['y'],r['x']))
    chosen=[first]
    while len(chosen)<count:
        choices=[r for r in candidates if r not in chosen]
        selected=min(choices,key=lambda r:(-min((r['x']-c['x'])**2+(r['y']-c['y'])**2
                                               for c in chosen),r['y'],r['x']))
        chosen.append(selected)
    return chosen
