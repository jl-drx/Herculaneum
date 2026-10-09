"""Fixed sparse PPM consistency, conditional on one invariant-valid byte order."""
import hashlib
import math
import struct
from common import require

TOLERANCE = 1e-5
NATIVE_SHAPE_ZYX = [8855,2260,6300]


def derive_points(width,height,offset,total):
    require(all(type(v) is int for v in (width,height,offset,total)) and width>=11 and height>=11 and offset>0,"Invalid structural dimensions")
    require(offset+width*height*48 == total,"PPM extent changed")
    points,ranges=[],[]
    for i in range(9):
        y=(i+1)*(height-1)//10
        for j in range(9):
            x=(j+1)*(width-1)//10
            require(1<=x<width-1 and 1<=y<height-1,"Neighborhood outside canvas")
            spans=[]
            for dy in (-1,0,1):
                start=offset+48*((y+dy)*width+x-1)
                span=[start,start+143]
                require(offset<=span[0]<=span[1]<total,"PPM sample outside object")
                spans.append(span);ranges.append(span)
            points.append(dict(lattice_ij=[i,j],center_yx=[y,x],row_ranges=spans,
                               potential_native_site=i in (1,4,7) and j in (1,4,7)))
    ordered=sorted(ranges)
    require(len(set(map(tuple,ranges)))==243 and all(a[1]<b[0] for a,b in zip(ordered,ordered[1:])),"Sample spans duplicate or overlap")
    return points,ranges


def assemble(body,parts,ranges):
    require(len(parts)==243 and len(ranges)==243,"Exactly 243 MIME ranges required")
    lookup={tuple(p["range_inclusive"]):p for p in parts}
    require(len(lookup)==243 and set(lookup)==set(map(tuple,ranges)),"MIME and scheduled ranges differ")
    rows=[]
    for span in ranges:
        item=lookup[tuple(span)]
        raw=body[item["payload_offset"]:item["payload_offset"]+item["payload_bytes"]]
        require(len(raw)==144 and hashlib.sha256(raw).hexdigest()==item["payload_sha256"],"MIME payload changed before assembly")
        rows.append(raw)
    result=b"".join(rows)
    require(len(result)==34992,"Wrong assembled PPM sample extent")
    return result


def sentinel(record):
    return all(v==0.0 for v in record)


def candidates(raw):
    require(len(raw)==34992,"Expected exactly 729 six-double records")
    summaries={}
    for name,order in (("little","<"),("big",">")):
        records=list(struct.iter_unpack(order+"6d",raw))
        zero=sum(sentinel(r) for r in records)
        nonfinite=nonunit=0;errors=[]
        for r in records:
            if sentinel(r):continue
            finite=all(math.isfinite(v) for v in r)
            nonfinite+=not finite
            norm=math.hypot(*r[3:])
            error=abs(norm-1.0)
            unit=math.isfinite(error) and error<=TOLERANCE
            nonunit+=not unit
            if math.isfinite(error):errors.append(error)
        summaries[name]=dict(record_count=len(records),zero_sentinel_count=zero,mapped_count=len(records)-zero,
             nonfinite_mapped_count=nonfinite,nonunit_mapped_count=nonunit,
             maximum_finite_normal_norm_error=max(errors) if errors else None,
             normal_absolute_tolerance=TOLERANCE,
             passed=zero<len(records) and nonfinite==0 and nonunit==0)
    passing=[k for k,v in summaries.items() if v["passed"]]
    return dict(records_sha256=hashlib.sha256(raw).hexdigest(),candidates=summaries,
                passing_candidates=passing,selected_byte_order=passing[0] if len(passing)==1 else None,
                status="CONDITIONAL" if len(passing)==1 else "STOP",
                interpretation="Invariant-consistent serialization only; source frame, producer platform and anatomy are not authenticated.")


def finite_or_none(v):
    return v if math.isfinite(v) else None


def vector_or_none(v):
    return [finite_or_none(x) for x in v]


def dot(a,b):
    products=[x*y for x,y in zip(a,b)]
    return sum(products) if all(math.isfinite(v) for v in products) else float("nan")


def cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def cosine(a,b):
    na,nb=math.hypot(*a),math.hypot(*b)
    if not (math.isfinite(na) and math.isfinite(nb) and na>0 and nb>0):return None
    return finite_or_none(dot([v/na for v in a],[v/nb for v in b]))


def describe(raw,validation,points):
    require(validation["status"]=="CONDITIONAL" and validation["passing_candidates"]==[validation["selected_byte_order"]],"Ambiguous serialization forbids geometry")
    require(hashlib.sha256(raw).hexdigest()==validation["records_sha256"],"Validated records changed")
    require(len(points)==81,"Fixed 81 centers required")
    records=list(struct.iter_unpack(("<" if validation["selected_byte_order"]=="little" else ">")+"6d",raw))
    samples=[];inside_count=mapped_count=0
    for index,point in enumerate(points):
        neighborhood=records[index*9:index*9+9]
        valid=[not sentinel(r) for r in neighborhood]
        summaries=[]
        for r,mapped in zip(neighborhood,valid):
            x,y,z=r[:3]
            inside=mapped and 0<=x<6300 and 0<=y<2260 and 0<=z<8855
            mapped_count+=mapped;inside_count+=inside
            summaries.append(dict(mapped=mapped,xyz=list(r[:3]) if mapped else None,
                                  normal=list(r[3:]),normal_norm=math.hypot(*r[3:]),
                                  inside_declared_primary_volume_bounds=inside if mapped else None))
        center=neighborhood[4]
        dx=[(a-b)/2 for a,b in zip(neighborhood[5][:3],neighborhood[3][:3])] if valid[3] and valid[5] else None
        dy=[(a-b)/2 for a,b in zip(neighborhood[7][:3],neighborhood[1][:3])] if valid[1] and valid[7] else None
        area=cross(dx,dy) if dx is not None and dy is not None else None
        center_normal=center[3:] if valid[4] else None
        samples.append(dict(**point,records_row_major=summaries,center_mapped=valid[4],complete_mapped_neighborhood=all(valid),
             jacobian_xyz_by_xy=None if dx is None or dy is None else [[finite_or_none(dx[k]),finite_or_none(dy[k])] for k in range(3)],
             tangent_x=vector_or_none(dx) if dx is not None else None,tangent_y=vector_or_none(dy) if dy is not None else None,
             tangent_x_norm=finite_or_none(math.hypot(*dx)) if dx is not None else None,
             tangent_y_norm=finite_or_none(math.hypot(*dy)) if dy is not None else None,
             local_area_scale=finite_or_none(math.hypot(*area)) if area is not None else None,
             tangent_cosine=cosine(dx,dy) if dx is not None and dy is not None else None,
             normal_dot_unit_tangent_x=cosine(center_normal,dx) if center_normal is not None and dx is not None else None,
             normal_dot_unit_tangent_y=cosine(center_normal,dy) if center_normal is not None and dy is not None else None,
             normal_dot_unit_raster_cross=cosine(center_normal,area) if center_normal is not None and area is not None else None))
    native=[dict(lattice_ij=s["lattice_ij"],center_yx=s["center_yx"],center_mapped=s["center_mapped"],complete_mapped_neighborhood=s["complete_mapped_neighborhood"]) for s in samples if s["potential_native_site"]]
    return dict(samples=samples,mapped_records=mapped_count,records_inside_declared_primary_bounds=inside_count,
                primary_shape_zyx=NATIVE_SHAPE_ZYX,bounds_interpretation="Compatibility only; same-shaped energy scans are not distinguished.",
                center_mapped_count=sum(s["center_mapped"] for s in samples),
                complete_mapped_neighborhood_count=sum(s["complete_mapped_neighborhood"] for s in samples),
                fixed_potential_native_sites=native,fixed_native_complete_count=sum(s["complete_mapped_neighborhood"] for s in native),
                minimum_six_fixed_native_sites_condition_met=sum(s["complete_mapped_neighborhood"] for s in native)>=6,
                stage_c_authorized=False,source_volume_binding=None,plane_offset_rule=None,
                anatomical_normal=None,physical_ink_depth=None,
                limitation="Sparse PPM-only consistency; derivatives use raster x-right/y-down. No CT reproduction, optical alignment, signed render depth or accuracy evaluation.")
