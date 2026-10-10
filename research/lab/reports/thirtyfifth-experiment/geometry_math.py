"""Descriptive local raster tangents and existing historical ray bounds."""
import numpy as np
from packet_common import require

def describe(q):
    require(q.shape==(3,3,6) and q.dtype==np.float64,'Nine float64 PPM records')
    mapped=np.any(q!=0,axis=2);m=q[mapped]
    require(np.isfinite(m).all(),'Nonfinite mapped record')
    error=np.abs(np.linalg.norm(m[:,3:],axis=1)-1)
    require(np.all(error<=1e-5),'Nonunit mapped normal')
    result=dict(mapped_records=int(mapped.sum()),complete=bool(mapped.all()),center_mapped=bool(mapped[1,1]),normal_norm_max_error=float(error.max()) if len(error) else None,
        center_XYZ=None,center_normal=None,tangent_x=None,tangent_y=None,tangent_x_norm=None,tangent_y_norm=None,
        tangent_cosine=None,area_scale=None,singular_values=None,condition_number=None,normal_dot_unit_tangent_x=None,
        normal_dot_unit_tangent_y=None,normal_dot_unit_raster_cross=None,valid_layer_count=None,layer_validity=None)
    if mapped[1,1]:
        point=q[1,1,:3];normal=q[1,1,3:];rays=point+np.arange(-31,32)[:,None]*normal
        valid=((rays>=0)&(rays<=np.array([6299,2259,8854]))).all(axis=1)
        result.update(center_XYZ=point.tolist(),center_normal=normal.tolist(),valid_layer_count=int(valid.sum()),layer_validity=valid.tolist())
    if mapped.all():
        dx=(q[1,2,:3]-q[1,0,:3])/2;dy=(q[2,1,:3]-q[0,1,:3])/2
        nx=float(np.linalg.norm(dx));ny=float(np.linalg.norm(dy));cross=np.cross(dx,dy);area=float(np.linalg.norm(cross));sv=np.linalg.svd(np.stack([dx,dy],axis=1),compute_uv=False)
        result.update(tangent_x=dx.tolist(),tangent_y=dy.tolist(),tangent_x_norm=nx,tangent_y_norm=ny,
            tangent_cosine=float(np.dot(dx,dy)/(nx*ny)) if nx*ny else None,area_scale=area,singular_values=sv.tolist(),
            condition_number=float(sv[0]/sv[1]) if sv[1]>0 else None,
            normal_dot_unit_tangent_x=float(np.dot(normal,dx)/nx) if nx else None,
            normal_dot_unit_tangent_y=float(np.dot(normal,dy)/ny) if ny else None,
            normal_dot_unit_raster_cross=float(np.dot(normal,cross)/area) if area else None)
    return result
