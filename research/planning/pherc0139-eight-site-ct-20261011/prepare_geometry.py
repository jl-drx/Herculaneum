"""CT-free inversion and complete finite-search support qualification."""
from common import *
def main():
    check_inputs();out=R/'geometry';out.mkdir(exist_ok=False);plan=read(PLAN);results=[]
    for w in ['w040','w041','w043']:
        native=G.load(w,'native');fine=G.load(w,'fine78');mat=G.to_native(G.read(G.I/'CATALOG_EXTRACT.json'),fine['volume'])
        for site in [s for s in plan['fixed_selected_sites'] if s['winding']==w]:
            sid=site['core_id'];d=out/sid;d.mkdir();cx,cy,ww,hh=site['scored_core_xywh'];assert ww==hh==128
            x0,y0=cx-8,cy-8;xs=np.arange(x0,x0+144);ys=np.arange(y0,y0+144)
            base,nd,ngood=G.render(native['mesh'],native['cache'],xs,ys,native['canvas'],0)
            fx,fy=G.mapped_axes(native,fine,[x0,y0,144,144]);u0,v0=np.meshgrid((fx-1.5)/4,(fy-1.5)/4)
            rect=site['fine_group2_xyxy'];b,n,ok=pooled_field(fine,rect,mat)
            qq=[];vv=[];ff=[];rr=[];movement=[]
            for j in range(4,24):
                initial=np.stack((u0.ravel(),v0.ravel(),np.full(u0.size,54+(j-13.5)*9.362/2.399)),axis=-1)
                target=(base+nd*(j-13.5)).reshape(-1,3)
                q,valid,failure,residual,steps=P.solve(b,n,ok,initial,target,nd.reshape(-1,3),ngood.ravel(),rect[0],rect[1])
                qq.append(q.reshape(144,144,3));vv.append(valid.reshape(144,144));ff.append(failure.reshape(144,144));rr.append(residual.reshape(144,144))
                movement.append(np.max(np.abs(q[valid]-initial[valid]),axis=0).tolist() if valid.any() else [0,0,0])
            q=np.stack(qq);valid=np.stack(vv);failure=np.stack(ff);residual=np.stack(rr)
            for name,a in [('queries_uvk',q),('support',valid),('failure_codes',failure),('residual',np.where(np.isfinite(residual),residual,np.nan))]:np.save(d/(name+'.npy'),a)
            bases=[]
            for x,y in FIT+CHECK+[(64,64)]:
                yy,xx=y+8,x+8
                bases.append(np.stack((nd[yy,xx],(base[yy+1,xx]-base[yy-1,xx])/2,(base[yy,xx+1]-base[yy,xx-1])/2),axis=1).tolist())
            supports=[];sets=[]
            for i,(template,search) in enumerate(patches()):
                sub=q[search];v=valid[search];ix=np.floor(sub).astype(int)
                within=bool((ix[...,0]>=rect[0]).all() and (ix[...,0]+1<rect[2]).all() and (ix[...,1]>=rect[1]).all() and (ix[...,1]+1<rect[3]).all() and (ix[...,2]>=5).all() and (ix[...,2]+1<=103).all())
                footprint=fine_indices(sub);sets.append(footprint)
                supports.append(dict(index=i,role='fit' if i<4 else 'check',xy=(FIT+CHECK)[i],geometry_complete=bool(v.all()),source_envelope_complete=within,
                    group2_XY_contributors=len(footprint),query_min_uvk=sub.min((0,1,2)).tolist(),query_max_uvk=sub.max((0,1,2)).tolist(),native_template_geometry_complete=bool(ngood[template[1:]].all())))
            overlaps=[dict(a=a,b=b,shared_group2_XY_pixels=len(sets[a]&sets[b])) for a in range(8) for b in range(a+1,8) if sets[a]&sets[b]]
            ready=not overlaps and all(s['geometry_complete'] and s['source_envelope_complete'] and s['native_template_geometry_complete'] for s in supports)
            item=dict(core_id=sid,winding=w,ready_for_CT_scoring=ready,native_halo_xywh=[x0,y0,144,144],samples=int(valid.size),accepted=int(valid.sum()),
                failure_counts={str(k):int((failure==k).sum()) for k in range(6)},maximum_accepted_residual=float(residual[valid].max()) if valid.any() else None,
                max_movement_uvk=np.max(movement,axis=0).tolist(),supports=supports,pairwise_footprint_overlaps=overlaps,bases_depth_y_x_nativeXYZ=bases,
                files=[receipt(p) for p in sorted(d.glob('*.npy'))])
            save(d/'RESULTS.json',item);results.append(item);print(sid,'ready',ready,'valid',int(valid.sum()),'/',valid.size,'overlap',len(overlaps),flush=True)
        del native,fine
    save(out/'RESULTS.json',dict(status='COMPLETE_GEOMETRY_ONLY',sites=results,ready=sum(s['ready_for_CT_scoring'] for s in results),prior_geometry_failures=plan['failed_geometry_sites']))
if __name__=='__main__':main()
