"""Fixed-cohort geometry and input-footprint audit. No intensity or target reads."""
from pathlib import Path
import argparse,ast,csv,json,time
import numpy as np
import geometry as G
R=Path(__file__).resolve().parent
I=R/'intake'

def distribution(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return None if not len(v) else dict(n=len(v),min=float(v.min()),median=float(np.median(v)),p99=float(np.quantile(v,.99)),max=float(v.max()))

def compare(native,fine,xywh,matrix,naive=False):
    x,y,w,h=xywh
    base,normal,ok=G.render(native['mesh'],native['cache'],np.arange(x,x+w),np.arange(y,y+h),native['canvas'],0)
    fx,fy=G.mapped_axes(native,fine,xywh,naive)
    fb,fn,fok=G.sampled_field(fine,fx,fy)
    factor=2**fine['group'];fb=(fb*factor)@matrix[:3,:3].T+matrix[:3,3];fn=(fn*factor)@matrix[:3,:3].T
    valid=ok&fok&np.isfinite(fb).all(-1)&np.isfinite(fn).all(-1)
    lengths=np.linalg.norm(fn,axis=-1);valid &= lengths>0
    residual=np.linalg.norm(fb-base,axis=-1)
    dot=(fn*normal).sum(-1)/np.maximum(lengths*np.linalg.norm(normal,axis=-1),1e-30)
    angle=np.degrees(np.arccos(np.clip(dot,-1,1)))
    rd=distribution(residual[valid]);ad=distribution(angle[valid])
    return dict(valid_pixels=int(valid.sum()),total_pixels=valid.size,position_residual_native_voxels=rd,
                normal_angle_deg=ad,geometry_screen=bool(valid.all() and rd['max']<=1 and ad['max']<=10),
                query_x_minmax=[float(fx.min()),float(fx.max())],query_y_minmax=[float(fy.min()),float(fy.max())]),fx,fy

def validate_inputs():
    for r in G.read(R/'ACQUISITION.json')['requests']:
        p=R/'coordinates'/(r['id']+'.tif');assert G.sha(p)==r['sha256'] and p.stat().st_size==r['bytes']
    for r in G.read(I/'COORDINATE_LEDGER.json')['requests']:
        p=I/'coordinates'/(r['id']+'.tif');assert G.sha(p)==r['sha256']
    for r in G.read(I/'METADATA_LEDGER.json')['responses']:
        if r['method']=='GET':assert G.sha(I/'metadata'/(r['id']+'.json'))==r['sha256']
    # Compare the exact upstream grid function without importing Torch/inference.
    tree=ast.parse((R/'sources/v42-inference.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_grid_1d')
    node.returns=None
    for a in node.args.args:a.annotation=None
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned-grid','exec'),ns)
    for length in [1,127,256,257,383,384,385,797,31040]:assert G.grid(length)==ns['_grid_1d'](length,256,128)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);validate_inputs()
    paths=[R/'PROTOCOL.md',R/'SOURCE_SCOPE_ADDENDUM.md',R/'ACQUISITION.json',Path(__file__),R/'geometry.py']
    paths += sorted(p for p in (R/'sources').glob('*') if p.is_file())
    paths += [I/'CATALOG_EXTRACT.json',I/'selection/RESULTS.json',I/'selection/RESERVED_CONTEXTS.json']
    G.save(out/'COMPUTE_BINDING.json',dict(files=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=G.sha(p)) for p in paths]))
    cat=G.read(I/'CATALOG_EXTRACT.json');cohort=G.read(I/'selection/RESULTS.json')['selected']
    records=[];site_contracts=[]
    for w,kind in [('w040','fine78'),('w041','fine78'),('w043','fine78'),('w024','fine59')]:
        native=G.load(w,'native');fine=G.load(w,kind);matrix=G.to_native(cat,fine['volume'])
        nm=native['mesh_asset'];fm=fine['mesh_asset']
        same=nm['creation_info']['derived_from']==fm['creation_info']['derived_from'];assert same
        an=np.eye(4);af=np.eye(4)
        an[:3]=json.loads(nm['creation_info']['provenance']['parameters']['affine-matrix'])
        af[:3]=json.loads(fm['creation_info']['provenance']['parameters']['affine-matrix'])
        discrepancy=float(np.max(np.abs(an-matrix@af)));assert discrepancy<1e-8
        teacher=G.asset(cat,w,'ink-detection',fine['volume']);tp=teacher['creation_info']['provenance']['parameters']
        assert tp['tile-size']=='256' and tp['stride']=='128'
        depth=[int(tp['start-layer']),int(tp['end-layer'])]
        assert depth==([23,85] if kind=='fine78' else [27,89])
        contract=dict(winding=w,kind=kind,source_group=fine['group'],native_mesh_shape=list(native['mesh'].shape),
            fine_mesh_shape=list(fine['mesh'].shape),fine_canvas=fine['canvas'],same_normalized_ancestry=same,
            affine_composition_max_error=discrepancy,source_pipeline_verified=(kind=='fine78'),
            teacher_tag='v42' if kind=='fine78' else 'v43',teacher_depth_half_open=depth,
            teacher_reverse=tp.get('force-reverse'),normal_norm_max_error=fine['normal_norm_max_error'])
        site_contracts.append(contract)
        for c in [r for r in cohort if r['winding']==w]:
            t=time.perf_counter();correct,fx,fy=compare(native,fine,c['core_xywh'],matrix)
            naive,_,_=compare(native,fine,c['core_xywh'],matrix,naive=True)
            fp=G.footprint(fine,fx,fy,kind=='fine78')
            bound=dict(complete_geometry=False,reason='no_contributing_tiles') if fp is None else G.context_bound(fine,fp['rect_xyxy'],depth,matrix)
            bounds_ok=bound.get('complete_geometry',False) and bound.get('within_catalog_source_extent',False)
            record=dict(core_id=c['core_id'],winding=w,kind=kind,role=c['role'],selection_order=c['selection_order'],
                core_xywh=c['core_xywh'],native_context_xywh=c['context_xywh'],native_bbox=c['bbox_conservative_xyz'],
                mapped_core=correct,naive_raster_resize=naive,teacher_footprint=fp,teacher_context=bound,
                geometric_footprint_screen=bool(correct['geometry_screen'] and bounds_ok),teacher_pipeline_verified=(kind=='fine78'))
            # Fixed future CT sites; all are retained even if a gate fails.
            if kind=='fine78' and c['selection_order'] in [0,16,32,48]:
                x,y,ww,hh=c['core_xywh'];halo=[x-8,y-8,ww+16,hh+16]
                hx,hy=G.mapped_axes(native,fine,halo)
                urange=[int(np.floor(hx.min()/4))-2,int(np.ceil(hx.max()/4))+3]
                vrange=[int(np.floor(hy.min()/4))-2,int(np.ceil(hy.max()/4))+3]
                rect=[4*urange[0],4*vrange[0],4*urange[1],4*vrange[1]]
                hbound=G.context_bound(fine,rect,[0,109],matrix)
                record['pilot']=dict(native_core_halo_xywh=halo,fine_group2_xyxy=[urange[0],vrange[0],urange[1],vrange[1]],
                    fine_full_depth_halo_bound=hbound,rule='fixed selection order0,16,32,48; no substitution')
            records.append(record)
        print(w,dict(cores=sum(r['winding']==w for r in records),core_screen=sum(r['winding']==w and r['mapped_core']['geometry_screen'] for r in records),
                     geometric_footprint_screen=sum(r['winding']==w and r['geometric_footprint_screen'] for r in records)),flush=True)
        del native,fine
    reserved=G.read(I/'selection/RESERVED_CONTEXTS.json')
    original=reserved[:268];assert len(original)==268
    ref_native=[r for r in cohort if r['role']!='training_candidate']
    w024=[r for r in records if r['winding']=='w024']
    for r in records:
        bound=r['teacher_context'];box=bound.get('bbox_native_xyz')
        if r['role']=='training_candidate':
            native_reserved=original+ref_native
            matches=[] if box is None else [z['core_id'] for z in native_reserved if G.intersects(box,z['bbox_conservative_xyz'])]
            alt=[]
            for ref in w024:
                rb=ref['teacher_context'].get('bbox_native_xyz')
                if rb is not None and (G.intersects(r['native_bbox'],rb) or (box is not None and G.intersects(box,rb))):alt.append(ref['core_id'])
            r['teacher_hits_reserved_native']=matches;r['conditional_w024_teacher_hits']=alt
            r['geometry_admitted_for_native_reference_CT_pilot']=bool(r['geometric_footprint_screen'] and not matches)
            r['clears_conditional_w024_reference_envelopes']=bool(r['geometry_admitted_for_native_reference_CT_pilot'] and not alt)
            if 'pilot' in r:
                b=r['pilot']['fine_full_depth_halo_bound'];bb=b.get('bbox_native_xyz')
                hmatches=[] if bb is None else [z['core_id'] for z in native_reserved if G.intersects(bb,z['bbox_conservative_xyz'])]
                r['pilot']['full_depth_halo_hits_reserved_native']=hmatches
                r['pilot']['geometry_ready']=bool(r['geometry_admitted_for_native_reference_CT_pilot'] and b.get('complete_geometry') and b.get('within_catalog_source_extent') and not hmatches)
        else:
            r['teacher_hits_old_or_blank_native']=[] if box is None else [z['core_id'] for z in original+[x for x in ref_native if x['winding']=='w053'] if G.intersects(box,z['bbox_conservative_xyz'])]
            r['alternate_reference_admitted']=False
    counts={}
    for w in ['w040','w041','w043','w024']:
        rows=[r for r in records if r['winding']==w]
        counts[w]=dict(total=len(rows),core_geometry_pass=sum(r['mapped_core']['geometry_screen'] for r in rows),
            complete_teacher_geometry=sum(r['teacher_context']['complete_geometry'] for r in rows),
            teacher_extent_pass=sum(r['teacher_context'].get('within_catalog_source_extent',False) for r in rows),
            geometry_and_footprint_pass=sum(r['geometric_footprint_screen'] for r in rows),
            geometry_with_native_references=sum(r.get('geometry_admitted_for_native_reference_CT_pilot',False) for r in rows),
            also_clears_conditional_w024=sum(r.get('clears_conditional_w024_reference_envelopes',False) for r in rows),
            pilot_selected=sum('pilot' in r for r in rows),pilot_geometry_ready=sum(r.get('pilot',{}).get('geometry_ready',False) for r in rows))
    result=dict(status='COMPLETE_GEOMETRY_ONLY',counts=counts,source_contracts=site_contracts,regions=records,
                fixed_cohort_sha256=G.sha(I/'selection/RESULTS.json'),v43_source_qualified=False,
                paired_training_admitted=False,CT_pixels_fetched=0,target_pixels_fetched=0,new_model_forwards=0,GPU_spend_GBP=0)
    G.save(out/'RESULTS.json',result)
    with (out/'ADMISSION.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['core_id','role','core_geometry','teacher_geometry','teacher_extent','clears_native_refs','clears_conditional_w024','pilot_geometry_ready'])
        for r in records:writer.writerow([r['core_id'],r['role'],r['mapped_core']['geometry_screen'],r['teacher_context']['complete_geometry'],r['teacher_context'].get('within_catalog_source_extent',False),r.get('geometry_admitted_for_native_reference_CT_pilot',False),r.get('clears_conditional_w024_reference_envelopes',False),r.get('pilot',{}).get('geometry_ready','not_selected')])
    print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':main()
