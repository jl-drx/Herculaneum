"""Target-free freeze of alternatives, controls and complete teacher inputs."""
from common import *
import ast

def main():
    verify(read(R/'INPUTS.json')['files'])
    ct=read(R/'inputs/ct/RESULTS.json');cat=read(G.I/'CATALOG_EXTRACT.json')
    old=read(R/'inputs/QUALIFICATION.json')['regions']
    reserved=read(R/'inputs/RESERVED_CONTEXTS.json')
    assert len(reserved)==332 and len({r['core_id'] for r in reserved})==332
    # Run the actual frozen upstream grid in isolation, without importing models.
    node=next(n for n in ast.parse((R/'sources/v42-inference.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_grid_1d')
    node.returns=None
    for a in node.args.args:a.annotation=None
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'source-grid','exec'),ns)
    for size in [1,127,256,257,383,384,385,797,31040]:assert G.grid(size)==ns['_grid_1d'](size,256,128)
    author=read(R/'sources/pinned-native_dense_report.json');rows=[];provenance=[]
    for w in ['w040','w041','w043']:
        fine=G.load(w,'fine78');matrix=G.to_native(cat,fine['volume'])
        a=G.asset(cat,w,'ink-detection',fine['volume']);p=a['creation_info']['provenance']['parameters']
        assert p['tile-size']=='256' and p['stride']=='128' and [int(p['start-layer']),int(p['end-layer'])]==[23,85]
        filename=Path(a['origins'][0]['path']).name;assert filename==author[w]['teacher_tif']
        assert a['creation_info']['derived_from']['source_path'].endswith(fine['render_asset']['origins'][0]['path'])
        provenance.append(dict(winding=w,canvas_xy=fine['canvas'],teacher_filename=filename,
            threshold=author[w]['theta'],threshold_source=author[w]['theta_source'],
            url='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'+a['origins'][0]['path'],
            teacher_asset=a,source_depth=[23,85],historical_binary_identity_verified=False,
            historical_manual_overrides_reproduced=False,teacher_training_ancestry_independent=False,
            current_native_annotation_release_in_catalog=any(z['type']=='ink-labels' and z['parameters']['target_volume']=='20250728140407' for z in cat['segments'][w]['data'])))
        for site in [s for s in ct['sites'] if s['scored'] and s['core_id'].startswith(w+'-')]:
            sid=site['core_id'];d=R/'inputs/ct'/sid;fit=site['fit_translation_zyx'];idx=tuple(np.array(fit)+6)
            landscapes=np.load(d/'landscapes.npy');q=np.load(d/'queries_uvk.npy',mmap_mode='r');v=np.load(d/'support.npy',mmap_mode='r')
            patches=[];bounds=[]
            for k,center in enumerate(CHECK,4):
                a=landscapes[k];coords=np.argwhere(a>=a[idx]-.05)-6
                assert any(np.array_equal(c,fit) for c in coords)
                patch=dict(index=k,centre_core_xy=list(center),at_fit_ncc=float(a[idx]),
                    alternatives=[dict(shift_zyx=c.tolist(),ncc=float(a[tuple(c+6)])) for c in coords])
                for j in PLANES:
                    for name,uv in queries(q,v,patch,j,fit).items():
                        for op in OPS:bounds.append(T.source_bounds(uv,np.ones(uv.shape[:-1],bool),op))
                patches.append(patch)
            bounds=np.asarray(bounds);b=[int(bounds[:,0].min()),int(bounds[:,1].min()),int(bounds[:,2].max()),int(bounds[:,3].max())]
            # Integer contributor rectangle: no extra +1 neighbor is needed here.
            starts=[[z for z in G.grid(size) if z<hi and z+256>lo] for size,lo,hi in zip(fine['canvas'],b[:2],b[2:])]
            assert all(starts)
            rect=[min(starts[0]),min(starts[1]),max(starts[0])+256,max(starts[1])+256]
            context=G.context_bound(fine,rect,[23,85],matrix);box=context.get('bbox_native_xyz')
            hits=[] if box is None else [r['core_id'] for r in reserved if G.intersects(box,r['bbox_conservative_xyz'])]
            conditional=[] if box is None else [r['core_id'] for r in old if r['winding']=='w024' and r['teacher_context'].get('bbox_native_xyz') is not None and G.intersects(box,r['teacher_context']['bbox_native_xyz'])]
            qualified=bool(context.get('complete_geometry') and context.get('within_catalog_source_extent') and not hits and not conditional)
            before=next(r for r in old if r['core_id']==sid)['teacher_footprint']['rect_xyxy']
            result=dict(core_id=sid,winding=w,fit_translation_zyx=fit,patches=patches,
                threshold=author[w]['theta'],source_bbox_fullfine_xyxy=b,teacher_tiles_xy=[[x,y] for y in starts[1] for x in starts[0]],
                teacher_full_input_rectangle_xyxy=rect,previous_unshifted_core_teacher_rectangle_xyxy=before,
                teacher_context=context,reserved_native_hits=hits,conditional_w024_hits=conditional,
                admitted_for_bounded_target_sensitivity=qualified,structural_registration_pass=False)
            rows.append(result);print(sid,'candidates',[len(p['alternatives']) for p in patches],'qualified',qualified,'source',b,'teacher',rect,flush=True)
    save(R/'ALTERNATIVES.json',dict(status='FROZEN_BEFORE_TARGET_PIXELS',sites=rows,
        retained_source_failures=[s['core_id'] for s in ct['sites'] if not s['scored']],
        total_patches=sum(len(s['patches']) for s in rows),admitted_sites=sum(s['admitted_for_bounded_target_sensitivity'] for s in rows),
        reserved_native_contexts=len(reserved),protocol_sha256=sha(R/'PROTOCOL.md'),inputs_sha256=sha(R/'INPUTS.json')))
    save(R/'PROVENANCE.json',dict(sites=provenance,author_report_pin='217e7521670db49cde97872ea6a8bdfb80c449f1',
        source_v42_pin='d4615b134ce1b62368bed1029ed47e5569ca03fb',scope='Current teacher-only conditional numeric sensitivity; no annotation accuracy or historical merged-label reproduction.'))

if __name__=='__main__':main()
