"""Independent analytic/index/footprint checks and saved-cohort invariants."""
from pathlib import Path
import ast,json
import numpy as np
import geometry as G
R=Path(__file__).resolve().parent

def scalar_axis(x,count,canvas,scale):
    # Scalar translation of the source statements, with explicit FP32 stores.
    f=np.float32;mesh_scale=f(.05000000074505806);r=f(scale)
    centre=f(float(count)/2/float(mesh_scale));origin=f(f(-.5)*f(f(canvas)-f(1)))
    start=(x//128)*128;local=x-start;off=f(origin+f(start))
    nominal=f(f(off/r)+centre);ul=f(nominal*mesh_scale)
    step=float(mesh_scale)/float(r);offset=f(float(ul)-4*step)
    return f(offset+f(f(local+4)*f(step)))

def main():
    checks=[]
    def need(ok,name):
        if not ok:raise ValueError(name)
        checks.append(name)
    for row in G.read(R/'qualification/COMPUTE_BINDING.json')['files']:
        p=R/row['path'];need(p.stat().st_size==row['bytes'] and G.sha(p)==row['sha256'],'bound '+row['path'])
    yy,xx=np.mgrid[:61,:65];mesh=np.stack([20*xx,20*yy,100+2*xx+3*yy],axis=-1).astype(np.float32);cache=G.N.normals(mesh)
    expected_n=np.array([.1,.15,-1]);expected_n/=np.linalg.norm(expected_n)
    for group in [0,1]:
        scale=2.**(-group);canvas=[int(round(65*20*scale)),int(round(61*20*scale))]
        xs=np.array([64,126,127,128,129,254,255,256,257,500]);ys=np.array([64,126,127,128,129,254,255,256,257,500])
        q=G.axis(xs,65,canvas[0],scale)[0]
        need(np.array_equal(q,np.array([scalar_axis(int(x),65,canvas[0],scale) for x in xs])),f'group{group} scalar tile-phase index replay')
        b,n,ok=G.render(mesh,cache,xs,ys,canvas,group)
        need(ok.all(),f'group{group} complete analytic plane support')
        ex=xs[None,:]+.5;ey=ys[:,None]+.5
        need(np.max(np.abs(b[...,0]-ex))<2e-4 and np.max(np.abs(b[...,1]-ey))<2e-4,f'group{group} analytic XY physical units')
        need(np.max(np.abs(b[...,2]-(100*scale+.1*ex+.15*ey)))<2e-4,f'group{group} analytic sloped Z base scaling')
        need(np.max(np.abs(n-expected_n))<1e-6,f'group{group} normals remain unit, not downscaled')
        if group==0:
            legacy=G.N.render(mesh,cache,xs,ys,canvas)
            need(all(np.array_equal(a,z) for a,z in zip((b,n,ok),legacy)),'byte-identical legacy group0 renderer regression')
        fake=dict(mesh=mesh,cache=cache,canvas=canvas,group=group,slices=116 if group else 109,source_shape_xyz_conservative=[10000]*3)
        bound=G.context_bound(fake,[120,120,144,144],[27,89] if group else [23,85],np.eye(4))
        rb,rn,rv=G.render(mesh,cache,np.arange(120,144),np.arange(120,144),canvas,group)
        box=np.asarray(bound['bbox_source_level_xyz']);start,end=bound['depth_half_open']
        need(rv.all() and bound['complete_geometry'],f'group{group} bounded plane full footprint')
        need(all(np.all(rb+(d-bound['centre_depth'])*rn-1>=box[0]) and np.all(rb+(d-bound['centre_depth'])*rn+1<=box[1]) for d in range(start,end)),f'group{group} every teacher depth and trilinear neighbor enclosed')
        fullbox=np.asarray(bound['bbox_native_xyz'])
        need(np.allclose(fullbox,box*(2**group)),f'group{group} level-to-full-resolution box units')
        damaged=mesh.copy();damaged[12,12]=np.nan
        bad=dict(fake,mesh=damaged,cache=G.N.normals(damaged))
        # Different render scales mean the same mesh vertex occupies a different raster location.
        origin=int(12*20*scale)-4
        need(not G.context_bound(bad,[origin,origin,origin+8,origin+8],[23,85],np.eye(4))['complete_geometry'],f'group{group} broken normal/coordinate stencil rejected')
    for length in [257,385,797,1000]:
        pixels=np.arange(length);covered=np.zeros(length,bool)
        for s in G.grid(length):covered[s:s+256]=True
        need(covered.all() and G.grid(length)[-1]==length-256,f'border grid covers length{length}')
        sample=np.array([1.,length/2+.25,length-2.2]);fake=dict(canvas=[length,length])
        fp=G.footprint(fake,sample,sample,True);ints=np.unique(np.r_[np.floor(sample),np.floor(sample)+1]).astype(int)
        expected={(sx,sy) for sy in G.grid(length) for sx in G.grid(length) if any(sx<=x<sx+256 for x in ints) and any(sy<=y<sy+256 for y in ints)}
        need(set(map(tuple,fp['tiles_xy']))==expected,f'all contributing border tiles length{length}')
        phasefree=G.footprint(fake,sample,sample,False)['rect_xyxy']
        need(all(phasefree[0]<=x and x+256<=phasefree[2] and phasefree[1]<=y and y+256<=phasefree[3] for x,y in expected),f'unknown-phase envelope includes verified grid length{length}')
    d=G.read(R/'qualification/RESULTS.json');rows=d['regions'];cohort=G.read(R/'intake/selection/RESULTS.json')['selected']
    need(len(rows)==224 and {r['core_id'] for r in rows}=={r['core_id'] for r in cohort if r['winding']!='w053'},'unchanged224 analyzed regions, w053 stays native-only')
    need(not d['paired_training_admitted'] and not d['v43_source_qualified'],'no accidental supervision/v43 admission')
    pilot=[r for r in rows if 'pilot' in r]
    need(len(pilot)==12 and all(r['selection_order'] in [0,16,32,48] for r in pilot),'exact predetermined pilot with no substitutions')
    reserved=G.read(R/'intake/selection/RESERVED_CONTEXTS.json')
    for w in ['w040','w041','w043']:
        rr=[r for r in rows if r['winding']==w]
        accepted=[r for r in rr if r['geometry_admitted_for_native_reference_CT_pilot']]
        need(all(not any(G.intersects(r['teacher_context']['bbox_native_xyz'],s['bbox_conservative_xyz']) for s in reserved) for r in accepted),w+' independently recomputed native-reference exclusions')
        need(len(accepted)==d['counts'][w]['geometry_with_native_references'],w+' admitted count')
    conflict=next(r for r in rows if r['core_id']=='w041-x1088-y1088')
    need(conflict['mapped_core']['geometry_screen'] and set(conflict['teacher_hits_reserved_native'])=={'w035-x00640-y00768','w035-x00768-y00768'} and not conflict['geometry_admitted_for_native_reference_CT_pilot'],'new full-teacher-context conflict is rejected')
    result=dict(status='PASS',count=len(checks),checks=checks,code_sha256=G.sha(Path(__file__)),results_sha256=G.sha(R/'qualification/RESULTS.json'),scope='Source geometry and input-support checks, no CT correspondence or ink claims')
    G.save(R/'VERIFICATION.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}))

if __name__=='__main__':main()
