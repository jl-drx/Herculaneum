"""Read JSON/source-coordinate metadata only; no CT/teacher pixel data, no network."""
import json,pathlib,itertools,hashlib
S=pathlib.Path('/workspace/herculaneum-lab/reports/pherc0139-supervision-20261011')
C=pathlib.Path('/workspace/herculaneum-lab/reports/pherc0139-correspondence-20261011')
O=pathlib.Path(__file__).parent
foot=json.loads((S/'SOURCE_FOOTPRINT.json').read_text())
plan=json.loads((C/'ct-study/REQUEST_PLAN.json').read_text())
head=json.loads((S/'provenance/TEACHER_TIFF_HEADER.json').read_text())
tags={r['tag']:r['value'] for r in head['tags']}
def bbox_union(bb):return [min(b[0] for b in bb),min(b[1] for b in bb),max(b[2] for b in bb),max(b[3] for b in bb)]
def starts(a,b):return [p for p in range(0,b,128) if p+256>a]
def tiles_for(b):return {(x,y) for x in starts(b[0],b[2]) for y in starts(b[1],b[3])}
def area_overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
out={'scope':'Metadata only; no new source or target pixel access','teacher_tile_native_width_nominal':256*7980/31040,'teacher_singlepixel_overlap_union_native_width_nominal':384*7980/31040,'supports':{},'pairs':[],'retained_ct':[],'prospective_teacher_full_footprint_upperbounds':[]}
for s in (4,5,6,7):
 rs=[r for r in foot['records'] if r['support']==s and (r['variant']=='fit' or r['variant'].startswith('candidate_'))]
 b=bbox_union([r['source_bounds_fullfine_xy_half_open'][op] for r in rs for op in ('mean16_bilinear','point_bilinear')])
 ts=set().union(*(tiles_for(r['source_bounds_fullfine_xy_half_open'][op]) for r in rs for op in ('mean16_bilinear','point_bilinear')))
 cb=bbox_union([[x,y,x+256,y+256] for x,y in ts])
 out['supports'][str(s)]={'native_support':rs[0]['native_xy_half_open'],'teacher_source_union':b,'conservative_inference_tiles_xy':sorted(map(list,ts)),'conservative_input_union_bbox':cb,'input_bbox_area':(cb[2]-cb[0])*(cb[3]-cb[1]),'boundingbox_is_conservative':True}
for a,b in itertools.combinations(out['supports'],2):
 x,y=out['supports'][a],out['supports'][b]
 shared=set(map(tuple,x['conservative_inference_tiles_xy']))&set(map(tuple,y['conservative_inference_tiles_xy']))
 out['pairs'].append({'supports':[int(a),int(b)],'shared_contributing_teacher_tiles':len(shared),'shared_tiles_xy':sorted(map(list,shared)),'teacher_context_bbox_overlap_fine_pixels':area_overlap(x['conservative_input_union_bbox'],y['conservative_input_union_bbox'])})
retained={r['index'] for r in json.loads((S/'teacher/REQUEST_LEDGER.json').read_text())['requests']}
for site in plan['sites']:
 s=site['site'];rr=[r for r in plan['requests'] if r['site']==s]
 out['retained_ct'].append({'site':s,'native_xywh':site['native_xywh'],'fine_level2_bbox':site['source_xy_half_open']['fine'],'bytes':sum(r['exact_bytes'] for r in rr),'fine_bytes':sum(r['exact_bytes'] for r in rr if r['acquisition']=='fine'),'native_bytes':sum(r['exact_bytes'] for r in rr if r['acquisition']=='native'),'chunks':len(rr)})
 # Retained source footprint includes all inverse-query locations plus CT interpolation/filter support.
 # Its full-resolution x4 rectangle is an overbound for teacher sampling on the original 272-square geometry domain.
 bb=[4*v for v in site['source_xy_half_open']['fine']]
 ts={ty*31+tx for ty in range(bb[1]//1024,(bb[3]-1)//1024+1) for tx in range(bb[0]//1024,(bb[2]-1)//1024+1)}
 out['prospective_teacher_full_footprint_upperbounds'].append({'site':s,'fullfine_bbox':bb,'tile_indices':sorted(ts),'new_tile_indices':sorted(ts-retained),'all_compressed_bytes':sum(tags[325][i] for i in ts),'new_compressed_bytes':sum(tags[325][i] for i in ts-retained),'caveat':'Current TIFF-header upper bound for original retained geometry domain. No new pixels fetched/decoded. Future shifted queries may extend this bbox; freeze exact chosen transfer/operator/context first.'})
out['source_metadata_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'SOURCE_FOOTPRINT.json',S/'provenance/TEACHER_TIFF_HEADER.json',C/'ct-study/REQUEST_PLAN.json']}
(O/'DATA_READINESS_METADATA.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='source_metadata_sha256'},indent=2))
