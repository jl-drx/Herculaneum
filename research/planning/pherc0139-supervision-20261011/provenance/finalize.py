import pathlib,json,hashlib,shutil,datetime
R=pathlib.Path(__file__).parent
h=json.load(open(R/'TEACHER_TIFF_HEADER.json'));tags={t['tag']:t['value'] for t in h['tags']};ranges=json.load(open(R/'TEACHER_HEADER_REQUESTS.json'))
offs=tags[324];lens=tags[325];assert len(offs)==len(lens)==775
metadata_ranges=[r['range'] for r in ranges if 'range'in r]
assert all(not(max(a,o)<min(b,o+n)) for a,b in metadata_ranges for o,n in zip(offs,lens))
assert all(o>=0 and n>0 and o+n<=h['length'] for o,n in zip(offs,lens))
assert all(o+n<=p for o,n,p in zip(offs[:-1],lens[:-1],offs[1:]))
assert tags[256]==[31040] and tags[257]==[24880] and tags[258]==[8] and tags[259]==[32946] and tags[262]==[1] and tags[277]==[1]
assert tags[322]==tags[323]==[1024] and 317 not in tags and 339 not in tags
for key,source in {
 'pinned-native_common.py':'/workspace/Herculaneum/research/planning/plinius-research-review-20261011/paired-data/sources/training__scripts__native_common.py',
 'pinned-build_native_dense_labels.py':'/workspace/herculaneum-lab/reports/frontier-methods-audit-20261008/source-evidence/training/421fb3ff94f21620cb2902d055f5c34be83cee23f57fa1cf23e88dbde4e131ff-nieuwlaar-training__scripts__build_native_dense_labels.py',
 'pinned-native_dense_report.json':'/workspace/Herculaneum/research/planning/plinius-research-review-20261011/paired-data/sources/training__native_dense_report.json',
}.items():shutil.copyfile(source,R/'sources'/key)
summary={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_target_or_label_pixels_fetched_or_decoded':True,'metadata_payload_bytes':sum(b-a for a,b in metadata_ranges),'metadata_ranges_disjoint_all_image_data_ranges':True,'all_tiles_inside_object':True,'all_tiles_disjoint':True,'tile_grid_yx':[25,31],'tile_count':len(offs),'shape_yx':[24880,31040],'tile_shape_yx':[1024,1024],'dtype':'uint8','compression':'TIFF Deflate 32946 (zlib)','predictor':'none/default1','samples_per_pixel':1,'photometric':'min-is-black1','orientation':'defaulttop-left1','expected_decoded_tile_bytes':1024*1024,'teacher_declared_canvas_contract_verified':True,'historical_teacher_bytes_identical_proven':False,'exact_v42_image_digest_verified':False,'historical_manual_override_identity_verified':False,'admissible_scope':'bounded teacher-only sensitivity using declared fine full raster; no accuracy or historical training target reproduction claim'}
json.dump(summary,open(R/'ADMISSION_SUMMARY.json','w'),indent=2)
# Explicit provenance for the public tag/source requests not captured by fetch_v42.py.
extra=[]
for file,url in [('villa-v42-ref.json','https://api.github.com/repos/ScrollPrize/villa/git/ref/tags/docker/ink-detection/v42'),('villa-v42-tag.json','https://api.github.com/repos/ScrollPrize/villa/git/tags/6b91f905080b09e06e6fd9c8980789886090d8cf'),('villa-paper-tree.json','https://api.github.com/repos/ScrollPrize/villa/git/trees/e583fb67468f483fadd73d52e06f0ab0fe5ba813?recursive=1'),('paper-build.sh','https://raw.githubusercontent.com/ScrollPrize/villa/e583fb67468f483fadd73d52e06f0ab0fe5ba813/ink-detection/optimized_inference/build.sh'),('paper-entrypoint.py','https://raw.githubusercontent.com/ScrollPrize/villa/e583fb67468f483fadd73d52e06f0ab0fe5ba813/ink-detection/optimized_inference/entrypoint.py')]:
 p=R/'sources'/file;extra.append({'path':str(p),'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'retrieved_date_utc':'2026-10-11','scope':'metadata/textonly'})
json.dump(extra,open(R/'ADDITIONAL_SOURCE_PINS.json','w'),indent=2)
print(json.dumps(summary,indent=2))
