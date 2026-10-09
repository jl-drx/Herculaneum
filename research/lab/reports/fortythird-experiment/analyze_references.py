"""Read-only source extraction and metadata checks; no image or model access."""
import csv,gzip,hashlib,io,json
from collections import Counter
from decimal import Decimal
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),sha256=h,size_bytes=p.stat().st_size)
def put(name,v):
 with (P/name).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def check(v,msg):
 if not v:raise ValueError(msg)
def main():
 ledger=json.loads((P/'REQUESTS.json').read_text());go=json.loads((P/'GO.json').read_text())
 for v in go['pins']:check(pin(v['path'])==v,'Frozen source identity')
 check(len(ledger)==24 and len({r['url'] for r in ledger})==24,'Bounded unique request count')
 for r in ledger:check(pin(r['raw']['path'])==r['raw'] and r['bytes']==r['raw']['size_bytes'] and r['complete'] and r['status']==200,'Complete retained response')
 total=sum(r['bytes'] for r in ledger);check(total==429661 and total<=go['max_delivered_bytes'],'Byte cap')
 catalog=json.loads(gzip.decompress((P/'sources/catalog.txt').read_bytes()))['samples'];targets={k:catalog[k] for k in ['PHercParis4','PHerc0500P2']}
 for name,v in targets.items():check(v==json.loads((P/(name+'-catalog.json')).read_text()),'Exact target extraction')
 source_tree=json.loads((P.parents[1]/'data/methods-review/paper-code-tree.json').read_text());check(not source_tree['truncated'],'Complete pinned code tree')
 blobs={r['path']:r for r in source_tree['tree'] if r['type']=='blob'}
 blob_checks=[]
 for tag,path in [('guided-dino','vesuvius/docs/guided_dinovol.md'),('guided-config','vesuvius/src/vesuvius/models/configuration/single_task/ps128_guided_dinovol_ink.yaml'),('fragments-doc','scrollprize.org/docs/02_data_fragments.md')]:
  body=(P/'sources'/(tag+'.txt')).read_bytes();gitsha=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest();check(gitsha==blobs[path]['sha'] and len(body)==blobs[path]['size'],'Pinned Git blob identity');blob_checks.append(dict(path=path,git_blob_sha1=gitsha))
 labels={}
 for name,data in targets.items():
  labels[name]=[dict(segment=k,**a) for k,s in data['segments'].items() for a in s.get('data',[]) if a['type']=='ink-labels']
 check(len(labels['PHercParis4'])==8 and len(labels['PHerc0500P2'])==1,'Catalog label count')
 check(all(r['properties']['registration']=='transferred' and r['properties']['annotated_on']=='unknown' for rows in labels.values() for r in rows),'Explicit label provenance status')
 front=targets['PHerc0500P2']['segments']['20250628074500'];check(front['suffix']=='500P2_front' and not any(r['type']=='ink-labels' for r in front['data']),'No front label entry in this catalog')
 attrs=json.loads((P/'sources/front-surface-attrs.txt').read_text());check(attrs['num_slices']==118 and attrs['canvas_size']==[15060,26440],'Front surface metadata')
 listings=[];ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
 for r in ledger:
  if 'list-type=2' in r['url']:
   root=ET.fromstring(Path(r['raw']['path']).read_bytes());truncated=root.findtext('s:IsTruncated',namespaces=ns);check(truncated=='false','Inspected S3 prefix page complete')
   listings.append(dict(tag=r['tag'],prefix=root.findtext('s:Prefix',namespaces=ns),truncated=False,directories=[x.text for x in root.findall('s:CommonPrefixes/s:Prefix',ns)],objects=[dict(key=x.findtext('s:Key',namespaces=ns),size_bytes=int(x.findtext('s:Size',namespaces=ns))) for x in root.findall('s:Contents',ns)]))
 manifest=list(csv.DictReader(io.StringIO((P/'sources/profilometer-manifest.txt').read_text())));check(len(manifest)==14,'Manifest case count')
 sampling=json.loads((P/'sources/profilometer-sampling.txt').read_text());config=json.loads((P/'sources/profilometer-experiment.txt').read_text());native=Decimal(str(sampling['native_lateral_sampling_um_per_pixel']));factors=sampling['resolution_factors']
 check(native==Decimal('0.68793625'),'Declared dataset sampling')
 for factor,effective in zip(factors,sampling['effective_lateral_sampling_um_per_pixel_exact']):check(native*factor==Decimal(effective),'Factor-derived sampling arithmetic')
 max_extent_error=Decimal(0);extent_residuals=[]
 for row in manifest:
  width,height=int(row['width_px']),int(row['height_px']);check(width*height==int(row['pixel_count']),'Pixel count')
  check(Decimal(row['lateral_sampling_x_um'])==Decimal(row['lateral_sampling_y_um'])==native,'Manifest sampling consistency')
  for extent,count in [('x_extent_um',width),('y_extent_um',height)]:
   signed=Decimal(row[extent])-(count-1)*native;error=abs(signed);max_extent_error=max(max_extent_error,error)
   extent_residuals.append(dict(papyrus=row['papyrus_id'],sample=row['sample_id'],axis=extent,signed_residual_um=str(signed),exceeds_initial_0p0001_check=error>Decimal('.0001')))
 check(factors==config['experiments']['matched_resolution']['train_factors']==config['experiments']['fixed_native_model']['test_factors'],'Declared experiment factors')
 tree=json.loads((P/'sources/profilometer-tree.txt').read_text());hfblobs={r['path']:r for r in tree if r['type']=='file'}
 for tag,path in [('profilometer-card','README.md'),('profilometer-manifest','manifest.csv')]:
  body=(P/'sources'/(tag+'.txt')).read_bytes();check(hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==hfblobs[path]['oid'],'HF pinned text Git blob')
 root_bucket=json.loads((P/'sources/ink-bucket-directories.txt').read_text());check(any(r['path']=='ink/0500p2' for r in root_bucket),'Public original-fragment annotation directory lead')
 first_page=json.loads((P/'sources/ink-bucket.txt').read_text());check(len(first_page)==1000 and all(r['path'].startswith('ink/0009b/') for r in first_page),'HF initial page limitation retained')
 findings=dict(status='CT_REFERENCE_NOT_ADMITTED; PROFILOMETRY_METADATA_LEAD',created_at_utc=datetime.now(timezone.utc).isoformat(),targets=list(targets),CT_reference_admitted=False,CT_reads=0,media_reads=0,model_calls=0,external_messages_sent=0,public_label_entries=labels,front_segment=dict(id='20250628074500',suffix='500P2_front',volume='20250526151718',voxel_um=2.215,released_depth_planes=118,paper_figure_depth_planes=65,canvas_yx=[15060,26440],same_as_paper_projection_verified=False,independent_reference=False),profilometry=dict(repo='scrollprize/profilometer',revision='a806bead2f3b9100c20e19de37814fb285cfeefd',manifest_regions=len(manifest),regions_by_papyrus=dict(Counter(r['papyrus_id'] for r in manifest)),height_pixels_in_manifest=sum(int(r['pixel_count']) for r in manifest),metadata_sampling_um=float(native),paper_reported_sampling_um=.34,metadata_to_paper_sampling_ratio=float(native/Decimal('.34')),maintainer_calibration_notice=True,scale_calibration_resolved=False,maximum_extent_rounding_error_um=float(max_extent_error),declared_factors=factors,PHerc500P2_samples=[r for r in manifest if r['papyrus_id']=='PHerc. 500P2'],actual_arrays_read=False,CT_mapping_provided_in_inspected_manifest=False,physical_sample_overlap_verified=False,possible_use='Supporting relative-factor morphology control after file, annotation and split QC; no CT inference claim'),limitations=['Only the frozen two-target search was performed; global public unavailability is not established','Original ink/0500p2 bucket exists but its contents were not inspected within this request budget','HF recursive ink listing was one 1000-file page, not a complete bucket inventory','The profilometry discovery concerns the same named PHerc500P2 material but its measured regions are not registered to the CT front surface','Internal consistency of sampling metadata does not resolve conflict with the paper'])
 put('FINDINGS.json',findings);put('LISTING_SUMMARY.json',listings);put('EXTENT_ARITHMETIC.json',dict(status='SMALL_RECORDED_RESIDUALS_NOT_CALIBRATION',maximum_absolute_residual_um=str(max_extent_error),initial_overstrict_check_failures=sum(r['exceeds_initial_0p0001_check'] for r in extent_residuals),rows=extent_residuals))
 elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(ledger[0]['started_at_utc'])).total_seconds();check(elapsed<=1800,'Reference search concluded within time bound')
 receipt=dict(status='PASS',created_at_utc=datetime.now(timezone.utc).isoformat(),GO=pin(P/'GO.json'),requests=24,bytes=total,successful_complete_responses=24,failed_requests=0,HTTP_retries=0,seconds_since_first_request=elapsed,Git_blob_checks=blob_checks,HF_text_blob_checks=2,exact_catalog_target_extractions=2,complete_S3_prefix_pages=len(listings),profilometry_manifest_rows_checked=14,derived_sampling_factors_checked=9,image_bodies_checked=0,CT_chunks_checked=0,reference_truth_established=False,scope='Metadata identity and arithmetic verification, not physical calibration, alignment, pixel-label QC or learning',analyzer=pin(__file__))
 put('SOURCE_VERIFICATION.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
