"""Assemble bounded metadata findings without decoding scientific arrays."""
import datetime, hashlib, json, pathlib, xml.etree.ElementTree as ET

R = pathlib.Path(__file__).resolve().parent
LAB = R.parents[2]
T = LAB/'reports/tenth-experiment/reference-audit'

def load(p): return json.loads(pathlib.Path(p).read_text())
def pin(p):
    p=pathlib.Path(p); b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(name,value): (R/name).write_text(json.dumps(value,indent=2)+'\n')

ledger=load(R/'request-ledger.json')
assert len(ledger['requests'])==24
assert ledger['body_bytes'] < 16*1024*1024
for entry in ledger['requests']:
    p=pin(entry['body_path'])
    assert p['sha256']==entry['body_sha256'] and p['bytes']==entry['body_bytes']
assert all(x['error'] is None for x in ledger['requests'])
reservation=load(T/'candidate-reservation-spec.json')
provenance=load(T/'retained-fragment-provenance.json')
paper_path=LAB/'data/native-reference-preflight-20261008/paper-ink-detection__metadata.json'
paper=load(paper_path)
assert pin(paper_path)['sha256']=='6cfaf7c64cd286da9e9f302df663e86d36573ec17b25a89dd64ff3c2644deaf8'
renderer=LAB/'data/experiments/pherc0800-native-ct-consistency/export-source/vc_render_tifxyz.cpp'
assert pin(renderer)['sha256']=='cda11ab76be310aa7e07fb09d9e01234f4aecc6f2977b1ee40966e77812fd002'
sources=[T/'LINEAGE.md',T/'candidate-reservation-spec.json',T/'retained-fragment-provenance.json',
 LAB/'reports/eighteenth-experiment/data-audit/DATA_AUDIT.md',paper_path,renderer,
 LAB/'data/experiments/pherc0800-native-ct-consistency/native-ct-plan.json',
 LAB/'reports/fifth-experiment/native-check/manifest.json',
 LAB/'data/experiments/pherc0800-native-ct-consistency/metadata.full.json.gz']
findings=[]
ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
for c in reservation['candidates']:
    sample=c['sample_id']; catalog=load(R/f'{sample}-full-retained-catalog-record.json')
    inv=ET.parse(R/'sources'/f'{sample}-package-inventory.xml')
    assert inv.find('s:IsTruncated',ns).text=='false'
    keys=[x.text for x in inv.findall('s:Contents/s:Key',ns)]
    assert len(keys)==39
    assert not any('mapping_validity' in x or 'provenance.json' in x for x in keys)
    metadata={}
    for asset in ['inklabels','supervision','validation']:
        root=load(R/'sources'/f'{sample}-{asset}-root.json')
        levelpath=(T/f'{sample}-labels_level2_metadata.json') if asset=='inklabels' else (R/'sources'/f'{sample}-{asset}-level2.json')
        a=load(levelpath)
        assert a['shape']==c['label_level2_shape_yx'] and a['data_type']=='uint8' and a['fill_value']==0
        metadata[asset]={'root':pin(R/'sources'/f'{sample}-{asset}-root.json'),'level2':pin(levelpath),'shape_yx':a['shape'],'dtype':a['data_type'],'fill_value':a['fill_value'],'dimension_names':a['dimension_names'],'storage_chunk_shape':a['chunk_grid']['configuration']['chunk_shape'],'shard_subchunk_yx':a['codecs'][0]['configuration']['chunk_shape'],'encoding_values_authenticated':False,'unique_values_not_read':True,'declared_scale_yx_um':root['attributes']['ome']['multiscales'][0]['datasets'][2]['coordinateTransformations'][0]['scale'],'published_translation':None}
        sources.append(levelpath)
    ids=['20250910185200','20250919125754-auto_grown_20250919055754487_inp_hr'] if sample=='PHerc0009B' else ['500p2a','-1']
    memberships=[{'segment':i,**paper['segments'][i],'configured_train':i in paper['training']['train_segments'],'configured_validation':i in paper['training']['val_segments']} for i in ids]
    aliases=['PHerc0009B','PHerc_0009B','0009b','9b','new_traces9b/david_9b',c['segment_long_id'],'0009b/auto_grown_20250919055754487_inp_hr_2um'] if sample=='PHerc0009B' else ['PHerc0500P2','PHerc_0500P2','PHerc_500P2','0500p2','500p2a','train_scrolls/500p2a','sean_hiddenlayers/-1','0500p2/-1',c['segment_long_id']]
    scan=catalog['scans']['20250718080859' if sample=='PHerc0009B' else '20250508033249']
    acq=scan['creation']['metadata']['metadata']['tomo']['acquisition']
    n=c['source_shape_zyx'][0]; pitch=c['source_spacing_zyx_um'][0]
    findings.append({'sample_id':sample,'status':'not_ready_for_confirmatory_accuracy_or_new_array_acquisition','current_local_exposure':'metadata_and_text_only; no scientific payload or prediction','future_role_if_contract_closed':'conditional supplied-reference agreement; not authenticated independent optical accuracy','confirmed_historical_checkpoint_holdout':False,'alias_exclusion':{'rule':'all representations and derived surfaces for entire parent, including unknown future aliases','known_text_aliases':aliases,'all_catalog_scan_ids':list(catalog['scans']),'all_catalog_volume_ids':list(catalog['volumes']),'all_catalog_segment_ids':list(catalog['segments']),'segment_count':len(catalog['segments']),'complete_global_alias_inventory_claim':False},'source_surface_url':c['source_surface_url'],'source_shape_zyx':c['source_shape_zyx'],'source_spacing_zyx_um':c['source_spacing_zyx_um'],'published_labels':metadata,'public_package_inventory':{'key_count':len(keys),'truncated':False,'mapping_validity_present':False,'linked_provenance_json_present':False,'keys':keys},'canvas_offset_source_yx':c['declared_mapping_geometry']['canvas_offset_yx'],'offset_interpretation':'base transfer code consumes offset while sampling source labels into target canvas; no evidence authorizes applying it again to published target labels','exact_dirty_publication_command_verified':False,'base_converter_matches_published_storage':False,'base_converter':'65-plane ZYX volume with LABEL_SLICE=32; actual arrays are 2D YX','renderer_geometry':{'pinned_commit':c['archived_renderer_metadata']['commit'],'mesh_center_index':(n-1)/2,'signed_offset_um_formula':f'(z - {(n-1)/2}) * {pitch} along the flipped unit normal','sampled_zero_mesh_plane':n%2==1,'flip_normals':True,'slice_step_source_voxels':1,'authenticated_annotation_plane':None,'anatomical_orientation_authenticated':False},'energy_provenance':{'scan_id':scan['id'],'catalog_keV':scan['creation']['metadata']['energy_keV'],'raw_recorded_acquisition_keV':acq['energy'],'scan_radix_literal':acq['scanRadix'],'detector_distance_mm':acq['sampleDetectorDistance'],'interpretation':'Use recorded acquisition energy; keep literal source names as aliases. No independent instrument calibration verification.'},'upstream_paper_config_membership':memberships,'upstream_membership_limit':'configuration membership is not checkpoint-specific or pixel-mask exposure proof','label_lineage':{'source_rasters':provenance['samples'][sample]['label_package']['creation_info']['source'],'annotated_on':'unknown','review_method':c['label_origin']['review_method'],'prediction_model_identity':None,'human_only_or_optical_truth_authenticated':False},'licenses':{'ct':c['source_scan_properties']['license'],'modern_label_redistribution_license_authenticated':False},'unresolved_before_array_acquisition':['exact dirty publication command and per-entry transfer report, including applied offset and any post-transfer realignment','public mapping-validity asset or documented alternative participation policy; cannot fabricate measured-only support','actual categorical ink/supervision/validation encoding and negative-label meaning','annotation reference surface and depth relation, and predeclared depth/window/normal handling','bounded no-prediction sampling/crop rule and source-to-target mapping validation','upstream model/teacher lineage sufficient for claimed evaluation role'],'array_acquisition_executed':False,'model_calls':0,'bbox_source_xywh':None,'depth_window':None,'pooling':None})
save('eligibility-matrix.json',{'schema':'phase19-scroll-metadata-audit/1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'network_requests':len(ledger['requests']),'network_body_bytes':ledger['body_bytes'],'http_status_counts':{str(s):sum(x['status']==s for x in ledger['requests']) for s in sorted(set(x['status'] for x in ledger['requests']))},'scientific_array_or_image_payloads_read':0,'model_calls':0,'paid_resources':0,'candidates':findings})
save('SOURCE_PINS.json',{'prior_sources':list({str(p):pin(p) for p in sources}.values()),'network_ledger':pin(R/'request-ledger.json'),'network_sources':[{k:x[k] for k in ['url','started_at_utc','finished_at_utc','status','body_path','body_sha256','body_bytes']} for x in ledger['requests']],'claims':'Prior local text hashes recomputed. No prior scientific array/checkpoint payload read.'})
print(json.dumps({'candidates':len(findings),'requests':len(ledger['requests']),'body_bytes':ledger['body_bytes'],'all_receipt_hashes_match':True}))
