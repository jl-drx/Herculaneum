"""Post-acquisition completeness report from the six retained metadata bodies only."""
import hashlib
import json
from pathlib import Path
import urllib.parse
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
ledger=json.loads((BASE/'run-v1/request-ledger.json').read_text())
plan=json.loads((BASE.parent.parent/'benchmark-inventory/METADATA_QUERY_PROPOSAL.json').read_text())
assert ledger['status']=='PASS' and len(ledger['requests'])==len(plan['queries'])==6
assert [r['url'] for r in ledger['requests']]==[r['url'] for r in plan['queries']]
assert [r['id'] for r in ledger['requests']]==[r['id'] for r in plan['queries']]
assert sum(r['observed_body_bytes'] for r in ledger['requests'])==59771
records=[]
for query,rec in zip(plan['queries'],ledger['requests']):
    body=Path(rec['body_path']).read_bytes()
    assert rec['status']=='PASS' and rec['http_status']==200
    assert len(body)==rec['observed_body_bytes'] and hashlib.sha256(body).hexdigest()==rec['body_sha256']
    assert len(body)<=query['max_body_bytes']
for inv_rec,root_rec in zip(ledger['requests'][::2],ledger['requests'][1::2]):
    xml=ET.fromstring(Path(inv_rec['body_path']).read_bytes())
    ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
    prefix=xml.findtext('s:Prefix',namespaces=ns)
    assert xml.findtext('s:IsTruncated',namespaces=ns)=='false'
    keys={e.findtext('s:Key',namespaces=ns):e for e in xml.findall('s:Contents',ns)}
    assert len(keys)==int(xml.findtext('s:KeyCount',namespaces=ns))==39
    relative=sorted(k[len(prefix):] for k in keys)
    roles=sorted({k.split('/')[0] for k in relative})
    assert roles==['inklabels.zarr','supervision.zarr','validation.zarr']
    expected=[]
    for role in roles:
        expected += [role+'/zarr.json']
        for level in range(6): expected += [f'{role}/{level}/zarr.json',f'{role}/{level}/c/0/0']
    assert relative==sorted(expected)
    root=json.loads(Path(root_rec['body_path']).read_bytes())
    attrs=root['attributes'];pub=attrs['ink_republish'];multiscale=attrs['ome']['multiscales'][0]
    root_key=urllib.parse.unquote(urllib.parse.urlsplit(root_rec['url']).path.lstrip('/'))
    assert int(keys[root_key].findtext('s:Size',namespaces=ns))==root_rec['observed_body_bytes']
    headers={k.lower():v for k,v in root_rec['headers'].items()}
    assert keys[root_key].findtext('s:ETag',namespaces=ns)==headers['etag']
    assert pub['atlas_sample']=='PHerc0841' and pub['provenance']=='provenance.json'
    assert [a['name'] for a in multiscale['axes']]==['y','x']
    assert [d['path'] for d in multiscale['datasets']]==[str(i) for i in range(6)]
    assert all([t['type'] for t in d['coordinateTransformations']]==['scale'] for d in multiscale['datasets'])
    records.append({'segment':pub['atlas_segment'],'source_entry':pub['source_entry'],'inventory_complete':True,'inventory_objects':39,
        'published_role_prefixes':roles,'mapping_validity_present_in_prefix':False,'linked_provenance_present_in_prefix':False,
        'literal_declared_canvas_size':attrs['canvas_size'],'canvas_size_axis_order':'not established by these root fields alone',
        'root_axes':['y','x'],'root_level2_scale_um':multiscale['datasets'][2]['coordinateTransformations'][0]['scale'],
        'root_has_translation_transform':False,'canvas_offset_yx':pub['canvas_offset_yx'],'package_commit':pub['package_commit'],
        'provenance_link':pub['provenance'],'transfer_report_path_kind':'publisher-local path, not a public retrievable URL',
        'source_image_path_kind':'publisher-local path, not a public retrievable URL',
        'root_size_and_etag_match_inventory':True,'root_body_sha256':root_rec['body_sha256'],'inventory_body_sha256':inv_rec['body_sha256']})
result={'status':'PASS_METADATA_ACCOUNTING','v2_attempts':6,'v1_preserved_failed_attempts':1,'total_stage_attempts_including_separate_reauthorization':7,
        'observed_body_bytes':59771,'scientific_arrays_read':0,'no_automatic_retries':True,'records':records,
        'limits':['Absence applies to these complete prefixes at request time, not all locations.',
                  'Array-level metadata/data were not read; dtype, actual shapes/value encodings, S/V semantics and negative support not verified.',
                  'No translation transform does not authenticate target registration or justify applying source canvas offsets again.',
                  'Model independence, label authorship, optical truth and teacher history remain unresolved.']}
(BASE/'PACKAGE_FINDINGS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'segments':len(records),'bytes':59771}))
