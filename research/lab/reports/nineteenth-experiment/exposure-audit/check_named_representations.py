"""Exact inspection of declared identifiers; no fuzzy or physical-identity inference."""
import hashlib,json,pathlib
O=pathlib.Path(__file__).resolve().parent
p=pathlib.Path('/workspace/herculaneum-lab/reports/sixth-experiment/model-contract/audit-v2/model-contract.json')
raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='6cbd873f6c41a01414351424d6d3e487b15f934597b838858e965faaab3714b5'
cfg=json.loads(raw)['models']['seed42']['embedded_config'];rows=[]
for d in cfg['datasets']:
 for name in d['segments']:
  rows.append({'sampling_scroll':d['sampling_scroll'],'segment':name,'physical_key':d['sampling_physical_segment_keys'][name],'representation_key':d['sampling_representation_keys'][name]})
assert len(rows)==29
aliases={'PHerc0009B':['PHerc0009B','PHerc9B','0009b','9b','20250910185200','20250919125754-auto_grown_20250919055754487_inp_hr','auto_grown_20250919055754487_inp_hr_2um'], 'PHerc0500P2':['PHerc0500P2','PHerc500P2','0500p2','500p2a','-1'], 'PHerc0051':['PHerc0051','PHerc51','0051','51','PHerc51Cr4Fr8','PHerc0051Cr04Fr08','Frag6']}
results={}
for parent,names in aliases.items():
 found=[]
 for i,row in enumerate(rows):
  values=[row['sampling_scroll'],row['segment'],row['physical_key'],*row['physical_key'].split(':')]
  matches=sorted(set(n.casefold() for n in names)&set(v.casefold() for v in values))
  if matches:found.append({'row':i,'matches':matches})
 results[parent]={'case_insensitive_exact_identifiers_checked':names,'matches':found,'conclusion':'absent_from_inspected_declared_identifiers' if not found else 'present'}
assert all(not v['matches'] for v in results.values())
(O/'named-representation-audit.json').write_text(json.dumps({'source_path':str(p),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_json_path':'models.seed42.embedded_config.datasets','representation_count':len(rows),'representations':rows,'candidate_checks':results,'scope_limit':'Identifier comparison only. Not an exhaustive alias graph, proof of historical execution, or teacher/pretraining ancestry audit. -1 is a collection-scoped marker and does not alone establish a parent.'},indent=2)+'\n')
print(json.dumps({'representations':len(rows),'candidate_identifier_matches':{k:len(v['matches']) for k,v in results.items()}}))
