"""Read retained text only and record model-specific provenance without loading models."""
from pathlib import Path
import datetime,hashlib,json
B=Path('/workspace/herculaneum-lab'); O=Path(__file__).resolve().parent
specs=[
('checkpoint_contract','reports/sixth-experiment/model-contract/audit-v2/model-contract.json',None),
('exposure_ledger','reports/seventh-experiment/model-exposure-ledger.json',None),
('phase18_data_audit','reports/eighteenth-experiment/data-audit/DATA_AUDIT.md',None),
('paper_pipeline_metadata','data/native-reference-preflight-20261008/paper-ink-detection__metadata.json','https://raw.githubusercontent.com/ScrollPrize/villa/e583fb67468f483fadd73d52e06f0ab0fe5ba813/ink-detection/metadata.json'),
('paper_model_source','data/native-reference-preflight-20261008/paper-ink-detection__train_resnet3d_lib__model.py','https://raw.githubusercontent.com/ScrollPrize/villa/e583fb67468f483fadd73d52e06f0ab0fe5ba813/ink-detection/train_resnet3d_lib/model.py'),
('paper_iteration0_card','data/native-reference-preflight-20261008/iteration-0-README.md','https://huggingface.co/scrollprize/PHerc.1667-iteration-0/raw/06c306449b39df42c745608ceade243498d24243/README.md'),
('paper_iteration0_config','data/native-reference-preflight-20261008/iteration-0-config.json','https://huggingface.co/scrollprize/PHerc.1667-iteration-0/raw/06c306449b39df42c745608ceade243498d24243/config.json'),
('paper_source_manifest','data/native-reference-preflight-20261008/source-manifest.json',None),
('paper_supplemental_manifest','data/native-reference-preflight-20261008/supplemental-source-manifest.json',None),
('curated_labels_snapshot','data/experiments/known-ink-control/upstream-curated-ink-labels-README.md','https://huggingface.co/buckets/scrollprize/datasets/resolve/ink-labels/README.md'),
('ink9um_labels_snapshot','reports/seventh-experiment/native-reference/c9a88ff970594fcef3788eb588bb75dbbd0e6436f4812f89074e26c0d9fef7a5-ink9um-README-hop1.body','https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/README.md'),
('russo_teacher_source','reports/frontier-methods-audit-20261008/source-evidence/official-contributions/russo-scripts-teacher_seg.py','https://raw.githubusercontent.com/DomRusso2/ink9um-dense/dd5bc8fb300c810c6ec042cb7e068be78829d086/scripts/teacher_seg.py'),
('russo_source_receipts','reports/frontier-methods-audit-20261008/source-evidence/official-contributions/sources.json',None),
('nieuwlaar_teacher_sources','reports/frontier-methods-audit-20261008/source-evidence/training/25d17fdb764e92fbea419b804bd1f0a831ed4555271b4bc8bccdd0aeab9acace-nieuwlaar-hf-training__teacher_sources.tsv','https://huggingface.co/Nieuwlaar/ink9um-dense-native/raw/b894a0493ab2291cc8dc3247020c0d7cb0967a58/training/teacher_sources.tsv'),
('nieuwlaar_teacher_receipts','reports/frontier-methods-audit-20261008/source-evidence/training/request-ledger.json',None),
('seed42_factory_source','vendor/villa-ink-baseline/vesuvius/ink_detection/models/model.py','https://raw.githubusercontent.com/ScrollPrize/villa/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/models/model.py'),
('seed42_factory_manifest','vendor/villa-ink-baseline/source-manifest.json',None)]
records=[]
for key,rel,url in specs:
 p=B/rel; raw=p.read_bytes();records.append(dict(id=key,path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),source_url=url,acquisition='retained text read only'))
assert next(r for r in records if r['id']=='checkpoint_contract')['sha256']=='6cbd873f6c41a01414351424d6d3e487b15f934597b838858e965faaab3714b5'
assert next(r for r in records if r['id']=='paper_pipeline_metadata')['sha256']=='6cfaf7c64cd286da9e9f302df663e86d36573ec17b25a89dd64ff3c2644deaf8'
contract=json.loads((B/specs[0][1]).read_text());seed=contract['models']['seed42'];cfg=seed['embedded_config'];pm=json.loads((B/specs[3][1]).read_text())
extract={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_checkpoint_loading':True,'seed42':{k:seed[k] for k in ['label','sha256','model_repository','model_revision','repository_path','step']},'seed42_embedded_config':cfg,'paper_metadata':pm,'interpretation':{'seed42_configured_backbone_pretrained_path':cfg['data_provenance']['backbone_pretrained_path'],'seed42_configured_backbone_pretrained_sha256':cfg['data_provenance']['backbone_pretrained_sha256'],'seed42_declared_parent_roots':sorted(set(d['sampling_scroll'] for d in cfg['datasets'])),'seed42_candidate_roots_absent':['PHerc0009B','PHerc0500P2','PHerc0051'],'paper_vs_canonical_model_identity':'different: paper ResNet3D-50; canonical production ResNet-152','actual_seed42_label_teacher_ancestry':'unresolved','paper_train_val_lists_prove_identical_pixels':False}}
(O/'provenance-extract.json').write_text(json.dumps(extract,indent=2)+'\n')
(O/'retained-source-pins.json').write_text(json.dumps({'scope':'Retained text only, fresh SHA-256 computation; no scientific image, array or checkpoint payloads read.','records':records},indent=2)+'\n')
print(json.dumps({'retained_text_files':len(records),'bytes':sum(x['bytes'] for x in records),'seed42_parent_roots':extract['interpretation']['seed42_declared_parent_roots'],'seed42_configured_backbone_pretraining':None}))
