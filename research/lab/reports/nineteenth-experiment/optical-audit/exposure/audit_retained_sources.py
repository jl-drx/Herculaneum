"""Read retained human-readable sources only; no scientific data or network."""
from pathlib import Path
import hashlib, html, json, re
LAB = Path('/workspace/herculaneum-lab')
OUT = LAB / 'reports/nineteenth-experiment/optical-audit/exposure'
rels = [
 'reports/fifteenth-experiment/source-audit/frag5-config.json',
 'reports/fifteenth-experiment/source-audit/frag6-config.json',
 'reports/fifteenth-experiment/BENCHMARK_DESIGN.md',
 'reports/eighteenth-experiment/data-audit/DATA_AUDIT.md',
 'reports/eleventh-experiment/optical-reference/annotation-provenance/educelab-arxiv-v4-complete.html',
 'reports/eleventh-experiment/optical-reference/annotation-provenance/inkid-paper-commit.json',
 'reports/eleventh-experiment/optical-reference/annotation-provenance/inkid-paper-tree.json',
 'reports/eleventh-experiment/optical-reference/annotation-provenance/inkid-dataset.py',
 'reports/eleventh-experiment/optical-reference/annotation-provenance/SOURCE_RECEIPTS.json',
 'data/native-reference-preflight-20261008/paper-ink-detection__metadata.json',
 'data/native-reference-preflight-20261008/paper-ink-detection__train_resnet3d_lib__config.py',
 'data/native-reference-preflight-20261008/paper-ink-detection__train_resnet3d_lib__datasets_builder.py',
 'data/methods-review/paper.txt',
 'data/models/ink9um-model-card.md',
 'reports/seventh-experiment/model-exposure-ledger.json',
]
patterns = {
 'parent_0051_named': r'\bp[.\s_-]*herc[.\s_-]*0*51\b',
 'fragment_51cr4fr8': r'\b(?:p[.\s_-]*herc[.\s_-]*)?0*51[.\s_-]*cr[.\s_-]*4[.\s_-]*fr[.\s_-]*8\b',
 'legacy_frag6_named': r'\bfrag(?:ment)?[.\s_-]*0*6\b',
}
records=[]
for rel in rels:
 p=LAB/rel;b=p.read_bytes();t=b.decode('utf-8')
 records.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'named_alias_matches':{k:[{'line':t[:m.start()].count('\n')+1,'match':m.group()} for m in re.finditer(v,t,re.I)] for k,v in patterns.items()}})
raw=(LAB/rels[4]).read_text();plain=html.unescape(re.sub('<[^>]+>',' ',raw));plain=re.sub(r'\s+',' ',plain)
excerpts={}
for term in ['16 scans of 6 fragments from 4 scrolls','four fragment surfaces','Some names are temporarily obfuscated','P.Herc.Paris. Objet 59','P.Herc. s.n. (Cass. 110)','fragments_all']:
 i=plain.find(term); assert i>=0, term
 excerpts[term]=plain[max(0,i-180):i+len(term)+600]
(OUT/'source-pins.json').write_text(json.dumps({'scope':'Named-alias screening of explicitly enumerated retained human-readable files; not a global training absence proof. No scientific arrays/checkpoints decoded; no network.','patterns':patterns,'sources':records},indent=2)+'\n')
(OUT/'paper-excerpts.json').write_text(json.dumps({'source_path':str(LAB/rels[4]),'source_sha256':records[4]['sha256'],'normalization':'HTML tags removed, entities unescaped, whitespace collapsed; excerpts selected by exact text string.','excerpts':excerpts},indent=2)+'\n')
print(json.dumps({'source_count':len(records),'source_bytes':sum(r['bytes'] for r in records),'alias_positive_files':[r['path'] for r in records if any(r['named_alias_matches'].values())]}))
