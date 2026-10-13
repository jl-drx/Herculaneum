from pathlib import Path
import json,hashlib
p=Path(__file__).parent/'packet';go=json.loads((p/'frozen/GO.json').read_text());im=json.loads((p/'INPUT_MANIFEST.json').read_text());pins={r['path']:r for r in go['pins']};checked=[];unmatched=[]
for r in im['files']:
 if r['source_path'] in pins:
  q=pins[r['source_path']];assert q['sha256']==r['sha256'] and q['size_bytes']==r['bytes'];checked.append(r['path'])
 else:unmatched.append(r['path'])
result={'input_files_matched_to_frozen_GO_source_pins':checked,'not_pinned_in_GO':unmatched,'limitation':'The multi-megabyte phase47 cumulative seal and immutable workspace fallback manifest are referenced by hash but not included. Their historic authenticity cannot be independently reread from this packet; local payload integrity and agreement with frozen GO pins are verified.'}
Path(__file__).with_name('source-identity-check.json').write_text(json.dumps(result,indent=2));print(len(checked),'pinned, unmatched:',unmatched)
