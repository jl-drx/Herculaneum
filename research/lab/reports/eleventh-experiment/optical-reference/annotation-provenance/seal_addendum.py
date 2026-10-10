from datetime import datetime, timezone
from pathlib import Path
import hashlib
import html
import json
import re

root = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    with (root / name).open('x') as out:
        json.dump(value, out, indent=2)
        out.write('\n')

ledger = json.loads((root / 'request-ledger.json').read_text())
assert len(ledger['attempts']) == 8 and ledger['response_body_bytes'] == 505925
assert ledger['response_body_bytes'] <= 512 * 1024
assert all(e['response_body_bytes'] <= 128 * 1024 for e in ledger['attempts'])
assert all(e['http_status'] in (200, 206) for e in ledger['attempts'])
assert all(digest(Path(e['saved_path'])) == e['sha256'] for e in ledger['attempts'])

paper = root / 'educelab-arxiv-v4-complete.html'
source = paper.read_text()
assert paper.stat().st_size == 341140
assert digest(paper) == 'db692cf31f8ea02e935e6df97840278046bc4f0f82205355b48176a71a952e54'

anchors = ['S2.p1.1', 'S2.SS1.p2.1', 'S2.SS1.p3.1', 'S4.SS3.p1.1',
           'A1.SS2.p1.1', 'A1.SS2.p2.1', 'A1.SS5.p1.1', 'A1.SS5.p2.1',
           'A1.F11', 'A1.SS6.p5.1', 'A1.SS6.p6.1', 'A1.SS6.p7.1',
           'A1.SS8.p1.1', 'A1.SS8.p3.1', 'A1.SS8.p4.1',
           'A1.SS9.p1.1', 'A1.SS9.p2.1', 'A1.SS10.p2.1', 'A1.SS10.p5.1']
anchors += [f'alg3.l{i}' for i in [2, 6, 9, 10, 11, 12]]

def cleaned(raw):
    raw = re.sub(r'<math\b[^>]*alttext="([^"]*)"[^>]*>.*?</math>', lambda m: m.group(1), raw, flags=re.S)
    return re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]+>', ' ', raw))).strip()

excerpts = []
for anchor in anchors:
    tag = 'p' if anchor.endswith('.1') and not anchor.startswith('alg3') else 'div'
    if anchor == 'A1.F11':
        tag = 'figure'
    match = re.search(fr'<{tag}\b[^>]*id="{re.escape(anchor)}"[^>]*>(.*?)</{tag}>', source, flags=re.S)
    if not match:
        raise ValueError(f'Anchor not found: {anchor}')
    excerpts.append({'anchor': anchor, 'source_url': f'https://arxiv.org/html/2304.02084v4#{anchor}',
                     'retained_html_line_start': source[:match.start()].count('\n') + 1,
                     'text': cleaned(match.group(1)),
                     'transformation': 'HTML tags removed, entities decoded, math rendered from published alttext; embedded images not fetched'})
write('paragraph-excerpts.json', {'paper_version': '2304.02084v4', 'paper_date': '2024-05-20',
                                 'paper_sha256': digest(paper), 'excerpts': excerpts})

code = root / 'inkid-dataset.py'
blob = hashlib.sha1(b'blob ' + str(code.stat().st_size).encode() + b'\0' + code.read_bytes()).hexdigest()
tree = json.loads((root / 'inkid-paper-tree.json').read_text())
entry = next(e for e in tree['tree'] if e['path'] == 'inkid/data/dataset.py')
assert not tree['truncated']
assert blob == entry['sha'] == '5e55c9b0b4196bb0e717984f7d6d4134d9c9f654'
commit = json.loads((root / 'inkid-paper-commit.json').read_text())
assert commit['sha'] == '58b1519551b73554168371a31b1ca27767ef6e33'

prior = root.parent / 'MANIFEST.json'
prior_manifest = json.loads(prior.read_text())
assert all(digest(root.parent / e['path']) == e['sha256'] for e in prior_manifest['files'])
write('SOURCE_RECEIPTS.json', {
    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'authorization': 'Separate root-authorized eight-attempt/512KiB official text-only annotation provenance lookup after initial20-request budget',
    'requests': ledger,
    'paper': {'url': 'https://arxiv.org/html/2304.02084v4', 'version': '2304.02084v4',
              'publication_version_date': '2024-05-20', 'article_license': 'CC-BY-NC-SA4.0; not substituted for data license',
              'byte_length': paper.stat().st_size, 'sha256': digest(paper),
              'assembly': [
                  {'file': 'educelab-arxiv-v4-bounded.html', 'byte_range': [0, 131071], 'initial200_bounded_prefix': True},
                  {'file': 'educelab-arxiv-v4-range131072.html', 'byte_range': [131072, 262143], 'http206_content_range_verified': True},
                  {'file': 'educelab-arxiv-v4-range262144.html', 'byte_range': [262144, 341139], 'http206_content_range_verified': True}],
              'same_etag': '"CIGsiOSXupYDEAE="', 'assembled_length_equals_declared_total': True,
              'first_two_response_blocks_marked_partial_by_hardcap': True, 'complete_assembled_html': True},
    'implementation': {'repository': 'educelab/ink-id', 'paper_commit_abbreviation': '58b1519',
                       'resolved_commit': commit['sha'], 'git_tree_sha1': commit['commit']['tree']['sha'],
                       'path': 'inkid/data/dataset.py', 'git_blob_sha1': blob,
                       'git_blob_verified_against_pinned_tree': True, 'sha256': digest(code),
                       'byte_length': code.stat().st_size,
                       'relevant_line_anchors': {'image_loading': [205, 211], 'feature_transform': [283, 289],
                                                 'ambiguity_and_sampling': [319, 356], 'surface_and_binary_classes': [410, 452]},
                       'source_read_not_executed': True, 'called_uint16_utility_implementation_not_fetched': True},
    'closed_at_declared_publication_protocol_level': [
        'four-fragment inclusion includingParis1Fr39', 'optical1000nm source and manual binarytracing underpapyrology',
        'manual PuppetWarp registration method', 'surfacebounds mask plus intended zero-negative/nonzero-positive class',
        'boundary exclusion FIND_EDGES plus squareMaxFilter(33) for radius16',
        'published65channel r32/interval1 trilinear centered-depth convention',
        'consumerhalfopen boundingbox semantics'],
    'limits_not_closed': ['exactPNGpublication hash and annotatoridentity', 'local opticalregistration residual/warp',
                          'exactstack generationbinary/argv', 'modernmodel normal convention',
                          'upstreamteacher/pretraining independence', 'source-supported9um adaptation to3.24um legacyinputs',
                          'large PPMbinaryformat/frame and nativeCTcorrespondence ifused'],
    'no_uint16_to_uint8_CT_recipe_established': True,
    'earlier_report_manifest_sha256': digest(prior), 'all_prior_manifest_files_unchanged': True,
    'arrays_images_imageheaders_model_gpu_credentials_outreach_termsacceptance': False,
})
write('MANIFEST.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                       'scope': 'additionalannotation-provenancetextlookup only; originalreport preserved',
                       'files': [{'path': p.name, 'byte_length': p.stat().st_size, 'sha256': digest(p)}
                                 for p in sorted(root.iterdir()) if p.is_file() and p.name != 'MANIFEST.json']})
print(json.dumps({'https_attempts': 8, 'body_bytes': 505925, 'paper_bytes': 341140,
                  'git_blob_verified': True, 'old_evidence_unchanged': True}))
