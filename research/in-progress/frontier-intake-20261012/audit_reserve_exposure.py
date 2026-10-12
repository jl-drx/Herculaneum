"""Read receipt/manifest metadata and artifact names, never CT/label/map arrays."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, unquote

TOKEN = re.compile(r'PHerc(?:0*175(?:A|B)?|1203)(?![0-9])', re.I)
URL = re.compile(r'https?://[^\s<>"\x27]+|s3://[^\s<>"\x27]+')
SKIP = ('frontier-intake-20261012', 'pherc0139-native-learning-20261011',
        'pherc0139-native-reference-20261011', '.git/')


def classify(url):
    path = unquote(urlsplit(url).path).rstrip('/')
    last = path.rsplit('/', 1)[-1]
    if last in ('.zarray', '.zattrs', '.zgroup', 'zarr.json') or path.endswith(('.json','.md','.txt','.html')):
        return 'metadata_or_document'
    if path.endswith('.zarr') or not last:
        return 'container_reference'
    if re.fullmatch(r'\d+(?:\.\d+)*', last) or path.endswith(('.npy','.npz','.nrrd','.tif','.tiff','.png','.jpg','.jpeg','.bin','.chunk')):
        return 'potential_scientific_payload'
    return 'unclassified_reference'


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values(): yield from walk(child)
    elif isinstance(value, list):
        for child in value: yield from walk(child)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', action='append', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    considered=parsed=0; matches=[]; artifacts=[]; errors=[]; oversized=[]
    for root in a.root:
        for path in root.rglob('*'):
            if not path.is_file():continue
            rel=path.relative_to(root).as_posix()
            if any(x in rel for x in SKIP):continue
            if TOKEN.search(rel) and path.suffix.lower() in ('.npy','.npz','.nrrd','.tif','.tiff','.png','.jpg','.bin','.chunk'):
                artifacts.append(str(path))
            if path.suffix not in ('.json','.jsonl'):continue
            considered+=1
            if path.stat().st_size > 16*1024*1024:
                oversized.append(str(path));continue
            text=path.read_text(errors='replace')
            if not TOKEN.search(text):continue
            try:
                values=[json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix=='.jsonl' else [json.loads(text)]
            except (ValueError,RecursionError):
                errors.append(str(path));continue
            parsed+=1
            for value in values:
                for obj in walk(value):
                    for key, field in obj.items():
                        if not isinstance(field,str):continue
                        for url in URL.findall(field):
                            if not TOKEN.search(url):continue
                            kind=classify(url)
                            # Receipt evidence is local to this dictionary. A URL in
                            # a planned source catalog is not a completed acquisition.
                            evidence=any(k in obj for k in ('sha256','response_sha256','response_bytes','bytes_received','body_sha256'))
                            if kind in ('potential_scientific_payload','unclassified_reference'):
                                matches.append({'file':str(path),'field':key,'url':url,'classification':kind,
                                                'receipt_like_identity_in_same_record':evidence})
    potential=[r for r in matches if r['classification']=='potential_scientific_payload' and r['receipt_like_identity_in_same_record']]
    result={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'status':'REVIEW_REQUIRED' if potential or artifacts or errors or oversized else 'NO_IDENTIFIED_PRIOR_PAYLOAD_IN_AUDITED_RECORDS',
            'metadata_files_considered':considered,'files_with_reserve_mentions_parsed':parsed,
            'possible_acquired_payload_receipts':potential,'named_scientific_artifacts':artifacts,
            'other_references_requiring_interpretation':matches,'parse_errors':errors,'oversized_metadata_unread':oversized,
            'exclusions':list(SKIP),'array_payloads_opened':0,
            'limits':'Receipt/name search is not an exhaustive proof of absent exposure in every anonymous cache, external device or upstream model. Full-parent embargo is prospective; unknown lineage remains unknown.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('other_references_requiring_interpretation','exclusions')},indent=2))


if __name__=='__main__':main()
