"""Prepare reviewed-ready or draft label-only plan; never access scientific arrays/network."""
from pathlib import Path
import argparse
import json
from label_common import *


def nested_records(value):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys() and ('bytes' in value or 'size_bytes' in value):
            yield value
        for child in value.values():
            yield from nested_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_records(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--review', action='append', default=[])
    parser.add_argument('--ready', action='store_true')
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    require(not output.exists(), 'Fresh plan directory required')
    design = record(ROOT.parent / 'BENCHMARK_DESIGN.md')
    source_record = record(ROOT.parent / 'SOURCE_CONTRACT.json')
    require(design['sha256'] == DESIGN_SHA and source_record['sha256'] == SOURCE_SHA, 'Root contracts changed')
    source = json.loads(verify(source_record).read_text())
    code = [record(ROOT / n) for n in CODE_NAMES]
    reviews = [record(p) for p in args.review]
    if args.ready:
        require(reviews, 'Ready plan requires technical GO')
        accepted = False
        for review in reviews:
            value = json.loads(verify(review).read_text())
            if value.get('status') == 'GO':
                pins = list(nested_records(value))
                require(all(any(p['sha256'] == c['sha256'] and Path(p['path']).resolve() == Path(c['path'])
                                for p in pins) for c in code), 'GO does not bind all executed source')
                accepted = True
        require(accepted, 'Source-bound technical GO is required')
    pins = {str(Path(p['path']).resolve()): record(verify(p))
            for p in list(nested_records(source)) + [design, source_record] + code + reviews}
    plan = dict(schema_version=1, ready=args.ready, design=design, source_contract=source_record,
                runtime=runtime_check(), limits=LIMITS, selection=SELECTION, code=code,
                technical_reviews=reviews, fragments=source['eligible_candidates'],
                objects=[{'fragment': c['legacy_alias'].lower(), 'kind': kind,
                          'url': c[key], 'shape_yx': c['shape_yx']}
                         for c in source['eligible_candidates']
                         for kind, key in [('labels', 'labels_url'), ('mask', 'mask_url')]],
                input_pins=list(pins.values()), scientific_calls=0,
                deferred=source['deferred_candidates'],
                limitations={'upstream_exposure': 'unknown complete teacher/pretraining exposure',
                             'registration_residual': 'unmeasured', 'CT_ready': False,
                             'roles': 'fixed before label/image acquisition; no fragment substitution'})
    output.mkdir(parents=True)
    path = output / 'label-plan.json'
    save(path, plan)
    load_plan(path, ready=args.ready)
    print(json.dumps({'plan': record(path), 'ready': args.ready, 'input_pins': len(pins)}))


if __name__ == '__main__':
    main()
