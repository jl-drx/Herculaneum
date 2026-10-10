"""Metadata-only pooling plan builder; root supplies complete raw acquisition and source GO."""
import argparse
import json
from pathlib import Path

from pooling_common import HERE,LIMITS,sources,record,read_json,write_json,require,validate_source_go,load_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw-provenance',type=Path)
    parser.add_argument('--source-go',type=Path);parser.add_argument('--production-root',type=Path,action='append',default=[])
    parser.add_argument('--review',type=Path,action='append',default=[]);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--ready',action='store_true');args=parser.parse_args();source,pin=sources()
    freeze=record(HERE/'code-freeze.json') if (HERE/'code-freeze.json').exists() else None
    raw=record(args.raw_provenance) if args.raw_provenance else None;go=record(args.source_go) if args.source_go else None
    plan=dict(kind='offline-Paris-CT-pooling-v1',ready_for_execution=args.ready,sources=source,source_pins=pin,
        raw_provenance=raw,source_go=go,code_freeze=freeze,limits=LIMITS,reviews=[record(p) for p in args.review],
        production_roots=sorted({str((HERE.parent/'production-acquisition').resolve(strict=True)),
            str((HERE.parent/'production-pooling').resolve(strict=True)),
            *[str(p.resolve(strict=True)) for p in args.production_root]}),
        code={p.name:record(p) for p in sorted(HERE.glob('*.py'))},
        guard='All153 layouts/1530 complete encoded bodies verified before first numeric decode; no fill/transform/model/network.')
    if args.ready:
        require(raw is not None and freeze is not None,'Complete raw input/freeze required')
        validate_source_go(plan);metadata=read_json(raw)
        require(metadata['status']=='PASS' and metadata['schema_version']==1 and metadata['stage']=='pixels'
            and len(metadata['tiff_contracts'])==153 and len(metadata['batches'])==1530,'Raw production acquisition incomplete')
    args.output_dir.mkdir(parents=True,exist_ok=False);path=args.output_dir/'pooling-plan.json';write_json(path,plan)
    load_plan(path,execute=args.ready);print(json.dumps(dict(status='PASS',plan=record(path),CT_pixel_reads=0,model_calls=0)))


if __name__=='__main__':main()
