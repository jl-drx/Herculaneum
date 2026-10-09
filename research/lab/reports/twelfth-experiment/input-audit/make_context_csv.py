"""Export descriptive fields from the fixed audit JSON; never load source arrays."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'run/input-statistics.json'
SHA = 'bbca418976914a26b466b0f416797e01403c26e37001f6818ab6ede993a815b5'


def record(path):
    path = Path(path)
    return dict(path=str(path.resolve()), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    before = record(SOURCE)
    assert before['sha256'] == SHA and before['size_bytes'] == 930244
    result = json.loads(SOURCE.read_text())
    assert result['status'] == 'PASS' and result['total_tile_count'] == 159
    rows = []
    for context in result['contexts']:
        domains = context['domains']
        raw = domains['whole_context']['raw']
        tiles = [row for row in context['tiles'] if row['normalizer_called']]
        normalized = [row['normalizer']['normalized'] for row in tiles]
        affine = [row['affine_scale_sanity'] for row in tiles if 'affine_scale_sanity' in row]
        row = dict(id=context['id'], role=context['role'], physical_group=context['physical_group'],
            selected_voxel_count=raw['count'], raw_median=raw['median'], raw_mean=raw['mean'],
            raw_std=raw['std'], raw_mad=raw['mad'], raw_zero_fraction=raw['zero_fraction'],
            raw_fractional_fraction=raw['fractional_fraction'],
            fixed64_zero_fraction=domains['fixed64_interior']['raw']['zero_fraction'],
            published_support_zero_fraction=domains['separately_named_published_support']['raw']['zero_fraction'],
            support_semantics=context['source_contract']['source_surface_support_semantics'],
            depth_correlation=domains['whole_context']['adjacent_depth_aggregate']['pearson_correlation'],
            fixed64_depth_correlation=domains['fixed64_interior']['adjacent_depth_aggregate']['pearson_correlation'],
            depth_mean_absolute_difference=domains['whole_context']['adjacent_depth_aggregate']['mean_absolute_difference'],
            X_correlation=domains['whole_context']['adjacent_XY']['X']['pearson_correlation'],
            Y_correlation=domains['whole_context']['adjacent_XY']['Y']['pearson_correlation'],
            normalized_tile_std_min=min(row['std'] for row in normalized),
            normalized_tile_std_median=statistics.median(row['std'] for row in normalized),
            normalized_tile_std_max=max(row['std'] for row in normalized),
            normalized_tile_count=len(tiles),
            scaled_mad_tiles=context['tile_summary']['denominator_branches'].get('scaled_mad', 0),
            fallback_tiles=sum(value for key,value in context['tile_summary']['denominator_branches'].items() if key!='scaled_mad'),
            affine_max_absolute_difference=max((row['maximum_absolute_difference'] for row in affine), default=None),
            affine_mean_absolute_difference=statistics.mean(row['mean_absolute_difference'] for row in affine) if affine else None,
            affine_compared_elements=sum(row['compared_elements'] for row in affine),
            nominal_spacing_um=context['source_contract']['pixel_spacing_um'],
            independent_replicates=False)
        rows.append(row)
    target = HERE/'context-summary.csv'
    with target.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert record(SOURCE) == before
    receipt = dict(status='PASS', source=before,
        script=record(__file__), outputs=[record(target), record(HERE/'RESULTS.md')],
        source_array_reads=0, label_array_reads=0, prediction_or_logit_reads=0,
        normalizer_calls=0, model_calls=0, checkpoint_reads=0, network_calls=0,
        computation='Descriptive scalar-field export and medians of existing per-tile summary values from the root-produced JSON only.',
        source_unchanged_before_after=True)
    with (HERE/'SUMMARY_RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
