"""Read-only verification of restored study52. Never imports a model or worker."""
from pathlib import Path
import datetime, hashlib, itertools, json
import numpy as np

ROOT = Path(__file__).resolve().parent
LAB = Path('/workspace/herculaneum-lab')
P = LAB / 'reports/fiftysecond-experiment'

def load(p):
    return json.loads(Path(p).read_text())

def pincheck(r):
    p = Path(r['path'])
    assert p.is_file() and p.stat().st_size == r['size_bytes'], p
    with p.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == r['sha256'], p

def main():
    index = load(ROOT / 'INDEX.json')
    go = load(P / 'INFERENCE_GO.json')
    results = load(P / 'INFERENCE_RESULTS.json')
    go_sha = hashlib.sha256((P / 'INFERENCE_GO.json').read_bytes()).hexdigest()
    assert go_sha == index['go_sha256'] == '6e9d6f48614767d232a4601785c6278b5d09a8d73380967fc9a7038990b421e8'
    assert go['depths'] == [-15, 15] and go['expected_maps'] == 32 and go['expected_batches'] == 3096
    assert results['status'] == 'PASS' and results['all_saved_logit_replays_exact'] is True
    assert results['gpu_spend_GBP'] == 0
    expected = set(itertools.product([m['name'] for m in go['models']], ['plain', 'padded'], ['forward', 'reverse'], [-15, 15]))
    identities, jobs = set(), []
    support = np.load(P / 'work/output-support.npz', allow_pickle=False)
    batches = tiles = output_files = 0
    for rec in results['maps']:
        ident = (rec['model'], rec['phase'], rec['direction'], rec['depth'])
        assert ident in expected and ident not in identities
        identities.add(ident)
        folder = P / 'work/inference' / f'{ident[0]}-{ident[1]}-{ident[2]}-d{ident[3]:+03}'
        assert load(folder / 'COMPLETE.json') == rec
        assert rec['status'] == 'PASS' and rec['go_sha256'] == go_sha
        assert rec['core_xywh'] == go['core_xywh'] == [512, 1024, 1536, 512]
        family = 'hecate' if ident[0] == 'hecate' else 'villa'
        grid = next(g for g in go['grids'] if (g['family'], g['phase']) == (family, ident[1]))
        expected_batches = grid['batches']
        receipts = sorted(folder.glob('batch-*.json'))
        assert len(receipts) == len(expected_batches) == rec['counts']['model_calls']
        replay = rec['read_only_replay']
        assert replay['status'] == 'PASS' and replay['new_forwards'] == 0
        assert replay['reused_forwards'] == replay['model_calls'] == len(expected_batches)
        positions = next(v for v in go['depth_positions'] if v['d'] == ident[3])
        for i, (path, cohort) in enumerate(zip(receipts, expected_batches)):
            b = load(path)
            assert path.name == f'batch-{i:05}.json'
            assert b['index'] == i and b['original_batch_index'] == cohort['original_batch_index']
            assert b['cohort_yx'] == cohort['cohort_yx']
            assert b['binding'] == dict(go_sha256=go_sha, job=list(ident), depth=ident[3], positions=positions)
            assert all(v['shape'][0] == len(cohort['cohort_yx']) for v in b['inputs'])
            pincheck(b['logits'])
            logits = np.load(b['logits']['path'], allow_pickle=False)
            assert logits.dtype == np.float32 and logits.shape[0] == len(cohort['cohort_yx'])
            assert np.isfinite(logits).all()
            batches += 1
            tiles += len(cohort['cohort_yx'])
        assert sum(len(c['cohort_yx']) for c in expected_batches) == rec['counts']['model_tiles'] == replay['model_tiles']
        for pin in rec['outputs'].values():
            pincheck(pin)
            output_files += 1
        pincheck(rec['support_mask'])
        aligned = np.load(rec['outputs']['aligned']['path'], allow_pickle=False)
        display = np.load(rec['outputs']['display']['path'], allow_pickle=False)
        supported = np.load(rec['outputs']['supported']['path'], allow_pickle=False)
        mask = support[family + '_' + ident[1]]
        assert aligned.shape == supported.shape == display.shape == mask.shape == (512, 1536)
        assert aligned.dtype == np.float32 and display.dtype == np.uint8
        assert np.isfinite(aligned).all() and ((aligned >= 0) & (aligned <= 1)).all()
        assert np.array_equal(display, np.rint(aligned * 255).astype(np.uint8))
        assert np.array_equal(supported[mask], aligned[mask]) and np.isnan(supported[~mask]).all()
        jobs.append(dict(model=ident[0], phase=ident[1], direction=ident[2], depth=ident[3], batches=len(receipts), source_recorded_replay_exact=True))
    assert identities == expected and len(list((P / 'work/inference').glob('*/COMPLETE.json'))) == 32
    assert batches == results['total_unique_model_batches'] == 3096
    assert tiles == results['total_model_tiles'] == 11976
    baselines = load(P / 'BASELINES.json')['maps']
    assert len(baselines) == 16
    for rec in baselines:
        pincheck(rec['crop'])
        a = np.load(rec['crop']['path'], allow_pickle=False)
        assert a.shape == (512, 1536) and a.dtype == np.float32 and np.isfinite(a).all()
    available, missing = [], []
    for pin in index['prerequisites']:
        if Path(pin['path']).is_file():
            pincheck(pin)
            available.append(pin['path'])
        else:
            missing.append(pin['path'])
    report = dict(status='PASS', verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), checkpoint_commit='02812de445f6ab0069174805bb14b6014bed159c', remote_completed_checkpoint_at_utc=index['updated_at_utc'], inference_results_sha256=hashlib.sha256((P / 'INFERENCE_RESULTS.json').read_bytes()).hexdigest(), completed_maps=32, baseline_maps=16, verified_batches=batches, verified_model_tiles=tiles, verified_map_output_files=output_files, source_recorded_replay_exact=True, new_model_execution=False, new_logit_accumulation_replay=False, available_prerequisites_verified=len(available), missing_prerequisites=len(missing), missing_prerequisite_explanation='Earlier study51 inputs and code are not restored here; no restart is needed because all completed outputs and their bound receipts are restored.', scope='Archive/member hashes, complete Cartesian map coverage, all raw-logit receipts and hashes, depth/cohort/GO bindings, map output hashes/numeric invariants, 16 baseline hashes. This is not fresh model inference or scientific interpretation.', jobs=jobs)
    (ROOT / 'COMPLETION_VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'jobs'}, indent=2))

if __name__ == '__main__':
    main()
