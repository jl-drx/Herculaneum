"""Fixed native CT cross-sections from retained observed cubes, with unknown masks."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sample(positions, arrays):
    """Diagnostic float64 trilinear interpolation; require every observed corner."""
    shape = positions.shape[:-1]
    xyz = positions.reshape(-1, 3)
    base = np.floor(xyz).astype(np.int64)
    fraction = xyz - base
    valid = np.ones(len(xyz), dtype=bool)
    values = np.zeros(len(xyz), dtype=np.float64)
    for delta in itertools.product([0, 1], repeat=3):
        indices = base + np.array(delta)
        chunk_xyz = indices // 128
        corner = np.full(len(xyz), np.nan, dtype=np.float64)
        for key, array in arrays.items():
            chosen = np.all(chunk_xyz == np.array(key[::-1]), axis=1)
            local = indices[chosen] % 128
            corner[chosen] = array[local[:, 2], local[:, 1], local[:, 0]]
        known = np.isfinite(corner)
        valid &= known
        weight = np.prod(np.where(np.array(delta), fraction, 1 - fraction), axis=1)
        values += np.where(known, corner, 0) * weight
    values[~valid] = np.nan
    return values.reshape(shape).astype(np.float32), valid.reshape(shape)


def main():
    result_path = ROOT / 'native-check/offline-revalidation.json'
    result_hash = sha(result_path)
    result = json.loads(result_path.read_text())
    plan_path = Path(result['frozen_plan_path'])
    if sha(plan_path) != result['frozen_plan_sha256']:
        raise ValueError('Native plan changed.')
    if len(result['points']) != 10:
        raise ValueError('Ten completed profiles are required for these fixed views.')
    output = ROOT / 'native-sections'
    output.mkdir(exist_ok=False)
    arrays, input_records = {}, []
    for record in result['native_chunk_sources']:
        p = Path(record['local_path'])
        if record.get('acquisition') != 'new_http200_payload' or record.get('http_status') != 200:
            raise ValueError('This study requires observed HTTP 200 cubes.')
        if p.stat().st_size != 128**3 or sha(p) != record['sha256']:
            raise ValueError('Native cube size/hash mismatch.')
        arrays[tuple(record['chunk_index_zyx'])] = np.frombuffer(p.read_bytes(), dtype=np.uint8).reshape(128,128,128)
        input_records.append({'path': str(p), 'sha256': record['sha256']})
    normal_offsets = np.arange(-24, 25, dtype=float)
    tangent_offsets = np.arange(-48, 49, dtype=float)
    peaks = [p for p in result['points'] if p['name'].endswith('.peak')]
    if [p['source_xy'] for p in peaks] != [[1278,504],[1100,628]]:
        raise ValueError('Fixed response peaks differ.')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad('#c7a7d8')
    records = []
    for row, point in enumerate(peaks):
        frame = point['visualization_orthogonal_frame_xyz']
        base = np.array(point['base_ct_xyz'], dtype=float)
        n = np.array(frame['n'], dtype=float)
        for col, axis in enumerate(['u','v']):
            tangent = np.array(frame[axis], dtype=float)
            if abs(np.linalg.norm(tangent)-1) > 1e-5 or abs(np.dot(tangent,n)) > 1e-5:
                raise ValueError('Diagnostic frame is not orthonormal.')
            positions = (base + normal_offsets[:,None,None] * n
                         + tangent_offsets[None,:,None] * tangent)
            values, known = sample(positions, arrays)
            name = point['name'].replace('.','-') + '-' + axis
            np.save(output / f'{name}-values.npy', values, allow_pickle=False)
            np.save(output / f'{name}-observed-mask.npy', known, allow_pickle=False)
            ax = axes[row,col]
            im = ax.imshow(values, cmap=cmap, vmin=0, vmax=255, interpolation='nearest',
                           origin='lower', extent=[-48.5*8.64,48.5*8.64,-24.5*8.64,24.5*8.64], aspect='equal')
            ax.axhline(0, color='#ef4444', lw=.9)
            for depth in [-8,8]: ax.axhline(depth*8.64, color='#22c55e', lw=.8, ls='--')
            for depth in [-15,15]: ax.axhline(depth*8.64, color='#38bdf8', lw=.8, ls=':')
            ax.axvline(0, color='#ef4444', lw=.5, alpha=.6)
            ax.set_title(f'{point["name"]}, source {point["source_xy"]}: {axis.upper()} / normal')
            ax.set_xlabel(f'In-plane {axis.upper()} offset (micrometers)')
            ax.set_ylabel('Recorded flipped-normal offset (micrometers)')
            fig.colorbar(im, ax=ax, shrink=.85, label='Observed CT intensity, fixed 0–255')
            records.append({'name':name, 'source_xy':point['source_xy'], 'shape':list(values.shape),
                'observed_pixels':int(known.sum()), 'unknown_pixels':int((~known).sum()),
                'base_ct_xyz':base.tolist(), 'unit_normal_xyz':n.tolist(), 'unit_tangent_xyz':tangent.tolist(),
                'center_sample':float(values[24,48]), 'center_surface_intensity':point['observed_profile_uint8'][15]})
    fig.suptitle('Fixed native CT cross-sections: rendering geometry diagnostics, not an ink reading\nPurple = incomplete retained-cube support; no additional downloads', fontsize=13)
    fig.legend(handles=[Line2D([0],[0],color='#ef4444',label='Base surface, depth 0'),
        Line2D([0],[0],color='#22c55e',ls='--',label='Detector depth window ±8 voxels'),
        Line2D([0],[0],color='#38bdf8',ls=':',label='Rendered depth extent ±15 voxels')],
        loc='outside lower center', ncols=3)
    fig.savefig(output / 'native-cross-sections.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True, constrained_layout=True)
    for ax, point in zip(axes, peaks):
        offsets = np.arange(-15,16)
        ax.plot(offsets, point['observed_profile_uint8'], 'o-', ms=3, label='Retained surface samples')
        ax.plot(offsets, point['rendered_profile_uint8'], 'x--', ms=5, label='Reconstructed native CT samples')
        ax.axvspan(-8,8,color='#22c55e',alpha=.1)
        ax.set_ylim(0,255)
        ax.set_ylabel('uint8 intensity')
        ax.set_title(f'{point["name"]}, source {point["source_xy"]}; max error {point["max_abs_error"]}')
        ax.legend(loc='upper right')
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Recorded normal offset (native voxels, 8.64 micrometers each)')
    fig.suptitle('Fixed peak depth profiles — local source/render correspondence only')
    fig.savefig(output/'native-profiles.png',dpi=160)
    plt.close(fig)
    if sha(result_path) != result_hash:
        raise ValueError('Native result changed during plotting.')
    summary = {'created_at_utc':datetime.now(timezone.utc).isoformat(), 'result_path':str(result_path),
        'result_sha256':result_hash, 'plan_sha256':result['frozen_plan_sha256'], 'code_sha256':sha(__file__),
        'native_inputs':input_records, 'sections':records, 'normal_offsets_voxels':normal_offsets.tolist(),
        'tangent_offsets_voxels':tangent_offsets.tolist(), 'network_requests':0,
        'interpolation':'Diagnostic float64 trilinear interpolation; all eight corners must be retained and observed. Distinct from archived float32/FMA equality test.',
        'limitations':['Plane is tangent to the cached local frame, not a curved-surface reconstruction.',
                       'Unknown cube support stays unknown; it is not empty tissue.',
                       'Matching rendered data does not establish ink or correct papyrus-layer placement.'],
        'letter':None,'transcription':None,'translation':None}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    files=[{'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]
    (output/'manifest.json').write_text(json.dumps({'files':files,'self_hash':'Omitted'},indent=2)+'\n')
    print(json.dumps({'sections':records,'network_requests':0},indent=2))


if __name__ == '__main__':
    main()
