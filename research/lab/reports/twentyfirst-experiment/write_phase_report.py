"""Render root's phase synthesis from completed, reviewed receipts only."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
geometry=json.loads((HERE/'pherc0139-geometry/public-geometry-implementation-v2/run/result.json').read_text())
assert geometry['status']=='PASS'
assert json.loads((HERE/'pherc0139-runtime-review/verification.json').read_text())['status']=='PASS'
assert json.loads((HERE/'native-header-review/verification.json').read_text())['status']=='PASS'
rows='\n'.join(f"| {c['id']} | {c['counts']['mesh_supported_level2']:,} / 147,456 | {c['counts']['mesh_and_bounds_supported_level2']:,} / 147,456 |" for c in geometry['contexts'])
report=f'''# Plinius phase 21: preservation allocation and geometry/source checks

**Keep unchanged seed42.** The frozen CPU experiment passed all five training-mechanism criteria but failed five of fourteen development criteria. Broader preservation on the selected retired-reference training support reduced drift; it did not produce the required regional performance. No model was promoted, no text was recovered, and paid GPU spend remains **£0**.

The [study report](ALLOCATION_RESULTS.md) gives the full protocol, both arms, all six regions and verification limits. The [next-study recommendation](NEXT_STUDY_DRAFT.md) prioritizes defensible independent evaluation and source-compatible inputs before more tuning or GPU spending. This phase also completed separately bounded source and geometry audits, without changing the study's masks or decisions.

## Model result

Both matched 5,296-parameter adapters trained for exactly 32 updates, from identical initialization on the same PHerc0139 manual schedule and frozen seed42 backbone. Control used manual BCE + 2×manual KL; candidate used manual BCE + manual KL + retired-reference KL. Each KL was a separate group mean from original seed42 outputs. Retired labels were not training targets.

Paris2 equal-fragment AUROC/AP was **0.657642/0.384286** for seed42, **0.668216/0.366364** for control, and **0.640727/0.364451** for candidate. Independent numerical verification passed, with no scientific retries, threshold fitting or numerical decision ambiguity. All six Paris regions remain reused development, not independent confirmation.

The support audit established supplied participation roles and complete source acquisition, not tissue validity. The two supplied masks do not overlap; simply removing the validation exclusion would add no supervised pixels. The separately frozen allocation comparison therefore tested an explicit M/V reallocation and excluded the unknown group. Full historical geometry validity was never substituted into that objective.

## PHerc0139 source and computational geometry

The official full metadata ties curated w016 to public segment `20250108000004-w029_2025010827`, source volume `20260102150214`, and renderer commit `650bd56663efb7225fdba73fe267ba86b8e29368`. The exact source gives a 109-plane render, one normal flip and the half-pixel raster origin. Selected source planes 21..88 correspond to offsets −33..+34 along that flipped direction. Two aligned rounded 2×2 reductions contribute all 16 fine XY samples to each level2 output.

Public transformed XYZ TIFFs contain a 1444×1404 mesh. With the observed scale this reconciles the 28880×28080 surface raster. The current complete public prefix has no `mask.tif`; this does not prove a mask was absent from historical private staging. The archived orchestration was marked dirty, so the source reconstruction is qualified.

After separate proposal, implementation and numerical-contract review, six fixed single-range requests acquired **2,739,744 response-body bytes**. Only 153,600 bytes were decoded into the two 80×80 XYZ windows; 2,586,144 gap bytes remain opaque. Frozen byte ranges, original object validators and the exact context footprints were enforced. No native CT chunks, ink labels or model outputs were used.

The reconstruction covered 4,718,592 fine XY samples and the complete 68-depth interval through a proved monotone endpoint rule. A coarse output requires all 16 contributing fine samples. Mesh/stencil validity and complete interpolation-corner bounds in the declared original native frame are retained separately:

| Fixed training context | Mesh/stencil support | Mesh plus declared native bounds |
|---|---:|---:|
{rows}

The [runtime review](pherc0139-runtime-review/verification.json) records independent coverage and limitations. The production run took {geometry['elapsed_seconds']:.2f} seconds and peaked at {geometry['peak_rss_bytes']/1048576:.2f} MiB RSS. These masks describe **current public computational support under a conservative declared arithmetic path**. They do not establish observed native voxels, masked tissue, authentic ink, historical private validity or exact private-renderer output equality. They remain separate from the completed learning experiment.

Neither audited training context has a missing mesh stencil or a failed declared-coordinate bound under these checks. CT reproduction and annotation alignment remain separate questions; this result does not explain the Paris2 Fr47 development losses.

A pre-launch technical STOP preserved the original reader: its atomic ledger writes lacked durable synchronization and its framing checks needed tightening. The additive v2 reader was corrected and reviewed before the sole live run; no acquisition retry or scientific change followed.

## Frag6 sparse geometry

Stage A acquired 1,504 bytes in 10 structural requests. The PPM is a 6205×8853 ordered grid with six double fields per record and a 73-byte header. The inspected TIFF metadata supplies no source-frame UUID, signed render offset or physical-depth rule.

Stage B used one multipart request: 64,747 body bytes containing 34,992 useful bytes across 243 exact ranges. Independent reassembly and geometry checking passed. Of 729 fixed records, 558 were mapped and 171 were zero sentinels; little endian alone passed the fixed finite/unit-normal conditions. Sixty-two of 81 neighborhoods were complete. Seven of nine predeclared possible native-comparison sites were complete; the other two were not replaced.

All mapped coordinates fit the declared primary bounds. Stored normals opposed the raster x-right/y-down cross product; tangency was approximate. These observations do not authenticate an energy scan, inward/outward anatomy or an ink-depth sign. Passing the minimum site-count condition does not supply missing source/render provenance. **Stage C remains closed.** See the [independent review](frag6-b-runtime-review/RESULTS.md).

## Native checkpoint and source correction

The pinned Nieuwlaar `dense_native_016000.safetensors` header matches all 508 expected F32 tensor names/shapes and 244 factory alias-name groups. Its 68,175,426 named elements and 272,701,704 disjoint data bytes are consistent with a 264-group factory projection of 34,546,498 parameters. The former parameter-count concern is resolved at the serialized-schema level; separate native tensor intervals do not prove equal values or shared storage.

This took three HTTPS attempts and 67,306 observed response-body bytes, including one redirect body retained only as a hash/count. No tensor values were requested. Full-file SHA, finiteness, alias-value equality, strict model loading and native preprocessing parity remain unverified. The [header findings](native-checkpoint-contract/query-run/RESULTS.md) and [independent check](native-header-review/verification.json) retain those boundaries. No checkpoint was silently adapted or partially loaded.

The source synthesis also corrects a sentence in the older immutable frontier audit: Nieuwlaar's pinned label builder explicitly adds manual support with `sup = sup | m_sup`. The [additive correction and evidence](research-context/EVIDENCE.md) preserve the earlier record and identify the precise source. Prior broader-KL studies also added pseudo-label BCE and changed other conditions; this phase's matched allocation contrast was a distinct question.

## Retention and readiness

The study, full unfavorable outcomes, two final adapter states, raw numerical traces, source receipts, failed pre-launch fixtures/reviews and final independent checks are retained locally. All first-twentieth evidence and tracked repository files remain unchanged; the existing untracked `.codex/` is preserved. The additive register and externally anchored retention manifest bind the completed phase.

The working runtime remains Python 3.12.14, NumPy 2.5.3 and Torch 2.12.0+cpu. No new installation or service is needed. Updated startup instructions preserve prior history and explain how to verify retained state; saving them is a configuration draft, not publication, external backup or demonstrated fresh-task restoration.

The **£10 total** first paid pilot cap remains intact, including failed starts/retries and attributable overheads. A useful frozen GPU workload, pinned CUDA environment, current pricing and margin, independent shutdown and verified bucket-scoped R2 worker-secret delivery are still required. The recorded temporary credential expiry is 2026-10-15 09:58:59 UTC. No credentials, uploads, external publication, outreach, new terms or prize submission were part of this batch.
'''
with (HERE/'RESULTS.md').open('x') as f:f.write(report)
print('Phase report written from verified results')
