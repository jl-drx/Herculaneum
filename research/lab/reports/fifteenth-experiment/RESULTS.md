# Plinius: expanded optical-reference benchmark

The frozen label-only construction passed. Three additional physical fragments now supply six deterministically selected regions with separate calibration and evaluation roles. An independent implementation reconstructed the support and selections from the six source PNGs. This is benchmark preparation; no CT pixels or model predictions were acquired in this stage, and no accuracy or text-reading result is claimed.

| Region | Role | Parent | Native XYWH | Primary positives | Primary negatives |
|---|---|---|---|---:|---:|
| frag1-context1 | evaluation | Paris2 | [2304, 1152, 1152, 1152] | 19,400 | 69,159 |
| frag1-context2 | evaluation | Paris2 | [0, 6912, 1152, 1152] | 12,633 | 70,681 |
| frag2-context1 | evaluation | Paris2 | [1152, 1152, 1152, 1152] | 19,049 | 61,660 |
| frag2-context2 | evaluation | Paris2 | [8064, 12672, 1152, 1152] | 20,248 | 66,318 |
| frag3-context1 | calibration | Paris1 | [2304, 1152, 1152, 1152] | 9,434 | 82,951 |
| frag3-context2 | calibration | Paris1 | [3456, 5760, 1152, 1152] | 5,771 | 83,490 |

Each region is 1152×1152 native pixels, sampled into 384×384 labels. Primary support excludes a 32-coarse-pixel context border, uncertain 33×33 native neighborhoods and pixels outside the surface mask. Every selected region passed the frozen 90% surface, 512-positive and 8192-negative gates. Selection used the first eligible grid region and the last eligible region separated by at least 2304 native y pixels; no region, grid or gate was substituted after inspection. All 24 arrays have retained byte hashes.

Frag3 (PHercParis1Fr34) is calibration. Frag1 (PHercParis2Fr47) and Frag2 (PHercParis2Fr143) are evaluation, reserved as whole fragments across aliases and scan versions. Both evaluation fragments share Paris2, giving one evaluation parent, distinct from calibration parent Paris1. Frag4 remains a previously seen Paris1 diagnostic. Evaluation labels were inspected for predefined support gates and are not blinded. Unknown upstream teacher/pretraining exposure prevents an upstream-independent generalization claim.

The source audit made 42 bounded metadata requests totaling 116,634 response-body bytes. Six complete PNG requests then downloaded 652,579 encoded bytes, with no retries or redirects. The selection controller took 7.985 seconds in total, with maximum measured worker RSS 862.0 MiB. The exact candidate tables, source encodings, unsupported-positive counts and selected array identities are retained in the machine-readable contract and worker receipts. Independent verification is linked below; synthetic preparation failures were retained and corrected before real downloads.

The next useful experiment is an efficient CT reader check, followed by a fixed-weight calibration-transfer comparison. The old full-width TIFF-band reader would fetch about 4.6 GiB for these six regions under the old layout assumption. The new TIFF layouts, Range behavior and stable object identities have not been probed. Freeze a bounded header/IFD audit and a transfer strategy before downloading CT. Then compare the unchanged baseline and both thirteenth states using only Frag3 to fit any new cutoffs, locking them before Paris2 evaluation. Do not start more training or rent a GPU based on benchmark construction alone.

Current legacy data-use terms match the retained version; redistribution/publication restrictions and EduceLab citations remain attached. No upload, publication, submission or new terms acceptance occurred. GPU spend is GBP 0; the total first-pilot cap remains GBP 10. Worker secret delivery remains unverified. Repository tracked files and all prior frozen research evidence remain unchanged.

Records: [benchmark contract](BENCHMARK_CONTRACT.json), [frozen design](BENCHMARK_DESIGN.md), [independent verification](/workspace/herculaneum-lab/reports/fifteenth-experiment/numeric-review/completed-review/independent-verification.json), [CT cost calculation](implementation/planning/CT_COST.md), [next study](NEXT_STUDY_DRAFT.md).
