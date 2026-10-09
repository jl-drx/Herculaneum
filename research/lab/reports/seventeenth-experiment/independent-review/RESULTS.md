# Independent phase-17 numerical review

PASS for the frozen CT, infrared and conditional PPM diagnostics across all six fixed contexts. No producer numerical helper, Torch, checkpoint or model execution was used. No network requests were made by this reviewer. Earlier phase artifacts were only read.

| Check | Independently verified result |
| --- | --- |
| CT source/pooling | 1,530 retained batches, 848,908,566 encoded body bytes, 406,093,824 native uint16 values; all 15,040,512 pooled float32 values bitwise equal |
| CT profiles | 306 native and 102 pooled profiles across four supports; 8,160 integer and 14,688 floating fields exactly equal |
| Infrared | All 18 saved maps/2,654,208 values; explicit separable Gaussian with half-sample reflection, maximum map error 1.580957587066223e-13; all 12 AUROC/AP pairs within 1e-12, class and tie-group counts exact |
| PPM acquisition records | All 2,916 row spans independently derived and multipart bodies reparsed; 419,904 useful bytes, 8,748 records, 972 centers |
| PPM arithmetic | Both endian candidates for all six objects; 972 point geometries, 486 same-point map comparisons and 12 support tables; 85,604 float, 86,532 Boolean and 408 integer comparisons, maximum float difference 1.8189894035458565e-12 |
| Fixed-site coverage | Eight result-map zero centers, all outside literal surface, known and primary masks; all 972 sites retained |

The CT implementation uses independent `struct.unpack` row decoding, explicit 27-offset pooling and separate profile kernels. Native support is the exact 3×3 expansion of each frozen coarse support. The infrared check implements Gaussian weights and the two convolution passes explicitly, without importing SciPy functions. Its tie-aware ROC area uses grouped pairwise wins and AP uses complete descending score groups. The PPM check reconstructs the coordinate schedule and parses physical multipart framing independently before interpreting records.

All six PPM objects have exactly one candidate consistent with the frozen invariants: little endian. This is conditional serialization evidence. It does not authenticate the historical producer hardware, renderer invocation, native CT coordinate frame, anatomical normal direction or optical registration. Result-map zero-record totals are 45, 18 and 9 for Frag1, Frag2 and Frag3, corresponding to five, two and one zero centers in their second contexts. No zero centers occur in the flat maps. These are sparse coverage checks, not whole-surface certification.

The retained acquisition ledgers reconcile exactly: nine IR/prefix requests retained 41,044,172 response-body bytes; twelve sparse requests retained 776,100. These figures exclude the separately executed historical-source request and do not represent total wire traffic. The reviewer verified file hashes, recorded request counts/statuses, reservations, byte totals, exact ranges, strong ETag links and sparse If-Match values. This is verification of retained acquisition evidence, not independent observation of remote server execution.

Independent numerical runs used `/workspace/.venvs/herculaneum/bin/python`, Python 3.12.14, NumPy 2.5.3 and Pillow 12.3.0. Installed SciPy 1.18.1 was not used for the independent convolution. The author used NumPy 2.3.5 and SciPy 1.17.0; all comparisons passed the frozen tolerances despite that runtime difference. The CT reviewer took 30.97 seconds with 125,812,736 bytes peak RSS; infrared took 2.80 seconds with 284,405,760 bytes peak RSS. No producer rerun was needed.

Seven grouped numerical fixture checks preceded the CT/IR kernels' scientific work. Five grouped PPM fixture checks cover both true endian encodings, wrong-endian subnormals, exact zero sentinels and nonvacuity, nonfinite/nonunit rejection, exact flat coordinates/sign, and both/neither-candidate ambiguity. Original PPM fixtures are retained beside the source-driven zero-sentinel revision. Four multipart parser checks preceded scientific PPM reconstruction.

Two PPM comparison attempts stopped on checker presentation assumptions: the first lacked the final `mapped` field; the second expected literal zero XYZ rather than the author's null XYZ for an unmapped sentinel. The original checker sources and separate STOP receipts are preserved. `compare_ppm_v3.py` corrects those expected presentation fields only; numerical formulas, data, source identities and tolerances are unchanged. Its full comparison passed. The null sentinel representation correctly avoids presenting an unmapped tuple as a physical origin.

No software indexing, raw decoding, pooling or reported-statistic mismatch was found in these checks. Agreement with IR-derived labels is not independent registration accuracy. Intensity profiles and sparse geometry do not establish the cause of regional model failure or locate authenticated ink. No physical adapter correction, model promotion or confirmatory generalization claim follows from this review. Root's visual review and source-provenance interpretation remain separate evidence.

Receipts: `ct-comparison-receipt.json`, `ir-reconstruction/independent-ir-verification.json`, `ppm-comparison-receipt.json`, `ppm-support-coverage.json`; consolidated identities and results are in `verification.json`. `retention-manifest.json` seals this directory's final files, excluding only the manifest itself. Retention is not a backup, publication or fresh-environment restoration test.
