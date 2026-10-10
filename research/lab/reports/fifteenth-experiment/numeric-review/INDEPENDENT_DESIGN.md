This verifier independently reconstructs the fifteenth label-only benchmark after root confirms that the six-object acquisition and deterministic selection completed. Preparation used source, metadata and synthetic inputs only. It imports no author acquisition or selection module, SciPy filter, Torch or model code; socket connections are blocked.

The active checker is `independent_check.py`, with `integral_support.py`, `png_decode.py` and `review_io.py`. Its final synthetic receipt is `preparation-tests-v2/synthetic-fixtures.json`: 24 cases pass, including brute-force square counts, zero-padded homogeneous support, sampled band boundaries, literal six-candidate row-major counts, canonical context extraction, exact integer 90% eligibility, separation failures and binary PNG palette/CRC acceptance or rejection.

The independent support calculation uses signed int32 two-dimensional prefix sums on 256 output-row bands plus 16-row halos. Each sum is checked against the int32 size bound. It reconstructs every native known pixel, counts native support and retains only the source-aligned coarse known raster. It never allocates a full-native int64 integral image. The largest fragment is processed alone; only the current labels and surface fields, its coarse support and small band temporaries coexist. Address space is capped at 12 GiB and observed peak RSS must remain at or below 3 GiB.

All six complete encoded PNG byte streams are hash/CRC/shape checked before any real pixel decoding. Decoding checks raw mode and binary values, used palette colors, absence of transparency and canonical nonzero-to-one conversion. Verification then compares every grid candidate and exact eligibility decision, the first/last eligible spacing rule and all 24 retained arrays, including their dtype, shape, bytes and SHA256. Fragment aliases, parent papyri, calibration/evaluation roles, deferred/seen exclusions, source-center mappings and bounding boxes are checked against the frozen design/source contract. Completed file byte caps and recorded worker resource/HTTP receipts are checked as well.

Preserved preparation history includes the first PNG synthetic failure and `png_decode_v1.py`. Calling Pillow `getpalette()` before `verify()` had implicitly loaded a palette image, preventing the verification call. The corrected decoder verifies immediately after its first open and retrieves the palette on the second load. This was a synthetic preparation failure; no real PNG or scientific output was read. Earlier standalone and first combined fixture receipts are retained as history; the v2 combined receipt binds the final active code.

Execution requires root-supplied exact hashes for the ready plan, acquisition provenance and panel provenance, plus `--authorized-retained-read`. A fresh owned output directory is required. The authorized saved-output pass writes a PASS receipt or preserves a STOPPED receipt. No acquisition, scientific selection, model call or network retry occurs in this verifier.

This establishes deterministic label-supported panel construction. It does not establish CT readiness, model accuracy, unseen pretraining exposure, optical registration residuals, or independent specimens for every context. The three physical fragments have two parent papyri: both evaluation fragments share Paris2; calibration Paris1 also contains the existing seen Frag4 diagnostic. HTTP/TLS and author resource execution claims remain pinned source and recorded receipt evidence because this reviewer performs no network or author worker execution.

The execution interface is:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/fifteenth-experiment/numeric-review/independent_check.py \
  --authorized-retained-read \
  --plan PLAN --plan-sha256 PLAN_SHA256 \
  --acquisition ACQUISITION --acquisition-sha256 ACQUISITION_SHA256 \
  --panel PANEL --panel-sha256 PANEL_SHA256 \
  --output-dir /workspace/herculaneum-lab/reports/fifteenth-experiment/numeric-review/completed-review
```
