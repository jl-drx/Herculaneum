# Frag6 sparse PPM Stage B

This implementation performs **one bounded multipart request** against the existing `result.ppm`, followed by a qualified serialization and sparse geometry check. It does not read CT pixels, TIFF pixels, labels, optical images or model outputs, and it cannot advance to Stage C.

The pinned successful Stage A contract and its independent runtime review establish a canvas of **8,853 rows × 6,205 columns**, a 73-byte PPM header, 48-byte records, total object length 2,636,777,593 bytes, and the existing quoted ETag. Every point and byte span is derived before acquisition and written into the execution plan:

- Lattice coordinates are `y = ((i+1)*(8853−1))//10` and `x = ((j+1)*(6205−1))//10`, for `i,j = 0…8`.
- All 81 centers contribute their 3 × 3 neighborhoods. Three contiguous records per neighborhood row produce **243 distinct 144-byte ranges**, **729 records** and **34,992 useful bytes**.
- The nine potential native-check sites are exactly the centers with `i,j ∈ {1,4,7}`. Invalid sites remain recorded and are never replaced. Having fewer than six complete neighborhoods blocks the proposed native comparison; having six or more does not authorize it.

The sole HTTP attempt reserves **222,640 response-body bytes before connection**, including room for an excess-byte check. A declared body length equal to or above that ceiling is rejected before reading. The request includes the exact retained `If-Match`, all exact ranges and identity encoding. HTTP 206, the pinned strong ETag and one bounded multipart boundary are required. Every part must match one requested range and the object total; coalesced, missing, duplicate, overlapping or unrequested parts stop. Optional part Content-Type must be `application/octet-stream`. Header, framing and closing-delimiter checks prohibit encoded parts, extra data and full-object fallback. There are no retries or source substitutions.

The multipart parser is adapted from the source-pinned phase 17 reader. It uses explicit lengths, so delimiter-like binary payload bytes are safe. Response order may differ from request order; reconstruction always follows the frozen point/row schedule. Raw MIME and assembled records are retained with hashes. Failure reservations and returned partial bytes remain recorded. These figures describe response-body accounting, not total wire traffic or recovery of bytes trapped inside a failed HTTP-library call.

For numerical interpretation, the code tests **exactly little-endian and big-endian float64** over all 729 records. Exact six-zero records are unmapped sentinels. Every other record must have six finite values and a normal norm within `1e-5` of one; there must be mapped data. Exactly one hypothesis must pass. Neither or both passing produces STOP with both diagnostics retained, without geometry or another sample request. A unique invariant-consistent byte order is conditional evidence, not authenticated historical producer provenance.

For that unique interpretation, the report retains every scheduled neighborhood and its mapping status, primary-volume bounds checks, central raster derivatives, local Jacobian, area scale and normal/tangent comparisons. Degenerate or nonfinite derived quantities are null rather than silently clipped. All-six-zero centers or required neighbors invalidate the relevant quantities. The raster cross-product orientation is relative to x-right/y-down; it does not identify an anatomical side. Primary native bounds are a consistency check and cannot distinguish the same-shaped energy scans. Source volume binding, render offsets, anatomical normal and physical ink depth remain null.

`check_fixtures.py` is completely offline, using synthetic planar geometry, adversarial MIME bodies and injected response objects with socket tripwires. It covers both byte orders, ambiguous palindromic values, all-zero/NaN/nonunit cases, sentinels without replacement, exact schedule assembly, partial failures, pre-body rejection, resource caps and absolute deadlines. The final execution plan pins the passing receipt and all nine implementation sources.

The worker uses a 60-second absolute request timer, 20-second connection timeout and 1 GiB address-space limit. Measured RSS is limited to 256 MiB, run artifacts to 32 MiB and external wall time to 1,800 seconds. The separate controller polls every 0.1 seconds and kills the worker process group on a breach. RSS/disk checks are acceptance limits, not allocation quotas. Every execution directory must be fresh.

**No live request is permitted before independent review and root's matching written GO.** Both controller and worker require these fields:

```json
{"status":"GO","stage":"B_sparse_ppm","plan_sha256":"ACTUAL_PLAN_SHA256","allow_live_requests":true,"allow_conditional_ppm_geometry":true}
```

Root may add its scope and evidence pins. Execution then uses:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python controller.py \
  --plan /ABSOLUTE/frag6-stage-b/execution-plan.json \
  --plan-sha256 ACTUAL_PLAN_SHA256 --go /ABSOLUTE/ROOT_GO.json \
  --output-dir /ABSOLUTE/frag6-stage-b/FRESH_RUN_DIRECTORY
```

A passing result establishes only conditional sparse PPM consistency. It does not bind the surface TIFFs to a native scan, recover their depth recipe, validate optical labels, demonstrate model accuracy or make Frag6 an independent holdout. The PHerc51 parent reservation and the £10 total first GPU pilot cap remain unchanged. This implementation and its fixtures used no live request or paid resource.
