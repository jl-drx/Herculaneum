# Original native stream qualified; added population still blocked

The result and cumulative checkpoint are now **published in private GitHub**.
The recovered push and a fresh selective archive download passed on 12 October
at 20:53 UTC; its length and SHA256 match the locally replayed archive. See
[PERSISTENCE.json](PERSISTENCE.json) and [REMOTE_RESTORE.json](REMOTE_RESTORE.json).
The [later source retry](../native-coverage-retry-20261012-2053/README.md) retained
72 more teacher ranges before HTTP 503 returned; full intake is still incomplete.

12 October 2026. The resumed acquisition completed **all 781 PHerc0139 CT
objects**, and **all 876 fixed original-stream candidates** pass the unchanged
support rule. A separate full-raster/integral implementation reproduces every
candidate, class count and input-support decision. No candidate was replaced
or selected for positive labels. The earlier partial result remains intact.

This removes the original-stream input blocker. It does not admit the full
matched coverage experiment: most PHerc0814 CT and teacher dependencies still
await transport, and no added-population supervision has been qualified.

## Complete original result

| Winding | Qualified patches | Supervised occurrences | Positive occurrences | Patches with no positive annotations |
|---|---:|---:|---:|---:|
| w035 | 191 | 1,909,065 | 594,034 | 59 |
| w039 | 104 | 1,033,015 | 330,293 | 35 |
| w040 | 304 | 2,889,262 | 1,292,579 | 64 |
| w041 | 91 | 977,657 | 413,367 | 22 |
| w044 | 186 | 2,125,149 | 788,468 | 28 |
| Total | **876** | **8,934,148** | **3,418,741** | **208** |

Occurrence totals count overlapping patches. Deduplicating within each stored
winding's UV grid gives **5,022,663 supervised positions**, including **1,914,193
positive annotation values**. The frozen patch union covers all 5,023,011 source
supervision-mask positions; centre-plane CT validity removes 348 distinct UV
positions, or 720 occurrences when overlapping patches are counted. These sums
across windings are not deduplicated physical sheet area or independent subjects.
Native/fine aliases, including w044/w028, retain their conservative ancestry.

All 208 zero-positive patches and the one zero-negative patch remain in the
population. A zero-positive annotation patch is not an independently verified
physical blank control. Outside-mask positive values remain ignored. Source
annotations are used under their declared masks; these checks do not authenticate
ink, certify surface placement or validate precise transferred fine-scan labels.

[ORIGINAL_VERIFICATION.json](ORIGINAL_VERIFICATION.json) verifies all 781 chunks
and all 876 rows. It uses the unchanged independent verifier from the partial
intake. [ORIGINAL_EXPOSURE.json](ORIGINAL_EXPOSURE.json) additionally checks
within-grid overlap and counts unique support separately from repeated exposure.
Its weighted raster sums reproduce the independently checked candidate totals.

## Consequences for the proposed control

Uniform sampling over all 876 patches would give w040 **34.70%** of original
draws and w041 **10.39%**. Sampling windings equally would give each **20%**.
The positive fraction of expected supervised occurrences is **38.27%** versus
**37.44%**, respectively. These are descriptive, unaugmented exposure calculations,
not two training arms, a selected sampler, or expected finite-batch gradients.

The practical point is to freeze winding quotas explicitly, then hold the
original patch IDs, augmentation seeds and stream loss coefficient identical
between the restricted and broader added-stream arms. Equal numbers of patches
alone do not guarantee equal source or class exposure. Preserve the already
qualified separate-stream loss normalization. Choose the full schedule after
the added population's class/support accounting is available, before training
or model-performance feedback. These checks demonstrate input readiness; they
do not measure learning power or justify a shorter optimization budget.

## Transport findings and current state

The resumed original intake retrieved 357 remaining chunks in about 39 seconds,
with no recorded retry or missing source. Added intake retained 566 further
objects before the same outbound proxy-tunnel HTTP 503 error returned.

The bounded persistent-connection diagnostic then retained seven more teacher
ranges and failed on the eighth request. The client observed the **same live
connection and socket** on that failure. Thus simply reusing the local HTTP
connection did not avoid the error. It does not establish the remote service's
internal cause. The remaining 56 diagnostic requests were not attempted, and
no failed source was called missing or supplied as zeros. Inherited proxy routing,
CA trust, TLS verification, URLs and scientific request identities were unchanged.

At this qualification's archived checkpoint the added cache has **1,522 complete objects**: 1,403 CT chunks and 119
teacher ranges, totalling 657,066,767 bytes. **16,048 requests remain**, with
5,833,369,469 nominal body bytes. The fixed first segment's CT is complete, but
its teacher union is not. It is not a substitute one-segment training experiment.

The qualification used a retained snapshot of 1,515 added receipts; the seven
later pooled successes are source progress only. [RECOVERY_STATUS.json](RECOVERY_STATUS.json)
distinguishes these snapshots and records the completed original acquisition.
Earlier `ACQUISITION_FAILURE.json` files remain historical evidence; the newer
original `ACQUISITION_COMPLETE.json` accounts for all its frozen requests.

## Next decision

Resume the **unchanged remaining added-source requests only after a meaningful
transport recovery**. Do not repeat the failed connection-reuse hypothesis,
shrink to the first segment, drop unavailable sources or reopen completed native
fits/evaluation. Once the full teacher/native population is qualified, quantify
the surviving coverage and class contrast, freeze the paired workload and obtain
approval for a concrete GPU job. Independent detail/negative endpoints and
geometry-reference adjudication remain separate dependencies.

No new model forward, optimizer update, reserved evaluation payload, GPU rental,
Actions job, external message or spare-VM assignment occurred. The £10 total
first-GPU-pilot cap remains in force. See [PORTABILITY.md](PORTABILITY.md) for
the cumulative private checkpoint and numerical replay.
