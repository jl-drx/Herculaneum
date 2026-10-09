# Pre-acquisition source correction: unmapped flat PPM records

2026-10-08 UTC, before sparse PPM acquisition or numeric decoding. Source review found LayerFromPPM.cpp:301–302 skips pixels without a mapping. Therefore a legitimate surface_volume.ppm may contain exact all-six-zero tuples. The earlier amendment's prohibition on such tuples was unnecessarily stricter than the source's valid-mapping guarantee.

For BOTH map variants, count/report exact all-six-zero tuples as unmapped, without discarding them from coverage reporting. Exclude only those exact tuples from geometric-invariant validation; require at least one nonzero tuple per object. Every nonzero surface_volume.ppm tuple must still equal scheduled(x,y,32,0,0,s) exactly with one consistent sign. Every nonzero result.ppm tuple must still satisfy the unchanged finite/unit-normal test. Never treat merely small/nonunit normals as empty.

This is the only change to PPM_PREDECODE_AMENDMENT.md. Both-or-neither candidate ambiguity remains unresolved, with no tolerance widening. Pin this source-driven clarification in acquisition and numerical plans. All scheduled samples remain unchanged and all unmapped points remain visible in results.
