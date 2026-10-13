# Soft supervision: averaging versus smoothing and centroid controls

Completed 11 October 2026. All64 frozen cases were calculated and independently
verified. No model was trained. This is a bounded development control on the
previously inspected four w040 supports, not an independent learning result.

**Decision:** retain ordinary continuous teacher scores as the default for a
future paired-learning baseline. This packet does not justify a new local
uncertainty architecture. The next learning study should first establish a
native-CT baseline against declared native-frame annotations on separate
physical supports. A gain against the training teacher alone would not answer
whether ink detection improved.

## What the controlled comparison found

A uses the fixed-fit soft teacher; B averages the local CT alternatives; C uses
one global kernel with equal mass for each of the four local kernels; D samples
once at the mean projected coordinate of the local alternatives. All four use
the same raw score scale and exactly the same retained pixels.

Primary j14 / mean16-bilinear results below are mean absolute differences on the
common positive union, expressed in original 0–255 source-score units:

| Support | Common pixels | Active union | A–B | B–C | B–D |
|---|---:|---:|---:|---:|---:|
|4|1024|0|undefined|undefined|undefined|
|5|1024|139|2.0093|1.0041|0.1737|
|6|1024|451|0.5055|1.2502|0.2710|
|7|992|155|1.1372|0.7977|0.0514|

Local averaging is very close to sampling at the local mean coordinate: B–D is
below one source quantization interval in all48 cases with a nonempty positive
union, including both sampling operators and all eight fixed planes. The maximum
is0.329 source units. On these smooth teacher fields, most of B's change from A
can therefore be represented by a small change of sampling location. This is
not a physical depth estimate or advice to recenter the CT input window.

B and the global control are numerically different. Their difference mixes
changes of kernel location, spread and local association. Support5 straddles
one source unit across controls (0.9825–1.0242); support6 is1.2446–1.2542 and
support7 is0.7718–0.8169. The one-unit comparison is a descriptive scale chosen
for this already-inspected packet, not a statistical threshold or evidence that
smaller changes cannot affect training. No predictive advantage was tested.

The global kernel goes outside the previously retained teacher crop on32 pixels
of support7. All four operators exclude the same32 pixels, leaving992, in every
depth/operator case. Geometry and native CT remain valid there. No missing
source value was filled with zero, and no new teacher pixel was read to repair
the comparison. The fourth support remains in every output despite lacking a
positive union. Denominators differ from the previous local-only sensitivity
study and must not be silently interchanged.

## Why training four patches would not answer the research question

There is no earlier equivalent native-CT/fine-teacher soft-learning comparison.
However, these four small supports do not supply adequate independent training
and evaluation: neighboring supports share possible teacher input context, all
belong to the same inspected winding, and there is no separate ink reference in
this packet. A random pixel split or scoring each arm against its own target
would produce a misleading learning result. A small full28-plane native learner
is technically feasible; scarce qualified physical coverage is the issue here.

Unweighted BCE against the mean target is exactly the mean of BCE against its
constituent targets. It is averaged supervision, not a new latent-geometry model.
Hard labels at score48 and raw soft scores/255 also have different calibration
and class-balance semantics. A hard-versus-soft result cannot be attributed
solely to registration robustness.

## Verification and preserved limitations

Seven synthetic groups preceded the real calculation, including constant/ramp
fields, weights, crop/support failures and the BCE identity. An independent
implementation regenerated all64 source cases, checked8,000 scalar summaries,
1,024 array comparisons and60 exact rational moment entries. Saved scalar
summaries reproduce exactly; a different bilinear evaluation order changes
individual transfers by at most4.27e-14 source units.

The reviewer preserved a nonprimary point-sampling boundary at support6/j11:
seven candidate values are exactly48.0 with nested arithmetic versus
47.99999999999999 with production arithmetic. It changes no common mask, active
union, arm threshold count or descriptive decision. See review receipts for the
exact scope and distinction between full geometry replay and compact replay.

All previous strict correspondence failures remain. Candidate sets are finite
descriptive sets, not posterior confidence regions. Teacher scores are not
calibrated probabilities or authenticated ink. No new CT/teacher acquisition,
model forward, optimizer update, GPU resource, Actions run or spare-VM task
occurred for this operator study. The separate native annotation intake is
documented in its own packet.
