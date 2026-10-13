# Study52 design and feasibility — no shifted inference yet

The bounded comparison is defined and its retained inputs pass feasibility.
All **16 centre-depth baseline maps reconstruct exactly** on the proposed strip
from **1,548 saved batches**, with **zero neural-model forwards**. No new GPU job,
CT download, training, shifted prediction or scientific ink result was produced.

The [protocol](PROTOCOL.md) fixes segment20251028222030, x512:2048/y1024:1536,
nominal depths−15/0/+15, all four models, both padding phases and both depth
orders. It retains 786,240 of786,432 pixels (99.9756%) on the inherited common
support; 192 remain unknown. All three cells were already inspected in study51.
The positive outcome is a candidate for independent review, not a reading or
accuracy score. See the [design assessment](DESIGN_REVIEW.md).

## Checks completed

- All958 inherited input/runtime/model pins matched before and after preparation.
- Native and Hecate depth windows and interpolation corners stay within layers
  9–56 of the retained66-plane render. No re-render or extra source is required.
- Exact full-grid contributors and original active cohorts were frozen. Analytic
  and OpenCV interpolation-support bounds agree; family masks reproduce the
  inherited strip masks. The original Hecate grid and denominators are retained.
- Six Hecate volumes were prepared with the unchanged author script: three
  depths × two phases. Both d0 preparations match every retained byte. All
  retained input patches remain nonempty at all three depths; no whole depth
  profile within these patches is zero. Cohorts therefore do not change with
  depth because of an empty-input skip.
- Saved-logit reconstruction matches all16 baseline aligned and display maps
  exactly:12,582,912 values per output type, including unscored strip pixels.
  Villa numerator/denominator arrays match too. The replay loaded no model
  checkpoint and made no neural-model forwards.

The first replay-helper attempt passed the12 Villa maps, then rejected a Hecate
input identity: NumPy's boolean reduction had produced bool support where the
original Torch uint8 reduction produced uint8. Using the original typed operation
restored exact input identity and all16 map replays passed. The failed helper
and [failure record](REPLAY_ATTEMPT_1.json) are preserved. This was an error in
the new checker; prior model inputs, scores and scientific evidence are unchanged.

## Frozen workload and decision

| Family / phase | Retained batches per model/order/depth | Retained tiles |
| --- | ---: | ---: |
| Villa, plain |57|228|
| Villa, padded |65|260|
| Hecate, plain |204|765|
| Hecate, padded |204|765|

Across the two new depths the production plan is **32 maps /3,096 batches**.
Scaling the measured study51 per-map times by cohort count gives **3h09m**.
Allow additional qualification, fixed replay/I/O and backup overhead; this is a
forecast, not observed shifted-run performance. CPU spend is unrestricted;
GPU spend remains **£0**, with the existing £10 total pilot cap unchanged.

The design and input preparation are complete. The next implementation step is
the [bounded worker qualification](EXECUTION_QUALIFICATION_DESIGN.md), followed
by a frozen execution GO and private checkpoint setup. Those gates have **not**
run; input readiness is not a launch receipt. No worker is running for study52.
The earlier `FEASIBILITY.json` lists the then-pending baseline replay gate; the
later [BASELINE_REPLAY.json](BASELINE_REPLAY.json) supplies its passing result.

At review, keep every depth/model/order/phase visible with fixed contrast. A
repeatable structured candidate at a nonzero offset earns separate image and
geometry review. Otherwise stop this three-position screen, without expanding
the search on brightness alone. Absence of a candidate does not establish
absence of ink or rule out intermediate centres or other physical regions.

The source code, design and receipts are a compact private Git packet. Prepared
arrays are local, reproducible derivatives of pinned study51 inputs; they are
not included in the compact packet. The retained study51 inputs/logits require
their separate private checkpoint chain and prior runtime/model prerequisites.
The cumulative formal research seal remains at study50. No claim of a complete
standalone study52 inference bundle or second-device validation is made.
