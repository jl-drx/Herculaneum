# Plinius: Paris1-to-Paris2 calibration transfer

The fixed CPU comparison is complete and independently verified. **None of the three models passed the preregistered requirement of at least 95% specificity in all four Paris2 regions using its Paris1-only cutoff.** The unchanged baseline remains the comparator; no candidate is promoted.

| Fixed model | Mean ROC AUC | Mean AP | Sensitivity, old → Paris1 cutoff | Specificity at Paris1 cutoff | Regions passing ≥95% specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed42 baseline | 0.6576 | 0.3843 | 2.60% → 4.55% | 97.46% | 3/4 |
| A preserve | 0.6617 | 0.3804 | 2.00% → 3.92% | 97.43% | 3/4 |
| B preserve | 0.6630 | 0.3787 | 1.76% → 3.80% | 97.40% | 3/4 |

These are equal-context means within each fragment, then an equal-fragment mean across the two Paris2 fragments. Mean supplied-positive prevalence is 21.02%. A mean specificity above 95% does not satisfy the per-region requirement. All three fail in Frag1 context1; the other three regions pass. The old fourteenth cutoffs pass specificity in all four regions, with still lower sensitivity. New cutoffs were fitted only from the two fixed Frag3/Paris1 negative supports and locked for all three models before any Paris2 forward.

Baseline cutoffs were 0.7558947206 previously and 0.7266893387 from Paris1; A changed from 0.7450202703 to 0.7076279521; B from 0.7422624826 to 0.7000159621. Exact values, confusions, all metrics and per-region results are in [the tables](reporting/completed-results/tables.md) and [metrics JSON](inference/analysis/metrics.json).

## What this tells us

Baseline ROC AUC is 0.4461, 0.9037, 0.5013 and 0.7795 across the four Paris2 contexts. The large within-fragment variation matters more than the small candidate mean-AUC gains (+0.0040 and +0.0053). Both candidates have lower mean average precision and sensitivity than baseline. A cutoff change leaves score ranking unchanged.

In the failing Frag1 context1, baseline specificity is 93.07% and sensitivity 0.31% at its Paris1 cutoff: 61 true positives, 4,796 false positives, 64,363 true negatives and 19,339 false negatives. A and B identify none of the 19,400 supplied-positive pixels there, while producing 4,886 and 4,924 false positives respectively. The outcomes warrant a focused source/alignment/depth audit before more continuation training. They do not yet identify whether the cause is representation, surface/annotation registration, acquisition-domain shift or a combination.

## Execution and independent checks

The one model run took 58.634 seconds: 150 calibration and 300 evaluation tile forwards, six final-role worker constructions and ten strict state loads (six initializers plus four candidate replacements). There were zero backward calls, updates, new checkpoints or model retries. All three cutoff locks preceded evaluation. Independent saved-output reconstruction checked normalized-input hashes, all 18 score maps/450 captures, blending, coverage, cutoffs, confusions, ranking and macro metrics, and receipt ordering. It used no Torch, checkpoints or model execution.

All six float32[17,384,384] inputs were independently reconstructed bit-for-bit from the retained CT ranges; all 24 label/support arrays remain literal fifteenth artifacts. The source TIFFs use mixed byte orders, and Frag1 strip offsets vary between 260 and 262 bytes. Each of 153 files was checked separately.

The production pixel download used 1,530 GETs and 848,908,566 encoded response-body bytes in 452.079 seconds. Including the original/revised probes and the 450 new header requests, this phase retained 861,973,578 response-body bytes over 2,001 GETs. These are body counts, not total wire traffic. The narrow-row production body total is about 83% below the earlier 4,955,143,680-byte full-width-band payload projection; full scans were not downloaded. No automatic network retries occurred.

Corrected pooling took 22.561 seconds with 138,268,672 bytes peak RSS. The whole M/P/C window, including planning and the correction, was 931.077 seconds under its 1,500-second cap. Producer and pooling outputs total 1,372,187,238 bytes under 2 GiB.

## Preserved technical stops

The first 8-byte header probe stopped on its little-endian-only assumption; a separately reviewed reader added explicit II/MM handling before further acquisition. The original pooling attempt stopped before numeric CT decoding because JSON tag keys were strings while reconstructed parser keys were integers. A separate V2 normalizes only those keys; all tag values and pooling arithmetic stayed exact. The first saved-output checker stopped before scientific array reads because it required absolute worker output arguments; the controller had retained valid relative arguments. Its separate correction resolves only that field against a hash-pinned, explicitly retrospective record of the actual root execution directory. Original sources, plans, failures and every scientific output remain retained.

## Limits and next step

Paris2 supplies one evaluation parent papyrus, not two independent scroll replications. Fixed support gates used labels; complete upstream model exposure and historical registration remain unresolved. Byte correctness does not authenticate the annotation or written surface. There are no pixel-independent confidence intervals, significance claims, first-in-class claim, transcription or translation.

The next recommended CPU task is [the source/alignment/depth diagnostic](NEXT_STUDY_DRAFT.md), using the fixed six contexts and retaining this result. Any subsequent method choice is informed by these Paris2 results and needs fresh evaluation material for a confirmatory claim. GPU spending remains **GBP 0**, with the **GBP 10 total first-pilot cap** unchanged. No new paid resource, upload, public redistribution, prize submission, new terms acceptance or outreach occurred. Retained data-use restrictions and EduceLab attribution continue to apply.

Exact completion evidence is bound in [COMPLETION_CONTRACT.json](COMPLETION_CONTRACT.json). Local retention and the environment startup draft are separate from backup, publication or a fresh-task restoration test.
