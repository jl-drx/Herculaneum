# Phase 20 independent pre-execution review

The numerical contract and synthetic fixture suite are prepared before scientific output reads. This independent role verifies saved outputs with NumPy only. It does not execute, import or decode the producer model; technical source review is a separate role.

The key mechanism change from phase 18 is explicit: the 5,296-parameter adapter returns 32 channels at the original fusion site and the frozen U-Net maps those altered features to logits. Therefore, no old teacher-plus-output-residual assertion is valid. The verifier checks saved diagnostic differences as student minus teacher, initial original and zero-adapter outputs bitwise against all cached teachers, full parameter traces against clipping and AdamW recurrences, and feature tensor fingerprints from separately reconstructed inputs. Fused cache and feature residual summaries remain execution receipts, not independent reconstruction of the frozen network.

Before review of any actual results, the producer must freeze an exact plan and code manifest, bind the root protocol/configuration, retain all 32 initial original/zero-adapter output batches, both historical teacher-cache identities, all pre/post/fixed logits and manual output gradients, all 5,296 parameters/gradient/moment entries at each of 32 updates, and final-only development logits on all six fixed contexts. Source-dependent fields and actual module hook counts must be adapted to the final schema before root execution GO.

Readiness requires the final schema preflight, source pins and synthetic fixture receipts. A preliminary mathematical fixture PASS is not a scientific execution GO. The independent training PASS is required before final development prediction. Published report arithmetic is checked after both stages. Every unfavorable result and failed checker attempt remains retained.

All sources before phase 20 and the tracked application are immutable. No network, new data, model calls, checkpoint decoding, training replay, GPU resources, external publication or startup configuration action is part of this role.
