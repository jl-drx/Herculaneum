# Proposed phase45: native-only budget continuation

Run one **native-only budget continuation check** before a teacher or another resolution comparison. Use the first native fold, preserve its inputs, objective, 192-pixel context and seed, and replay to 2,048 updates to verify exact checkpoint equality while recovering optimizer state. Continue that same trajectory to 4,096 and 8,192, saving optimizer state and training-fit checkpoints. Do not change loss/calibration/architecture at the same time. Stop immediately for integrity/nonfinite failure; otherwise use the 8,192 endpoint. Training AP ≥ 0.80 on both training papyri would justify completing native qualification across all parents. Failure with < 0.01 AP improvement on both in the last interval would redirect effort toward representation/context or objective design. Continued material improvement below the fit gate remains budget-inconclusive, not an information-loss result. Do not use another held-out score to choose the endpoint.

This recommendation isolates budget while preserving the documented objective caveat. It is a proposal, not a new training GO or a started run. In parallel, the coordinate-linked CT reference remains necessary for the main research objective; the Paris4 request is still unsent. The stopped Paris4 panel and historical full-depth contract are unchanged.

GPU spend for this study: **£0**. No GPU job, R2 write, maintainer message, transcript or translation. The £10 total first-GPU-pilot cap remains, and GPU work stays deferred.


At the 2,048-step replay boundary, compare every model tensor bit-for-bit with the retained checkpoint; serializer-container hashes can differ despite identical tensors. Preserve AdamW moments, step counters and random-generator state for the continuation. The first native fold trains PHerc.250 and PHerc.500P2 and reserves PHerc.248, whose phase44 results are already development-exposed. No new held-out selection is proposed.

Reference checkpoint SHA-256: `1f86124061a4dc3a115535124cb38ad2d4ba79a9d608eebb083c7c1221789c1c`. Prior training GO SHA-256: `de35f74f1b8baa9d929fcb82e3e09dd684609d3e0fce2b9083cbda8d269b0496`. These anchors specify the predecessor, not a new execution authorization.
