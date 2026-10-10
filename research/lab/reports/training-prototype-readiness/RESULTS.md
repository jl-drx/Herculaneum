# Synthetic CPU training mechanics passed

One fixed-seed artificial input of shape [1,1,17,128,128] passed forward, masked BCE backward and one SGD update using the unchanged strict seed42 factory/checkpoint (34,546,498 parameters). The root executed the frozen plan once on 8 October 2026; exit status was 0.

The check took **3.123 seconds** including setup and measured **799,980 KiB peak RSS (781.23 MiB)**. All 264 gradient tensors present were finite. A retained parameter comparison confirmed 16 changed elements in `depth_fusion.features.1.weight`. The participation mask contained 1,920 synthetic positives and 4,480 negatives; all 9,984 unknown pixels had exactly zero gradient with respect to their output logits. Positive and negative anchors were present in both artificial regions.

The step optimized plain masked BCE only. An illustrative equal-region/equal-class anchor loss was calculated but not used for the optimizer step. No scan data, new teacher targets, model predictions or updated weights were saved. The source checkpoint remained unchanged and the modified in-memory model was discarded. There were no network requests, new dependencies or paid resources.

This establishes CPU training mechanics and a small-workload memory observation. It does not validate a new loss, real-data training behavior, Adam/AMP memory, CUDA compatibility, full training throughput or model quality. The optimizer settings are smoke-check settings, not a chosen research recipe. No research model was trained.

Evidence: [frozen plan](synthetic-plan.json), [checker](check_synthetic_step.py), [package freeze](package-freeze.json), [execution log](execution.log), [machine-readable result](synthetic-result.json). The separate [training contract](TRAINING_CONTRACT_DRAFT.md) remains a draft: data, teacher, split, loss weights, optimizer, update budget and success/stop criteria must be frozen before adaptation.
