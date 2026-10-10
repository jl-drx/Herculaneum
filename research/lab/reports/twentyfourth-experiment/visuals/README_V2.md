# Final phase24 plotting contract

Use `plot_depth_reversal_v2.py`. The earlier preparation source and freeze remain unchanged as historical evidence. V2 binds root protocol `f9cc08b99144f55e5842b94e08cb92a8524badafd58d5bf7483254953641c3ee` and requires the producer's explicit per-case fields `adapter_read_direction="forward"` and `model_depth_permutation=[16,15,...,0]`, plus the correct model/case identity. The old generic `direction="reverse"` shorthand does not establish the actual intervention and is not used by this guard.

The implementation agent confirmed that normalization receives the original forward patch and that the resulting normalized tensor is then reversed. V2 figure captions call this a normalized depth-order permutation. It is a counterfactual input ordering, not a corrected physical surface or a renormalized reversed scan.

All remaining plot requirements in the preparation README persist: four figures, both models and all six whole contexts, supplied labels and original primary masks, common fixed scales, gray unsupported pixels, no thresholds/crops/contrast tuning, exact file/raw-array pins, full evaluated coverage, 60 source-array receipts and a 32 MiB whole-stage cap. No source arrays or outputs have been read during preparation. Root must release the verified reverse-result artifacts before execution.

Use the earlier documented command with executable name `plot_depth_reversal_v2.py`. Both runtime result hashes must be supplied from the verified actual results; no placeholder or automatic file discovery is allowed. Root owns image review and scientific interpretation.
