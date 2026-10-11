# Independent check of study52 completed-map reporting

Result: **PASS**, 11 October 2026, 08:48 BST (07:48 UTC). This is a mathematical and image-export check, not an ink assessment.

The frozen reporting plan has SHA-256 `0706ea76bdec21fc925f0a5ee6615c8a872050aae593275e06c7566b0982d143`. Its source pins, all 48 saved float32 map inputs, generator pin, and all 71 output-manifest entries verified. The checker read the generator to understand the reporting interface and panel layout, but did not import or execute it. No inference, network request, source edit, or paid compute occurred.

Checks performed:

- Reconstructed the full 4-model × 2-padding × 2-order × 3-depth factorial map set and verified unique coverage.
- Derived comparisons by enumerating pairs with exactly one different experimental factor. This independently recovered all 168 distinct required pairs: 24 padding, 24 order, 72 cross-model, and 48 depth pairs. Both the plan and output contain exactly this set.
- Reconstructed the common mask from the intersection of the four inherited family/phase masks. It contains 786,240 known and 192 unknown pixels in the 512 × 1536 strip. Each pair uses that mask and all three fixed 512 × 512 cells.
- Recomputed all 672 pair/scope scores using long-double raw-moment covariance, rather than the generator's centred float64 reduction. Maximum Pearson difference was 3.33e−16; maximum MAD difference was 1.42e−14 uint8-equivalent levels.
- Checked every image pixel for all three depth rows in all 64 exact panels: 192 map-region views, with one source pixel per displayed pixel. The declared fixed float32 probability × 255, nearest-even rounding mapping is exact; all unsupported pixels use RGB (139, 90, 43). No local contrast fit or intensity rescaling is present in these exported source-image rectangles.
- Checked all 48 tiles of the four overview sheets against uniform 4× BOX spatial reduction of those same fixed-display images. These are overview images, not the source-resolution evidence.
- Verified the exact-panel and contact-sheet dimensions, the three cell coordinate contracts, and the absence of missing or unmanifested reporting files.

No discrepancy was found. All quantitative scores and source-pixel panels are suitable for the separately required scientific review under the frozen protocol. This check does not reconstruct the underlying CT, prove support geometry, measure ink accuracy, authenticate letters, or establish that any response passes the scientific candidate criterion. In particular, correctness of model-output export is independent of the biological or textual validity of those outputs.

`verify.py` is the independent replay; `VERIFICATION.json` contains all independently recomputed values and checks; `SUMMARY.json` gives the compact outcome. Use the existing NumPy/Pillow environment and one numerical thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/herculaneum/bin/python /workspace/scratch/phase52-completion-independent-20261011/verify.py
```

The replay reads only the completed evidence; it writes its own verification results in this directory. The paths are intentionally explicit and do not trigger data retrieval.
