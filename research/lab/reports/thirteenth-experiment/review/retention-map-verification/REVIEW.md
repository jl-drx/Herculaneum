# Independent retention-map reconstruction

PASS on the first completed-output check: all eight new score maps and all 200 retained tile logits were verified. All evaluated masks matched exactly, including every pixel of each 384-by-384 canvas. Maximum absolute score difference was **2.384185791015625e-7**, below the **2e-6** tolerance declared and approved before any new scientific array was read.

The checker independently computes the mathematical nonperiodic Hann window and stable sigmoid in NumPy float64, casts weights and probabilities to float32, and reconstructs the original ordered float32 tile accumulation and division. It checks exact tile placement/order, forward/depth metadata, source hashes, all finite score/logit contracts, and unchanged inputs before and after verification. Eight synthetic checks passed before the retained-array run.

Score arrays are **not bitwise identical** to the original Torch-kernel results. This review establishes bounded numerical agreement and exact coverage, not exact transcendental-kernel replay. The earlier eleventh checker used Torch elementwise functions; those were deliberately not imported here. No model, Torch, checkpoint, original CT/input-array or network operation occurred. Metrics and biological interpretation belong to the separate reviews.

The method and checker were frozen in `METHOD.json` (SHA256 `807850a171c26f83753ad4c937eef171f7a30ccd7b62c27a80f422bf563efc94`). The completed receipt is `verification.json` (SHA256 `6dfe19dbeb6991bfecf3fe4d47fba868c061757bd06281545178f1851a62cda9`). Earlier sealed evidence was unchanged.
