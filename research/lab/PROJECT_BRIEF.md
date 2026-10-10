# Herculaneum: experiment record for review

Prepared 7 October 2026. The immediate goal is a reproducible, evidence-based workflow that can progress toward the Vesuvius Challenge First Letters prize. We should first recover a small region with independently supported letters, then transcribe and translate it. A full-scroll Grand Prize pipeline also requires substantial automated geometry work and is a later target.

Updated 8 October 2026: the [second experiment](reports/second-experiment/RESULTS.md) reproduces released PHerc1667 annotations and compares three PHerc0800 contexts chosen by a frozen geometry/data rule. It supports the scoring pipeline, with training-overlap and annotation limits, but establishes no target letters or translation. Review [the comparison panels](reports/second-experiment/review.html). The next research gate is a larger separately selected mesh-backed region and fixed boundary/depth diagnostics.

The [third experiment](reports/third-experiment/RESULTS.md) now covers a geometry-selected 1536×1536 context with four fixed depth controls. No coherent target reading is supported. The [June 2026 methods audit](reports/methods-review/PHERC1667_METHODS.md) distinguishes the team's finer-resolution, adapted network and specialist reading from our later 9 µm baseline. The current research gate is native-resolution reference-method validation and credible target supervision; the inventory screen verifies no finer eligible substitute.

## Recommended target and method

Use **PHerc0800, eligible volume `20250521135224`**, provisionally. The official bundled data index lists six segments and surface predictions for this 8.640 µm scan, making it a useful starting candidate. This index is a snapshot: confirm the live inventory and mesh/render compatibility before downloading a region. The eligibility attaches to this exact scan ID.

The maintained tutorial's **[ink_9um](https://huggingface.co/scrollprize/ink_9um)** cross-scroll detector is a candidate baseline at this resolution. Its stated training scrolls are PHerc0139, PHerc1667, PHercParis4, and PHerc0814. PHerc0800 is outside that list, but we must check the actual checkpoints and datasets before claiming a held-out result. Model agreement is a diagnostic, not proof of writing.

Our first scientific experiment will read a bounded coarse CT region, inspect a mesh-backed flattened surface, and compare unprocessed views with conservative ink predictions. Keep overlapping crops and layers in the same data split. Only image-supported letters enter the diplomatic transcription; supplied letters, word divisions, and translation are separate fields. Seek independent papyrological review before interpreting a passage as recovered text.

## Prepared and verified

- Isolated Python 3.12 lab with NumPy, SciPy, Pillow, Matplotlib, TIFF/image codecs, scikit-image, Zarr, fsspec/S3 support, and JupyterLab. The lighter official open-data workflow supports this runtime. The full current villa package requires Python 3.14 and has not been installed.
- A complete, hash-pinned dependency lock and repeatable setup command. Caches stay in writable workspace paths.
- Two checksum-verified **published reference illustrations**, with fixed official source URLs and source commit. They are not raw CT inputs, independent training ground truth, or new discoveries.
- Nine functional checks covering original hashes, preview/provenance generation, TIFF and Zarr exact pixel round trips, SciPy processing, and rejection of bad hashes/crops.
- A notebook executed in the isolated kernel, displaying the reference images and recording runtime versions and processing choices. See [the executed notebook](reports/reference-review.executed.ipynb) and [the exported review](reports/reference-review.html).

The selected Git checkout at `/workspace/Herculaneum` remains unchanged. This review lab and its environment are retained outside it at `/workspace/herculaneum-lab` and `/workspace/.venvs/herculaneum`.

## Reproduce

```bash
bash /workspace/herculaneum-lab/setup.sh
source /workspace/herculaneum-lab/activate.sh
cd /workspace/herculaneum-lab
python -m jupyter nbconvert --to notebook --execute notebooks/reference-review.ipynb \
  --output reference-review.executed.ipynb --output-dir reports \
  --ExecutePreprocessor.timeout=120 --ExecutePreprocessor.kernel_name=python3
python -m jupyter nbconvert --to html reports/reference-review.executed.ipynb \
  --output reference-review.html --output-dir reports
```

Each setup run verifies retained inputs and writes a fresh report directory; source files are never overwritten. `data/manifest.json` records hashes and attribution. `data/provisional-target.json` records the target and the limits of the inventory snapshot. The requirements lock and local helper files are part of the prepared filesystem snapshot.

## Award context and remaining work

The official source at commit `e0bbb8b40a2db58b1d71864f286eb85717e59e64` describes First Letters awards of **$50,000 per eligible scroll for ten legible letters within one 4 cm² area**, with a deadline of **26 June 2027 at 07:59 Europe/London** (25 June, 11:59pm Pacific). This is an initial research direction, not a claim of eligibility or likely success. See [the official rules](https://scrollprize.org/prizes#first-letters-prizes) and [the dated research brief](CHALLENGE_RESEARCH.md) for evidence, other awards, and relevant conditions.

Live AWS scan and public model access now work. The first real-data experiment and its limits are documented in [RESULTS.md](reports/first-experiment/RESULTS.md), with a [standalone visual review](reports/first-experiment/review.html).

The live inventory, bounded coarse CT reads, native small-surface download, mesh mapping and two-checkpoint CPU inference have now been exercised. Native CT reproduced all 93 tested surface-depth values at the hotspot and two neighbors, establishing local rendering consistency. The current scientific gate is determining whether model responses represent ink or other material features; no full-scroll dataset download is needed for this pilot.

Remote CT access, released surface/mesh inspection and the focused CPU inference adapter are validated in the current instance. Full unrolling, ink accuracy and new transcription/translation remain **unvalidated**. Any novel text findings should stay private for review under the prize's publication rules. No prize submission or external message has been sent.
