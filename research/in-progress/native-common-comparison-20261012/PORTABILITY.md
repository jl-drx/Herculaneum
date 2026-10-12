# Download and verify the completed comparison

Use the existing private `jl-drx/Herculaneum` repository and branch
`codex/plinius-pherc343-reproduction`. The result is an ordinary Git LFS bundle;
no Actions job, GPU, RunPod/R2 secret or model inference is needed for review.
Keep downloads private. A fresh checkout can avoid unrelated large objects:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch codex/plinius-pherc343-reproduction git@github.com:jl-drx/Herculaneum.git
cd Herculaneum
git lfs pull --include='research/workspace/bundles/native-common-comparison-review-20261012.tar.gz' --exclude=''
```

For an existing checkout, fetch/pull the branch and run only the final command.
The review archive is 79,494,713 bytes (about 75.8 MiB), with SHA256
`4ae4082fad47e336c6b408fb88fab411cd81002202b480ca2f616611d8a399ac`.
The [packaging receipt](BUNDLE.json) pins its contents and implementation.

Use an isolated Python environment. The retained run used Python 3.14.7,
NumPy 2.4.4 and opencv-python-headless 5.0.0.93. Only the latter two libraries
and the standard library are needed for saved-result verification; torch and
CUDA are unnecessary. Install those exact dependencies, then:

```bash
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-common-comparison-review-20261012.tar.gz \
  --expected-sha256 4ae4082fad47e336c6b408fb88fab411cd81002202b480ca2f616611d8a399ac \
  --output /tmp/plinius-common-review
python /tmp/plinius-common-review/run/evaluation-implementation/verify_results.py \
  --inputs /tmp/plinius-common-review/inputs \
  --run /tmp/plinius-common-review/run \
  --output /tmp/plinius-common-review-verification.json
```

Use fresh output paths. Extraction verifies the archive SHA, safe member paths,
all 3,600 manifest entries and SHA256SUMS. The second command verifies saved
outputs and recomputes 238 metric records without inference. It checks the
producer's exact replay receipts, not a new execution of its tile accumulator.
All four fixed-scale review panels and all 24 core records are included.

Source CT and model weights are deliberately separate. For input construction
or model replay, also use
`research/workspace/bundles/native-common-support-inputs-20261012.tar.gz`
(SHA256 `b6632e6798d808af7252cc372f9e567d3338adba947e64e54755c895af236968`)
and the pinned checkpoint/runtime requirements in GO.json. The review bundle's
input receipts alone cannot establish that transferred labels are physically
registered or independent of upstream model training.
