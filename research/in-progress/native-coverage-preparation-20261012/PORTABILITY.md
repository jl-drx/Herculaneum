# Private selective download and offline review

Use the existing private `jl-drx/Herculaneum` repository, branch
`codex/plinius-pherc343-reproduction`. The preparation bundle is **25,267,149
bytes (24.1 MiB)**, SHA256
`2262563b70de9c1bafdf215a97a4abbd977011a3069d10b696d8e023a4497218`.
[BUNDLE.json](BUNDLE.json) pins all 308 payload files and the packaging script.
This is a separate supplement; historical bundles and their manifests are unchanged.

With access to the private repository, avoid downloading unrelated LFS objects:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch codex/plinius-pherc343-reproduction https://github.com/jl-drx/Herculaneum.git
cd Herculaneum
git lfs pull --include='research/workspace/bundles/native-coverage-preparation-20261012.tar.gz' --exclude=''
```

For an existing checkout, update the branch and run only the selective LFS pull.
No Actions job, RunPod credentials or model inference is needed. Keep the archive
and extracted data private.

Extract into a new directory, checking the archive, every manifest entry and
the checksum list with the existing standard-library verifier:

```bash
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-coverage-preparation-20261012.tar.gz \
  --expected-sha256 2262563b70de9c1bafdf215a97a4abbd977011a3069d10b696d8e023a4497218 \
  --output /tmp/plinius-coverage-review
```

The archive expands to approximately 50 MiB. It includes the actual bounded CT
and teacher byte ranges, TIFF headers, XYZ meshes, source plans/receipts, seven
decoded source fixtures, the four native training fixtures, both support
reservations, preserved first attempts, code and reports. It also includes the
single retained PHerc0814 geometry CT context and every new review figure.

In an isolated environment with NumPy and tifffile, recompute the reservation
with the separate arithmetic implementation:

```bash
python /tmp/plinius-coverage-review/repo/research/in-progress/native-coverage-preparation-20261012/verify_support.py \
  --work /tmp/plinius-coverage-review/native \
  --output /tmp/plinius-coverage-review-check.json
```

The retained check used Python 3.14.7, NumPy 2.4.4 and tifffile 2026.7.14.
It enumerates all 11,887 broader candidates, verifies the 2,886 restricted
members, and recomputes both physical-context proxy counts. It needs no GPU,
torch, network, source predictions or training. OpenCV is unnecessary for this
review; original source assembly used opencv-python-headless 5.0.0.93.
[LOCAL_RESTORE.json](LOCAL_RESTORE.json) records a fresh extraction passing all
308 file checks and reproducing the support-verification receipt exactly.

The disposable mechanics checkpoint and released model weights are excluded;
the checkpoint hash/length receipt and saved test results remain. The package
alone cannot independently rerun the neural forwards or reread that checkpoint.
A model replay needs the separately pinned dense-native weights and retained
runtime, a new output directory, and an explicitly recorded path relocation of
historical GO files. Do not execute write-once training code inside the evidence.
The checkpoint test was CPU save/reload, not a simulated process crash.

Byte verification and arithmetic replay do not prove physical teacher alignment,
ink accuracy, independent sheet identity or final training admission. Read the
scope and next steps in [README.md](README.md). This preparation is not a new
assignment to the spare VM; the checks are short enough to run in this workspace.
