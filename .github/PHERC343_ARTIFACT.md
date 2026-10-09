# Download the retained PHerc343 review files

The **Package PHerc343 review files** workflow has only a manual `workflow_dispatch` trigger. Once registered on the default branch, run it from the repository's Actions tab or with:

```bash
gh workflow run package-pherc343-review.yml --repo jl-drx/Herculaneum --ref main
```

The repository must remain private. The job uses only `contents: read`; every other repository-token permission is disabled. Official checkout/upload actions are pinned to full commit hashes. There are no schedules, push/PR triggers, deployments, package installations, caches, model loads or inference steps. A five-minute Ubuntu job uploads one artifact with **one-day retention**. Downloading requires repository access.

The snapshot is pinned to `f85c6ab579da054d3ca94c3b7e4b58699f7c6a3a`, with its manifest independently pinned by SHA-256. Exactly three LFS objects are fetched (49,059,957 bytes total):

| LFS object | Bytes | Purpose |
| --- | ---: | --- |
| `pherc343-results-001.tar.gz` | 37,569,445 | Included unchanged |
| `pherc343-reference-001.tar.gz` | 9,780,724 | Included unchanged |
| `pherc343-inputs-004.tar.gz` | 1,709,788 | Extract only the 17,203,328-byte `valid.npy` member |

The input archive also contains other retained files. They are neither extracted nor included in the artifact. No live lab, Hecate process or Hecate model is accessed. The results archive is the existing dated snapshot; it is not a request to collect new results from the running study.

The artifact contains the two original archives, `valid.npy`, `PROVENANCE.json` and `SHA256SUMS`. Every downloaded object's length and SHA-256 is verified, then every payload is checked before publication. Provenance records the source commit, manifest hash, source object hashes and exact mask member. After downloading and extracting the artifact ZIP:

```bash
sha256sum -c SHA256SUMS
```

## Availability and potential charges checked on 9 October 2026

The repository is private. Authenticated workflow/run listing succeeds and initially showed no existing workflows or runs. Reading Actions policy and account billing usage returns HTTP403 (`Resource not accessible by integration`), so remaining allowances, spending limits and policy enablement cannot be confirmed from those endpoints. Successful workflow registration and dispatch must still be checked; API listing alone is not proof that jobs may execute.

GitHub's current published rates are $0.006/minute for a standard two-core Linux runner, $0.25/GB-month for artifact storage beyond included allowance, and $0.0875/GiB of LFS download bandwidth beyond allowance. The five-minute job is approximately **$0.03 compute**, the 49.1 MB LFS fetch approximately **$0.004**, and about 64.6 MB of artifact storage for one day approximately **$0.0006**, if all usage is billable. These are estimates before tax/currency effects and billing rounding, not an account spending cap or a promise of free execution. Additional manual runs consume additional usage. Existing repository/LFS storage is not newly created by this workflow.

Sources checked: [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions), [runner pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing), [LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs), and the LFS bandwidth rate in [GitHub's pricing calculator](https://github.com/pricing/calculator). The user's existing authorization permits CPU costs; the separate £10 first-GPU-pilot cap is unchanged. No GPU is involved.
