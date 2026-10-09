# R2 and Runpod architecture source audit

Official documentation reviewed October 8, 2026. Read-only research: no account access, purchases, resource provisioning or external feedback.

**The proposed split is practical:** GitHub for code, configuration and experiment manifests; public Vesuvius S3 for bounded source reads; R2 Standard for persistent results and checkpoints; Runpod for measured GPU jobs. Storage lifecycle, request billing and explicit compute teardown determine the costs.

## Cloudflare R2

The [official pricing page](https://developers.cloudflare.com/r2/pricing/), last updated October 1, 2026, lists:

| Charge | Standard | Infrequent Access |
| --- | ---: | ---: |
| Storage | $0.015 / GB-month | $0.010 / GB-month |
| Class A operations | $4.50 / million | $9.00 / million |
| Class B operations | $0.36 / million | $0.90 / million |
| Data retrieval | None | $0.01 / GB |
| Direct internet egress | Free | Free |
| Minimum storage duration | None | 30 days |

The monthly Standard free allowance is 10 GB-month, 1 million Class A operations and 10 million Class B operations. It does not apply to Infrequent Access. The allowance is account usage, not a guarantee that this project will incur no charge.

Material pricing details from the same page:

- Storage uses the average of daily peak usage over the billing period. Billable usage is rounded upward to the next unit; the docs explicitly give 1,000,001 operations billed as 2 million and 1.1 GB-month billed as 2 GB-month.
- PUT, LIST and multipart creation/parts/completion are Class A; GET and HEAD are Class B. DELETE and aborting multipart uploads are free operations. Repeated object listings and many tiny uploads can therefore matter even when bandwidth is free.
- Direct R2 egress through the S3 API, Workers API and r2.dev has no bandwidth charge. Other metered services connected to the bucket may still charge; source providers may also charge when migration copies their objects.
- Infrequent Access retrieval charges apply when objects are read or copied, and its 30-day minimum applies even to deleted, moved or replaced objects. **Standard is the initial choice for active checkpoint and result iteration.**

For transfers, the [official boto3 example](https://developers.cloudflare.com/r2/examples/aws/boto3/) uses `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` with region `auto`. Use managed S3 uploads/downloads for ordinary artifacts, local checkpoint caching, bounded concurrency and retry limits. The example recommends low-level multipart with a thread pool for multi-GB objects. [Upload documentation](https://developers.cloudflare.com/r2/objects/upload-objects/) specifies at least 5 MiB per part except the last, at most 5 GiB per part, no more than 10,000 parts and equal sizes except the last. Incomplete multipart uploads are automatically aborted after seven days by default; the uploader should explicitly abort a failed upload rather than rely on that delay.

Multipart ETags are composite hashes, not the whole-file SHA-256. Retain a client-computed SHA-256 and byte length in the run manifest, verify transferred bytes, and publish the completion manifest after its artifacts. Use immutable run/content keys. The [current S3 compatibility table](https://developers.cloudflare.com/r2/api/s3/api/) does not implement AWS bucket-versioning or replication APIs; S3 compatibility is not a promise of every AWS feature.

[Presigned URLs](https://developers.cloudflare.com/r2/api/s3/presigned-urls/) can grant object-specific temporary GET/PUT access for one second to seven days. Scoped credentials or short-lived URLs should be injected at execution time, while Git records nonsecret identifiers and manifests. Checkpoint transfer time can still occupy paid GPU time even when R2 egress itself is free; stage/cache inputs before the expensive phase where practical.

## Runpod Pods and persistent storage

[Pod storage types](https://docs.runpod.io/pods/storage/types) and [Pod pricing](https://docs.runpod.io/pods/pricing) distinguish:

| Storage | Persistence | Published standard rate |
| --- | --- | --- |
| Container disk | Lost when stopped/restarted | $0.10 / GB-month while running |
| Pod volume disk, usually `/workspace` | Survives stop; deleted on termination | $0.10 / GB-month running; $0.20 stopped |
| Network volume | Independent of Pod; survives termination | $0.07 / GB-month for first TB; $0.05 beyond |
| Global volume | Independent, read-oriented, beta | Separate storage and IOPS pricing |

Pod compute is billed by the second, with no Pod ingress/egress fee listed. Container and volume disk billing is per second; network storage billing is hourly. GPU prices and availability should be checked during deployment; this audit does not quote a live GPU lease price. On-demand resources have no multi-month commitment; savings plans require three or six months prepaid and are not the first choice for an unmeasured experimental workload.

[Pod lifecycle documentation](https://docs.runpod.io/pods/manage-pods) states that stopping releases the GPU, clears the container disk and keeps the volume disk, which continues to incur storage charges. Termination deletes Pod-local data; network volumes remain. Editing a running Pod resets it and erases data outside persistent mounts. Do not treat a script's completion as confirmation of provider teardown: explicitly stop or terminate the resource and verify its state after outputs are exported and verified.

[Network volume documentation](https://docs.runpod.io/storage/network-volumes) states that volumes are attached during deployment, constrain GPU placement to their data center, and cannot be detached later without deleting the Pod. They can persist and bill independently of compute. Long-term unpaid charges may eventually lead to their deletion; they are not the authoritative archive. Runpod itself recommends backing up critical data to a dedicated storage provider. R2 fits that role; Pod disks are working space or disposable cache.

For reproducibility, [custom template documentation](https://docs.runpod.io/pods/templates/create-custom-template) recommends versioned image tags over mutable `latest`, a tested Docker image and `linux/amd64` builds. The architecture should additionally retain the resolved image digest, exact Git commit, dependency lock, CUDA/PyTorch versions, GPU model, checkpoint SHA-256 and the frozen experiment manifest. CUDA host compatibility is a deployment requirement in [Manage Pods](https://docs.runpod.io/pods/manage-pods). A GPU run needs its own verified execution path; existing CPU reproducibility does not establish identical GPU numerics.

## Serverless tradeoffs

[Worker lifecycle documentation](https://docs.runpod.io/serverless/workers/overview) and [Serverless pricing](https://docs.runpod.io/serverless/pricing) distinguish Flex workers that scale to zero from Active workers that remain running and bill continuously. Container-image pulls and cached-model downloads in Initializing are unbilled. Once Running, handler model loading, execution and the post-request idle timeout are billed; its default timeout is five seconds. A worker's ephemeral files disappear on scale-down, according to the current [Serverless storage page](https://docs.runpod.io/serverless/storage/overview).

The docs disagree on Serverless container-disk charges: the storage page says they are included in running cost, while the pricing page separately lists approximately $0.10 / GB-month in five-minute intervals. Treat that cost as unresolved until the actual deployment/billing view confirms it; do not promise free worker disks.

[Endpoint settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations) permit long jobs: execution timeout defaults to 600 seconds and can reach seven days. Job TTL is separate, defaults to 24 hours and starts at submission. [Async request documentation](https://docs.runpod.io/serverless/endpoints/send-requests) says completed async results remain accessible for 30 minutes. Durable results therefore belong in R2, with the response returning artifact identifiers and hashes, rather than relying on endpoint result retention.

**Start with a bounded on-demand Pod for the first GPU batches**, measure runtime/memory and export verified artifacts to R2 before teardown. This agrees with Runpod's [Pod-first development guidance](https://docs.runpod.io/serverless/development/dual-mode-worker). After the batch runner is reproducible, Flex Serverless can suit repeated independent jobs, with zero Active workers initially, bounded concurrency, explicit timeouts/TTL and durable output writes. The documentation does not establish which mode is cheaper for this workload; that requires a measured job and an actual deployment quote.

The audit supplies architecture evidence only. Public CT and annotation terms still govern what may be copied or shared; provider storage does not change those rights. No cloud infrastructure or transfer was created by this review.
