# Connecting private cloud services

The current environment has working platform GitHub authentication. No new GitHub token is required. R2 and RunPod configuration was absent during the variable-name/presence check; values were not printed.

For R2, supply the nonsecret account ID and an existing private bucket name. The artifact client expects `R2_ACCOUNT_ID`, `R2_BUCKET`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`. Add keys using a secure runtime secret mechanism with bucket-scoped permissions. Do not paste keys in chat or save them in Git.

S3 signing requires actual signing credentials inside the SDK runtime. A proxy placeholder intended for later header substitution is not automatically a usable signing key. Verify the supported credential delivery path before a live SDK test. GPU-worker secret injection or a securely provided credential profile are possible deployment mechanisms; neither is configured by this repository.

After credentials are usable, test a small immutable private run, verify its stored bytes, and exercise interruption/resume before relying on remote persistence. Confirm bucket privacy and storage retention separately. Local/fake-client tests do not establish account connectivity.

A future RunPod launcher needs usable account/API access, a pinned GPU environment, current pricing and verified independent shutdown. Account access alone does not make the scientific reference or the £10 pilot ready. This repository includes a validator and specification, not a paid-job launcher.
