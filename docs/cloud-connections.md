# Connecting private cloud services

The cloud environment has working platform GitHub authentication. No new GitHub token is required. Local R2 access and local Runpod OAuth have been verified separately; that does not inject credentials or provider tools into this cloud process.

For R2, supply the nonsecret account ID and an existing private bucket name. The artifact client expects `R2_ACCOUNT_ID`, `R2_BUCKET`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`. Temporary credentials additionally require `R2_SESSION_TOKEN`. Add the complete credential set through a secure runtime mechanism with bucket-scoped permissions. Keep account-administration credentials outside the worker. Do not paste keys in chat or save them in Git.

S3 signing requires actual signing credentials inside the SDK runtime. A proxy placeholder intended for later header substitution is not automatically a usable signing key. [Runpod Secrets](https://docs.runpod.io/pods/templates/secrets) documents substitution into the worker environment at startup. The prepared temporary-credential mapping is:

| Worker variable | Runpod environment value |
| --- | --- |
| `R2_ACCESS_KEY_ID` | `{{ RUNPOD_SECRET_HERCULANEUM_R2_ACCESS_KEY_ID }}` |
| `R2_SECRET_ACCESS_KEY` | `{{ RUNPOD_SECRET_HERCULANEUM_R2_SECRET_ACCESS_KEY }}` |
| `R2_SESSION_TOKEN` | `{{ RUNPOD_SECRET_HERCULANEUM_R2_SESSION_TOKEN }}` |

The stored secret names are the unprefixed `HERCULANEUM_R2_*` names shown above; `RUNPOD_SECRET_` belongs to the reference syntax. Account ID and bucket name remain ordinary configuration. Import and actual worker delivery still require verification in the intended account/team. Check that all references resolve before using the client.

Temporary credentials expire. Record their conservative expiry outside the secret values and ensure it covers the entire job and artifact-export window. The artifact client captures credentials at construction and does not renew them. Updated Runpod Secrets take effect on a later worker boot; refreshing an issuer credential or changing a stored secret does not refresh a running client.

After credentials are usable, test a small immutable private run, verify its stored bytes, and exercise interruption/resume before relying on remote persistence. Confirm bucket privacy and storage retention separately. Local/fake-client tests do not establish account connectivity.

A future Runpod launcher needs usable account/API access, a pinned GPU environment, current pricing and verified independent shutdown. Account access alone does not make the scientific reference or the £10 pilot ready. This repository includes a validator and specification, not a paid-job launcher.
