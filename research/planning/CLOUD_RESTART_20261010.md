# Unexplained cloud runtime replacement — 10 October 2026

This is a support-ready incident record, not a finding about the platform's shutdown policy. No support request has been sent.

The user asked for status on a long CPU inference job. That turn arrived with the attached environment already marked `starting`, before the assistant executed commands. After waiting for readiness, the existing files were present but the old supervisor and inference processes were absent. The assistant did not invoke restart, stop, configuration update or publication during the preceding assessment or this status check. The precise platform event that ended or replaced the old runtime is not exposed by the available tools.

| Observation | Time, Europe/London |
| --- | --- |
| Last retained original heartbeat | 10 October, 09:59:10 |
| Status-check turn start, from the chat service's `startedAt` | 10 October, 10:06:18 |
| Approximate current kernel start, inferred from `/proc/uptime` | 10 October, 10:07:32 |
| First successful inspection of restored files; old PIDs absent | 10 October, 10:08:01 |
| Recovery worker started | 10 October, 10:10:16 |

The old log ended at 8,904/14,162 padded Hecate tile origins. Seven complete map files survived and passed encoded/decoded hashes. An unfinished prediction was not recoverable from the restored files. The source actually uses temporary disk-backed numerator/weight arrays under `/tmp`, in addition to a final output array in memory. No matching old temporary arrays were present when inspected; only the recovery worker's newly created arrays existed. Earlier descriptions that attributed all partial state to RAM were incomplete.

The last *retained* heartbeat is not an exact process-death timestamp. A restored filesystem may lag actual execution. The chat service subsequently supplied the turn-start timestamp in [CLOUD_RESTART_20261010_TIMELINE.json](CLOUD_RESTART_20261010_TIMELINE.json). The replacement runtime began approximately74 seconds after that turn started. This links startup in time to the request; it does not establish why the old runtime ended. The turn timestamp is not an infrastructure event or the exact message-receipt timestamp. The evidence cannot distinguish request-triggered replacement of a live worker from restoration of an already stopped runtime, nor determine whether withholding the message would have allowed completion.

Later turns began at10:12:36 and10:15:06 BST. The replacement worker's same PID982 and10:10:16 start remained present through those messages and the checkpoint setup. Thus a message does not invariably replace the worker; this observation does not exclude a conditional lifecycle policy affecting the earlier request.

Platform identifier: `ccarenv_b64_Y2NhcmVudl9iOTM3ZTg3ZWM2NTQ4MTkxYWUwY2FkNDdkMTRlZGE4Mw`. Configuration/source revision observed after recovery: `29`. The status tool reported `observed_phase: running`, `connectivity: connected`, `observations_current: true`, and `failure: null`. These describe the replacement runtime, not the reason the previous runtime ended. The replacement runtime's zero OOM counters do not diagnose the old runtime.

Thread: `01a1185d-e84a-75ee-b597-093417b43048` on host `durable`; status-check turn: `01a12510-2914-75ad-bb7a-3ffbfb84ef50`.

Questions for the platform operator:

1. What stop/suspend/replace/start events occurred for this environment around 08:59–09:08 UTC on 10 October 2026, with their reason codes and initiating actor/request?
2. Did the user's status message initiate replacement, restoration or reassignment of an environment that still had a live CPU worker?
3. What lifetime or inactivity rules apply to detached background CPU processes between messages, and is active computation considered?
4. What persistence guarantees apply to `/workspace` and `/tmp` on those transitions? Which supported execution mode permits an unattended multi-hour job?

Do not describe the event as an idle timeout, maintenance restart, OOM, user-caused restart or expected wake-on-message behavior without the operator's event record. The engineering mitigation is durable progress checkpointing and verified recovery; it cannot prevent the platform replacing a runtime.
