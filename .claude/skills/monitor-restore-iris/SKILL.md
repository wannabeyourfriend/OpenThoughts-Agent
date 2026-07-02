---
name: monitor-restore-iris
description: Re-register the every-3-hours Iris job-monitor cron (status check + datagen auto-rescue/keep-2-in-flight) if it has been lost. Primarily the marin TPU datagen/eval jobs ("iris" = the marin TPU cluster); also queries CoreWeave (cw-us-east-02a) GPU-RL as monitor-only. The cron is session-only and recurring crons auto-expire after 7 days, so it's routinely lost on a session restart. Use at the start of a new session, after a restart, or when the user asks to restore/check the iris monitoring cron. The sweep PROCEDURE the cron runs lives in monitor-cron-sweep-iris.
---

# monitor-restore-iris

> **📍 Iris orientation — read first.** Before acting on anything in this skill, read the Iris **tools
> catalog** (`.claude/ops/iris/iris_tools.md`) and the Iris **ops directory** (`.claude/ops/iris/` — the
> CoreWeave GPU particulars in `coreweave_gpu_ops.md`, the TPU `marin` particulars in `iris_job_lifecycle.md`).
> They carry the binding access/preamble/gotchas and the helper-script inventory the steps below rely on.

The recurring cron that watches all of `benjaminfeuer`'s Iris jobs is
**session-only** (it lives in the Claude session, not on disk) and recurring
crons **auto-expire after 7 days** — so it is routinely lost on a session
restart. This skill is the durable source of truth for re-creating it: the
**canonical cron prompt below is what gets (re-)installed — copy it verbatim
into `CronCreate`.** The per-tick sweep *methodology* the prompt executes is
**monitor-cron-sweep-iris**.

(This is the lightweight marin-TPU-centric monitor. The separate broader
tri-cluster monitor — Leonardo + CoreWeave + TACC — is restored via
**monitor-restore** / **monitor-cron-sweep**.)

## When to run
- Start of a new session where Iris jobs are in flight.
- The user says the monitor/cron is gone, down, or "not firing."
- After ~7 days (expiry).

## Steps
1. **Check if it already exists** — call `CronList`. If a recurring job whose
   prompt mentions "status check on ALL Iris jobs for user benjaminfeuer" is
   present, do nothing (a duplicate causes redundant SQL/tunnel load). If a stale
   **datagen-only** variant exists (prompt mentions only `qwen3.5-122b-32k-%`),
   `CronDelete` it and recreate with the all-jobs prompt below.
2. **If absent, call `CronCreate`** with:
   - `cron`: `23 */3 * * *`  (every 3 h at :23 — off the :00/:30 marks)
   - `recurring`: `true`
   - `prompt`: the exact text in the fenced block below.
3. Tell the user the new job id + the two caveats: **session-only** (dies when
   this Claude session exits — re-run this skill next session) and **7-day
   auto-expiry**.

## Notes
- `durable: true` is NOT honored in this harness (it still creates a
  session-only job), so don't rely on it — this skill IS the persistence layer.
- The cron only fires while the REPL is idle (not mid-task). If it reliably
  misses, the fallback is the user pasting the prompt manually, or an external
  launchd monitor (out of scope here).
- It tracks ALL `/benjaminfeuer/%` jobs but the autonomous write actions
  (auto-rescue, keep-2-in-flight) are **datagen-only**; eval jobs are
  monitor-only (they self-sync to Supabase+HF). See **datagen-launch-iris** and
  **eval-agentic-launch-iris**.
- **Two clusters.** The cron queries both the **marin** TPU cluster and the
  **`cw-us-east-02a`** CoreWeave GPU cluster. The marin `.venv` iris carries the
  `[controller]` deps so it drives CoreWeave too — but the CoreWeave query MUST be
  prefixed `KUBECONFIG=~/.kube/coreweave-iris-gpu`, else iris falls back to the
  shell-default kubeconfig (`~/.kube/lambdaconfig`) and errors with
  `Invalid kube-config file … Expected object with name`. GPU-RL jobs on CoreWeave
  are **monitor-only** (no rescue, no keep-2); their pods GC on terminal, so logs
  come from the persistent finelog server. Other CoreWeave GPU configs exist
  (`coreweave*` = US-WEST-04A, CI/smoke) but are NOT in scope unless the user runs
  jobs there.
- **The methodology each step encodes** (how to run the analyzer, classify, rescue,
  refill) is **monitor-cron-sweep-iris** — read it when actually executing a tick;
  this skill is just the (re)install wrapper + the canonical prompt.

## Canonical cron prompt (copy verbatim into CronCreate)

```
Every-3-hours status check on ALL Iris jobs for user benjaminfeuer (datagen + eval + GPU-RL + anything else), across BOTH the marin TPU cluster and the CoreWeave GPU cluster.

1. Active jobs (query BOTH clusters):
   1a. marin (TPU):
       /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin query "SELECT job_id, state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%' ORDER BY job_id DESC LIMIT 20" -f csv
   1b. cw-us-east-02a (CoreWeave GPU) — KUBECONFIG prefix is REQUIRED (else iris uses the wrong shell-default kubeconfig and errors):
       KUBECONFIG=~/.kube/coreweave-iris-gpu /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=cw-us-east-02a query "SELECT job_id, state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%' ORDER BY job_id DESC LIMIT 20" -f csv
   For EACH cluster also query state IN (4,5,6) LIMIT 8 to catch jobs that went terminal since the last tick. If the cw query errors (cluster down / creds), report that and continue with marin — do not fail the whole tick.

2. For each ACTIVE marin (TPU) datagen/eval job, run the harbor analyzer (TPU/harbor-shaped — does NOT apply to CoreWeave GPU-RL jobs; handle those per class D). Use the analyze-job-history-iris skill (the reliable foreground-and-wait recipe; dispatch a patient subagent for the big task-history jobs):
   /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_job_history.py <job_id> --output /tmp/$(basename <job_id>)_history.md --refresh
   Report from the .json sidecar: runtime_h, iris_preemption_count, cycles total/served, samples (serving_summary.gen_tps.n), gen tok/s mean/peak, Running mean/peak, non_empty/total trials = rate, t_first_serve, top harbor_exception_stats. ALSO report mean reward + completed/total tasks from the harbor progress line (NOT in the sidecar): /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin job logs <job_id> --max-lines 8000 | grep -aoE '[0-9]+/[0-9]+ Mean: [-0-9.]+' | tail -1

3. Print `## Iris jobs status — <ISO UTC>`: one line per job (name + state + CLOSED/PARTIAL/OPEN/DEAD), a compact metrics block, and a survival check (past cold compile? throughput sane? traces/results landing on HF?). Classify each job by job_id prefix and apply the right treatment:

   A. **Datagen** (`qwen3.5-122b-32k-%`): S1 baseline gen tok/s mean~400/peak~1115 (short-task datasets run lower — judge by productive trial rate). HF repo `penfever/<slug>-qwen3.5-122b-32k-traces`. Image `ae085bc8`+ auto-uploads on state-4 success — verify the repo self-created before rescuing. Watch the OOM-fix (heavy datasets) and stuck-PENDING (unpinned relaunch via --gcs-output-dir gs://marin-models-us/ot-agent). **The standing actions in §4-5 below apply to datagen jobs ONLY.**

   B. **Eval** (`eval-%`): launched per the eval-agentic-launch-iris skill. These **auto-sync to Supabase + HF on completion** (`--upload_to_database`), build sandboxes at runtime (force_build:true, MAIN Daytona org) and need **NO rescue and NO keep-2-in-flight** — they are one-off. **ALWAYS report the leading metric (running accuracy / mean reward) for each in-flight eval** — pull the `<done>/<total> Mean: <X>` figure from the harbor progress line in `/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin job logs <job_id>` (the analyzer sidecar does NOT capture it). Also report productive trial rate + harness exceptions, and on a terminal job whether results landed (Supabase/HF). Do NOT auto-relaunch eval jobs.

   C. **Other** job types: report state + a one-line health read; take no autonomous write action.

   D. **GPU-RL** (CoreWeave `cw-us-east-02a`, e.g. `rl-iris-%` / `rl-%` — MarinSkyRL GRPO on whole H100x8 nodes, possibly gang-scheduled multi-node `replicas>1`): **monitor-only — NO rescue, NO keep-2-in-flight, NO auto-relaunch** (those are TPU-datagen concepts and do not apply). The harbor analyzer in §2 does NOT apply (no harbor trial sidecars). For each in-flight GPU-RL job report state + the latest RL progress by reading the persistent finelog (pods GC on terminal): `KUBECONFIG=~/.kube/coreweave-iris-gpu /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=cw-us-east-02a job logs <job_id> --max-lines 100000 --no-tail` then grep `WANDB_MIRROR kind=train step=` for the latest `trainer/global_step`, `loss/avg_raw_reward`, and `generate/num_failed_trajectories`/`generate/errors`. For multi-node confirm `All N Ray node(s) joined`. On a terminal job report exit state (4=SUCCEEDED). Note: finelog retention is finite — older runs' step lines may have aged out (report what survives). NEVER kill/relaunch GPU-RL jobs.

STANDING ACTIONS — DATAGEN JOBS ONLY (override read-only; see memories auto_rescue_banked_trials, datagen_keep_two_in_flight):
4. AUTO-RESCUE (datagen only) — covers two cases, both autonomous:
   4a. TERMINAL rescue: if a datagen job is terminal (4/5/6) with productive trials banked in GCS that did NOT auto-upload (HF repo missing/stale), rescue automatically — no need to ask.
   4b. ZOMBIE kill-then-rescue (datagen only — killing the zombie IS the precondition for the rescue, so it falls UNDER this datagen-rescue authority, NOT under the §6 no-kill rule): if a datagen job is **state 3 AND harbor progress frozen ≥3h (harbor_updated_at stale) AND the task log shows ONLY `[fd-monitor]` heartbeats in that window (no vLLM/harbor/trial activity)**, it is a confirmed zombie — `iris --cluster=marin job stop <job>` it, then rescue its banked trials as in 4a. The fd-monitor-ONLY clause is the safety gate: a healthy cold-compiling/preempt-recompiling job emits XLA/vLLM compile logs (NOT fd-monitor-only), so it will NOT match (see memory datagen_watchdog_kills_healthy_jobs — a loose stale-only detector once killed a HEALTHY recompiling job; the fd-monitor-only requirement + ≥3h prevents that). If unsure whether it's a wedge vs a slow long-task cycle, do NOT kill — report and ask.
   Rescue mechanics (both cases): rsync gs://marin-models-{us,eu}/ot-agent/<job>/<job>/ → /tmp/<job>_traces, then /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/harbor/make_and_upload_trace_dataset.py --job_dir /tmp/<job>_traces --repo_id penfever/<slug>-qwen3.5-122b-32k-traces --episodes last --filter none --skip_register (source /Users/benjaminfeuer/Documents/secrets.env first). Report final row count + update the tracker.
5. KEEP TWO DATAGEN IN-FLIGHT: if the count of active DATAGEN jobs (`qwen3.5-122b-32k-%`, state 1/2/3) is < 2, auto-launch the next `pending` dataset from /Users/benjaminfeuer/Documents/experiments/active/datagen/qwen3.5-122b-tt/tracker.md using the datagen launch template (S1 config, ctx32k_verified.yaml, --tpu v5p-8 --preemptible, --gcs-output-dir gs://marin-models-us/ot-agent [unpinned], repo penfever/<slug>-qwen3.5-122b-32k-traces). Flip its tracker row to RUNNING. (Eval jobs do NOT count toward this 2 and are never auto-launched.)
   SNAPSHOT-CAP HYGIENE (every tick, datagen only): the cli Daytona org (DAYTONA_API_KEY) has a hard cap of 60 snapshots. If a datagen launch fails with SnapshotCapExceeded OR the org is >=~58/60, reclaim idle harbor__ env snapshots (idle >120min) then retry the launch: `source /Users/benjaminfeuer/Documents/secrets.env && /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/daytona/daytona_snapshot_manager.py --api-key-env DAYTONA_API_KEY --stale-days 0.0833 --delete-stale --yes` (run from the OT-Agent dir). This deletes ONLY idle harbor__ snapshots (rebuilt on demand by harbor auto_snapshot); the --name-prefix harbor__ default GUARDS the shared base images (daytonaio/sandbox:*, daytona-*, windows-*) which must never be deleted. This supersedes the old "delete only MISSING harbor__" rule (which stalls at 0 MISSING when all are ACTIVE). See skill datagen-reclaim-stale-snapshots.

6. NEVER kill/restart/bounce a RUNNING job or the cluster without express user permission — with ONE exception: a **confirmed zombie DATAGEN job** per §4b (the kill is the precondition of a datagen rescue, which IS autonomously authorized). The autonomous write actions are: datagen rescue (incl. the §4b zombie kill-then-rescue), datagen refill-launch. This exception is **datagen-only** — GPU-RL and all other RUNNING jobs stay strictly no-touch (flag for the user, never kill). If a job is stuck PENDING (no capacity), report it and surface the unpinned-relaunch option — do not kill a running/placed job unprompted.
```

If you change the cadence or scope, update BOTH the `cron`/`prompt` above and the
live job (delete + recreate), and keep **monitor-cron-sweep-iris** (the procedure)
in sync — so this skill stays the canonical copy.
