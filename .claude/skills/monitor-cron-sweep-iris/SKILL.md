---
name: monitor-cron-sweep-iris
description: >-
  The PROCEDURE for one every-3-hours Iris job-status sweep — primarily the marin TPU datagen/eval jobs
  ("iris" here = the marin TPU cluster), plus CoreWeave GPU-RL as monitor-only. Query both clusters, run the
  harbor analyzer on each active TPU datagen/eval job, classify every job (datagen / eval / other / GPU-RL)
  and apply its treatment, then take the standing DATAGEN-ONLY autonomous actions (auto-rescue, keep-2-in-flight).
  This is the methodology the recurring cron prompt runs; the cron itself is (re)installed via monitor-restore-iris.
  Use for "run an iris sweep / cluster sweep now" or as the reference behind each cron tick.
---

# monitor-cron-sweep-iris

> **📍 Iris orientation — read first.** Before acting on anything in this skill, read the Iris **tools
> catalog** (`.claude/ops/iris/iris_tools.md`) and the Iris **ops directory** (`.claude/ops/iris/` — the
> CoreWeave GPU particulars in `coreweave_gpu_ops.md`, the TPU `marin` particulars in `iris_job_lifecycle.md`).
> They carry the binding access/preamble/gotchas and the helper-script inventory the steps below rely on.

The per-tick procedure for the lightweight Iris monitor: the **marin TPU**
datagen/eval pipeline (the autonomous-action surface) plus **CoreWeave**
(`cw-us-east-02a`) GPU-RL as **monitor-only**. This is distinct from the broader
multi-cluster **monitor-cron-sweep** (Leonardo + CoreWeave + TACC SLURM/eval
campaign). The recurring 3-hour cron that fires this is installed/restored by
**monitor-restore-iris** (which holds the verbatim cron prompt); this skill is
the methodology behind it and can also be run ad-hoc.

> Autonomous WRITE actions are **datagen-only** (§4 rescue, §5 keep-2). Eval is
> monitor-only (self-syncs), GPU-RL is monitor-only, everything else is read-only.
> Never kill/restart a RUNNING job without express permission — the ONE exception
> is a confirmed zombie DATAGEN job (§4b).

## 1. Query both clusters (active + recently-terminal)
- **marin (TPU):**
  `/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin query "SELECT job_id, state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%' ORDER BY job_id DESC LIMIT 20" -f csv`
- **cw-us-east-02a (CoreWeave GPU)** — the `KUBECONFIG` prefix is REQUIRED (else iris uses the shell-default kubeconfig and errors):
  `KUBECONFIG=~/.kube/coreweave-iris-gpu /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=cw-us-east-02a query "… same …" -f csv`
- For EACH cluster also query `state IN (4,5,6) LIMIT 8` to catch jobs gone terminal since the last tick.
- If the cw query errors (cluster down / creds), report it and continue with marin — don't fail the whole tick.
- States: 1=PENDING 2=starting 3=RUNNING 4=SUCCEEDED 5=FAILED 6=KILLED.

## 2. Metrics for each ACTIVE marin datagen/eval job → use **analyze-job-history-iris**
Run the harbor analyzer via the **analyze-job-history-iris** skill — do NOT eyeball a `--tail`. That skill
carries the reliable recipe (it's SLOW: paginates the full history, minutes per job — run it foreground with a
long timeout and WAIT; offload to a patient subagent for big 16k/46k-task jobs, per its dispatch template).
Report from the JSON sidecar: runtime_h, iris_preemption_count, cycles total/served, gen tok/s mean/peak (n),
Running mean/peak, non_empty/total trials = productive rate, t_first_serve, top harbor_exception_stats — PLUS
the two log-only fields that skill extracts (NOT in the sidecar): **mean reward** and **completed/total tasks**
(`iris … job logs | grep -aoE '[0-9]+/[0-9]+ Mean: [-0-9.]+'`). (The analyzer is harbor-shaped; it does NOT
apply to CoreWeave GPU-RL — see class D.)

## 3. Report + classify (`## Iris jobs status — <ISO UTC>`)
One line per job (name + state + CLOSED/PARTIAL/OPEN/DEAD) + a compact metrics block + a survival check (past
cold compile? throughput sane vs S1 baseline gen mean~400/peak~1115? traces/results landing on HF?). Classify by
job_id prefix:

- **A. Datagen** (`qwen3.5-122b-32k-%`): HF repo `penfever/<slug>-qwen3.5-122b-32k-traces`; image `ae085bc8`+
  auto-uploads on state-4 — verify the repo self-created before rescuing. Short-task datasets run lower gen
  tok/s — judge by productive trial rate. Watch the heavy-dataset OOM-fix + stuck-PENDING (unpinned relaunch).
  **§4-5 standing actions apply to datagen ONLY.**
- **B. Eval** (`eval-%`): auto-syncs to Supabase + HF on completion (`--upload_to_database`); NO rescue, NO
  keep-2. ALWAYS report the leading metric — the `<done>/<total> Mean: <X>` harbor progress line from
  `iris --cluster=marin job logs <job_id>` — plus productive rate + harness exceptions; on a terminal job, whether
  results landed. Never auto-relaunch eval. (See eval-agentic-launch-iris.)
- **C. Other** (anything matching none of A/B/D/E — e.g. `serve-%` inference jobs): report state + a one-line
  health read; no autonomous write action.
- **E. Executor/Levanter training** (a marin-executor training run — a CPU coordinator `<run>-coord` PLUS its
  nested v5p training child `<run>-coord/checkpoints-<step>-<hash>`; e.g. the `delphi-%` midtraining runs):
  **monitor-only — NO rescue, NO keep-2, NO auto-relaunch.** The §2 harbor analyzer does NOT apply (training
  has no harbor trial sidecars, like GPU-RL). Run the **analyze-training-run-iris** skill on the CHILD job and
  report its compact line: `step=<cur>/<total> (X%) loss=<L> ~<T>tok/s preempts=<P> gaps=<G>/<H>h ckpt=step-<C>`
  + a health read (past setup/compile? step rate sane vs last tick? loss finite & trending down? preemptions
  resuming cleanly — checkpoint advancing? ETA to the K-budget target). That skill reads W&B per-step history
  (`nyu-dice-lab/delphi-midtraining`, run = the GCS output-path hash) + `iris job summary` (preemptions) + GCS
  `step-*` checkpoints; empty W&B history = **pre-first-step** (still HF-download/XLA-compile), not a gap. Track
  progress in steps + MAJOR gaps every tick, just as datagen tracks productive trials. NEVER kill/relaunch.
- **D. GPU-RL** (CoreWeave, `rl-%` / `rl-iris-%` / MarinSkyRL GRPO on H100×8, possibly multi-node `replicas>1`):
  **monitor-only — NO rescue, NO keep-2, NO auto-relaunch.** The §2 analyzer does NOT apply (no harbor trial
  sidecars). Report state + latest RL progress from the persistent finelog (pods GC on terminal):
  `KUBECONFIG=~/.kube/coreweave-iris-gpu iris --cluster=cw-us-east-02a job logs <job_id> --max-lines 100000 --no-tail`
  then grep `WANDB_MIRROR kind=train step=` for the latest `trainer/global_step`, `loss/avg_raw_reward`,
  `generate/num_failed_trajectories`/`errors`; for multi-node confirm `All N Ray node(s) joined`. On terminal,
  report exit state. For a NEW/untested RL run, deep-probe via **rl-job-health-deep-dive** (KILL/NO-KILL). NEVER
  kill/relaunch GPU-RL. (Finelog retention is finite — report what survives.)

## 4. AUTO-RESCUE — DATAGEN ONLY (autonomous; overrides read-only)
- **4a. TERMINAL rescue:** a datagen job terminal (4/5/6) with productive GCS trials that did NOT auto-upload
  (HF repo missing/stale) → rescue automatically, no need to ask.
- **4b. ZOMBIE kill-then-rescue (datagen only):** state 3 AND harbor frozen ≥3h (`harbor_updated_at` stale) AND
  the task log shows ONLY `[fd-monitor]` heartbeats in that window (no vLLM/harbor/trial activity) → confirmed
  zombie: `iris --cluster=marin job stop <job>`, then rescue per 4a. The fd-monitor-ONLY clause is the safety
  gate (a healthy cold-compiling/preempt-recompiling job emits XLA/vLLM logs, so it won't match — see memory
  `datagen_watchdog_kills_healthy_jobs`). Unsure wedge vs slow long-task cycle → do NOT kill; report and ask.
- **Rescue mechanics (both):** `gsutil rsync` `gs://marin-models-{us,eu}/ot-agent/<job>/<job>/` → `/tmp/<job>_traces`,
  then `make_and_upload_trace_dataset.py --job_dir /tmp/<job>_traces --repo_id penfever/<slug>-qwen3.5-122b-32k-traces
  --episodes last --filter none --skip_register` (source `secrets.env` first). Report row count + update the tracker.
  (Full launch/rescue detail: **datagen-launch-iris**.)

## 5. KEEP TWO DATAGEN IN-FLIGHT (datagen only)
If active datagen (`qwen3.5-122b-32k-%`, state 1/2/3) < 2, auto-launch the next `pending` dataset from
`/Users/benjaminfeuer/Documents/experiments/active/datagen/qwen3.5-122b-tt/tracker.md` via the datagen launch
template (S1, `ctx32k_verified.yaml`, `--tpu v5p-8 --preemptible`, `--gcs-output-dir gs://marin-models-us/ot-agent`
unpinned, repo `penfever/<slug>-qwen3.5-122b-32k-traces`) — see **datagen-launch-iris**. Flip its tracker row to
RUNNING. Eval jobs do NOT count toward the 2 and are never auto-launched.

**Snapshot-cap hygiene (every tick):** audit the cli-org (`DAYTONA_API_KEY`) snapshot count. If a datagen refill
is blocked by `SnapshotCapExceeded` OR the org is ≥ ~58/60, **reclaim idle `harbor__` snapshots** (idle > 120 min)
via `daytona_snapshot_manager.py --api-key-env DAYTONA_API_KEY --stale-days 0.0833 --delete-stale --yes`, then retry
the launch. This deletes ONLY idle `harbor__` env snapshots (rebuilt on demand) — the `--name-prefix harbor__`
default guards the shared base images (`daytonaio/sandbox:*`, `daytona-*`, `windows-*`), which must never be deleted.
This supersedes the old MISSING-only rule (which stalls at 0 MISSING). Full procedure: **datagen-reclaim-stale-snapshots**.

## 6. No-kill guardrail
Never kill/restart/bounce a RUNNING job or the cluster without express permission — the ONLY exception is the §4b
confirmed-zombie DATAGEN kill (the precondition of its rescue). Autonomous write actions = datagen rescue (incl.
§4b) + datagen refill. GPU-RL and everything else stay strictly no-touch (flag, never kill). Stuck PENDING (no
capacity) → report + surface the unpinned-relaunch option; don't kill a placed job unprompted.

## Related skills
- **monitor-restore-iris** — (re)installs the recurring 3-hour cron that runs this procedure (holds the verbatim prompt).
- **analyze-job-history-iris** — the §2 analyzer recipe for class A/B datagen+eval (foreground-and-wait; mean-reward + completed/total extraction).
- **analyze-training-run-iris** — the class-E recipe for executor/Levanter training runs (step progress + major-gap detection via W&B + `iris job summary` + GCS checkpoints).
- **datagen-launch-iris** — launch / rescue / snapshot-cleanup mechanics for §4–§5.
- **rl-job-health-deep-dive** — deep per-RL-job KILL/NO-KILL probe for class-D GPU-RL in new/untested settings.
- **monitor-cron-sweep** / **monitor-restore** — the separate broader tri-cluster (Leonardo+CoreWeave+TACC) campaign.
