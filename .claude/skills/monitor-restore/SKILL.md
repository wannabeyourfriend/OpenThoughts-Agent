---
name: monitor-restore
description: >-
  Re-create the local 3-hour tri-cluster cluster-sweep loop (Leonardo + CoreWeave(iris) + TACC(Vista); Jupiter
  SKIPPED until ~Jul 12) — the autonomous ML-ops monitor — if it has been lost. The loop is session-only and is
  dropped on any session restart, so re-establish it at the start of a new session or whenever the user asks to
  restore/restart the 3h sweep/cron/monitor. Sets a /loop 3h (or equivalent recurring cron) whose task is the
  canonical sweep prompt below: status-table active/pending/completed jobs, auto-cleanup+DB-register completions,
  diagnose+remediate failures via subagents, log to agent_logs + claude_experiments.md. The prompt block here is
  the source of truth — copy it verbatim.
---

# monitor-restore

Re-creates the recurring **3-hour tri-cluster sweep** — **Leonardo + CoreWeave(iris) + TACC(Vista)** (Jupiter
is SKIPPED, down for MDC maintenance until ~2026-07-12 — it rejoins as a 4th cluster when it returns) — the
autonomous ML-ops monitor. The loop is **session-only** (it lives in the Claude session, not on disk) and is
lost on a session restart, so this skill is the durable source of truth: the canonical prompt below is what gets
(re-)installed.

## When to run
- Start of a new session where Leonardo / CoreWeave / TACC jobs are in flight (or expected).
- The user says the monitor/cron/sweep is gone, down, "not firing," or asks to "restart the 3h loop."
- After ~7 days if running it as a `CronCreate` recurring job (auto-expiry).

## How to (re-)establish it
1. **Check for an existing one first** — `CronList` (and note any active `/loop`). If a recurring sweep whose prompt mentions "3-hour … status … leonardo, coreweave, tacc" (or the legacy "jupiter, leonardo") is already present, do **not** create a duplicate (duplicate monitors double the ssh/SQL/Daytona/iris load). Otherwise:
2. **Start the loop.** Two equivalent mechanisms — the user's phrasing is "**/loop 3h or equivalent, active session only, maximum duration**":
   - **Preferred — `/loop`:** invoke the `/loop` skill with interval **3h** and the **maximum duration** the harness allows, passing the canonical prompt below as the loop task.
   - **Equivalent — `CronCreate`:** `cron: 17 */3 * * *` (every 3 h, off the :00 mark to avoid contention), `recurring: true`, `prompt:` = the canonical block below verbatim.
3. Tell the user it's set + the two caveats: **session-only** (re-run this skill next session) and, for the cron variant, **7-day auto-expiry**.

## Supporting skills/docs the sweep leans on (read these; the prompt references them)
- **`monitor-cron-sweep`** — the sweep *procedure* (gather → bucket → render → flag), now with per-cluster Leonardo / CoreWeave / TACC gather+triage sections; **`monitor-job-tables`** + `/Users/benjaminfeuer/Documents/notes/ot-agent/job_monitor_table.md` — the exact per-type table formats + metric/red-flag definitions (the unified RL table now spans Leonardo + CoreWeave).
- **Cleanup:** `rl-job-cleanup` (agentic), `rl-standard-job-cleanup` (standard GRPO), `sft-job-cleanup`, `datagen-job-cleanup`, `eval-agentic-cleanup` (+ `eval-standard-cleanup`).
- **Launch:** `rl-agentic-launch-iris` (CoreWeave RL), `rl-standard-launch-leonardo`, `sft-launch-leonardo`, `datagen-launch`, `eval-agentic-launch`, `eval-standard-launch`. (`*-jupiter` skills apply when Jupiter returns.)
- **Cluster particulars** (access, paths, caps, preamble, binding gotchas): `.claude/ops/leonardo/ops.md`, `.claude/ops/iris/coreweave_gpu_ops.md` (+ `coreweave_h100_cloud_hardware.md`), `.claude/ops/tacc/ops.md`, `.claude/ops/local/ops.md`. **Dependency facts:** `.claude/projects/{marinskyrl,harbor,vllm,llama-factory,daytona}/`.
- The repo `CLAUDE.md` is the user's cited reference for table format + launch + cleanup; **the canonical prompt below overrides any memory/skill if they conflict** (per the user).

## Notes
- `durable: true` is not honored in this harness — this skill IS the persistence layer.
- A cron/loop only fires while the REPL is idle (not mid-task); if it reliably misses, fall back to the user pasting the prompt.
- **Path correction baked into the prompt:** the local SkyRL checkout is `/Users/benjaminfeuer/Documents/MarinSkyRL` (the user's usual prompt says `…/SkyRL`). All other paths are verbatim.
- **Jupiter is SKIPPED** for now (MDC hard downtime Jun 23 – ~Jul 12 2026). When it returns, re-add it as a 4th cluster (its STEP-0 ops doc is `.claude/ops/jupiter/ops.md`; its launch/cleanup skills are the `*-jupiter` variants) in both this block and the live loop.
- **CoreWeave per-run Monitors are a COMPLEMENTARY finer-grained layer** for actively-debugging runs (bring-up/wedge watch via `scripts/iris/watch_job_state.py`) — the 3h cron is the baseline; don't treat a per-run Monitor as a substitute for the sweep, or vice-versa.

---

## Canonical sweep prompt (copy verbatim into `/loop 3h` or `CronCreate`)

```
3-HOURLY CLUSTER SWEEP — clusters: leonardo, coreweave(iris), tacc (active session only, run to max duration).
[Jupiter is SKIPPED — down for MDC maintenance until ~2026-07-12; re-add it as a 4th cluster when it returns.]

STEP 0 (do this FIRST, every sweep): read EACH cluster's ops doc for its BINDING gotchas before touching it —
- leonardo → `.claude/ops/leonardo/ops.md`: hardened `ssh Leonardo` ControlMaster (host keys rotate, benign;
  use login02/03/04 on false-drain); GPFS — no `find`/`du`, locate logs via `scontrol show job <id> -o`
  StdOut/`%Z`; ptrace LOCKED (`ptrace_scope=2`) → no py-spy/gdb; the `~/.ssh/leonardo_daytona` step-ca cert
  expires ~12h (a fast publickey-denied upload/eval = refresh it); WRITE-PATH MANDATE (ckpts/exports → `$WORK`
  / `$CHECKPOINTS_DIR`, NEVER `$SCRATCH_FAST`).
- coreweave(iris) → `.claude/ops/iris/coreweave_gpu_ops.md` (+ `coreweave_h100_cloud_hardware.md`): NO ssh/login —
  drive via the iris SDK from the Mac; `export KUBECONFIG=~/.kube/coreweave-iris-gpu` is a HARD prereq in the same
  shell (the Mac default kubeconfig points at a DIFFERENT cluster → wrong-context "0 pods/not found"); use the
  OTAGENT-ENV iris binary `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris` (the marin `.venv` iris has a
  broken `kubernetes` import and CANNOT drive cw); all `iris`/`kubectl` calls SYNCHRONOUS (never background);
  CoreWeave nodes have egress (NO `HF_HUB_OFFLINE`).
- tacc(Vista) → `.claude/ops/tacc/ops.md`: `ssh TACCVista` (ControlMaster live, single-string hardened ssh);
  `salloc` is BLOCKED → use sbatch; compute nodes have FULL internet (NO proxy/SOCKS/step-ca cert needed —
  contrast Leonardo); GPUs are NOT a SLURM gres (whole-node alloc) and RealMemory is misreported; uv OOMs on the
  shared login node → any build/install goes in a CPU `-p gg` sbatch, never the login node.

PER-CLUSTER GATHER (validate each before trusting it; procedure = `monitor-cron-sweep`, particulars = each ops doc):
- LEONARDO — `squeue -u bfeuer00 -t RUNNING` + `sacct -u bfeuer00 -S now-3hours -X` via `ssh Leonardo`. VALIDATE
  squeue succeeded before trusting an empty result (slurmctld timeout / login-node fork-saturation false-drain →
  re-check via sacct / another login node). For each RUNNING job stat its StdOut mtime — RUNNING is NOT proof of
  progress (silent wedge). Leonardo eval log-path trap: if `scontrol` `StdOut=` doesn't exist, read
  `data_<jobid>.out` in the job's `%Z` workdir before declaring it dead.
- COREWEAVE(iris) — STATE-POLL the authoritative iris lifecycle, NOT a log-string watch (a clean kill/eviction/
  preempt emits no terminal log line + reaps the pods):
    KUBECONFIG=~/.kube/coreweave-iris-gpu; PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
    $PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json   # per active job (auth state now)
    /Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster=cw-us-east-02a job summary --json   # authoritative
  Treat "running-but-0-pods / record disappeared" as TERMINAL (the silent-wedge signature). `iris … query` over the
  jobs table lists live jobs (state 1/2/3). Full log (init→crash) via `iris … job logs --since-ms <submitted_at_ms>
  --no-tail` (finelog keeps the WHOLE log; only `--tail` caps lines).
- TACC(Vista) — `squeue -u penfever` + `sacct -u penfever -S now-3hours -X` via `ssh TACCVista` for job state.

IN-FLIGHT / ACTIVE jobs → report in a UNIFIED TABLE per job type, spanning all clusters (the RL table is
cross-cluster: Leonardo standard-GRPO rows + CoreWeave agentic/MoE rows together). Structure = `monitor-job-tables`
/ notes/ot-agent/job_monitor_table.md (box-drawing, not markdown). RL rows MUST include entropy + collapse
signals (grad_norm / log_ratio), not just step+reward.

CHAIN-RESTART TIMEOUTs are NORMAL (note the afterany successor), not failures. On CoreWeave, `--max-retries`
re-brings-up the gang on a transient HF-weight-resolution flake — a single retry is a normal time-cost, not a fault.

LEONARDO CAMPAIGN DRIVER (the priority — drive it every sweep):
- The 4 rl_dlp Delphi standard-GRPO RL cells (`rl-standard-launch-leonardo`; logger=console, no trace_jobs) —
  table them with entropy+grad+log_ratio. On a clean COMPLETED at max_steps → `rl-standard-job-cleanup` (model +
  metric CSVs only; size suffix DERIVED FROM THE EXPORTED WEIGHTS; DB-register only if the series is DB-registerable),
  then fire the Delphi downstream eval suite (`eval-standard-launch` §5b).
- The **SummarizationTimeoutError-deflated re-eval campaign (a1-`<benchmark>` models)** — re-evaluating
  `DCAgent/a1-<benchmark>` evals that the SummarizationTimeoutError bug deflated (zhuang1 + richard.zhuang owners),
  scored on tb2 / swebench / dev_set_v2. **Tracker (source of truth):
  `~/Documents/experiments/active/flawed_summ_evals/reeval_tracker.md`** (NOT the old `notes/ot-agent/…` path);
  the affected universe is `affected_evals.md` in the same dir. **Driver** (per the tracker's "🚦 CAMPAIGN DRIVER"
  section): each sweep, (1) HARVEST every leg gone terminal since the last pass — read new_score, compute
  `delta = new_score − old_deflated`, flip the row `🚀 → ✅/⚠️`, flag genuinely-negative deltas beyond ~1 stderr;
  (2) REFILL the next still-`⏳ pending` **Section A** rows (Section A before B) until **32 RUNNING/PENDING legs on
  Leonardo** (raised 16→32 on 2026-06-26 per user directive). **⚠️ DAYTONA-CAP GUARD (HARD):** 32 concurrent ≈
  4× the original Daytona sandbox churn; the ORG2 (`DAYTONA_API_KEY`) snapshot cap is **60 — never raise it**.
  Snapshots DON'T scale with leg count (legs reuse `harbor__*` snapshots); at 32 the binding limit is concurrent
  **SANDBOXES** (~32 legs × ~32 trials). Clean stale sandboxes (`utils-cleanup-stale-sandboxes` /
  `scripts/daytona/cleanup_stale_sandboxes.py --delete`) for headroom before/while refilling; if 32 won't fit,
  launch as many as fit and report the binding limit. Classify every live leg against the tracker + affected list FIRST — KEEP valid in-flight affected
  re-evals, CANCEL only confirmed duplicates of already-✅ rows or off-campaign (not-in-affected) legs; do NOT
  `--force-reeval` an already-✅ row (the duplicate trap). Launch via the canonical `eval-agentic-launch` listener
  (after the `ops/leonardo` preamble, in tmux, ONE listener per preset, ~40s stagger):
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"   # repo root — the listener imports first-party top-level packages
    python eval/unified_eval_listener.py --cluster-config eval/clusters/leonardo.yaml --preset <tb2|swebench|v2|...> \
      --baseline-model-configs eval/configs/baseline_model_configs_minimal.yaml \
      --require-priority-list --priority-file <list>.txt \
      --config-yaml dcagent_eval_config_no_override.yaml --force-reeval --once --verbose
  (thinking is per-model authoritative — sourced from the baseline model config
  `eval/configs/baseline_model_configs_minimal.yaml` per model; presets carry none, there is no
  `--enable-thinking` flag. Override a model with `--agent-kwarg 'extra_body={…}'` (CLI > per-model > preset).
  `--config-yaml dcagent_eval_config_no_override.yaml` resolves from `hpc/harbor_yaml/eval/configs/`; the cluster
  sbatch is `eval/leonardo/eval_harbor.sbatch`. `--force-reeval` is REQUIRED — every campaign row has a prior
  Finished/deflated DB row; the "Failed to create Pending DB entry: Job already finished" warning is benign.)
  `--baseline-model-configs` is LOAD-BEARING — omitting it SILENTLY drops every per-model serve override (the
  qwen3_5_moe legs then fall back to the otagent env and FAIL `model type … not recognized`); the listener resolves
  it CLI > cluster-config > None. `--require-priority-list` is LOAD-BEARING too — without it the listener floods the
  queue with every unevaled model in the lookback window. To refill, create/extend the priority list with the legs
  to (re)run and point `--priority-file` at it; use the SINGLE multi-model listener (1s internal submission_delay),
  do NOT fire one `&`-backgrounded listener per leg (the login-side conda-plugin circular-import race) — if multiple
  listeners are truly needed, stagger ~30–45s apart. On harvest → `eval-agentic-cleanup` only if auto-upload/register
  failed. (No-`Mean:`-lines: a Leonardo standard-eval results JSON without `Mean:` lines is the expected shape, not
  a broken run. AgentTimeoutError / ContextLengthExceeded are EXPECTED passthrough exceptions in agentic eval — the
  verifier still scores; never the cause of a hang.)

COREWEAVE RL (agentic SkyRL/MoE via `rl-agentic-launch-iris`):
- Report BRING-UP for fresh launches: gang/leafgroup admission (Kueue, pods SchedulingGated until atomically
  admitted is normal), `apply_ep` / mesh-load, weights resolving. `shm_broadcast: …60s` + a transient ghcr EOF →
  ImagePullBackOff self-heal are BENIGN bring-up noise.
- EP=8 science greps (the 131k arm) — `sel_rows` / `EPDIAG` via `scripts/iris/analyze_job_history.py`
  (log-content greps are for SCIENCE/throughput ONLY, never liveness — liveness = the state-poll above).
- On a COMPLETED run → route by flavor: AGENTIC (Harbor/Daytona/terminal_bench) → `rl-job-cleanup` (FULL checklist
  incl. trace upload + metrics); STANDARD/non-agentic GRPO → `rl-standard-job-cleanup`. Do this WITHOUT asking.
- The per-run Monitors (bring-up/wedge watch) are a COMPLEMENTARY finer-grained layer for active-debugging runs;
  this 3h cron is the baseline — don't let one substitute for the other.

TACC EVAL HARVEST (when present — newly-integrated, currently validated by a canary):
- TACC agentic eval runs through the v6 listener with `--cluster-config eval/clusters/tacc.yaml` (its
  `sbatch_script` = `eval/tacc/eval_harbor.sbatch`, `eval_jobs_dir` = `/scratch/10635/penfever/eval_jobs`; whole-node
  alloc, no `--gres`/`--mem`; compute nodes have egress so NO proxy/cert). Once a leg is RUNNING, harvest finished
  TACC evals the same way as Leonardo (`eval-agentic-cleanup` if auto-register failed). Treat the path as
  newly-integrated — sanity-check the canary's traces uploaded + registered before relying on it.

ON SUCCESSFUL COMPLETION (SFT / RL / datagen / eval) on ANY cluster → note it + summary stats, then:
- RL or SFT → manual HF upload + DB registration per CLAUDE.md. Route RL by flavor: AGENTIC RL
  (Harbor/Daytona/terminal_bench) → `rl-job-cleanup`; STANDARD / non-agentic GRPO (Delphi/rlvr/dapo math cells) →
  `rl-standard-job-cleanup` (model + metric CSVs only; size suffix from the exported weights; DB-register only if
  the series is DB-registerable). SFT → `sft-job-cleanup`. Do this WITHOUT asking. (Leonardo HF upload = the
  sbatch-tunnel path, NOT the login node — it SIGKILLs long processes at ~100s; needs the fresh step-ca cert.)
- Datagen → verify the traces uploaded to HF (penfever org); if NOT, dispatch a subagent to upload manually
  (`datagen-job-cleanup`).
- Eval where DB registration FAILED for a technical reason → WITHOUT asking, dispatch a subagent through ALL steps
  of `eval-agentic-cleanup`; confirm every step completed, dispatching another if any were missed.
- INODES (Leonardo/GPFS): every cleanup MUST `rm` the on-disk artifact tree (`trace_jobs/`/`tasks/`) after the HF
  upload is confirmed + verify reclaim — the #1 inode leak. (CoreWeave artifacts go to HF / R2, not POSIX scratch;
  no on-disk tree to reap there.)

ON ANY JOB THAT FAILED since the last check → dispatch a subagent to determine the cause + propose fixes. Then
ANNOUNCE the choices, SELECT one yourself, and apply the changes + relaunch via another subagent. Keep a running
DATED log of failures (job ID + remediation history) in /Users/benjaminfeuer/Documents/agent_logs/.
- If an RL job has EXHAUSTED all restarts WITHOUT reaching max steps AND the failure looks recoverable (transient)
  → queue 5 more restarts (Leonardo) / re-launch with `--max-retries ≥1` (CoreWeave). Spike-mitigation ablations
  are exempt from auto-cancel — observing the recovery IS the experiment (see `monitor-cron-sweep`).

CODE / CONFIG EDITS → edit LOCALLY on the active branches:
/Users/benjaminfeuer/Documents/OpenThoughts-Agent, /Users/benjaminfeuer/Documents/vllm,
/Users/benjaminfeuer/Documents/harbor, /Users/benjaminfeuer/Documents/MarinSkyRL.
Local clones are GROUND TRUTH — clusters never diverge (no untracked/divergent changes, no hand-editing, no
patch-by-rsync). Sync the Python repos by commit+push then `git pull` on the SLURM clusters (editable installs,
live after pull); CoreWeave has NO clone to pull — the iris launcher uploads the local workspace to `/app` so a
local commit takes effect on the next launch (no push/pull). EVERY SWEEP, run `git status --short` on each SLURM
cluster repo (leonardo, tacc) and triage accumulated untracked/modified drift back to local: TRACK reusable files
(commit local → push), GITIGNORE recurring transient junk (`*.bak`, `*_manifest.txt`, `&1`, ephemeral
`reeval_priority_*`); reconcile with `git pull`, NEVER `git reset --hard` while live jobs depend on uncommitted
state (procedure in `monitor-cron-sweep` §4). vLLM (compiled fork) → commit+push the fork, then BUILD FROM SOURCE
on each cluster from that commit (never rsync edits / hand-patch); CoreWeave rebuilds the gpu-rl image (bump the
digest) only when the compiled vLLM fork changes — first-party + MarinSkyRL fixes go live without a rebuild.

ACTIVELY-DEBUGGING jobs → monitor more closely than stable ones. For any FRESH launch, set one-time checks at
15 min and 30 min after launch to catch new failures early.

LAUNCHING FRESH JOBS → follow the per-job-type launcher instructions in CLAUDE.md (+ the `*-launch-*` skills and
`.claude/projects/ot-agent/ot-agent.md`). If the instructions are unclear, ASK for guidance.

EXPERIMENT LOG → maintain an up-to-date, date-indexed log of experiments launched (referenced by launch command)
in /Users/benjaminfeuer/Documents/notes/claude/claude_experiments.md.

STANDING CONSTRAINTS (do not violate without explicit permission): enable_db_registration stays false in YAMLs
(manual DB register only); Daytona RUNNING RL ≤ 6 per cluster; a3 series is CONCLUDED (no launch/refill/auto-
advance); Daytona snapshot caps are HARD (clean stale, never raise the cap); cross-user FK safety pre-check before
any Supabase delete/mutate; HF uploads default PUBLIC to laion/. NEVER kill/restart a RUNNING job (or
`iris cluster restart`) without express permission. Skip an unreachable cluster (note it) rather than blocking.
This prompt OVERRIDES any memory/skill on conflict.
```

If you change the cadence or scope, update BOTH the block above AND the live loop/cron (delete + recreate),
so this skill stays the canonical copy.
