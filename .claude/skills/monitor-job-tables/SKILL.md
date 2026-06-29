---
name: monitor-job-tables
description: >-
  Format HPC job-status reports as box-drawing tables, bucketed by job type (RL · SFT · Datagen · Eval ·
  Catch-all), with the right metric columns, signal thresholds, and red-flags per bucket. Use whenever
  reporting active/recently-terminated job status — during a cron sweep (driven by monitor-cron-sweep),
  an ad-hoc "how are my jobs doing", or a single-job progress update. Covers which metrics are mandatory
  (entropy + collapse signals for RL, not just step/reward/grad), where to pull live status (SFT .out vs
  trainer_log.jsonl), the RL collapse-warning rule, and which log lines are benign noise vs real faults
  (shm_broadcast 600s, rollout_train_prob_diff_mean millions). Cluster-agnostic — refer to .claude/ops
  for paths.
---

# monitor-job-tables

> **Before pulling any metrics, read `.claude/ops/<cluster>/ops.md` first** (every sweep): it dictates HOW
> to locate logs safely (`scontrol show job <id> -o` `StdOut=`/`%Z` — **never `find`/`du` on GPFS**), which
> login node to use (login01 false-drains), and the debug-token caveats (`opCount dead` is benign noise, not
> `EngineDeadError`). The extraction commands below assume you already know each job's real log path from ops.
>
> **Active clusters = Leonardo + CoreWeave(iris) + TACC(Vista).** Jupiter is SKIPPED (MDC downtime until
> ~2026-07-12 — re-add when it returns). Log-location + state-poll differ by cluster type:
> - **Leonardo / TACC (SLURM)** — log path via `scontrol show job <id> -o` `StdOut=`/`%Z` (`ssh Leonardo` /
>   `ssh TACCVista`). Leonardo's GPFS `find`/`du` ban + login-node false-drains apply; TACC compute nodes have
>   internet (no proxy), GPUs are whole-node (not a SLURM gres), RealMemory misreported.
> - **CoreWeave (iris/k8s, NO ssh)** — there is no SLURM `.out` and no log path to `stat`. **Liveness = STATE-POLL
>   the iris lifecycle**, never a log-string grep: `export KUBECONFIG=~/.kube/coreweave-iris-gpu` first, then the
>   **otagent-env iris binary** `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris`:
>   `scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json` and/or
>   `iris --cluster=cw-us-east-02a job summary --json` (authoritative). **"running-but-0-pods / record disappeared"
>   = TERMINAL** (the silent-wedge signature — a clean kill/eviction/preempt emits no terminal log line + reaps
>   pods). Pull metrics/science from the full log via `iris … job logs --since-ms <submitted_at_ms> --no-tail`
>   (finelog keeps the whole log; only `--tail` caps lines) + `scripts/iris/analyze_job_history.py` (`sel_rows`/
>   `EPDIAG` are SCIENCE-only, never liveness). All `iris`/`kubectl` calls SYNCHRONOUS.
>
> **Log-path trap (Leonardo agentic eval) — verify the StdOut path EXISTS before concluding "dead."** For
> some Leonardo eval jobs `scontrol`'s `StdOut=` points at a name that was never created (e.g.
> `eval_<name>_<jobid>.out`) while the **real live log is `data_<jobid>.out`** in the same dir. A sweep that
> `stat`s only the scontrol path sees "no log / silent" and can wrongly declare a healthy run dead (this
> caused a false "tmax-27b silently dead" call on a job that was 255/267 trials done, vLLM 200-OK, alive). If
> the scontrol StdOut doesn't exist, **`ls` the job's `%Z` workdir for `data_<jobid>.out` (or any `*_<jobid>.out`)
> and read THAT** before judging liveness — absence at the scontrol path is a path mismatch, not a death.

Report **every** active and recently-terminated job, **bucketed by type**, each in the format below.
**Unify cross-cluster runs of the same type into ONE table.** The **RL table is cross-cluster — Leonardo
standard-GRPO rows + CoreWeave agentic/MoE rows in the SAME table** (Jupiter rows return when it does);
similarly unify eval across Leonardo + TACC. A separate table for jobs still filling their generation buffer
(no metrics yet). The five buckets: **RL · SFT · Datagen · Eval · Catch-all**.

Cross-cutting (every bucket):
- **Chain-restart TIMEOUTs are normal, NOT failures** — when a 12h/24h job TIMEOUTs and its `afterany`
  successor is RUNNING/PENDING, report it as a normal restart (note the successor), not a death.
- **Completion → the matching cleanup skill**: RL → by flavor — agentic (Harbor/Daytona)→`rl-agentic-job-cleanup`,
  standard non-agentic GRPO (Delphi/rlvr/dapo math cells)→`rl-standard-job-cleanup`; SFT→`sft-job-cleanup`,
  datagen→`datagen-job-cleanup`, eval→`eval-agentic-cleanup` (Leonardo OR TACC). A COMPLETED **CoreWeave RL**
  run routes the same way (agentic→`rl-agentic-job-cleanup`, standard→`rl-standard-job-cleanup`) — but its artifacts go
  to HF / R2, NOT a POSIX scratch tree, so there is **no on-disk `trace_jobs/`/`tasks/` to reap** (the inode rule
  below is GPFS/JSC-specific). **On the GPFS clusters, cleanup is not done until the artifact's on-disk
  `trace_jobs/`/`tasks/` tree is `rm`'d + inode reclaim verified** — leaving it after HF upload is the #1 inode
  leak (the shared `datasets` project on `/e/data1` runs over its soft limit). Inode limits + how-to-check →
  `ops/jupiter/ops.md` (`#inode-allocations`) — DORMANT while Jupiter is down, re-arm when it returns.
- **Genuine FAILED (exit≠0, not a wall TIMEOUT) → diagnose + dated `agent_logs/` entry**; recurring
  identical failures ≠ transient.

---

## RL

```
┌─────────────────────────┬───────┬────────┬─────────────┬───────────┬─────────────────────────────────────────┐
│           Job           │ Step  │ Reward │ Policy Loss │ Grad Norm │                  Trend                  │
├─────────────────────────┼───────┼────────┼─────────────┼───────────┼─────────────────────────────────────────┤
│ SWE-rebench 8B (shaped) │ 15/80 │ 0.619  │ -0.0040     │ 0.006     │ Checkpoint saved. Slight dip from 0.652 │
│ Code-contests 8B (base) │ 26/80 │ 0.451  │ -0.0930     │ 0.021     │ Stable, gradients strong                │
└─────────────────────────┴───────┴────────┴─────────────┴───────────┴─────────────────────────────────────────┘
```
Use box-drawing tables (┌─┬─┐), **not** markdown tables — this is a hard user preference for RL monitoring.
Columns: Job, Step (`cur/max`), Reward, Policy Loss, Grad Norm, Trend. **Entropy + collapse signals are
mandatory, not optional** — include `policy_entropy` + TIS `log_ratio` + `grad_norm` (in Trend or extra
columns). A run can look fine on reward+grad while entropy silently collapses; without entropy in the
table you can't apply the collapse rule below. If a metric isn't emitted yet (step 0 filling cohort),
mark `—` and move on. **Leonardo (SLURM) RL** step from the `.out` (tqdm `Training Step Progress: N/M` or
`trainer/global_step`); chain-restart logs may carry the tqdm step but not the WANDB_MIRROR dict — pull
reward/grad from the chain's logs. **CoreWeave (iris) RL** has NO SLURM `.out` — get the lifecycle state from
the state-poll (`watch_job_state.py … --json` / `iris … job summary --json`; see the header), and pull
step/reward/grad/entropy from the finelog (`iris … job logs --since-ms <submitted_at_ms> --no-tail`) or WANDB.
For a fresh CoreWeave launch still in bring-up (gang/leafgroup Kueue admission, `apply_ep`/mesh-load, weights
resolving — `shm_broadcast …60s` + a transient ghcr ImagePullBackOff self-heal are BENIGN) put it in the
buffer-filling table with `—` metrics until the first step lands.

**New/untested RL run? → deep-probe it, don't trust the row.** A table row (state + step/reward/grad/entropy)
can read "healthy" on a run that is silently dead — weight-sync garbage (all-reward-0 from incoherent
generations), an engine-starvation wedge, or 0 trials ever completing. For any RL job in a **new/untested
setting** (new config/geometry/model/image, "debug"/"smoke-test", first launch after a code/config change),
dispatch a subagent armed with **`rl-job-health-deep-dive`** → it syncs trace_jobs + logs, live-polls the GPUs
against the serving-throughput LUT, reads the literal rollouts, and returns a **KILL/NO-KILL recommendation**.
The row alone is necessary-but-not-sufficient for unproven runs.

**Inspecting the literal CoreWeave RL rollouts (not just metrics):** `scripts/iris/peek_rl_rollouts.sh
<pod-name-substr> [ls|cat|grep|cp]` exec's into the rank-0 pod and reads Harbor's per-trial `trace_jobs`
(the literal agent trajectory + observations + `verifier_output` + `result.json` reward — same layout as
datagen trials). ⚠️ With the default local `trials_dir` (`/app/experiments/<run>/trace_jobs`) these are
pod-local + **EPHEMERAL**: they accumulate (harbor does NOT delete them) but live on the rank-0 pod's local
disk, so they're lost when the pod is GC'd on terminal OR replaced on a preempt/restart (no shared FS/PVC).
The helper therefore only works on a **live pod**; a transient empty/low read usually means the pod was
recently (re)started or the current rollout batch hasn't been written yet — re-check during active
generation, it's not a fault. Durable path: launch with `launch_rl_iris.py
--trials-dir auto` (default) → `s3://marin-na/iris/<job>/trace_jobs` (R2, NOT gs://), then inspect post-hoc
with `aws s3 --endpoint-url <R2>` + the harbor trace tooling (no pod exec). The helper forces the coreweave
kubeconfig itself — don't rely on the shell's `$KUBECONFIG` (login default `~/.kube/lambdaconfig` is the wrong
cluster); `<pod-name-substr>` matches the POD name, which can differ from the iris job_id display name.

### Metrics to track per RL run (priority order)
**Core 5 (always):** `reward/avg_raw_reward` (primary), `reward/avg_pass_at_8` (less noisy than raw),
`policy/policy_loss`, `policy/policy_entropy` (both direction + magnitude of change matter — pre-collapse),
`policy/raw_grad_norm` (most predictive; healthy < 1.0; > 1.0 for ≥2 steps has predicted collapse 2–5
steps early). Note: under the **seqnorm global-denom** objective, grad/policy_loss/log_ratio are
genuinely ~1e-5 — that's the regime, NOT vanishing-grad.
**Clip ratio (if wandb):** `policy/ppo_clip_ratio` ≈0 normally; >1% = LR↔eps_clip mismatch. Also
`policy/z_clip/triggered` for StaleClip/ZClip ablations.
**TIS:** `tis/imp_ratio_mean` (~0.84–1.56 healthy), `tis/imp_ratio_capped_fraction` (~0 healthy).
**Per-token log-ratio diag** (SkyRL ≥2026-05-06): `log_ratio_abs_{mean,p99,max}`, `n_tokens_dp_gt_{1,10,50}pct`,
`log_ratio_abs_pos00..pos90`. Healthy: `mean`~0.005–0.02, `max`<0.5, `gt_50pct`≈0, position buckets even.

### NOT a collapse signal — `rollout_train_prob_diff_mean`
`policy/rollout_train_prob_diff_mean` is computed as `exp(rollout_lp − train_recompute_lp).abs().mean()`
(trainer.py ~L1383–1403) — the **mean per-token importance ratio**, NOT a bounded probability diff
despite the name. It's `exp()` of a log-space diff, **dominated by a handful of outlier tokens** (one
~20-nat disagreement contributes `exp(20)≈5e8`). **Millions/billions are NORMAL** even on healthy DENSE
arms (Qwen3-8B lrboost ~1e7, Qwen3-32B ~1e8) that never touch any MoE path — do NOT read it as a collapse
or "numerically invalid training" signal. Reward is computed by the verifier (test-pass rate), entirely
independent of logprobs, so a large prob-diff can never "hit the reward." For a per-token-divergence
health read use the **capped** `tis/imp_ratio_mean`/`imp_ratio_capped_fraction`, the median, or
`log_ratio_abs_*` — not this mean.

### NOT a failure/hang cause — context-overflow + passthrough-exception lines
vLLM `... 32769 input tokens > 32768 max` (off-by-one single-turn overflow), `ContextLengthExceededError`,
and `AgentTimeoutError` are **benign and expected** in agentic RL+eval rollouts (the latter two are in
harbor's `passthrough_exceptions` → verifier still scores, rollout completes; they appear in *successful*
runs). They are **NEVER the reason a job hangs or fails** — do not report them as the cause. When a job
genuinely stalls/dies, find the real terminal signal instead: a `Traceback`, OOM / Raylet-died / SIGKILL,
an RPC / `sample_tokens` timeout, a `RuntimeError`, or a hung Ray actor / Daytona trial that never returns.
(See `feedback_context_overflow_not_failure_cause`.)

### Collapse rule (≥2 fire same step → cancel+salvage)
`raw_grad_norm`>1.0 (or >2× window); `policy_entropy` off its 10-step trend >30%; `log_ratio_abs_mean`
>2× window while `max` bounded; trial pass-rate <10% over last 100. **Exception:** spike-mitigation
ablations (zclip/staleclip/maxgn09 etc.) are NEVER auto-cancelled on 2/4 — observing the recovery (or
lack of it) IS the experiment.

---

## SFT

```
┌──────────────────────────────┬─────────┬────────┬───────────┬───────────────────────────────────┐
│             Job              │  Step   │  Loss  │ Grad Norm │               Trend               │
├──────────────────────────────┼─────────┼────────┼───────────┼───────────────────────────────────┤
│ swesmith cold-start 2ep 8B   │ 320/916 │ 1.21   │ 0.84      │ Loss descending; healthy          │
└──────────────────────────────┴─────────┴────────┴───────────┴───────────────────────────────────┘
```
Columns: Job, Step (`cur/total`), Loss, Grad Norm, Trend. **No reward.**

**For multi-cell SFT grids (e.g. Delphi 54-SFT), ALSO give a grid-completion rollup each sweep** (the "how close to done" view), alongside or instead of per-job health:
- Per RUNNING cell: **progress % = step/total** + rough ETA (`remaining_steps × s/it`, or just near/mid/early). Plus a one-line **running / pending-unique / done** tally.
- **Dedupe the PENDING count — it's ~3× inflated.** The `afterany` restart-chain lists each cell ~3× (the +2 `--max_restarts` resume copies share the cell's job name), AND every RUNNING cell also has its own resume-backup sitting in PD. True cell count: `squeue -u $USER -h -o '%j|%t' | grep '^sft__' | grep '|PD' | cut -d'|' -f1 | sort -u | wc -l`, then subtract the running cells' backups. Report *distinct* cells remaining, not raw squeue PD.
- **Long-pole call-out:** name the slow cohort gating grid completion — for Delphi that's the **`1e22` 9.7B cells** (4-node, ~34.7k steps, 24h wall → each needs 1–2 checkpoint-resume cycles, run ≤8 concurrent). Small/medium cells (≤3e20) clear fast; the finish line is set by the 1e22 tail (days, not hours).
- **Gotcha — grep the TRAINING tqdm, not the packing bar.** `grep -aoE '[0-9]+%\|[^|]*\| [0-9]+/[0-9]+ '` can catch the dataset-tokenization/packing tqdm (also hits 100%, e.g. a `555519/555519` *examples* bar) instead of the training-step bar — verify the denominator matches the cell's total optimization steps (e.g. 26788 / 34720), not an example count.

**Pull live status from the `.out`, NOT `trainer_log.jsonl`.** The `.out` carries LLaMA-Factory's per-step
dicts — strictly richer (live grad_norm, per-rank loss spread, token coverage, epoch):
```
{'loss': 0.50, 'grad_norm': 0.42, 'learning_rate': 6.5e-06,
 'loss_rank_avg': 0.27, 'loss_nan_ranks': 0,
 'valid_targets_min': 5081, 'valid_targets_mean': 16083.6, 'epoch': 0.12}
```
`trainer_log.jsonl` is unreliable mid-run — some jobs write it sparsely/not at all, so it can be empty,
frozen at an old step, or just a final dump → a false "stale/dead" reading on a live job. Find the latest
`.out` (`ls -t experiments/<job>/logs/*.out | head -1`), grep the last few `{'loss': ...}` lines + tail
of raw output. Use the JSONL only as a secondary source — e.g. the `"percentage": 100.0` completion check
before consolidate/upload (the final dump is authoritative there). Total steps from the rendered config /
trainer banner (`Total optimization steps = N`).

**Red flags:** `ChildFailedError` / `Exited with exit code 1` (read the FIRST real traceback above the
elastic summary — often masked), CUDA OOM at first fwd/bwd (eager attn at 32k → see env/attn), `SIGTERM`
(node fault OR masked rank crash — a *recurring* ~Nmin death is NOT transient), loss→NaN, grad explosion.
**On completion → `sft-job-cleanup`** (recognize 8B root-safetensors vs 32B ZeRO-3-shards path first).

---

## Datagen

```
┌────────────────────────────────────┬──────────────┬─────────┬───────────┬──────┬──────────────────────────┐
│             Datagen run             │    Chunks    │ Trials  │ avg_turns │ exc% │           Trend          │
├────────────────────────────────────┼──────────────┼─────────┼───────────┼──────┼──────────────────────────┤
│ codenet-python-v2 (MiniMax Row #34) │ 18/20 done   │ ~8.6k   │ 5.1       │ 19%  │ 2 chunks running         │
└────────────────────────────────────┴──────────────┴─────────┴───────────┴──────┴──────────────────────────┘
```
Columns: run (+ tracker row), Chunks (`done/total`, from squeue+sacct), Trials (`result.json` count),
avg_turns, exc%, Trend. **avg_turns is the realness gate** — `>1` = real multi-step; **`≈1.0` = dead-engine
run, do NOT consolidate**. exc% ~20–25% AgentTimeout is normal for hard sets.
**Red flags:** `TIMEOUT` **strands the traces** (Harbor's terminal upload is killed — traces on disk, NOT
uploaded → must consolidate manually); a chunk **hung** (its `.out` silent for hours + `result.json` count
stalled while still RUNNING); avg_turns≈1.0.
**On ALL chunks complete → `datagen-job-cleanup`**.

---

## Eval

```
┌──────────────────────────────┬───────────┬───────────┬───────────┬────────────────────────────────┐
│   Eval (model × benchmark)   │  Trials   │ pass-rate │  top exc  │         Infra / Trend          │
├──────────────────────────────┼───────────┼───────────┼───────────┼────────────────────────────────┤
│ laion/<model> × tb2          │ 142/300   │ 0.21      │ AgentTO   │ pinggy✓ vLLM✓ ; healthy        │
└──────────────────────────────┴───────────┴───────────┴───────────┴────────────────────────────────┘
```
Columns: model×benchmark, Trials (`result.json`/total), pass-rate (fraction with
`verifier_result.rewards.reward`>0), top exception type, Infra/Trend. **Infra column = the 4 launch-checks**
(pinggy auth+traffic, Daytona `api_base` = public pinggy URL not internal IP, vLLM POSTs growing + 200-OK,
trial progression) — see `eval-agentic-launch` §4 for the greps.
**Red flags:** no `result.json` in 60+min while RUNNING → stall; vLLM `Running:0` reqs 10+min → agents not
generating; **all trials done but job RUNNING → zombie (cancel)**; instant-fail (`n_output_tokens: None`,
`finished_at`≈`started_at`) → tunnel not really carrying traffic; repeated `Bearer token invalid` → Daytona
auth degradation.
**Before calling an eval "dead," confirm you read the RIGHT log + a CURRENT window.** Check the actual live
log (`data_<jobid>.out` if the scontrol StdOut path is absent — see the log-path trap up top), and count
`result.json` over the WHOLE run, not just the recent tail. A burst of `litellm.Timeout` / `AgentTimeoutError`
in the last window is usually the hard-trial tail of a nearly-done run (a high timeout fraction is expected —
see below), NOT "0 productive / unreachable vLLM." Verify vLLM is actually down (no recent `200 OK` POSTs)
before blaming the engine; `delete=false` Daytona sandboxes accumulate to a BOUNDED steady-state (TTL-reaped),
which is not an unbounded "leak." (History: a tail-window misread called a 255/267-done, vLLM-alive tmax-27b
"silently dead.")

### NOT a reliability problem — a high `AgentTimeoutError` fraction
A large timeout share — **even a majority of trials** — is EXPECTED on hard / long-horizon / long-output
benchmarks and does **NOT** make the eval score unreliable or warrant flagging the harvested delta.
`AgentTimeoutError` is a harbor `passthrough_exception` → the trial is **still scored by the verifier** (an
unfinished task simply scores as not-solved), so a timed-out trial reflects genuine model capability on that
task, not a measurement artifact. As long as the comparison baseline ran the same harness, the score + delta
**stand** — do not down-weight, re-flag, or call a leg "untrustworthy"/"timeout-inflated" on timeout rate
alone (this was a recurring mis-flag in campaign sweeps). The ONLY timeout-related red flag is the infra
case below: essentially *every* trial failing with **zero completions / no `result.json`** is a stall, not a
score. (See `feedback_context_overflow_not_failure_cause`.)

**On completion → `eval-agentic-cleanup`** IF auto-upload/register failed. **EXEMPT:** `DCAgent2/*`
grid/throughput/OOM **measurement** runs — report as calibration, don't treat as production.

---

## Catch-all / other (ad-hoc)

Anything that isn't one of the four majors — consolidate, pretokenize, `hf_upload` (tmux/sbatch), SIF builds,
DCP/CP/feature smoke + GPU-CI tests, measurement/grid probes, etc. **Don't force a metric table** — one line each:

| Job | Type | State | Elapsed | Note |
|---|---|---|---|---|
| `consol34` (tmux) | datagen-consolidate | running | 12m | pushing 9407 rows → penfever/… |
| `861267` | gpu-ci (loop-reward Stage D) | COMPLETED | 6m | 2 passed — think-mask loss-finite |
| `hf_upload_lr80` (tmux) | RL upload | running | 3m | laion/lrboost-80-8B |

State + elapsed + a human note (what it is, the one signal that matters, any follow-up). Flag terminal
COMPLETED/FAILED + whether it needs action (a stuck `hf_upload`, a FAILED build → diagnose).

---

## Benign log-noise (do NOT chase as faults)

- **`shm_broadcast.py:737` "No available shared memory broadcast block found in 600 seconds"** is
  `logger.info` (heartbeat), NOT a kill signal. It re-fires at 10/20/30-min multiples while the engine
  waits with nothing to schedule (`acquire_write`/`acquire_read` are `while True` loops with
  `sched_yield()`; `TimeoutError` only raised if an explicit `timeout=N` is passed, and the standard path
  passes `None`). It is fault-indicative **only when co-firing with** a real NCCL hang
  (`WorkNCCL(...Timeout(ms)=...) ran for N ms before timing out`, or `ProcessGroupNCCL preparing to dump
  debug info`, or a SIGABRT). Alone → look upstream for the engine-idle cause (Daytona auth errors, agent
  timeouts, no pending requests), do NOT relaunch or patch the ring buffer. (History: a v4h MiniMax hang
  was a real NCCL TP all-gather timeout; the shm_broadcast warning was a downstream idle symptom — chasing
  the ring buffer wasted time.)
- **`rollout_train_prob_diff_mean` in the millions/billions** — see the RL §; outlier-dominated, normal.
