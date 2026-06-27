---
name: monitor-rl-job-health
description: >-
  Deep single-RL-job health probe → a KILL / NO-KILL recommendation for the supervisor. Dispatched as a
  subagent on every monitor tick for RL jobs in NEW/UNTESTED settings (new config/geometry/model, "debug" or
  "smoke-test" flavor, first launches after a code/config change). Goes BEYOND state-poll + the table metrics:
  syncs the job's trace_jobs + stderr/stdout + Ray logs to ~/Documents/experiments/traces via the EXISTING
  capture tool (CoreWeave: scripts/iris/peek_rl_rollouts.sh `pull`), then runs four gates — (1) liveness
  (tail stdout/stderr: zombie/wedged/dead?), (2) resource utilization (live-poll GPUs: all inference engines
  alive + generating at a hardware/model-size-reasonable cadence per the serving LUT; training stage not
  VRAM/RAM-OOM), (3) rollout quality (trace_jobs: trials init/complete, non-zero rewards, turns completing,
  agent outputs sane, tasks hard, verifiers firing), and emits ONE verdict with evidence + next steps. The
  subagent NEVER kills — it recommends; the supervisor owns the kill (standing guardrail). Cluster-agnostic;
  defers access/hardware/log-path particulars to .claude/ops/<cluster>/ and dependency facts to
  .claude/projects/<dep>/. Reference: scripts/iris/peek_rl_rollouts.sh, scripts/iris/watch_job_state.py,
  .claude/ops/iris/coreweave_gpu_ops.md, .claude/skills/rl-agentic-launch-iris §8.
---

# monitor-rl-job-health

A **deep, single-RL-job** health probe that ends in **one recommendation to the supervisor: `KILL` or
`NO-KILL`**, backed by hard evidence and next steps. This is the heavier, per-job complement to the breadth
sweep: `monitor-cron-sweep` looks at *every* job briefly; **this skill looks at ONE RL job hard** and is the
right tool when a run is in a **new or untested setting** (new config / geometry / model / image, a "debug" or
"smoke-test" launch, or the first launch after a code/config change) where state-poll + table metrics are
**necessary but not sufficient** to tell "genuinely progressing" from "silently dead."

> **You are a SUBAGENT producing a recommendation — you do NOT execute the kill.** Standing guardrail
> (`supervisor-init`, every launch skill): **never kill a RUNNING job without explicit permission.** Your
> deliverable is the `KILL`/`NO-KILL` verdict + reasoning + recommended next steps. The supervisor decides and,
> if KILL, runs `iris job kill …` / `scancel …` themselves. The ONLY exception is the supervisor's own standing
> autonomy over **our own deterministically-doomed / wedged** jobs — and even then the *decision* is theirs, on
> your evidence. **When in doubt, recommend NO-KILL + escalate** (a wrongly-killed healthy run wastes a whole
> bring-up; a wrongly-kept dead one wastes one more sweep — asymmetric).

## Resources you must use (don't re-derive)

- **`.claude/ops/<cluster>/`** — *machine/cluster particulars*: access (kubeconfig/ssh), log-path discovery,
  GPU-poll mechanics, gpu-mem ceilings, the binding gotchas. **CoreWeave** → `ops/iris/coreweave_gpu_ops.md`
  (+ `coreweave_h100_cloud_hardware.md` for the node shape); **Leonardo** → `ops/leonardo/ops.md`; **TACC** →
  `ops/tacc/ops.md`. **Read the relevant one FIRST** — it tells you how to reach the job and read its logs
  safely (GPFS `find`/`du` ban, login-node false-drain, the kubeconfig export, the otagent iris binary).
- **`.claude/projects/<dep>/`** — *what each codebase is + its facts/gotchas*: `marinskyrl/` (the SkyRL/GRPO
  trainer — log-line vocabulary, sel_rows/EPDIAG, weight-sync), `vllm/` (the serve engine — the fork's MoE/DCP/R3
  flags, throughput knobs, enforce_eager), `harbor/` (the rollout/trial layout + `passthrough_exceptions`),
  `daytona/` (sandbox/reward-0 failure modes). Use these to read the logs correctly, not by guesswork.
- **`scripts/iris/peek_rl_rollouts.sh`** — the EXISTING capture tool for CoreWeave (§1). Do NOT hand-roll an
  ad-hoc R2/kubectl pull.
- **`scripts/iris/watch_job_state.py`** — the authoritative CoreWeave lifecycle state-poll (§2).
- **`rl-agentic-launch-iris` §8** — the per-rung CoreWeave bring-up ladder this skill operationalizes; cite it
  rather than restating every milestone string.

---

## 0. Inputs + setup (gather these first)

You need, from the dispatching supervisor (ask only if genuinely missing — most are derivable):

| Input | How to get it | Used for |
|---|---|---|
| **Cluster** | from the dispatch | which `ops/<cluster>` to read + which capture/poll path |
| **Job id** | from the dispatch (`/benjaminfeuer/<job>` on CoreWeave; SLURM jobid on Leonardo/TACC) | state-poll, log path |
| **Pod-name substring** (CoreWeave) | the `--job-name` slug (pod name = `iris-benjaminfeuer-<slug>-<rank>-<hash>-<gen>`) | `peek_rl_rollouts.sh` + `kubectl exec` |
| **Model + size (B, dense vs MoE active-B)** | from the config / `--model_path` | the serving-throughput LUT (§3) |
| **Stage** (bring-up / inference / training) | from the logs you capture in §1–2 | which §3 check applies |
| **What's "new/untested"** | from the dispatch (e.g. "TP=2+EP=2 first run", "R3+DCP unvalidated", "post-weight-sync-fix") | what to scrutinize hardest in §4 |

**Environment (CoreWeave example — adapt per `ops/<cluster>`):**
```bash
source /Users/benjaminfeuer/Documents/secrets.env          # HF/WANDB/DAYTONA (+ R2 creds injected pod-side)
export KUBECONFIG=~/.kube/coreweave-iris-gpu                # HARD prereq — Mac default points at the WRONG cluster
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python  # otagent env (symlinks fail in the sandbox)
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris  # NOT the marin .venv iris (broken kubernetes import)
```
All `iris`/`kubectl` calls are **SYNCHRONOUS — never background them** (`ops/iris/coreweave_gpu_ops.md`).

**Restart-burn check (do this EARLY — it sets the syncdown scope in §1 and feeds the verdict).** Before anything
else, find out whether this job has already **burned a restart/retry** — because (a) a run that has *already
failed once* is a sharply different risk than a clean first attempt, (b) the prior-attempt failure is
high-signal for the verdict, and (c) the **remaining** retry budget bounds urgency (a run on its last retry
that's misbehaving is closer to a KILL than one with headroom). Cheap signals (no heavy capture yet):

```bash
# CoreWeave: failure_count + per-task retry state (authoritative), and the pod GENERATION suffix.
$PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json     # → failure_count (retries burned so far)
$IRIS --cluster=cw-us-east-02a job summary /benjaminfeuer/<job> --json     # per-task state + restart/generation
# pod name = iris-benjaminfeuer-<slug>-<rank>-<hash>-<GEN>; GEN -0 = first attempt, -1/-2 = after a re-bring-up.
kubectl get pods -n iris -o name | grep "<slug>" | sed -E 's/.*-([0-9]+)$/gen \1/' | sort -u   # max GEN > 0 ⇒ restarts burned
```
Compare burned-count to the launch's **`--max-retries K`** (we launch with `K=1`) → `remaining = K − burned`.
**SLURM (Leonardo/TACC):** the restarts are the **`afterany` chain-restart legs** — `sacct -u <user> --name <jobname>
-X --format=JobID,State,Start,End,ExitCode` lists every chain link; each prior **terminal (FAILED/TIMEOUT/CANCELLED)**
leg before the running one is a burned attempt (a clean TIMEOUT→successor is a *normal* chain restart, NOT a burn —
distinguish: a wall-clock TIMEOUT with a healthy successor ≠ a crash-and-retry). `scontrol show job <id>` `Restarts=`
also counts requeues.

Record **`restarts burned = B / max K` (remaining = K−B)** and, if `B>0`, the **terminal state + exit/error of each
prior attempt** (from the summary/sacct). If `B>0`, set the syncdown in §1 to **include the failed runs** (next
section), and carry `B`, the remaining budget, and "same failure each attempt?" into Gate A (§2) and the verdict
(§5): the same crash repeating across every restart = **deterministically doomed → KILL**; a restart burned on a
genuine transient (e.g. an HF-weight-resolution flake the retry-wrapper now catches) that is now healthy = benign.

---

## 1. Sync the artifacts (use the existing tool — never ad-hoc)

Pull **trace_jobs + stderr/stdout + (if available) the Ray logs** into the canonical capture dir
**`~/Documents/experiments/traces/`**. Use the dependency's own capture tool — do not hand-roll a kubectl/R2
sync (the tool already handles per-object size-verify, the latest-generation pod, the REMOTE-R2-vs-node-local
trials_dir split, and a provenance MANIFEST).

**CoreWeave:**
```bash
IRIS_BIN=$IRIS bash scripts/iris/peek_rl_rollouts.sh <pod-name-substring> pull
#   → ~/Documents/experiments/traces/<slug>_<UTC-stamp>/
#       logs/iris_finelog.log      (complete iris finelog, --no-tail — the full bring-up + driver log)
#       logs/pod_rank*.log         (per-rank container stdout; rank0 = Harbor coordinator + driver; 1..N-1 = Ray workers)
#       trace_jobs/<trial>/…       (ALL Harbor trials synced from R2: config/prompt/conversation + result.json reward)
#       MANIFEST.md                (provenance: job id, rank-0 pod, trials_dir, started/completed counts)
```
The per-rank `pod_rank*.log` files **are** the engine/Ray stderr+stdout (k8s merges container stdout/stderr).
For deeper Ray actor logs, `kubectl exec -n iris <pod> -c task -- bash -lc 'ls /tmp/ray/session_latest/logs'`
and pull any `worker-*.err` / `raylet.*` of interest (the finelog usually surfaces the real traceback first).

> **If the §0 restart-burn check found `B>0` burned restarts → the syncdown MUST include the FAILED runs, not
> just the live generation.** The default `pull` captures the *current* (latest-generation) pods' logs + the
> shared trials — but the prior attempts are exactly the evidence you need. Capture them:
> - **iris finelog spans ALL generations** — it is the durable record of every burned attempt's bring-up +
>   crash (the GC'd prior-generation pods' stdout is gone, but the finelog kept it). `peek … pull` already
>   pulls it `--no-tail` full-history; **confirm it covers the original submission** (if it was tailed, re-pull
>   with `$IRIS … job logs /benjaminfeuer/<job> --since-ms <ORIGINAL submitted_at_ms> --no-tail`, using the
>   *first* submission time, not the latest generation's). Then **grep it for each prior attempt's terminal
>   traceback** — the real reason each restart was burned (and whether it's the SAME failure every time).
> - **The REMOTE R2 trials_dir is shared across generations** (`s3://marin-na/iris/<job>/trace_jobs`), so
>   `pull` already grabbed failed-generation trials too — segregate/label them by episode/timestamp so the
>   §4 rollout read doesn't conflate a dead generation's reward-0 trials with the live one's.
> - **SLURM:** the failed legs are *separate job IDs* in the `afterany` chain — `rsync` the **`.out`/`.err` of
>   each prior terminal leg** (its own jobid, located via `scontrol`/`sacct` per §0) into the capture dir,
>   alongside the running leg's. Don't sync only the live leg.

**SLURM (Leonardo/TACC):** there is no `peek` tool — the equivalent is: locate the `.out`/`.err` via
`scontrol show job <id> -o` (`StdOut=`/`StdErr=`/`%Z` workdir — **never `find`/`du` on GPFS**, per
`ops/<cluster>`), then `rsync`/`scp` the `.out`, `.err`, and the run's `trace_jobs/` tree into
`~/Documents/experiments/traces/<job>_<stamp>/`. (Most "new/untested" RL right now is CoreWeave; the four
gates below are cluster-agnostic once you have logs + trials + a GPU view.)

> **If `pull` returns 0 trials / no logs:** that is itself a signal, not a tool failure — usually the rank-0
> pod is gone (terminal job → node-local trials GC'd) or was just (re)started. Cross-check with §2's state-poll
> before concluding; a fresh launch legitimately has 0 completed trials for a while (long episodes).

---

## 2. Gate A — Liveness (is it alive, or zombie/wedged/dead?)

**Liveness = authoritative STATE-POLL + a log-freshness read — NEVER a single log-string grep** (a clean
kill/eviction/preempt emits no terminal string and reaps pods, so a content-watch sits idle while the job is
gone — the exact miss `rl-agentic-launch-iris` §8 warns about).

```bash
# CoreWeave: authoritative lifecycle (state + per-task + pod cross-check; "no record AND 0 pods" = TERMINAL absent)
$PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json
# SLURM: squeue -u <user> -t RUNNING ; sacct -j <id> -X --format=State,ExitCode  (validate vs false-drain — ops/<cluster>)
```
Then read the captured logs (§1) for **freshness + wedge signatures**:
- **Log freshness.** Compare the newest meaningful log line's timestamp to "now" against the run's expected
  cadence (a generating engine prints `Avg generation throughput …` every few seconds; a training step lands
  on the order of minutes). **Materially-stale-beyond-cadence = suspected wedge** (engine deadlock, NCCL
  stall, generation-buffer starvation) — a job can hold its whole gang for hours while hung and still read
  `running, pods=N`.
- **Wedge / death signatures** (grep the finelog + `pod_rank*.log`): a real `Traceback`, `CUDA out of memory`
  / `OutOfMemoryError`, `RayActorError` / `Raylet … died` / worker SIGKILL/SIGABRT, an NCCL hang
  (`WorkNCCL(...Timeout(ms)=...) ran for N ms before timing out`, `ProcessGroupNCCL preparing to dump debug
  info`), an RPC / `sample_tokens` / `execute_model` timeout, `EngineDeadError`.
- **Benign noise — do NOT read as death** (per `marinskyrl`/`vllm` projects + `monitor-job-tables`):
  `shm_broadcast … No available shared memory broadcast block found in 60/600s` (engine idle-waiting — heartbeat,
  not a kill), a transient `ghcr.io` blob EOF → `ImagePullBackOff` self-heal, the `Unknown vLLM environment
  variable VLLM_ALLOW_ROUTED_EXPERTS_DCP` whitelist note, `… 32769 input tokens > 32768 max` /
  `ContextLengthExceededError` / `AgentTimeoutError` (harbor passthrough — appear in *healthy* runs),
  `rollout_train_prob_diff_mean` in the millions (outlier-dominated, normal). `opCount dead` is benign debug-token
  noise, not `EngineDeadError`.

**Gate A verdict:**
- **DEAD / TERMINAL** (state-poll says failed/killed/absent, or 0 pods) → there's nothing to kill; report
  TERMINAL + the root-cause traceback + whether it's transient (relaunch-worthy) or deterministic.
- **WEDGED** (RUNNING/pods present but a real hang signature + stale-beyond-cadence logs, no benign explanation)
  → **lean KILL** (it's burning a multi-node gang doing nothing). Carry the evidence to §5.
- **ALIVE + fresh logs** → proceed to Gate B.

---

## 3. Gate B — Resource utilization (are the GPUs actually working?)

**Live-poll the GPUs** (don't infer from logs alone). CoreWeave — exec `nvidia-smi` on **every** rank:
```bash
for p in $(kubectl get pods -n iris -o name | grep "<job>" | sed 's#pod/##'); do
  echo "== $p =="
  kubectl exec -n iris "$p" -c task -- nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
done
```
(SLURM: `srun --jobid=<id> --overlap -w <node> nvidia-smi …`, or read the `.out`'s periodic util/mem prints —
see `ops/<cluster>`.) Interpret by **stage** (which stage the §1–2 logs put you in):

### B-inference (rollout/generation stage)
**Every inference engine must be alive, fed, and generating at a cadence reasonable for the hardware + model
size.** Concretely:
1. **All engines present + busy.** Engine count = `num_inference_engines × TP` GPUs should show **non-trivial
   `utilization.gpu`** (decode is bursty, but a generating engine is not pinned at 0%). A GPU at **0% util with
   memory resident** across several polls = an engine that loaded weights but is **not generating** (starved /
   wedged / never received requests).
2. **vLLM actually emitting tokens.** In the logs: `Avg generation throughput: >0 tokens/s, Running: R reqs,
   Waiting: W` recurring on **each** engine. This is the literal "is vLLM firing" check.
3. **Throughput is hardware-reasonable** — compare the aggregate `tokens/s` (summed across an engine's `Running`
   reqs) to the **serving LUT below**. Below-floor throughput while `Running > 0` is a red flag (eager mode,
   bad geometry, oversubscription thrash, or a degraded weight-sync producing pathological outputs).
4. **Queue is draining, not thrashing.** `Waiting ≈ 0` (or falling) = healthy. **`Waiting` persistently ≫
   `Running` with flat/low throughput = the throughput-starvation WEDGE** (oversubscribed: lower
   `n_concurrent_trials` / raise `max_num_seqs`) — this was the original `rl-131k-cpdcp2r3` silent death.

> **Serving-throughput LUT — H100-80GB SXM, vLLM, batched continuous-decode (the RL rollout regime).** These
> are **order-of-magnitude health FLOORS, not benchmarks** — the diagnostic is the SHAPE (throughput > 0,
> scales with `Running`, `Waiting ≈ 0`), not hitting a number. Decode is HBM-bandwidth-bound (H100 ≈ 3.35 TB/s);
> single-stream ≈ `HBM_BW / (2 × active_param_bytes)`; aggregate rises with batch until compute-bound.

| Model (active params) | Single-stream decode | **Healthy aggregate / engine** (dozens of `Running`) | "Awful" — investigate |
|---|---|---|---|
| Dense 7–8B | ~90–140 tok/s | **~2,000–6,000 tok/s** | < ~500 tok/s w/ many Running |
| Dense 14B | ~60–100 tok/s | ~1,500–4,000 tok/s | < ~400 |
| Dense 32B | ~30–55 tok/s | ~800–2,000 tok/s | < ~200 |
| **MoE 30B-A3B / 35B-A3B (~3–3.5B active)** | ~80–160 tok/s | **~2,000–6,000 tok/s** (low active → high) | **< ~500 tok/s** |
| MoE ~235B-A22B (~22B active) | ~25–45 tok/s | ~700–1,800 tok/s | < ~200 |

> **`enforce_eager: true` (CUDA graphs OFF) divides decode by ~3–10×** (kernel-launch-bound at small decode
> batch) — this is exactly why the CoreWeave MoE arms read **15–75 tok/s** and the supervisor called it
> "awful": the fix was `generator.enforce_eager: false` (CUDA graphs) in the iris config, NOT a serve-geometry
> change. **So: if you measure floor-or-below throughput, FIRST check `enforce_eager` in the resolved config**
> before blaming weights/geometry. Long context (131k) lowers steady-state tok/s (KV pressure) but should not
> floor it. TP/EP geometry: more, smaller engines (e.g. TP=2+EP=2 ×16) should *raise* aggregate cluster
> throughput vs few big TP=8 engines — if it doesn't, suspect the EP/R3 path.

### B-training (optimizer/update stage)
- **Not VRAM-OOM:** no `CUDA out of memory` / `OutOfMemoryError` in logs; `memory.used` not pinned at
  `memory.total` (≈80 GB) **while progress is stalled**. (Transient near-ceiling at peak activation is normal
  *if steps advance*.)
- **Not RAM/host-OOM:** no `OOMKilled` in pod state
  (`kubectl get pod <p> -o jsonpath='{.status.containerStatuses[*].lastState.terminated.reason}'`), no
  oom-killer / `Killed` in the rank logs, no repeated pod restarts.
- **GPUs actually computing:** during a step the policy/ref ranks show **high `utilization.gpu` + power draw**
  (a forward/backward is compute-bound), not 0%. All-0% with no log advance during "training" = wedge.

**Gate B verdict:** an engine resident-but-0%/no-tokens, throughput floored-or-zero with `Running>0` and
`enforce_eager:false`, `Waiting≫Running` flat, or a training-stage OOM / all-ranks-0%-no-progress → **lean
KILL** (carry evidence to §5). All engines generating ≥ floor with `Waiting≈0`, or training steps advancing
without OOM → **healthy on Gate B**, proceed to Gate C.

---

## 4. Gate C — Rollout quality (read the actual trace_jobs; use judgment)

State + GPUs can be green while the run produces **garbage** (e.g. a degraded FSDP→vLLM weight-sync serving a
policy that emits token-salad → 100% reward-0 → no learning signal). So **read the literal rollouts** under
`~/Documents/experiments/traces/<slug>_<stamp>/trace_jobs/`. (`peek_rl_rollouts.sh <substr> cat <trial-dir>` /
`grep <regex>` also read R2 directly for spot checks.) Work through, **using your best judgment** — this gate is
qualitative:

1. **Are trials INITIALIZING?** trial dirs being created (config/prompt present). Zero new trial dirs while
   engines generate ⇒ the Harbor RolloutCoordinator isn't dispatching (look for `TerminalBenchGenerator
   initialized … Concurrent trials: K` in the log).
2. **Are trials COMPLETING?** count `result.json` (the completed-trial marker carrying the reward). At +15/30
   min a 131k arm legitimately has **ZERO** completed (long episodes) — report that as *"rollouts executing, 0
   completed yet,"* **never** as "healthy/done." Completing at a steady rate ⇒ the loop is closing.
3. **Are any rewards NON-ZERO?** `grep result.json` for `"reward"`. **All-zero / all-timeout** is the headline
   failure mode — then ask *why*:
   - **Agent output is incoherent** (token-salad, repetition loops, wrong-language, empty) ⇒ **serving/weight-sync
     or geometry fault** (the FSDP→vLLM sync or the vLLM-fork build for this model/geometry). This is a **KILL**
     signal on a new/untested geometry — the policy being served is not the trained policy. (Check the
     tokenizer is right first — but a *correct* tokenizer + salad output = sync/build, per the prior CoreWeave
     diagnosis.) **Known CoreWeave MoE cause to check first:** the FusedMoE `w13` gate/up swap not re-applied
     on the disaggregated RL update (H100/FlashInfer-CUTLASS) → confirm `SKYRL_W13_RELOAD_BRACKET` is on
     (default 1) and the engine log shows `finish_weight_reload` (fix MarinSkyRL `2bb70a88`; marinskyrl doc).
   - **Agent output is COHERENT but wrong / runs out of turns** ⇒ tasks genuinely hard or the harness/verifier
     mis-set — NOT necessarily a kill; this can be a real (if low) learning signal. Read several conversations.
   - **Every trial ends in an environment/infra exception** (`VerificationNotCompletedError` everywhere =
     Daytona sandbox never came up / `DAYTONA_*` not forwarded; `Bearer token invalid` = auth) ⇒ infra, not
     model — **KILL + fix the infra**, relaunch.
4. **Are TURNS completing?** in `conversation`, count `role=="assistant"` turns. `avg_turns ≈ 1` (agent makes
   one move then stops/errors) is the dead-engine / broken-loop signature; multi-turn = real agent behavior.
5. **Are the AGENT OUTPUTS REASONABLE?** actually read 3–5 trajectories: is the model issuing sensible tool
   calls / code, or looping / emitting garbage / ignoring the task? (The task framing for the active CoreWeave
   arms — `pymethods2test` etc. — is simple code-contract Python; it should NOT need massive context, so a
   context-overflow storm on those is a red flag, not task difficulty.)
6. **Are the TASKS too hard / the VERIFIERS failing?** sample `verifier_output`: is it scoring a genuine
   attempt as fail, or erroring/timing out itself (`VerifierTimeoutError`)? A verifier that never returns a
   real score ⇒ no learning signal even with good generations.

**Gate C verdict:** incoherent generations / all-reward-0 from a serving-or-sync fault on a new geometry, or a
verifier/infra path that yields **zero learning signal** with no transient explanation → **lean KILL** (it
cannot learn in this state). Trials completing with *some* non-zero rewards (or coherent multi-turn attempts on
genuinely-hard tasks even at low pass-rate) → **NO-KILL, healthy/learning**.

---

## 5. Deliver ONE recommendation (the whole point)

Emit exactly one verdict to the supervisor, in this shape:

```
RL-JOB-HEALTH — /benjaminfeuer/<job>  (<model>, <geometry>, <stage>)   captured: <traces dir>

VERDICT: KILL | NO-KILL          confidence: high|medium|low
Restarts: <B/K burned, remaining K−B> — <none | same failure each attempt: … | transient, recovered>

Gate A (liveness):   PASS|FAIL — <state-poll verdict + log-freshness + any wedge/death signature>
Gate B (resources):  PASS|FAIL — <per-engine util/mem; aggregate tok/s vs LUT floor; Waiting/Running; enforce_eager; OOM?>
Gate C (rollouts):   PASS|FAIL — <trials started/completed; reward distribution; turns; output coherence; verifier sanity>

REASONING: <2–4 sentences — the load-bearing evidence, esp. for whatever was "new/untested" in this run>
NEXT STEPS (if KILL): <root cause + the concrete fix — config knob / weight-sync / image rebuild / infra —
                      and whether to relaunch on the corrected setting or hold for the supervisor's call>
NEXT STEPS (if NO-KILL): <what to watch next sweep + the specific signal that would flip it to KILL>
```

**Verdict rules:**
- **KILL** if ANY gate is a hard FAIL with **no transient/benign explanation**: terminal/wedged (A); engines
  resident-but-not-generating or floored throughput w/ `enforce_eager:false` or training-OOM (B); incoherent
  generations / all-reward-0 from a serving/sync/verifier fault on a new geometry (C). Give the root cause +
  fix — a KILL recommendation without a "what to change before relaunch" is incomplete.
- **KILL (restart-burn corroboration, from §0)** if the job has burned restarts repeating the **SAME** failure
  each attempt — it is **deterministically doomed**, and the remaining retry budget will only burn more
  nodes-hours reproducing it. State `B/K burned, remaining K−B, same failure: <traceback>` and the fix that
  must land before any relaunch. (A restart burned on a genuine *transient* the run has since recovered from is
  NOT a kill — say so and weigh the other gates.)
- **NO-KILL** if all three gates pass, **OR** the only failures have a legitimate transient/early-bring-up
  explanation (e.g. 0 completed trials at +15 min on a long-episode arm; an HF-weight-resolution flake that the
  `--max-retries`/retry-wrapper is catching; gang still admitting). Say what you're waiting on and the signal
  that would change the call.
- **Default to NO-KILL + escalate on genuine ambiguity** (low confidence) — the asymmetry favors not killing a
  possibly-healthy bring-up. Hand the supervisor the evidence and let them decide.

**You never run the kill.** If KILL, the supervisor executes `iris job kill /benjaminfeuer/<job>` (CoreWeave) /
`scancel <id>` (SLURM) with permission, then relaunches per `rl-agentic-launch-iris` / `rl-*-launch-*` on the
corrected setting. Log the probe + verdict to `~/Documents/agent_logs/` (dated) so the diagnosis isn't lost.

---

## Operating notes
- **This skill is for the HARD per-job read; `monitor-cron-sweep` is the breadth pass.** Don't duplicate the
  full sweep here — probe the ONE job you were handed, deeply, and return a verdict.
- **Cluster-agnostic by design:** the four gates (liveness / resources / rollouts / verdict) hold on any
  cluster; only the *mechanics* (capture tool, state-poll, GPU-poll, log paths) differ — and those live in
  `.claude/ops/<cluster>/`. CoreWeave is the worked example because that's where the new/untested RL runs.
- **Read logs through the dependency docs.** `.claude/projects/{marinskyrl,vllm,harbor,daytona}/` define the
  log vocabulary, the benign-vs-fault line, and the known failure modes — use them so you don't misread a
  heartbeat as a hang or a passthrough exception as a crash.
- **Never patch/hand-edit on a cluster.** If the fix is a config/code change, it goes in the LOCAL clone →
  commit → (push / next-launch upload) per CLAUDE.md — your job here is diagnosis + recommendation, not a
  cluster-side edit.
