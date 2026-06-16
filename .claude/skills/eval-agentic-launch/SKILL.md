---
name: eval-agentic-launch
description: >-
  Launch agentic Harbor evals through the OT-Agent unified eval listener
  (eval/jupiter/unified_eval_listener.py) on any cluster: select models (query_unevaled_models.py /
  priority lists), wire the pinggy served-model tunnel, submit with the right preset + flags in tmux,
  then VERIFY the launch actually works via the 15-min infra sanity check (pinggy auth, Daytona→cluster
  api_base, vLLM POSTs, trial progression — catches "RUNNING but silently dead" jobs). Cluster-AGNOSTIC:
  per-cluster particulars (sbatch script, gpu-mem ceiling, concurrency, cert/tunnel, conda env, paths,
  Daytona key, pre-download) live in `.claude/ops/<cluster>/`. Use when asked to launch/relaunch agentic
  evals, or eval a model on a benchmark (terminal_bench_2 / dev_set_v2 / swebench / bfcl / aider).
---

# eval-agentic-launch

Launch agentic evals via the **unified eval listener** (`eval/jupiter/unified_eval_listener.py`, shared
across clusters). This skill is the **cluster-agnostic process**; for the cluster you're on, read its
ops notes first.

> **Cluster particulars → `.claude/ops/<cluster>/ops.md`** (and `ops/all/` for shared, e.g. `hf_tmux.md`).
> What lives there, NOT here: the `--sbatch-script` path, `--gpu-memory-util` ceiling (A100-64GB needs a
> lower one than H/GH-class), `--n-concurrent` value, SSH-tunnel/step-ca cert refresh, conda env + code
> paths, whether `--pre-download` is needed (no-internet clusters), and the **Daytona eval-org key**
> (SWE-bench presets need the eval key, not the RL-org default — see ops/CLAUDE.md). Read it before launching.

## 1. Select the models
- **Priority list** (the default mode): a file in `eval/lists/` (`models_8b_*.txt`, `models_32b.txt`,
  `models_131k.txt`, …). Launch with `--require-priority-list --priority-file eval/lists/<file>`.
- **Find unevaled models** to build/refresh a list — `scripts/database/query_unevaled_models.py` (resolves
  benchmark families via the Supabase `duplicate_of` field, e.g. `dev_set_v2` ⊇ `DCAgent_dev_set_v2` /
  `dev_set_v2_2.0x` / `openthoughts-tblite`):
  ```bash
  python scripts/database/query_unevaled_models.py --benchmark <fam> --size <8|32> -o eval/lists/<file>.txt -v
  # needs SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY
  ```
- **Benchmark** = a `--preset` (`tb2`=terminal_bench_2, `v2`=dev_set_v2, `dev`=dev_set_71_tasks,
  `swebench`, `bfcl`, `aider`) OR explicit `--datasets` — **use one, not both**. 32k vs 131k models take
  different harbor configs (`--harbor-config hpc/harbor_yaml/eval/eval_ctx{32k,131k}_non_it*.yaml`).
  - Note: `--preset swebench` is the **random-100 subset** (`DCAgent/swebench_verified_eval_set` is
    aliased to `swebench-verified-random-100-folders` in every cluster's `unified_eval_harbor.sbatch`,
    n_concurrent 32) — NOT the full SWE-bench-verified set.

### "ID evals" — the in-distribution shorthand (launch all three)
**"ID evals" = launch these three presets** (each a separate listener invocation — they have different
`n_concurrent`/harbor-config, so they don't combine into one `--datasets`):

| shorthand leg | `--preset` | dataset (post-alias) | n_concurrent |
|---|---|---|---|
| SWE-bench-verified random-100 | `swebench` | `swebench-verified-random-100-folders` | 32 |
| dev_set_v2 | `v2` | `DCAgent/dev_set_v2` | 128 |
| terminal_bench_2 | `tb2` | `DCAgent2/terminal_bench_2` | 64 |

When asked to "run the ID evals" on a model/list, fire one `unified_eval_listener.py` per leg (§3) with the
shared flags, then run the §4 infra check on each. (Full SWE-bench-verified and other benchmarks are **OOD**.)

> **Terminology caveat:** `crud-otagent-supabase` defines the DB/scoring **ID** set as the **2-member**
> `{swebench-verified-random-100-folders, terminal_bench_2}` (dev_set_v2 is excluded there — it's
> partial-credit with no clean SE, see `analyze-rl-behavior`). This **launch** shorthand is the **3-member**
> `{swebench, v2, tb2}`. Same name, different membership by design (what to *launch* ≠ what to *score on*) —
> don't reconcile them silently.

## 2. Wire the pinggy served-model tunnel
The served model is exposed to Daytona cloud sandboxes via a **pinggy** persistent tunnel — pass
`--pinggy_persistent_url <URL> --pinggy_token <TOKEN>`. Standing rule: **use pairs 8/9/10 by default** (the
user keeps 1–7 for sibling experiments; confirm before borrowing 1–7).

> **The actual URL+token bank is privileged — NOT stored in this committable skill.** Read the pairs from
> **`.claude/secret.md`** (untracked) or the canonical source of truth
> `/Users/benjaminfeuer/Documents/notes/ot-agent/pinggy_bank.md` (assignments shift — re-read before launch).

## 3. Launch (in tmux — the listener is long-running)
General shape (fill the cluster-specific values from `ops/<cluster>/ops.md`):
```bash
# inside a tmux session (the listener runs minutes/model — pre-download; nohup/disown are unreliable)
python eval/jupiter/unified_eval_listener.py \
  --preset <preset> \
  --sbatch-script <ops/<cluster> value> \
  --require-priority-list --priority-file eval/lists/<file>.txt \
  --n-concurrent <ops value> --gpu-memory-util <ops value> \
  --harbor-config hpc/harbor_yaml/eval/<ctx>.yaml \
  [--pre-download] [--pinggy_persistent_url <URL> --pinggy_token <TOKEN>] \
  --once --verbose --batch-size <N> 2>&1 | tee eval/<cluster>/logs/<preset>_listener_$(date +%Y%m%d_%H%M%S).log
```

## 4. VERIFY the launch — the 15-min infra sanity check (do NOT trust "RUNNING")
A job can report RUNNING while nothing happens (pinggy locked, launcher missing `--pinggy_*`, dead vLLM
engine). **After launching, schedule a 15-min (`ScheduleWakeup delaySeconds: 900`) infra check** and
re-arm it each pass until the eval terminates / you have a verdict / the user says stop. The four checks
(infra, not results):

1. **Pinggy tunnel** — `grep` `experiments/<run>/logs/*pinggy.log`: `You are authenticated as …` = live;
   `A tunnel with the same token … is already active` = server-side lock → cancel + relaunch on a
   DIFFERENT pair; long silence after auth → confirm the traffic counter (`RB:/SB:/TC:`) is growing.
2. **Daytona → cluster** — a trial's `config.json` `api_base` MUST be the public `https://*.a.pinggy.link/v1`,
   NOT an internal IP (`10.*.*.*`). Internal IP = the launcher didn't wire pinggy → relaunch with `--pinggy_*`.
3. **vLLM serving** — `POST /v1/chat/completions` count grows ≥ a few/min, `200 OK` dominates. `400` ratio
   > 15% → context overflow (`VLLMValidationError: input tokens …` → lower `max_input_tokens`/`max_output_tokens`
   in the harbor yaml) or other validation error.
4. **Trial progression** — count trials with `agent/` populated (active) and `result.json` (done). 30+ min
   with zero `agent/command-0/` (OpenHands) → setup stalled (Daytona env build / agent install). Completions
   with `n_output_tokens: None` and `agent_execution.finished_at` ≈ `started_at` (instant-fail) = the tunnel
   isn't really carrying traffic despite a healthy-looking job.

(Ongoing per-sweep eval *reporting/monitoring* is a separate skill — this section is just the immediate
post-launch "did it actually start working" gate. The coarse 2h cron is too slow to catch eval-infra silent failures.)

Quick post-submission liveness (≈15 min after submit): `ssh <cluster> "squeue -u $USER --format='%.18i %.50j %.8T %.10M'"`
then tail the newest log — look for vLLM health-check pass, (Leonardo) SSH tunnel up, `trial`/`reward` lines, no OOM / repeated DaytonaErrors.

## 5. Trial directory layout (for the checks above + cleanup)
`<run_tag>/<task>__<trial_id>/`: `config.json` (mtime≈start, has `api_base`), `trial.log`, `result.json`
(timestamps + `verifier_result.rewards.reward` + `exception_info`), `exception.txt`, `agent/trajectory.json`,
`verifier/{reward.txt,detailed_scores.json}`. Eval **cleanup + manual DB register + trace upload** when
auto-upload fails → the **`eval-agentic-cleanup`** skill.

---

## Operating notes (folded from memory 2026-06-14)

- **Eval-job submission defaults** (apply automatically unless the user overrides): `--require-priority-list` (always), `--n-concurrent 48` (always), `--harbor-config hpc/harbor_yaml/eval/eval_ctx32k_non_it.yaml` for 32k models / `eval_ctx131k_non_it.yaml` for 131k. (Prevents evaluating non-priority models and uses the correct non-instruct configs.)
