---
name: eval-agentic-launch
description: >-
  Launch agentic Harbor evals through the OT-Agent unified eval listener
  (eval/unified_eval_listener.py) on any cluster: select models (query_unevaled_models.py /
  priority lists), wire the pinggy served-model tunnel, submit with the right preset + flags in tmux,
  then VERIFY the launch actually works via the 15-min infra sanity check (pinggy auth, Daytona→cluster
  api_base, vLLM POSTs, trial progression — catches "RUNNING but silently dead" jobs). Cluster-AGNOSTIC:
  per-cluster particulars (sbatch script, gpu-mem ceiling, concurrency, cert/tunnel, conda env, paths,
  Daytona key, pre-download) live in `.claude/ops/<cluster>/`. Use when asked to launch/relaunch agentic
  evals, or eval a model on a benchmark (terminal_bench_2 / dev_set_v2 / swebench / bfcl / aider).
---

# eval-agentic-launch

Launch agentic evals via the **unified eval listener** (`eval/unified_eval_listener.py`, shared
across clusters). This skill is the **cluster-agnostic process**; for the cluster you're on, read its
ops notes first.

> **Cluster particulars → `.claude/ops/<cluster>/ops.md`** (and `ops/all/` for shared, e.g. `hf_tmux.md`).
> What lives there, NOT here: the `--sbatch-script` path, `--gpu-memory-util` ceiling (A100-64GB needs a
> lower one than H/GH-class), `--n-concurrent` value, SSH-tunnel/step-ca cert refresh, conda env + code
> paths, whether `--pre-download` is needed (no-internet clusters), and the **Daytona eval-org key**
> (SWE-bench presets need the eval key, not the RL-org default — see ops/CLAUDE.md). Read it before launching.

> **⚠ SECRETS COME FROM `secrets.env`, NEVER HARDCODED IN AN SBATCH/SCRIPT.** The eval `eval_harbor.sbatch`
> (leonardo/tacc/jupiter) sources the secrets env (`$DC_AGENT_SECRET_ENV`, default `~/secrets.env` / TACC
> `$SCRATCH/keys.env`) and reads the two Daytona eval-org keys from it: **`DAYTONA_API_KEY` (org1) +
> `DAYTONA_DATA_API_KEY` (org2)**, 3:1-weighted (3/4 org2). The script fails loudly (`:?`) if either is unset —
> so **ensure your secrets env is populated + sourced before launch** (same model as `hpc.launch`), never paste
> a `dtn_…` key into a script/config/commit. (`data/sbatches/register_snapshots.py` reads the same two env vars.)
> If you ever find a literal key committed, treat it as a leak: replace with the env read AND get the key
> **rotated/revoked** — a fix-forward edit does not un-leak it from git history.

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
  `swebench`, `bfcl`, `aider`) OR explicit `--datasets` — **use one, not both**.
  - **Harbor config: do NOT override `--harbor-config` for standard terminus-2 evals.** Omit it — the
    listener **selects the canonical config by model size** (8B-class → `hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml`
    [timeout_multiplier 2.0]; 32B-class → `dcagent_eval_defaults_32b.yaml` [16.0]; the multiplier lives IN
    the config file — see §3b below) and sets `EVAL_HARBOR_CONFIG` per-model. The sbatch's own fallback
    default (when the listener sets nothing) is `dcagent_eval_defaults.yaml`. The old
    `eval_ctx{32k,131k}_non_it*.yaml` / `ctx32k_non_it_16x_eval_.yaml` configs are **deprecated** (they
    carried `penfever/temp-override`-era `mean-drop-ei`/`accuracy-drop-ei` metrics that no Marin-branch
    harbor supports → JobConfig ValidationError). Only pass `--harbor-config` for a genuine non-default need
    (131k context window, `openhands_*` installed-harness, etc.) — an explicit `--harbor-config` overrides
    the size selection for every model, so confirm that config has no stale metrics first.
  - Note: `--preset swebench` is the **random-100 subset** (`DCAgent/swebench_verified_eval_set` is
    aliased to `swebench-verified-random-100-folders` in every cluster's `eval_harbor.sbatch`,
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

> **Intentional re-eval / parity test? Pass `--force-eval`** (see gotcha #6). By default the listener
> **Skips** any model with a `Finished`+metrics row (`reason=job finished`) — correct for normal cohort
> fill, but it blocks a deliberate re-run. `--force-eval` bypasses the dedup so all three legs (re)submit.

> **Scoring note:** `crud-otagent-supabase` now uses the **same 3-member** ID set
> `{swebench-verified-random-100-folders, terminal_bench_2, dev_set_v2}` (unified 2026-06-16). One caveat
> carries over: `dev_set_v2` is **partial-credit** (no clean binomial SE, see `analyze-rl-behavior`), so on
> the scoring side it counts toward the ID **mean** but is **excluded from the ID SE** and from any
> model-vs-model ranking (rank on a binary benchmark — swebench-100 or tb2). The launch shorthand fires all
> three regardless.

## 2. Wire the pinggy served-model tunnel — ONLY for installed agent harnesses (NOT terminus-2)
> **Skip this whole section for the default `terminus-2` agent.** pinggy is needed **only for installed
> agent harnesses** (e.g. `openhands` — the `openhands_*` harbor configs) that run inside the Daytona
> sandbox and must call back out to the served model over a public tunnel. The default **`terminus-2`**
> agent (every standard `eval_ctx*`/`*_non_it*` config; the listener's hardcoded default) does **NOT** use
> pinggy — do not pass `--pinggy_*`, do not consume a pinggy pair. The `--preset {swebench,v2,tb2,dev,...}`
> presets are terminus-2 → no pinggy. Only reach for it when you've deliberately selected an
> `openhands_*` (or other installed-harness) harbor config.

For an installed-harness launch only: the served model is exposed to Daytona via a **pinggy** persistent
tunnel — pass `--pinggy_persistent_url <URL> --pinggy_token <TOKEN>`. Standing rule: **use pairs 8/9/10 by
default** (the user keeps 1–7 for sibling experiments; confirm before borrowing 1–7).

> **The actual URL+token bank is privileged — NOT stored in this committable skill.** Read the pairs from
> **`.claude/secret.md`** (untracked) or the canonical source of truth
> `/Users/benjaminfeuer/Documents/notes/ot-agent/pinggy_bank.md` (assignments shift — re-read before launch).

> **RESUMING a Cat-3 (installed-harness) eval also needs the tunnel — see `eval-agentic-cleanup` check 4.**
> The resume path runs through `eval/resume_chunked.py` (not this listener launch flow), and as of PR #31 it
> MUST forward four passthroughs — `--pinggy-url` / `--pinggy-token` / `--config-yaml` / `--agent-parser` —
> so the resume sbatch starts a Pinggy tunnel (gated on `EVAL_PINGGY_URL`) instead of leaving the sandboxed
> agent pointing at a dead URL → `[Errno 111] Connection refused`. Full resume command + the swe-agent
> retry-policy change live in **`eval-agentic-cleanup`** (don't duplicate here).

## 3. Launch (in tmux — the listener is long-running)

> **🚧 SUBMIT FROM THE REPO DIR WITH `DCFT` SET — the sbatch WORKDIR guard hard-fails otherwise.**
> Run the cluster preamble (`cd <repo>` + `source hpc/dotenv/<cluster>.env`, which exports `DCFT`) before
> launching the listener. The generated `universal_eval.sbatch` resolves `WORKDIR` from `DCFT_PRIVATE →
> DCFT → $PWD`; if the listener submits eval jobs from `$HOME` or a scratch subdir with `DCFT` unset, the
> guard trips (missing `hpc/shell_utils/triton_cache.sh` marker) and each eval job **`exit 1`s
> immediately** with a `FATAL: WORKDIR=... is not the OpenThoughts-Agent repo root` message. If you see
> that FATAL in an eval `.out`, the listener was started from the wrong place — re-run the preamble and
> relaunch.

General shape — use the **canonical unified listener `eval/unified_eval_listener.py` with a
`--cluster-config`** (the retired pre-v6 per-cluster listener subsystem has been removed; it lacked the
`conda_env` / `limit_mm_per_prompt` / `--config-yaml` / `--agent-kwarg` / `--agent-parser`
wiring and would mis-serve per-model-conda_env models — e.g. the qwen3_5 tmax models would fall back
to `otagent`/vLLM-0.16 and crash). The cluster config (`eval/clusters/<cluster>.yaml`) supplies
sbatch_script / hardware / conda_envs / paths, so you no longer pass `--sbatch-script` /
`--n-concurrent` / `--gpu-memory-util`:
```bash
# inside a tmux session (the listener runs minutes/model — pre-download; nohup/disown are unreliable)
# PYTHONPATH MUST include the repo root — the listener imports first-party top-level packages
# (`database`, `eval`, `hpc`). A bare `python eval/unified_eval_listener.py` from a one-liner that
# didn't set it dies with `ModuleNotFoundError: No module named 'database'`. Run from the repo root
# (cd $WORKDIR) AND export it:
export PYTHONPATH="$PWD:${PYTHONPATH:-}"   # $PWD = the OpenThoughts-Agent repo root
python eval/unified_eval_listener.py \
  --cluster-config eval/clusters/<cluster>.yaml \
  --preset <preset> \
  --require-priority-list --priority-file eval/lists/<file>.txt \
  --config-yaml dcagent_eval_config_no_override.yaml \
  [--agent-kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'] [--agent-parser json] [--max-output-tokens 16384] \
  [--pre-download] [--force-reeval] [--pinggy_persistent_url <URL> --pinggy_token <TOKEN>] \
  --once --verbose 2>&1 | tee eval/<cluster>/logs/<preset>_listener_$(date +%Y%m%d_%H%M%S).log
```
Per-model serve settings (`conda_env` — e.g. `eval-qwen35` for qwen3_5 — `tensor_parallel_size`,
`data_parallel_size`, `max_model_len`, `limit_mm_per_prompt`, and the optional `max_output_tokens`
serve-token budget) now come from the **shared model-config registry**
(`eval/configs/model_configs.yaml`), which is the **DEFAULT** resolution path for every cluster
(no flag needed). The cluster yaml's `hardware_profile:` (e.g. `gh200` on TACC/Vista) selects the
per-cluster recipe; a model that diverges on intrinsic fields per cluster is a flat
`name@<profile>` standalone entry, a pure hardware delta is a `variants: {<profile>: {…}}` block.
The preset forwards `--config-yaml dcagent_eval_config_no_override.yaml` so harbor inherits per-task
sandbox sizes (no per-cluster config). **Thinking is per-model authoritative** — sourced from the
registry, where each thinking-capable model carries `agent_kwargs: [extra_body={…enable_thinking:true}]`;
presets do NOT carry thinking, so a preset can never force thinking on a non-thinking
model. There is **no `--enable-thinking` flag**. To override a model's resolved
kwargs, pass `--agent-kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'`
(precedence: CLI `--agent-kwarg` > per-model registry > preset).

> **✅ Per-model serve overrides come from the registry BY DEFAULT — `--baseline-model-configs` is now a DEPRECATED optional override.**
> As of the Stage-4 cutover the shared registry (`eval/configs/model_configs.yaml`) is the default
> resolution path for ALL clusters, so omitting `--baseline-model-configs` is now **correct** — the
> registry supplies every per-model `conda_env` / `tensor_parallel_size` / `data_parallel_size` /
> `max_model_len` / `limit_mm_per_prompt` (the cluster's `hardware_profile:` picks the right variant /
> `name@profile` recipe). The earlier failure mode (omitting the flag → fell back to vanilla `otagent`
> → `model type qwen3_5_moe not recognized`, 2026-06-25) **no longer applies**: the registry, not a CLI
> flag, now carries those entries. Confirm it loaded: the listener logs `Model-config registry ENABLED
> (default-on): …` + `Loaded model registry: N model config(s)` + `Using conda env '<env>' for <model>`.
> Passing `--baseline-model-configs <file>` STILL works (loads via the legacy per-cluster baseline file)
> but is **deprecated** — it emits a `DeprecationWarning` + a `DEPRECATED: --baseline-model-configs is
> superseded by the shared model-config registry …` log line, and is treated as an explicit opt-OUT of
> the registry for that launch. Migrate any model into the registry instead.

> **🚧 Concurrent-submit guard — ONE listener enqueues many legs; do NOT fire N concurrent `--once`
> processes.** A refill of several legs (e.g. all flawed-summ legs, or the three ID legs across a list)
> is **one** listener invocation that submits every leg internally — `run_iteration()` loops the legs
> and sleeps `submission_delay` (default 1.0s) between each `sbatch`
> (`eval/unified_eval_listener.py` L3204–3205). A `--priority-file` with multiple models, or a
> multi-leg `--datasets`, all flow through that single serial loop. **Never launch a separate
> `python eval/unified_eval_listener.py … --once` process per leg "to go faster."** Each invocation is
> a fresh Python interpreter that re-runs conda's activation/plugin-discovery; firing N of them within
> the same second on the **login node** makes them race on conda's lazily-imported plugin registry →
> a **conda-plugin circular-import at activation** (observed on a Leonardo flawed-summ refill: a batch
> fast-failed on a Python circular-import inside a conda plugin during env activation). The canonical
> path already avoids this. If you genuinely must run more than one listener process (different presets
> with incompatible `n_concurrent`), **serialize / stagger them ~30–45s apart** so their conda
> activations don't collide — do NOT background them simultaneously with `&`. (Per-job `conda activate`
> *inside* the sbatch runs on independent compute nodes and does **not** race — the contention is purely
> login-side, at listener startup. See gotcha #8.)

## 3b. Timeout multiplier policy (automatic, config-by-size selection — usually nothing to do)

Harbor's `--timeout-multiplier` scales every agentic-rollout timeout. A larger model decodes its rollout
much more slowly, so a flat multiplier (the old 1× default) made big models spuriously hit
**AgentTimeout** → deflated scores. The multiplier now **lives IN the canonical config file**, and the
listener **selects the config by model size**, so a normal launch gets the right value with no flag:

| model size (param count) | selected config | timeout multiplier |
|---|---|---|
| **8B-class** (≤ ~14B; includes 1.5B/7B/14B) | `hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml` | **2×** (in the file) |
| **32B-class** (~28–42B; includes MoE like `30b-a3b`) | `hpc/harbor_yaml/eval/dcagent_eval_defaults_32b.yaml` | **16×** (in the file) |
| out-of-band (e.g. 70B / 80B) **or no size token in the name** | base default (`dcagent_eval_defaults.yaml`) | 2× + a logged note (set one explicitly if you want more) |

**Where it's set:** `eval/unified_eval_listener.py` — `select_harbor_config()` / `resolve_model_eval()`,
applied per-model in the gathering pass. It reads the **param-count size token from the HF model name**
(largest `\dB` token wins, so MoE `…-30b-a3b` → 30B → 32B band). The selected config path flows as
`EVAL_HARBOR_CONFIG` into the sbatch, and its `timeout_multiplier` flows as `EVAL_TIMEOUT_MULTIPLIER`
(→ harbor `--timeout-multiplier`) and is recorded in the Pending DB row's config so dedup stays consistent
with what actually ran (a 32B's 16.0 row dedups against 16.0, an 8B's against 2.0).

**Resolution order (first wins):**
1. **Explicit `--harbor-config` / preset `harbor_config`** — overrides the size selection for **every**
   model (use it for 131k context / `openhands_*` installed-harness needs). Its `timeout_multiplier` is
   used as-is.
2. **Per-model entry** — a `timeout_multiplier:` under a model (or pattern) in the shared registry
   `eval/configs/model_configs.yaml`. Overrides the selected config's value. Use this for models whose
   **name has no size token** (e.g. `laion/GLM-4_7-swesmith-…` is really a Qwen3-8B → add an entry with
   `timeout_multiplier: 2.0`) or for out-of-band sizes you want a deliberate value for.
3. **Size-based config selection** — the table above, derived from the name.

**Rule for sizes the table doesn't name:** ~28–42B → `_32b.yaml` (16×) and everything else → the base
default (2×) are applied automatically (1.5B is *covered* by the base 2× config; 80B / 70B fall to the base
default with a logged note). To give an out-of-band model a deliberate multiplier, add a per-model entry in
the shared registry `eval/configs/model_configs.yaml`. For a one-off manual `harbor jobs start`, point `--config` at
`dcagent_eval_defaults.yaml` (8B) / `dcagent_eval_defaults_32b.yaml` (32B) (see `docs/EVAL_GUIDE.md`).

## 4. VERIFY the launch — the 15-min infra sanity check (do NOT trust "RUNNING")
A job can report RUNNING while nothing happens (pinggy locked, launcher missing `--pinggy_*`, dead vLLM
engine). **After launching, schedule a 15-min (`ScheduleWakeup delaySeconds: 900`) infra check** and
re-arm it each pass until the eval terminates / you have a verdict / the user says stop. The four checks
(infra, not results):

> **Checks 1–2 are pinggy-path (installed-harness) ONLY** — skip them for the default `terminus-2` agent
> (it doesn't use a pinggy tunnel). For terminus-2, served-model reachability is proven by **check 3**
> (POSTs arriving from the sandboxes) — if check 3 is healthy and trials progress (check 4), the model is
> reachable. Checks 3–4 apply to every launch.

1. **Pinggy tunnel** (installed-harness only) — `grep` `experiments/<run>/logs/*pinggy.log`: `You are authenticated as …` = live;
   `A tunnel with the same token … is already active` = server-side lock → cancel + relaunch on a
   DIFFERENT pair; long silence after auth → confirm the traffic counter (`RB:/SB:/TC:`) is growing.
2. **Daytona → cluster** (installed-harness only) — a trial's `config.json` `api_base` MUST be the public `https://*.a.pinggy.link/v1`,
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

- **Eval-job submission defaults** (apply automatically unless the user overrides): `--require-priority-list` (always), `--n-concurrent 48` (always). **Do NOT pass `--harbor-config`** for standard terminus-2 evals — let the listener select the size-appropriate canonical config (8B-class → `hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml` [2.0]; 32B-class → `dcagent_eval_defaults_32b.yaml` [16.0]; see §3b). (Overriding it with the deprecated `eval_ctx*_non_it*`/`ctx32k_non_it_16x_eval_` configs injects the stale `*-drop-ei` metrics → JobConfig ValidationError; those were removed from the configs 2026-06-16 but the size-selected canonical defaults remain the right choice.) Use `--harbor-config` ONLY for 131k context or installed-harness (`openhands_*`) needs — it overrides the size selection for every model.

## Launch gotchas discovered in practice (2026-06-16)

Three traps that each silently break a launch — check these first when an eval misbehaves:

1. **`--require-priority-list` is LOAD-BEARING, not just a default — omitting it floods the queue.** `--priority-file` alone does **not** restrict which models get evaled; it only changes *sort order* (`unified_eval_listener.py` ~L1002: priority models sort first). The actual filter "skip models not in the list" lives behind `--require-priority-list` (~L978: `if args.require_priority_list and hf_model not in priority_models: skip`). Without the flag the listener submits an eval for **every unevaled model in the lookback window** (routinely 700+). ALWAYS pass `--require-priority-list` together with `--priority-file` for a targeted launch. If you ever launch without it by accident: kill the listener **before** it leaves pre-download (submission happens *after* the per-dataset `Pre-downloading…`), then `squeue`/`sacct --starttime=now-Nmin` to confirm nothing stray was submitted. Note the listener python is a child of the `sshd: …@notty` session and **survives the local ssh client being killed** — `pkill -9 -f unified_eval_listener.py` (or kill the notty parent) on the cluster to actually stop it.

2. **`PermissionError: [Errno 13]` at `harbor/job.py … job_dir.mkdir()` = a `jobs_dir` in the harbor config that another user owns.** The canonical `hpc/harbor_yaml/eval/dcagent_eval_defaults*.yaml` configs deliberately ship **no** `jobs_dir` (so this can't happen from the default), but a hand-rolled `harbor jobs start --config <other.yaml>` against a config carrying e.g. `jobs_dir: /e/data1/.../mmlaion/shared/guha1/eval_jobs` (guha1's tree) builds `job_dir = config.jobs_dir / job_name`, so a non-owner's mkdir is denied and **every** eval dies at job creation (before any rollout). Fix is already in `eval_harbor.sbatch`: it passes `--jobs-dir "$EVAL_JOBS_DIR"` (the per-user writable `…/ot-baf/eval_jobs`), which overrides the config (`harbor jobs.py:1071 config.jobs_dir = UPath(jobs_dir)`). `resume` is unaffected (it takes `-p $RUN_DIR` directly). If you see this error, confirm the sbatch on the cluster actually has the `--jobs-dir` line (commit `19f54df8`); a stale sbatch or a hand-rolled `harbor jobs start` will reintroduce it.

3. **A crashed eval leaves a non-terminal DB row that blocks resubmission for 24h** (`reason=job in progress`). When a job dies before writing a terminal status (e.g. the PermissionError above), its Supabase row stays `started`/in-progress. The listener's dedup only resubmits a `started` row once it's older than `--stale-started-hours` (**default 24h**, `EVAL_LISTENER_STALE_HOURS`); pending rows use `--stale-pending-hours` (default 6h, auto-cancels the stale SLURM job). So after fixing a crash-bug, a normal relaunch will **Skip** with `reason=job in progress (started_at=…)`. To force the resubmit of the just-crashed attempt, pass a small `--stale-started-hours` (e.g. `0.05` = 3 min) so the stuck row counts as stale. Safe to combine with `--require-priority-list` (only the targeted model is in scope). Orphaned in-progress rows otherwise age out at 24h or can be cleaned via `crud-otagent-supabase`.

4. **On Jupiter, pass `--reservation reformo` or eval jobs starve behind RL.** `eval/jupiter/eval_harbor.sbatch` sets `--account reformo` but **no** `#SBATCH --reservation`, so without the flag the job lands in the *general* booster pool — which is empty because the `reformo` reservation holds ~128 nodes (`IGNORE_JOBS`), leaving the eval `PENDING Reason=Priority` indefinitely even while ~90 reservation nodes sit free. The listener already supports it: `--reservation reformo` (or env `EVAL_LISTENER_RESERVATION=reformo`), wired to the `sbatch --reservation=` line. **So Jupiter ID-eval launches should always pass `--reservation reformo`** — *until the reservation expires* (currently `EndTime=2026-06-21`; after that, `scontrol show reservation` for the live name, or drop the flag if none is active — passing a dead reservation name errors the submit). Rescue already-PENDING jobs without resubmitting: `scontrol update jobid=<j> reservation=reformo` (flips them to RUNNING immediately if the reservation has free nodes).

6. **A `Finished`+metrics row makes the listener Skip with `reason=job finished` — that is correct for cohort fill, but blocks an intentional re-eval. Force it with `--force-eval`.** `should_start_job()` returns `(False, "job finished")` for any benchmark that already has a `Finished` row carrying non-null `metrics`. Crucially, **`--stale-started-hours` does NOT override this** — that flag only re-ages `Started` (in-progress) rows; a *completed* eval is never "stale". So for a deliberate **re-eval / parity test** (re-running a benchmark that already has a real score), there is exactly one launch-time flag: **`--force-eval`** (`eval/unified_eval_listener.py`). It bypasses ALL dedup (`should_start_job(..., force=True) → (True, "force-eval (dedup bypassed)")`) and submits a **fresh** `sandbox_jobs` row — it does **not** touch the existing row, so no metrics-clearing and no cross-user DB write is needed (important when the prior row is owned by another user — clearing it would violate the FK-safe / own-rows-only guardrail). **When to force vs respect the skip:**
   - **FORCE (`--force-eval`)** — the user explicitly asks for a re-run / parity test / repeatability check, or you must overwrite a known-bad-but-non-cleared score. ALWAYS pair with `--require-priority-list` + `--priority-file <single-model list>` so only the intended model(s) are forced (without it, `--force-eval` would resubmit *every* model in the lookback window, including already-scored ones — a queue flood).
   - **RESPECT the skip (no flag)** — normal cohort/sweep fill, where `reason=job finished` correctly means "already have this number, don't waste GPUs". This is the default and should stay the default.
   - Alternative (only if you genuinely want to *replace* an existing **own** row's metrics rather than add a sibling): clear that row's `metrics` to null (→ listener returns `(True, "finished but metrics cleared")`) via `crud-otagent-supabase`, FK-safe and **own-rows-only** (`.eq("username","bfeuer00")`). `--force-eval` is preferred — it needs no DB mutation and works regardless of row ownership.
   - **Don't confuse `--force-eval` with `--force-reeval`** — they are two different flags. `--force-eval` (this gotcha) bypasses the `should_start_job` *dedup* on a fresh launch. **`--force-reeval`** bypasses the DB *status check* and (PR #31) the `active_pairs` *resume filter*; it's the flag the RESUME path uses (`resume_chunked.py` passes it automatically). Use `--force-reeval` only when deliberately re-submitting/resuming — see `eval-agentic-cleanup` check 4.

7. **`hosted_vllm/<org>/<model>` evals need TWO things from harbor commit #339 (`e44d3822`, 2025-12-29), or every trial dies.** That commit added two hard gates to `harbor/agents/terminus_2` for `hosted_vllm/` models; org-model evals worked before it. Both fail FAST (~9 min, **0 vLLM POST 200s, 0 trajectories**, all N trials raise identically — looks like a silent infra death, not a model problem):
   - **(a) Org-qualified name rejection** — `validate_hosted_vllm_model_config` (`llms/utils.py`) demanded exactly one `/`, so `hosted_vllm/laion/<model>` (2 slashes) raised `ValueError: hosted_vllm model names must contain exactly one '/'`. **FIXED** in harbor commit **`0f5a6e9e`** (relax to allow `hosted_vllm/<model>` *and* `hosted_vllm/<org>/<model>`; pulled to the cluster editable install). If it recurs, confirm that commit is in the cluster's harbor clone.
   - **(b) `model_info` hard-requirement** — the **same** `validate_hosted_vllm_model_config` (`llms/utils.py:~123`, called from `lite_llm.py:454` during terminus-2 LiteLLM init) ALSO raises `ValueError: hosted_vllm models require model_info specifying token limits and costs` when `model_info` is absent. (Note: `_resolve_model_info` in `terminus_2.py` only *warns* + returns None — it is NOT the raiser; don't waste time there.) **Fix = SUPPLY model_info, do NOT relax the guard** (token limits genuinely drive terminus-2 context management): pass it via the existing `--agent-kwarg` channel on the `harbor jobs start` line (alongside `api_base`/`key`) — harbor's `parse_kwargs` **JSON-decodes** the value to a dict, so this works directly: `--agent-kwarg model_info='{"max_input_tokens": <served max_model_len>, "max_output_tokens": <gen budget>, "input_cost_per_token": 0, "output_cost_per_token": 0}'` (costs 0 = self-hosted; token limits from the served vLLM `max_model_len`). **DONE — OT-Agent commit `d0064011`** wired this into all 3 (`jupiter`/`leonardo`/`perlmutter`) `eval_harbor.sbatch` via `EVAL_VLLM_MAX_MODEL_LEN` (default 32768) + `EVAL_MAX_OUTPUT_TOKENS` (default 16384), so the listener path carries it by default for **every** hosted_vllm agentic eval. Verified: POST 200s climbing, trajectories written, 0 model_info ValueErrors.

8. **Conda-plugin circular-import at concurrent activation = N listener processes launched at once on the login node, NOT a broken canonical path.** Symptom: a multi-leg refill (e.g. flawed-summ legs relaunched close together) fast-fails on a Python *circular-import originating in a conda plugin during environment activation* — a failure mode the normal flow never hits. **Root cause is the SUBMISSION PATTERN, not first-party code.** The conda activation that races is **login-side, at listener startup**: the canonical launch is `conda activate otagent` once (the `ops/leonardo` preamble) then **one** long-running `unified_eval_listener.py` that submits every leg internally with a 1s `submission_delay` (L3204–3205). The submit step itself is just `sbatch …` (`submit_eval`, L2476–2572) — it does **no** conda work; and each job's own `conda activate otagent` happens **inside the sbatch on its own compute node** (`eval/leonardo/eval_harbor.sbatch:124–125`), so per-job activations are independent and never race. **The race only appears when an agent fires several `python eval/unified_eval_listener.py … --once` processes near-simultaneously from the login node** (one per leg, backgrounded with `&` or launched in a tight loop) — each fresh interpreter re-runs conda's activation hook + plugin auto-discovery (entry-points / pluggy) at the same instant, and conda's lazily-imported plugin registry is not safe under concurrent first-import → circular-import. **Guard (durable):** use the single-listener / multi-model `--priority-file` path (canonical — see §3's Concurrent-submit guard); if multiple listener processes are truly required, **stagger them ~30–45s apart**, never `&` them together. The ~45s manual stagger the agent used to work around this is exactly the right spacing — codify it instead of improvising. **`CONDA_NO_PLUGINS=true` is NOT the durable fix and was deliberately NOT baked into the sbatch:** (i) it would target the *compute-node* activation, which never races, so it can't fix the *login-side* collision; (ii) globally disabling conda plugins is a blunt band-aid that could mask other (legitimate) plugin behavior in the eval env. If you ever DO hit the race on a forced multi-listener launch and want belt-and-suspenders, export `CONDA_NO_PLUGINS=true` **in the login-node shell before the listener `conda activate`** (where the race actually is), not in the sbatch — but the stagger guard above is the recommended fix and removes the need for it.
