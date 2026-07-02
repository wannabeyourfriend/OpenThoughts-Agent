# MarinSkyRL — framework facts & gotchas

The RL training framework. Source-of-truth, code constraints, and resume/MoE/FSDP gotchas. Folded out of
memory 2026-06-14. Cluster runtimes/SIFs live in `.claude/ops/<cluster>/`; live project trackers
(rollout-fanout, PR tracking, log-ratio v3, ray-workercrashed) stay as memories.

---

## Source of truth = `marin-community/MarinSkyRL` branch `penfever/working`

Consolidated 2026-06-08. The marin repo has exactly two branches: `main` + **`penfever/working`** (the
strict-superset SoT).

- **`github.com/penfever/SkyRL` is OBSOLETE** (archived 2026-06-08 at `archive/skyrl-pre-marin-consolidation-20260608`, tip `5376d8f`). Do NOT use it as SoT, do NOT merge it into marin — every production feature it had (seqnorm-global loss, `policy_strict_spread_pg`, Stage-7 streamed-EP weight-sync, Qwen3-Next GDN kernel routing) is already in marin under squashed SHAs, and marin *additionally* has TIS exact-alignment (`align_logprobs_by_token_ids`) + mp-backend that the fork lacked. Merging it would re-apply OLD copies (worker.py/trainer.py conflicts) and risk regressing the live SoT.
- **Cluster clones** (all track marin `penfever/working`, editable-installed):
  - Jupiter: `/e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL` (origin repointed 2026-06-08; editable from `.../SkyRL/skyrl-train`).
  - Leonardo: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/MarinSkyRL` (duplicate `/code/SkyRL` removed 2026-06-08).
  - Perlmutter / NYU Torch: not yet realigned (Perlmutter dropped from cron anyway) — realign when next used.
- **Sync:** commit+push to marin `penfever/working` from the laptop (Leonardo can't push), then `git pull` on the cluster. SoT tip at consolidation: `e5315d5`. See `.claude/ops/jupiter/ENVIRONMENT_MAP.md` for the baked-SIF facts.

---

## `strategy.all_reduce(status)` requires IDENTICAL keys on every rank

`DistributedStrategy.all_reduce(data: dict)` iterates `for k,v in data.items(): ret[k]=all_reduce(v)` —
a separate NCCL all_reduce per key. If keys differ across ranks, the per-key calls don't line up → NCCL
**watchdog timeout** and process-group abort.

- **Symptom:** `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=N, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)` → `NCCL communicator was aborted`. The **`NumelIn=1` scalar** is the giveaway.
- **Bug history:** burned twice on `compute_log_ratio_diagnostics` — v2 (rank-0-only gating) and v3 (per-rank early `return {}` on empty/all-padded micro-batches). v4 fix (`69294ba5`): a `_log_ratio_diag_zero_metrics()` 16-key zeros fallback used on early return + try/except at the worker call site.
- **Rule whenever you touch `status` dict keys in `worker.py`:** (1) every rank contributes the SAME key set every iteration — no conditional skips; (2) data-dependent values that might fail get a sentinel (0.0/NaN) under the same key, never omitted; (3) wrap risky helpers in try/except with a full-keyset fallback; (4) unit-test the helper with empty / all-padded / normal input and assert `sorted(keys())` is identical.

---

## uvloop/libuv SIGABRT → force stock asyncio (NOT a libuv version bump)

RESOLVED 2026-05-29. The libuv epoll SIGABRT crash-looping RL drivers (`uv__epoll_ctl_prep` io_uring abort
+ sibling `uv__io_poll` EPOLL_CTL_ADD/EEXIST abort, under Daytona sandbox-teardown socket churn) is fixed
by **forcing CPython's stock asyncio SelectorEventLoop**, not by changing libuv.

- **The version chase was a dead end** (uvloop 0.19→0.22.1, custom 1.49 — all still abort; the io_uring deferred-EPOLL_CTL race spans libuv 1.45–1.48 and 1.49's adjacent assert fires anyway; `UV_USE_IO_URING=0` doesn't gate it). The custom libuv-1.49 wheels are stashed but UNUSED.
- **Root cause:** Ray installs uvloop in every worker via `try_install_uvloop()` (gated by `RAY_USE_UVLOOP`, default True). The SkyRL orchestrator is RTT-bound (vLLM/Daytona), so uvloop's throughput edge is moot — we get all its fragility, none of its benefit.
- **THE FIX (driver):** reset the policy at the top of **`BasePPOExp.run()`** (`main_base.py`), before its `asyncio.run()` calls — SkyRL `77fb0074`:
  ```python
  import asyncio
  asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
  ```
  **Placement is the trap:** must be on the SHARED `BasePPOExp.run()`, NOT a per-example `skyrl_entrypoint` (there are 26+; terminal_bench jobs use `TerminalBenchExp` which doesn't override run() — patching main_base's entrypoint, commit 38390079, was inert and they kept aborting).
- **THE FIX (actors), 2026-06-09 SkyRL `9e04851`:** the driver reset does NOT protect Ray ACTOR processes (they still install uvloop and can SIGABRT in a RolloutCoordinator). Also set `env_vars["RAY_USE_UVLOOP"]="0"` in `prepare_runtime_environment` (utils.py) so `try_install_uvloop` is a no-op in every worker/actor. **Use BOTH.**
- **Future-proofing:** `set_event_loop_policy()` is deprecated Py3.12+; when removed, switch to `asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)`.
- Related but SEPARATE bug class: the refcount-SIGABRT fix (harbor `ec508562` orphan-task reap + SkyRL `3b1708a0` gc backstop) on AgentTimeout-heavy datasets.

---

## Resume overshoots `max_steps` — a step past the data ceiling is spurious

For the pymethods2test-large RL family (explore-tis / a3-style): `epochs=2` × dataset/`train_batch_size=64`
= exactly **80 optimizer steps**; configs set `max_steps=80` because that's the data ceiling. With
`resume_mode=latest` + chained restarts, `global_step` does **not** hard-stop at 80 — it **overshoots**
(observed 86). Steps 81→86 are **spurious** (re-runs exhausted data, "eternal-retry").

- A run reaching **step ≥80 is COMPLETE, not failed** → RL Cleanup Checklist, NOT fix-and-requeue.
- **Best-checkpoint selection must CAP candidates at step ≤80** (use ≤78 if a step-79+ greedy/eval-pass reward jump is present, e.g. 0.23→0.74 — that's an eval-checkpoint artifact, not learning), or the EMA picks an inflated step.
- Errors in the 81–86 tail are noise (e.g. `VLLMValidationError: 32769 > 32768` near-budget BPE +1) — don't mis-diagnose a past-80 boundary error as a training failure.

---

## a3 RL resume: `--dry_run` regenerates the dedup config (RESOLVED in code)

> ✅ RESOLVED 2026-06-08 — OT-Agent `penfever/working` `0b01a273`. The launcher now (1) auto-resumes from
> the canonical run dir's `checkpoints/global_step_*` when present and `--overwrite_output_dir` was NOT
> passed (pins `ckpt_path`/`export_path`/`resume_mode=latest` as last-wins hydra overrides), and (2)
> routes `--dry_run` to a `<name>__dryrun` sibling so it can't seed the real dedup config. The manual
> move-config-aside dance is **no longer needed**; pass `--overwrite_output_dir true` to force a fresh fork.

Historical trap (pre-fix): to resume in-place you moved `configs/<job>_rl_config.json` aside and launched
with the un-suffixed `--job_name`; running `--dry_run` *after* the move **regenerated** the JSON →
the real launch saw a collision and forked to a fresh `<job>_2` dir at step 0 (ckpt_path pointing at a
nonexistent dir). Other confirmed a3-resume facts: series `n_concurrent_trials=675` (900 is
`override_timeout_sec`, not concurrency — verify against the prior chain's config backup); run from the
`otagent` conda env. See [[project-a3-series-concluded]] (a3 is CONCLUDED — no relaunch).

---

## Checkpoint pathing: ckpts are NESTED at `<rundir>/<job_name>/checkpoints/`, NOT the rundir top-level

When hunting for a run's resumable checkpoints (e.g. to confirm what a chain restart will resume from),
look at the **doubly-nested** path, not the rundir root:
```
<rundir>/                              # e.g. /e/data1/.../ot-baf/stageC-pbs/   (configs/ exports/ logs/ ray_logs/ sbatch/ wandb/ + an EMPTY top-level exports/)
└── <job_name>/                        # the run subdir (same name as rundir leaf, e.g. stageC-pbs/)
    ├── checkpoints/                   # ← FULL RESUMABLE CKPTS LIVE HERE (trainer.ckpt_path)
    │   ├── global_step_2/  global_step_4/  ...   # each: policy/ (FSDP shards) + trainer_state.pt + data_consumption_state.pt + generation_buffer_state.pt (~34MB)
    │   └── latest_ckpt_global_step.txt           # ← the step the restart resumes from
    ├── exports/                       # HF-format exports (trainer.export_path; cadence = hf_save_interval)
    └── trace_jobs/                    # per-rollout traces (can be HUNDREDS of thousands of files — NEVER find/du it)
```
- **Cadences are independent:** `trainer.ckpt_interval` (full resumable, e.g. **2** → ckpt every 2 steps) vs
  `trainer.hf_save_interval` (HF export, e.g. 5). The rendered values live in `configs/<job>_rl_config.json`
  (`trainer.ckpt_interval=…`, `trainer.ckpt_path=…`, `trainer.export_path=…`).
- **A glob of the rundir top-level finds nothing** — the top-level `exports/` is empty and there is no
  top-level `checkpoints/`. To check resume state: `cat <rundir>/<job_name>/checkpoints/latest_ckpt_global_step.txt`
  and `ls <rundir>/<job_name>/checkpoints/`. (Bit a 2026-06-15 sweep: a top-level glob wrongly concluded
  "no checkpoint, resumes from step 0" when `global_step_6` was present one level down.)
- Each ckpt persists `generation_buffer_state.pt` (the async rollout buffer), so `resume_mode=latest`
  restores the buffer too — relevant if a hang is *in* the buffer state (resume can re-trigger it).

---

## MoE + EP sharding: `fsdp_size` MUST divide `num_experts // ep_size`

When a model has MoE experts and SkyRL shards the expert dim (dim-0 = `num_experts`) over BOTH the EP mesh
axis AND FSDP:

> **`fsdp_size` must evenly divide `num_experts // ep_size`.**

For Qwen3-Next-80B-A3B: `num_experts=512`, `ep_size=8` → 64 experts/EP-rank. Valid `fsdp_size` ∈
{1,2,4,8,16,32,64}. **`fsdp_size=6` is INVALID** (64/6 uneven → [11,11,11,11,11,9]).

- **Failure signature if violated:** completes rollout + all policy_train fwd/bwd, then dies at the FIRST Adam step with `RuntimeError: The size of tensor a (10) must match the size of tensor b (9) at non-singleton dimension 0` (`adam.py _single_tensor_adam`, `exp_avg.lerp_(grad)`) — a=even-PADDED FSDP2 local shard, b=UNpadded EP-backward reduce-scatter grad, disagreeing by one on a boundary rank. Deterministic, ~2.3h in (after full step-1 fwd+bwd) → expensive to hit.
- History: Stage-7 validated EP8×FSDP4 (64/4=16 even); a 2026-06-08 FSDP4→6 bump (to clear a step-2 OOM) silently introduced the uneven split. Fix = **FSDP=8** (64/8=8 even, AND more sharding fixes the OOM too); verified 2026-06-10 (job 674828 completed step-1 end-to-end). Note: a fresh-run-dir policy_train iter-0 was ~2h (cold torch.compile + DeepEP + R3-replay warmup) — expect a long first step.
- **Guard in place:** `distributed/fsdp_utils.py` (`63cd2eb`) raises a fail-fast assertion at `create_device_mesh` if `(num_experts // ep_size) % fsdp_size != 0`. 80B yaml `fsdp_size=8` at OT-Agent `41379072`. Full detail: `agent_logs/80b_failures.md`.

---

## MoE served-policy token-salad on the RL update path → `SKYRL_W13_RELOAD_BRACKET` (FIXED `2bb70a88`)

A served MoE policy emitting incoherent CJK token-salad (100% reward-0) on EVERY generation after a
disaggregated weight sync = the FusedMoE **`w13` gate/up halves are in the wrong kernel order**. Root cause:
vLLM's initial from-disk load runs `process_weights_after_loading`, which for FusedMoE under
**FlashInfer-CUTLASS / TRTLLM** (auto-selected on H100) applies `swap_w13_to_w31` (`[gate;up]→[up;gate]`).
The RL update path did per-chunk `model.load_weights` with **no finalize**, reverting `w13` to checkpoint
`[gate;up]` and never re-swapping → the kernel reads the wrong halves → salad. **TRITON / AITER backends do
NOT swap** → the same skip is harmless there (the likely Jupiter-GH200-OK reconciliation).

- **Env var `SKYRL_W13_RELOAD_BRACKET`** (default `1`): brackets the multi-chunk sync in
  `fsdp_worker.broadcast_to_inference_engines` with `WorkerWrap.skyrl_begin/finish_weight_reload`
  (= vLLM `initialize/finalize_layerwise_reload`) so `process_weights_after_loading` runs **exactly once**
  post-sync (re-applies the swap; avoids the #1737 per-chunk re-finalize/absent-layer hazard). Set `0` for
  the exact prior behavior — it is **swap-inert on triton/dense → byte-identical there**, so leave it on.
- **Scope:** only the non-IPC, non-`_fuse_weights` broadcast path is bracketed. Diagnosis was **MoE-specific
  × FlashInfer-CUTLASS × disaggregated-RL-update** — NOT NCCL (a P2P/NVLS-disable A/B was falsified), NOT the
  gather (EP=8 on-GPU gather proven bit-exact vs disk), NOT placement/broadcast (engine-held == disk).
- **Bring-up check:** confirm the bracket engaged — engine log shows `initialize_layerwise_reload` /
  `finish_weight_reload`. Full account: `agent_logs/2026-06-27_coreweave_moe_ep_garbage_debug_cycle.md`.

---

## 80B RL is TRAINING-bound, and SkyRL FSDP is ALWAYS cross-node

The Qwen3-Next-80B-A3B production RL step (EP=8×FSDP=8, 32k, R3+TIS) is **training-bound, NOT
gen-bound**. Measured step ≈ 17,000s (~4.7h): `policy_train` ~48%, `fwd_logprobs_values_reward` ~31%,
`sync_weights` ~13%, `wait_for_generation_buffer` only ~7.5%. (The "1 step/12h" was step-time + a one-time
~4.5h cold-start buffer fill.) Earlier "generation-bound" sweep notes were WRONG.

- **SkyRL FSDP is cross-node regardless of `fsdp_size`.** `create_device_mesh` (`fsdp_utils.py:814`) builds the mesh with **`ep` innermost/contiguous** (`mesh_shape=(ddp,fsdp,ep)`), so on 4-GPU/node Jupiter an FSDP group = 1 GPU per node spanning `fsdp_size` nodes; EP is the intra-node dim. The yaml comments claiming FSDP is "intra-node" are FACTUALLY WRONG. The ordering is deliberate for a **correctness** reason (fsdp must precede ep so the composed expert DTensor `[Shard_fsdp, Shard_ep]` slices ascending; reverse → `KeyError: Mesh dim indices should be in ascending order`), NOT topology.
- **EP=16×FSDP=4 does NOT restore intra-node FSDP** — launchable but throughput-neutral-to-worse (FSDP stays cross-node; EP all-to-all widens 8→16). **Do NOT switch EP/FSDP for the speed objective.**
- **The real speed lever** = make FSDP intra-node via a mesh-dim-reorder CODE change (fsdp last, working around the DTensor ascending-slice constraint) — attacks the dominant 48% `policy_train`. Secondary: the 13% `sync_weights` (`broadcast_to_inference_engines` full_tensor gather — known 80B pain point) and 31% `fwd_logprobs`. (Investigation `a4e4b933`, 2026-06-11.)

---

## FSDP2 Context-Parallel Stage 2 — attn-backend pivot (DONE 2026-06-12)

Branch `feuer/fsdp2-cp`, HEAD `18c2606`. Stages 0–2 done.

- Added `trainer.attn_backend ∈ {auto,flash_attention_2,sdpa,flex}` (default `auto` = byte-identical to pre-Stage-2). `model_wrapper.py`: guarded flash import (`_HAS_FLASH` + shims that raise only if called) + `resolve_attn_implementation(...)`; CP (`context_parallel_size>1`) forces sdpa/flex, rejects flash varlen. Wired through policy/critic/ref. Tests: `test_attn_backend.py` (11 pass), `test_sdpa_flash_parity.py`.
- **Parity (Qwen2.5-0.5B, torch-2.11 SIF) — SDPA pivot is CORRECT:** sdpa@fp32 vs eager@fp32 logp 2.29e-3 (tight = correct); bf16 cross-kernel diffs (5e-2) are the bf16-quantization floor, not flash error. Did NOT loosen the spec tol silently — used a tight fp32 sdpa-vs-eager gate + bf16 tol at the measured floor.
- **GOTCHAS for Stage 3+ GPU CP runs:**
  - **/opt/SkyRL baked-module shadow:** the SIF bakes SkyRL at `/opt/SkyRL`; bare `python script.py` imports it (not a worktree clone) → new kwargs silently ignored via `**kwargs`, both arms fall to eager (false pass). FIX: `apptainer exec --env PYTHONPATH=<worktree>/skyrl-train`; the GPU test now asserts `model.config._attn_implementation` actually engaged.
  - Triton JIT gcc `-l:libcuda.so.1` link fail on compute node: `--env LIBRARY_PATH=/.singularity.d/libs`.
  - HF offline: prefetch into `/e/scratch/jureap59/feuer1/hf_cache`, `HF_HUB_OFFLINE=1`; `-p no:cacheprovider --confcutdir tests/cpu/<sub>` dodges the session-autouse `ray_init()` login-node hang.
  - Worktree `/e/scratch/jureap59/feuer1/cp_stage2_wt`; reformo srun (account+reservation=reformo, partition=booster, `--gres=gpu:1`). Log: `agent_logs/2026-06-12_cp_stage2.md`.

---

## Saturating vLLM engines in agentic fully_async RL — scale `n_concurrent_trials` AND `num_parallel_generation_workers` TOGETHER

**(User / system-author guidance, 2026-07-02 — validated ground truth; overrides any "duty-cycle-bound" inference.)**
For agentic RL (terminal_bench / Harbor rollouts) under **fully-async** training, the vLLM inference engines saturate
only when the TWO offered-concurrency levers scale up **together, in proportion** — they are INDEPENDENT knobs (distinct
from vLLM serving config):

1. **`terminal_bench_config.harbor.n_concurrent_trials`** — # concurrent agentic episodes (= Daytona sandboxes in flight).
   The PRIMARY offered-load lever; **scale this FIRST** to feed the engines. Engine idleness is driven first and foremost
   by this.
2. **`trainer.fully_async.num_parallel_generation_workers`** — # concurrent `generate()` calls in flight (the LLM-call
   concurrency; `skyrl_train/examples/terminal_bench/rollout_coordinator.py`, bounded in `fully_async_trainer.py`). **Must
   scale WITH `n_concurrent_trials`, in proportion** — it is the SkyRL-side cap on concurrent generation. Raising
   `n_concurrent_trials` alone with the worker pool fixed does NOT lift concurrency; both go up together, THEN the engines
   saturate. (Assert: `mini_batch_size ≤ num_parallel_generation_workers`.)

**Tuned reference — Jupiter `hpc/skyrl_yaml/jupiter/56GPU_seqnorm_tis.yaml`:** `n_concurrent_trials: 675` +
`num_parallel_generation_workers: 338` (**~2:1**, workers ≈ n/2), co-tuned ("450→338 −25%, paired with
n_concurrent_trials 900→675").

**⚠ MISDIAGNOSIS to avoid:** engine idleness (`Waiting=0`, low KV, `Running ≪ per-engine ceiling`) is **NOT** an intrinsic
"agentic duty-cycle" ceiling — it is an **offered-concurrency SUPPLY** shortfall. A timid `n_concurrent_trials` (e.g.
96–192) with an un-scaled worker pool *looks* starved/duty-cycle-bound but is simply under-offered. To saturate: raise
`n_concurrent_trials` toward the tuned reference (hundreds) **with `num_parallel_generation_workers` scaled proportionally
(~n/2)**, THEN measure Running/Waiting/KV. (This corrects the 2026-07-02 moe-grid Stage-0 "duty-cycle-bound" reading —
that was measured at n=96/192 without scaling the worker pool; see `experiments/active/moe_sharding_grid_131k`.)

## GDN + Context-Parallel (CP>1) HARD-CRASHES the 35B GatedDeltaNet MoE at forward

The **Qwen3.6-35B-A3B** MoE has 30 GatedDeltaNet (GDN) linear-attn layers with **no CP-aware kernel** → running it
under `context_parallel_size > 1` **hard-crashes at the forward pass** (not merely numerically wrong). Proven
2026-07-02 by moe-grid cell 35B-B (EP8×FSDP2×CP2): `RuntimeError: The expanded size of the tensor (33784) must match
the existing size (16892) at non-singleton dimension 3` in `FSDPPolicyWorkerBase.forward()` → `fwd_logprobs_values_reward`
— 33784 = 2×16892 = the CP=2 sequence-shard doubling the GDN attention mask; dies at step-1 forward (0 training steps).
⇒ **the 35B (GDN) model is CP=1 ONLY**; its CP>1 configs (EP8×CP2, EP8×CP4, EP16×CP2) are infeasible-by-inference. The
**30B Qwen3-Coder-30B-A3B is full-attention (no GDN)** so CP>1 is fine there. A trainable 35B-CP config would need a
CP-aware GDN kernel + a CP1-vs-CP2 logprob-parity smoke (`tests/gpu/test_cp_logprob_parity.py`), not yet built. (The
Stage-2 attn-backend pivot above forces sdpa/flex under CP for the *full-attn* path — it does NOT make GDN CP-correct.)

---

## 80B placement init-OOM = two-PACK-PG race → `policy_strict_spread_pg`

Qwen3-Next-80B-A3B init-OOM was a **two-PACK-PG race** (inference PG + lazy policy PG, no anti-affinity,
exactly-full 24 nodes → a policy worker lands on a vLLM-occupied GPU), **NOT** a ref-model issue (ref is
correctly not instantiated, `use_ref_model=False`). Fix = opt-in **`policy_strict_spread_pg`** flag (SkyRL
`6e3afc34`, OT-Agent `96df706f`): reserves the policy PG up front with STRICT_SPREAD. Default-off, so all
other RL (a3/shaped/A-B) is byte-identical. The 80B yaml
`hpc/skyrl_yaml/jupiter/extra/128GPU_qwen3_next_80b_a3b.yaml` sets it true.

---

## TIS exact-alignment hardening (2026-06-07) — VALIDATED

**Root cause of TIS rollout-logprob misalignment:** the SkyRL generator REBUILT the training response by
RE-TOKENIZING assistant message text (`apply_chat_template`/`encode_messages_subset`), then string-LCS
matched vLLM logprobs onto those ids — **never** using the exact `completion_token_ids` Harbor already
captured. Served-vs-training chat-template divergence (thinking tokens, tool-call serialization, BPE
boundaries) made the two tokenizations differ; LCS silently masked it. Plus the float logprob format was
treated as "legacy" → `extract_logprobs_from_rollout_details` returned None → TIS self-disabled.

**FIX (positions exact by construction):** `align_logprobs_by_token_ids()` zips vLLM logprobs onto training
tokens **by token id** (Harbor `completion_token_ids`); `extract_token_ids_from_rollout_details()`; float
format no longer disables TIS; LCS is last-resort and RECORDS every fallback in `AlignmentStats` → metrics
`generate/tis/{exact_match_fraction,lcs_fallback_fraction,unaligned_fraction,alignment_fail_count}`;
`worker.py` emits per-step `tis/{imp_ratio_mean,imp_ratio_capped_fraction,log_ratio_abs_mean}`
**keyset-identical on every rank** (all_reduce-safe — see the status-dict-keys section above). vLLM needs NO
change (already emits token_ids+logprobs over `/chat/completions`). Commits: MarinSkyRL
`consolidate-skyrl-20260606` `11285333`+`d32022ee`; harbor `marin/penfever/working` `8737426c` (per-turn
logprob/token-id length-parity guard in `Chat._accumulate_rollout_details` → empty-list-on-mismatch keeps
index alignment).

**SMOKE-VALIDATED** (job 651842, Qwen3-0.6B agentic harbor+daytona+vLLM, `use_tis=true`, COMPLETED rc=0,
40min): exact token-id path aligned 10882/11343 tokens (95.9%) with 0 LCS fallbacks; 46/47 assistant
messages exact, 1 failed-LOUD (462-vs-461 off-by-one) → REPORTED via `alignment_fail_count`, not masked. At
an on-policy step (`staleness_max=0`, `log_ratio_abs_mean=0.0`): `tis/imp_ratio_mean=0.79`,
`tis/log_ratio_abs_mean=0.094` nats (= inherent vLLM↔FSDP bf16 precision gap, what TIS corrects — not
misalignment), `imp_ratio_capped_fraction≈0`. (3rd commit `d32022ee`: `concatenate_generator_outputs` was
re-aggregating via `get_rollout_metrics` (reward/len only) → dropped `generate/tis/*` on the fully-async
path → never reached wandb; fixed via token-weighted merge.) Smoke config:
`hpc/skyrl_yaml/jupiter/extra/tis_smoke_0p6b.yaml`.

## Rollout/generate path is uniformly `AttributeError`-guarded → a deterministic rollout `AttributeError` is image/env, not first-party code

Investigated 2026-06-23 (32/32 iris seqnorm+TIS smoke rollout episodes hit `generate/errors/AttributeError` → reward 0.0). Two independent traces found the ENTIRE agentic rollout/generate path is robustly guarded against `AttributeError`:
- `examples/terminal_bench/terminal_bench_generator.py`: `_process_trial_result` extraction is wrapped in `except (KeyError, AttributeError, TypeError)` (~L1284); an outer handler (~L799, commit `fb102ed`) catches errors raised DURING processing (e.g. jinja2 `TemplateError` from `apply_chat_template`) and coerces them to masked.
- `skyrl_train/generators/utils.py`: `get_response_ids_and_loss_mask_from_messages` (1031) uses dict-indexing/asserts; `extract_{logprobs,token_ids,routed_experts}_from_rollout_details` (757-895) are None/dict/object-safe via `getattr`+`isinstance`.
- harbor `agents/terminus_2/terminus_2.py` + `llms/lite_llm.py`: `LLMResponse` fields all declared; response parsing is `getattr`/`.get`-safe; the TIS logprob diagnostic (commit `d16e8f49`) is in try/except.

DECISIVE on that smoke: the vLLM GENERATION engine ran FLASH_ATTN v3 + enforce_eager (the smoke's `trainer.flash_attn=false` only touches the FSDP/TRAINING side, NOT the engine) → NOT on a degraded native-rotary path → rules out a missing flash/rotary engine attribute.

**Heuristic:** a deterministic rollout `AttributeError` here is almost certainly raised inside a rollout DEPENDENCY (litellm / openai-SDK / daytona / uvloop) under the image's package pins, returned as an Exception from `asyncio.gather` and classified to `generate/errors/AttributeError`. Treat it as an IMAGE/env issue (a deps rebuild likely fixes it), NOT a MarinSkyRL/harbor source bug, unless a verbatim traceback points at first-party code. Confirm on the rebuilt gpu-rl image with rollout logs captured **live, in-window** — CoreWeave finelog retains only the init-phase log post-mortem.

## ⚠️ The RANK-0 LOGGING phenomenon — "only rank 0 logged X" ≠ "only rank 0 RAN X" (per-rank debug trap)

Most skyrl-train worker diagnostic logs are **gated to rank 0** — `if torch.distributed.get_rank() == 0:` (e.g. `init_weight_sync_state`'s `[weight-sync] hostname=/node_ip=/get_node_ip=` lines, `worker.py:452-463`) or tqdm `disable=not self.strategy.is_rank_0()` (`worker.py:1038,1533`). The collective itself runs on **all** ranks (its docstring: "called on all ranks in the worker group simultaneously"); only rank 0 emits the message. The iris finelog AGGREGATES every Ray actor's stdout into one stream tagged by actor `ip=`/`pid=`, so a gated log shows up as a single rank-0 (head-node ip) line.

**THE TRAP (bit the 2026-06-29 gs-1 wedge debug):** in that aggregated finelog you CANNOT infer "only rank 0 reached code X" from "only rank 0 logged X" for a **gated** log — all ranks ran it; only rank 0 logged. (We briefly mis-cited "only rank 0 logged `init_weight_sync_state`" as evidence the peers diverged at weight-sync; that line is rank-0-gated → an ARTIFACT, since weight-sync provably completed = all ranks participated.) Also note the per-NODE pod logs (`pod_rank1..N`) only show the `start_rl_iris_controller` "Worker rank N joined … parking until the head finishes" bootstrap — the actual rank-actors run on those nodes but log to the HEAD/finelog, so per-pod logs are NOT per-FSDP-rank views (only the head pod + finelog carry actor logs).

**The one RELIABLE per-rank signal: `WORKER_FORWARD_ENTER rank={self._rank}` (`worker.py:534`) is deliberately UNGATED** — every rank that reaches `worker.forward` prints its own rank. So "only rank 0 printed `WORKER_FORWARD_ENTER`" genuinely DOES mean only rank 0 reached the gs-1 forward (the MoE-RL async-dispatch wedge: peers' `forward` tasks never scheduled). Trust THIS marker; do not trust the gated ones.

**To localize a per-rank hang reliably:** (1) use UNGATED instrumentation (log `self._rank` unconditionally, no `if rank==0`); (2) capture per-rank FR dumps for ALL pods, not just rank 0 (`TORCH_NCCL_DEBUG_INFO_TEMP_FILE` writes `/tmp/nccl_fr_rank<N>` on every node — `peek_rl_rollouts.sh pull` grabs all pod logs, but the FR must be `kubectl cp`'d from each pod BEFORE the kill reaps them); (3) per-rank faulthandler/SIGUSR1 py-stacks (ptrace is locked cluster-wide). A SUCCESSFUL run's logs (e.g. a 32k-ctx arm, or a Jupiter run that reaches step 2-3) make the best diff — if its gated logs look identical to the failure's, that confirms gating is the artifact and the real signal is WORKER_FORWARD_ENTER.
