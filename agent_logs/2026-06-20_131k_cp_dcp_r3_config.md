# 2026-06-20 — 131k CP+DCP+R3 config: placement-group fix (smoke 934104)

## Failed run
- Job **934104**, `16node_qwen3_30b_a3b_131k_cp_dcp_r3_SMOKE.yaml`, 16 nodes (jpbo-117-[01-16]).
- FAILED at Ray placement-group formation, BEFORE any CP/DCP/R3 code ran.
- Log: `/e/data1/datasets/playground/ot-baf/rl_30b_a3b_131k_cpdcpr3_SMOKE/logs/rl_30b_a3b_131k_cpdcpr3_SMOKE_934104.out`

```
RuntimeError: Failed to create placement group with 8 bundles (requiring 8.0 GPUs, 8.0 CPUs total) in 180 seconds.
(autoscaler) Error: No available node types can fulfill resource request {'GPU': 8.0, 'CPU': 8.0}
  ...create_ray_wrapped_inference_engines... ray_wrapped_inference_engine.py:334
```

## Root cause (REFINED — NOT the supervisor's ref-budget hypothesis)

The supervisor's hypothesis was "policy(8 nodes/32 GPU) + ref(8 nodes/32 GPU) = all 64 GPU,
0 left for inference -> 96 > 64 oversubscription." **That mechanism is wrong: the ref model is
INERT.** `use_ref_model = use_kl_loss or use_kl_in_reward` (skyrl_train/utils/utils.py:372,
main_base eligibility :39). This config has `use_kl_loss=false`, so **no RefWorker is built and
`ref_num_nodes=8` allocates 0 GPUs**. The actual GPU budget is policy 32 + ref 0 + inference 32
= 64 = 16 nodes, which FITS. The log confirms: `get_policy_pg` reserved "8 node(s) x 4 GPU"
(32 GPU, PACK, per-GPU bundles) and SUCCEEDED, claiming 8 of 16 nodes; the failure was the
*next* step — the inference-engine PG.

**The real bug is intra-engine node-atomicity, not total budget.** With
`inference_engine_mp_backend=false` (RAY executor, mandatory for the R3+DCP path) and TP=8 > 1,
`create_ray_wrapped_inference_engines` takes the `use_per_engine_strict_pack` branch
(ray_wrapped_inference_engine.py:322): it creates one **STRICT_PACK** placement group per engine,
each demanding `per_engine_gpu_count = TP = 8` × {GPU:1} bundles that MUST co-locate on **one
node**. Jupiter nodes have only **4 GH200 each** (.claude/ops/jupiter/ops.md:6), so an 8-GPU
STRICT_PACK PG is UNPLACEABLE -> the exact `{'GPU': 8.0}` request that no node type can fulfill.

This is a HARD geometric impossibility, independent of node count:
- DCP ceiling: `dcp <= tp // num_kv_heads` (utils.py:909, vLLM init). Model has **4 KV heads**,
  so **dcp=2 REQUIRES tp >= 8**.
- A TP=8 engine needs 8 GPUs on one node. RAY executor -> 8×{GPU:1} STRICT_PACK on one node
  (impossible on 4-GPU nodes). mp executor -> one atomic {GPU:8} bundle per engine (equally
  impossible on a 4-GPU node).
- Therefore **DCP=2 / TP=8 cannot be placed on Jupiter for this 4-KV-head model at ANY node
  count.** No budget/node-count tweak fixes it.

This is exactly the conclusion the 64GPU 131k parent
(`64GPU_qwen3_30b_a3b_longctx131k_cp_dcp.yaml`) reached on 2026-06-17 (USER-APPROVED): "Option A"
(4 engines × TP8, DCP=2) died on cross-node-TP decode; "Option B" (8 engines × TP4, DCP=1) is the
proven-placeable geometry. The new r3 config had silently reverted to the dead Option-A geometry.

## Fix — adopt Option-B geometry (mirror the validated 64GPU parent)

| knob | before (dead) | after (fixed) |
|------|---------------|---------------|
| `inference_engine_tensor_parallel_size` | 8 | **4** |
| `num_inference_engines` | 4 | **8** |
| `inference_engine_decode_context_parallel_size` | 2 | **1** |

Everything else UNCHANGED: policy mesh EP8×FSDP2×CP2=32 GPU/8 nodes, `mp_backend=false` (RAY),
`async_scheduling=false`, R3 ON, 131k, CP=2, SIF, extra_env, batch sizes, node count = 16.

Applied to BOTH `16node_..._131k_cp_dcp_r3.yaml` and `..._SMOKE.yaml` (+ header rewrites).
**Filename node count unchanged (still 16 nodes) -> no rename.**

## Mesh arithmetic (post-fix)
- Policy: EP=8 × FSDP=2 × CP=2 = 32 GPU = 8 nodes.
- Ref: 0 GPU (inert — use_kl_loss=false).
- Inference: 8 engines × TP=4 (DCP=1) = 32 GPU = 8 nodes. Each TP=4 engine = exactly one
  4-GPU node -> per-engine STRICT_PACK places cleanly, on-node NVLink decode all-reduce.
- **Total: 32 + 0 + 32 = 64 GPU = 16 nodes. <= 64. FITS.**

## Divisibility re-checks (all hold)
- MoE dim-0 guard (128 experts // EP8) % FSDP2 = 16 % 2 = 8 even -> VALID.
- EP×FSDP×CP = 8×2×2 = 32 = policy GPU -> VALID.
- CP G4: 131072 % (2·2=4) = 0 -> OK.
- vLLM TP=4 divides 32 attn heads (8/GPU), 4 KV heads (1/GPU, valid GQA), 128 experts (32/GPU).
- DCP ceiling: dcp=1 <= tp//num_kv_heads = 4//4 = 1 -> LEGAL.
- Batch (unchanged, fixed in b5ce5812/53e6e6bc): policy_dp_size=32; SMOKE train==mini==32,
  32>=32, 32%32==0, mini_per_gpu = 32·2//32 = 2 (>0, %micro_train(1)==0). Production
  train==mini==64, 64>=32, 64%64==0. ref inert so lcm_dp_size=policy_dp_size=32.

## Axes status
- 131k context: KEPT (max_model_len 131072, no YaRN — Coder-Instruct native 262144).
- CP=2 ring-SDPA: KEPT (policy/ref fsdp_config unchanged).
- R3 routed-experts capture: KEPT (enable_return_routed_experts=true, moe_router_replay=true).
- DCP: RESHAPED 2 -> 1. **DCP=2 is geometrically impossible on Jupiter's 4-GPU nodes for a
  4-KV-head model** (needs an 8-GPU node). DCP=2 KV-sharding can only be validated on a
  >=8-GPU-node cluster or a <=2-KV-head model. Flagged to supervisor as the one unresolvable knob.

## Relaunch (supervisor — after push + `git pull` on Jupiter)
```
python -m hpc.launch --job_type rl \
  --rl_config ./hpc/skyrl_yaml/jupiter/extra/16node_qwen3_30b_a3b_131k_cp_dcp_r3_SMOKE.yaml \
  --model_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --train_data '["DCAgent/exp_rpt_pymethods2test-large"]' \
  --num_nodes 16 --reservation reformo --time_limit 02:00:00 \
  --job_name rl_30b_a3b_131k_cpdcpr3_SMOKE
```

## 2026-06-20 (supervisor) — relaunch after placement fix (job 934175)
- User approved running the **DCP=1** CP+R3+131k smoke on Jupiter (DCP≥2 is geometrically impossible on Jupiter for Qwen3-30B-A3B: 4 KV heads → DCP≥2 forces TP≥8 = 8 GPUs co-located on one node, Jupiter has 4 GPU/node). Real DCP≥2 deferred to ≥8-GPU-node hardware.
- Supervisor-controlled (per the no-subagent-push rule): pushed `005db144` → origin; Jupiter ff-pulled to `005db144`; config confirmed TP4 / 8 engines / DCP1, CP=2, R3 on, 131k.
- Relaunched via `/e/scratch/jureap59/feuer1/launch_131k_smoke.sh` (cloned from the proven cp2 script, 3 params changed) in tmux `cp131k_smoke` → **job 934175 RUNNING, 16 nodes jpbo-117-[01-16]**.
- Watching init past the 934104 failure point: Ray placement-group formation → vLLM TP4 engine bind → R3 capture → first 131k rollout.

## 2026-06-20 (subagent) — EngineCore init failure (job 934175): nested-Ray setup-hook leak — DIAGNOSED + FIXED (local-only)

### Symptom
Placement succeeded (policy workers loaded weights), then ALL 8 vLLM TP=4 EngineCores died at init:
```
(EngineCore) AssertionError: The env var, __RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR, is not permitted because it is reserved for the internal use.
→ RuntimeError: Engine core initialization failed. (wait_for_engine_startup)
```
Log: rl_30b_a3b_131k_cpdcpr3_SMOKE_934175.out (lines ~780-893).

### Confirmed root cause (exact code path where the reserved var leaks)
This config runs R3 (`enable_return_routed_experts=true`) with `inference_engine_mp_backend=false`, so each TP=4 vLLM engine uses vLLM v1's **`ray` distributed_executor_backend**. Chain:
1. SkyRL's top-level `ray.init` (MarinSkyRL `skyrl_train/utils/utils.py:1413`, `initialize_ray`) registers a `worker_process_setup_hook: "skyrl_train.utils.utils._force_stock_asyncio_in_worker"` (the uvloop-SIGABRT fix). Ray encodes this into the cluster's runtime_env, injected into EVERY worker process via the env var `RAY_JOB_CONFIG_JSON_ENV_VAR` (carries the hook + `env_vars["__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR"]`).
2. Inside the SkyRL `AsyncVLLMInferenceEngine` actor, vLLM's `RayDistributedExecutor._init_executor` calls `initialize_ray_cluster` → **`ray.init(address=ray_address, runtime_env=parallel_config.ray_runtime_env)`** where `ray_runtime_env` defaults to `None` (`vllm/v1/executor/ray_utils.py:592`, `vllm/config/parallel.py:222`).
3. With `runtime_env=None`, Ray's `init()` rebuilds `job_config` from the inherited `RAY_JOB_CONFIG_JSON_ENV_VAR` (`ray/_private/worker.py:1782-1809`). In `connect()` (`worker.py:2601`), `job_config.runtime_env` is truthy and contains `worker_process_setup_hook` → `upload_worker_process_setup_hook_if_needed` → `export_setup_func_module` (`ray/_private/runtime_env/setup_hook.py:73`) asserts the reserved env var is NOT already in `env_vars` — but it IS (inherited from the parent) → AssertionError, ×8 engines.

The validated cp2 rung never hit this because it ran R3-OFF on the `mp` executor backend (no nested `ray.init`). This is the first time the R3-on + ray-executor path was exercised in this stack.

### Fix (narrow, gated; MarinSkyRL editable — no SIF rebuild)
File: `MarinSkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py`, in `setup_envvars_for_vllm`'s existing `distributed_executor_backend == "ray"` branch (runs in `BaseVLLMInferenceEngine.__init__` BEFORE `_create_engine` spawns EngineCore). Added:
```python
os.environ.pop("RAY_JOB_CONFIG_JSON_ENV_VAR", None)
os.environ.pop("__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR", None)
```
This drops the inherited parent job_config + the encoded reserved var from this process (EngineCore subprocesses inherit the stripped env), so the nested `ray.init` builds a clean runtime_env (no stale hook to re-upload) and still connects via `RAY_ADDRESS`. The asyncio-policy hook already ran in this worker at process start, so dropping it for vLLM's internal TP workers is harmless (they are not RolloutCoordinators).

**Why safe / no regression:** gated on the `ray` executor backend ONLY. The validated `mp` path (Qwen3-Next R3) and the `uni` path (TP=1) never call a nested `ray.init`, so they are byte-identical.

### Local commit (NOT pushed)
- MarinSkyRL `penfever/working` @ **e0f52bb** — `fix(rl): strip inherited Ray job_config/setup-hook env before vLLM nested ray.init (R3 ray-executor path)`.
- py_compile clean. **Editable install → live on Jupiter after the supervisor pulls; NO SIF rebuild.**

### Supervisor TODO before relaunch
1. `git push` MarinSkyRL `penfever/working` (commit e0f52bb).
2. **`git pull` MarinSkyRL on Jupiter** (REQUIRED — editable install; the fix is live immediately after pull, no SIF rebuild, no OT-Agent change).
3. Relaunch the smoke (job_name rl_30b_a3b_131k_cpdcpr3_SMOKE), watch the 8 TP=4 EngineCores past init.

---

## R3 ray-executor EngineCore init — FIX-2 (e0f52bb was INSUFFICIENT) — 2026-06-20 PM

### Symptom (job 934209, same as 934175)
Identical `AssertionError: The env var, __RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR, is not permitted...` STILL fired, but in the **EngineCore SUBPROCESS** (`(EngineCore pid=...)` ... `core.py:1133 EngineCoreProc(...)` -> `ray_executor_v2._init_executor` -> `ray_utils.py:592 ray.init(address=..., runtime_env=parallel_config.ray_runtime_env)` where `ray_runtime_env` is **None** -> `worker.py:2610 connect()` -> `upload_worker_process_setup_hook_if_needed` -> `setup_hook.py:73 export_setup_func_module` assert). Timing-checked: e0f52bb was pulled to Jupiter @16:36 local, job 934209 started 16:36:47, actors imported the engine module @16:41 — so the run genuinely ran e0f52bb. The pop in `setup_envvars_for_vllm` ran in all 8 actors (line-182 log "[repeated 7x]").

### Confirmed mechanism (in-SIF, `skyrl_megatron_vllm0202rc0_r3_cp_fixb3.sif`)
- The EngineCore is a plain `spawn` multiprocessing child of the `AsyncVLLMInferenceEngine` actor (`get_mp_context()` in `vllm/v1/engine/utils.py:144 context.Process(...)`); **no explicit `env=`** is passed, so it inherits the actor's `os.environ` snapshot at `proc.start()`.
- The assert is on the **runtime_env dict** rebuilt from `RAY_JOB_CONFIG_JSON_ENV_VAR` (`worker.py:1781`): if that inherited job_config carries `worker_process_setup_hook` + the reserved `__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR` in its `env_vars`, the child's `connect()` (SCRIPT_MODE/driver) re-uploads the hook and asserts.
- **In-SIF Ray-actor repro (poisoned RAY_JOB_CONFIG, child does the real nested `ray.init(runtime_env=None)`):**
  - **NO-POP:** child env has both vars -> nested ray.init -> **exact AssertionError**.
  - **WITH-POP (pop the 2 vars in the actor BEFORE the spawn):** child env clean -> nested ray.init -> **OK**.
  This proves the os.environ-strip mechanism is correct AND that a `spawn` child inherits the popped (clean) env.
- Searched vLLM `v1/engine/{async_llm,core_client,utils}.py` + `v1/executor/ray_utils.py,ray_executor_v2.py`: **nothing re-writes the two vars** between a SkyRL pop and the spawn; `_maybe_force_spawn` only sets `RAY_ADDRESS`.

### Why e0f52bb didn't reach the EngineCore
e0f52bb pops at the very START of `BaseVLLMInferenceEngine.__init__` (`setup_envvars_for_vllm`), long before the EngineCore spawn. The var that matters (`RAY_JOB_CONFIG_JSON_ENV_VAR`, carrying the poisoned hook env_vars) is (re-)present in the actor's `os.environ` at the *spawn moment* — injected by Ray's job/runtime-env machinery during engine init, AFTER the early pop ran (vLLM provably does not inject it). So the early pop was effectively a no-op for the spawn; the child inherited the poisoned env.

### FIX-2 (lever 1: spawn-site strip — the LAST SkyRL-controlled point before the EngineCore spawn)
File `MarinSkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py`, `AsyncVLLMInferenceEngine._create_engine`, immediately before `vllm.AsyncLLMEngine.from_engine_args(...)` (the call that spawns EngineCore), gated on `distributed_executor_backend == "ray"`:
```python
if kwargs.get("distributed_executor_backend") == "ray":
    os.environ.pop("RAY_JOB_CONFIG_JSON_ENV_VAR", None)
    os.environ.pop("__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR", None)
```
This is strictly LATER than e0f52bb's pop — it runs after all of vLLM's pre-spawn config (and after any Ray-side re-injection during engine init), and immediately before the `spawn` `proc.start()`. Since vLLM provably never re-writes the two vars in the spawn path after this point, the EngineCore inherits a clean env -> its nested `ray.init` builds a clean runtime_env and connects via `RAY_ADDRESS`.

**Verification it reaches the EngineCore:** the in-SIF WITH-POP repro above (pop in the spawning process immediately before the `spawn` child) is exactly this position and yields child env JOB=False/HOOK=False and nested ray.init OK.

**Gating / mp-path safety:** gated on `distributed_executor_backend == "ray"` (same key `setup_envvars_for_vllm` uses). The `mp` (Qwen3-Next R3) and `uni` (TP=1) paths never enter this branch -> byte-identical. `os` is module-level imported; the pop is inside the EADDRINUSE retry loop (idempotent). e0f52bb is KEPT as belt-and-suspenders (also gated, harmless if a no-op).

**Did NOT use lever 3 (strip in `_force_stock_asyncio_in_worker`):** that hook runs in EVERY worker (policy/ref/inference + children), so stripping there is NOT byte-identical for the non-R3 paths — violates the gating constraint. The gated spawn-site strip is the minimal lever that satisfies "non-R3 paths byte-identical".

### Local commit (NOT pushed)
- MarinSkyRL `penfever/working` — see SHA below. py_compile clean. Editable install -> live on Jupiter after pull; **NO SIF rebuild**.

### Supervisor TODO before relaunch (UNCHANGED from FIX-1, push the new commit)
1. `git push` MarinSkyRL `penfever/working`.
2. **`git pull` MarinSkyRL on Jupiter** (REQUIRED — editable install, live after pull, no SIF rebuild).
3. Relaunch: `python -m hpc.launch --job_type rl --rl_config ./hpc/skyrl_yaml/jupiter/extra/16node_qwen3_30b_a3b_131k_cp_dcp_r3_SMOKE.yaml --model_path Qwen/Qwen3-Coder-30B-A3B-Instruct --train_data '["DCAgent/exp_rpt_pymethods2test-large"]' --num_nodes 16 --reservation reformo --time_limit 02:00:00 --job_name rl_30b_a3b_131k_cpdcpr3_SMOKE` — watch the 8 TP=4 EngineCores past init.

---

## R3 ray-executor EngineCore init — FIX-3 (the actual cure, IN THE vLLM FORK) — 2026-06-20 (subagent)

### Why FIX-1 (e0f52bb) and FIX-2 (05f23c7) both failed
Both popped `RAY_JOB_CONFIG_JSON_ENV_VAR` / the reserved var in the **parent SkyRL vLLM actor**
(`vllm_engine.py`, `setup_envvars_for_vllm` / `_create_engine`). Ray re-injects the job-config env
into worker/child processes via its own machinery, so by the time the spawned EngineCore performs ITS
OWN nested `ray.init`, its `os.environ` again carries the poisoned job_config. The assertion fires in
the **EngineCore process**, during its own `ray.init` — so the only point that provably stops the leak
is a strip IN THAT PROCESS, at the nested-init call site inside vLLM's `ray_utils.py`.

### Confirmed Ray mechanism (read in-SIF, ray 2.51.1)
`initialize_ray_cluster` (`vllm/v1/executor/ray_utils.py:592`) calls
`ray.init(address=..., runtime_env=parallel_config.ray_runtime_env)` with `ray_runtime_env=None`.
In `ray/_private/worker.py:init` (≈1778-1822): when `RAY_JOB_CONFIG_JSON_ENV_VAR` is in `os.environ`,
Ray rebuilds `injected_job_config = JobConfig.from_json(json.loads(env))` and **`_merge_runtime_env`-merges**
its `runtime_env` with the driver's (default `override=False`; env_vars merged per-key). So passing an
explicit `runtime_env` dict does NOT bypass it — the injected `worker_process_setup_hook` +
`env_vars["__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR"]` survive (and an explicit env_vars could even raise
the merge-conflict ValueError). `connect()` -> `upload_worker_process_setup_hook_if_needed` ->
`export_setup_func_module` (`setup_hook.py:73`) then asserts the reserved var is NOT already in env_vars
— it IS -> AssertionError. **The only provably-correct minimal fix is to pop `RAY_JOB_CONFIG_JSON_ENV_VAR`
in this process before the nested ray.init** (Ray then takes its `job_config is None` path, no injected hook
to re-upload; connection is still via `RAY_ADDRESS`).

### The fix (vLLM fork)
- File: `vllm/v1/executor/ray_utils.py`. Added `_strip_leaked_setup_hook_job_config()` (before
  `initialize_ray_cluster`) and call it immediately before BOTH nested `ray.init` call sites (the CUDA `else`
  branch line ~595, and the rocm/xpu branch). The helper pops `RAY_JOB_CONFIG_JSON_ENV_VAR` +
  `__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR`.
- **Guarded / minimal:** no-op unless an inherited `RAY_JOB_CONFIG_JSON_ENV_VAR` is present AND actually
  references a setup hook (`"worker_process_setup_hook"`/reserved-var substring check) -> non-leaked paths
  (and the `ray.is_initialized()` early-return, which does no nested init) are byte-identical. The two
  env-var name literals == the Ray constants' values (verified in-SIF).
- **Local commit (NOT pushed):** vLLM fork `penfever/working` @ **c5832db29e933cafde40d06ebffb50f3b8ceab57**,
  based on **4d167a4af** (the commit baked into `_cp_fixb3`; verified c5832db29 descends from it; delta is
  ray_utils.py ONLY, Python-only -> surgical rebake, no recompile).
- Note: the two SkyRL-side fixes (e0f52bb, 05f23c7) stay; they're harmless gated no-ops now. This fork fix is
  the cure.

### In-SIF VERIFICATION (faithful, on GH200) — required gate, PASSED
On a `booster` GH200 (`srun`, throwaway scratch dir, cleaned up), inside `_cp_fixb3.sif`:
constructed the EXACT poisoned `RAY_JOB_CONFIG_JSON_ENV_VAR` via Ray's own `setup_hook.export_setup_func_module`
(`{"runtime_env":{"worker_process_setup_hook":"...","env_vars":{"__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR":"..."}}}`),
exported it into the process env (mimicking the EngineCore inheriting the SkyRL actor's env), started a real
ray head, then drove the REAL `initialize_ray_cluster` (nested `ray.init(address=...)`, `PLATFORM_CUDA=true`):
- **STOCK** (`/opt/vllm_build/.../ray_utils.py`): `NESTED_INIT=ASSERTERROR`, exact message
  `The env var, __RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR, is not permitted because it is reserved for the internal use.`
- **PATCHED** (the fixb4 ray_utils.py): same poisoned start state (`JOB_CONFIG_HAS_HOOK_AT_START=true`,
  `RESERVED_VAR_AT_START=true`) -> `NESTED_INIT=OK`, `JOB_CONFIG_AFTER=false`, `RESERVED_AFTER=false`
  (nested ray.init connected to the running cluster).
Repro fidelity: drives the actual failing call (`initialize_ray_cluster` -> `worker.py:1782` job_config
rebuild -> `connect()` -> `export_setup_func_module`) on a GH200 with the exact poisoned env Ray injects.
The only simplification vs the real run: world_size=1 / no real TP engine spawn — irrelevant, the failure is
purely in the nested-init/job_config path, which is reproduced with the real code and the real assertion.

### Supervisor TODO before relaunch
1. `git push` vLLM fork `penfever/working` (commit c5832db29).
2. Rebake: run `.claude/ops/jupiter/sif_build/recipes/rebake_cp_fixb4_r3_rayhook.sh` on a Jupiter login node
   (fetches+checks out c5832db29 on `/e/scratch/jureap59/feuer1/vllm`, swaps ray_utils.py into a NEW
   `skyrl_megatron_vllm0202rc0_r3_cp_fixb4.sif` built alongside _cp_fixb3, clears pyc, validates import + helper).
3. Config repoint `_cp_fixb3.sif` -> `_cp_fixb4.sif`:
   - `hpc/skyrl_yaml/jupiter/extra/16node_qwen3_30b_a3b_131k_cp_dcp_r3_SMOKE.yaml` line 320 (`sif:`)
   - `hpc/skyrl_yaml/jupiter/extra/16node_qwen3_30b_a3b_131k_cp_dcp_r3.yaml` line 486 (`sif:`)
   (commit+push the YAML edits, then `git pull` OT-Agent on Jupiter so the launcher reads the new path.)
4. Relaunch the SMOKE; watch the 8 TP=4 EngineCores past init -> first 131k rollout.
