# #232 cp2 — residual uvloop SSL abort fix (job 930208 → relaunch) — 2026-06-20

This log tracks the #232 cp2 production run's uvloop/libuv abort saga. The
CP-MoE `dict.ndim` forward bug and the actor-concurrency-group uvloop abort were
already fixed (SkyRL `b02d758`/`6a1379f` and `47bf11f` respectively). This entry
covers the REMAINING uvloop leak that re-triggered the libuv-1.48.0 io_uring
abort via a DIFFERENT path in job 930208.

## Symptom (job 930208, CANCELLED, exp dir rl_30b_a3b_32k_cp2_3)
- Reached **genbuf 2/64**, then a `RolloutCoordinator` actor (pid 237783 /
  235352, ip 10.128.27.62/.64) hit `Fatal Python error: Aborted`.
- Traceback terminal frame: **`uvloop/sslproto.pyx:517 SSLProtocol._on_handshake_complete`**
  — the litellm→Daytona HTTPS handshake running on a **uvloop SSL transport**.
- The actor's loaded extension modules include both **`uvloop.loop`** and
  **`aiohttp._http_*`** → a uvloop.Loop() was live in the actor and aiohttp
  (litellm's transport) was driving SSL on it.
- This is DESPITE the prior fixes: BasePPOExp.run() driver reset, the
  `worker_process_setup_hook` `_force_stock_asyncio_in_worker` (`47bf11f`, sets
  the asyncio *policy*), `RAY_USE_UVLOOP=0`, and the `RolloutCoordinator.__init__`
  reset. A uvloop loop STILL existed in the actor.

## Definitive root cause (two independent facts)
1. **A policy reset cannot stop a uvloop loop.** `uvloop.new_event_loop()`
   constructs `uvloop.Loop()` DIRECTLY — it does NOT consult asyncio's
   event-loop policy. So `set_event_loop_policy(DefaultEventLoopPolicy())` (the
   `47bf11f` hook + the `__init__` reset) only governs `asyncio.new_event_loop()`
   / `get_event_loop()`; anything that calls `uvloop.new_event_loop()` outright
   (Ray's C++ CoreWorker, or aiohttp/litellm bringing up a loop) still gets a
   uvloop loop → the SSL transport runs on libuv 1.48.0 → the io_uring
   `uv__epoll_ctl_prep` abort path is reachable from the SSL handshake.
   Verified locally + in-SIF: pre-fix `uvloop.new_event_loop()` returns
   `uvloop.Loop`.
2. **`UV_USE_IO_URING=0` was inert because it never reached the actor's env.**
   libuv 1.48.0 DOES honor `UV_USE_IO_URING` — `uv__use_io_uring()` reads it via
   `getenv()` at first use and caches it atomically (confirmed against libuv
   v1.48.0 `src/unix/linux.c`). `"0"` → io_uring is never armed and the buggy
   path is dead. But the var lived only in the driver/launcher (host) env
   (`rl_launch_utils.py` only *comments* that host env should survive into
   apptainer; it is never actually exported, and grep found `UV_USE_IO_URING`
   nowhere in any of the three repos as a real export). Ray ACTOR processes
   derive their environment from `runtime_env["env_vars"]`, NOT arbitrary driver
   env — so even a host-set value never reached the RolloutCoordinator's libuv.

## The fix (belt-and-suspenders, no SIF rebuild) — SkyRL `0554dae` (penfever/working)
File: `skyrl-train/skyrl_train/utils/utils.py`
- `prepare_runtime_environment`: add `env_vars["UV_USE_IO_URING"] = "0"` (right
  after `RAY_USE_UVLOOP="0"`, ~line 1048) so it reaches EVERY actor process env
  before libuv's first `uv__use_io_uring()` getenv+cache.
- `_force_stock_asyncio_in_worker` (the worker-boot hook): (a) set
  `os.environ["UV_USE_IO_URING"]="0"` FIRST (before any libuv init; guards an
  import-order race vs the runtime-env injection); (b) after the policy reset,
  NEUTRALIZE uvloop in-process — alias `uvloop.new_event_loop` /
  `uvloop.install` / `uvloop.EventLoopPolicy` / `uvloop.Loop` to the stock
  asyncio equivalents (SelectorEventLoop / DefaultEventLoopPolicy), so NO uvloop
  loop can be created at all — covering the SSL path the bare policy reset
  missed. Guarded for uvloop-not-imported + best-effort on attribute drift;
  idempotent.

File: `skyrl-train/examples/terminal_bench/rollout_coordinator.py`
- `RolloutCoordinator.__init__`: the secondary backstop now CALLS the hardened
  `_force_stock_asyncio_in_worker()` (env var + uvloop neutralization) instead of
  only resetting the policy.

(2) makes sure no uvloop loop exists; (1) is the fallback that disables the
buggy libuv io_uring path even if some uvloop loop survives. Either alone kills
the abort; together they cover the C++ CoreWorker AND the litellm/aiohttp SSL
paths.

## Validation
- `ast.parse` clean on both files.
- Local (uvloop 0.22.1): after the hook, `UV_USE_IO_URING=0` is set;
  `uvloop.new_event_loop()`, `uvloop.Loop()` → `_UnixSelectorEventLoop`;
  `uvloop.EventLoopPolicy` → stock `DefaultEventLoopPolicy`.
- **In-SIF on Jupiter** (`_cp_fixb3.sif`, uvloop 0.22.1 = the SIF's actual
  version, editable skyrl_train via PYTHONPATH): baseline `uvloop.new_event_loop()`
  = uvloop loop; after the hook `UV_USE_IO_URING=0` and
  `uvloop.new_event_loop()`/`uvloop.Loop()` both yield SelectorEventLoop. VERIFY_OK.

## Sync
- Commit `0554dae` on marin `penfever/working`, pushed.
- `git pull` on Jupiter SkyRL clone (`/e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL`),
  fast-forward `47bf11f..0554dae`, HEAD = `0554dae`. Editable install → live.

## Relaunch
- cp2 (32k / CP2 / R3-off), `_cp_fixb3.sif`, `6node_qwen3_30b_a3b_32k_cp2.yaml`,
  detached tmux `cp2prod`, `--num_nodes 6 --reservation reformo
  --time_limit 11:59:00 --max_restarts 5`, db_reg false (auto-injected).
- RL concurrency at launch: 2 RUNNING (927673 lever1, 925740 swesmith) → 3 with
  cp2, under the ≤6 cap. (eval_SERA jobs are eval, not RL.)
- **Head jobid 930367** (chain 930368-930372 = 5 afterany restart links). Exp
  dir forked to `rl_30b_a3b_32k_cp2_4`. RUNNING 2026-06-20T00:57:05 on
  jpbo-004-[01,03-07]. Rendered sbatch confirmed `_cp_fixb3.sif`, --nodes=6,
  --reservation=reformo, and (host-level) `export UV_USE_IO_URING=0` (line 182)
  — but the FIX is the runtime_env env_vars + worker-boot hook injection, which
  is what actually reaches the actor; the host export was already present in the
  sbatch yet inert at the actor, exactly the gap diagnosed above.

## Verdict
PENDING — monitoring 930367 for genbuf advance past 2/64 (where 930208 died)
with NO `uvloop/sslproto` / `Fatal Python error: Aborted` in any
RolloutCoordinator. Will update with the genbuf milestone + abort-free
confirmation.

---

# 2026-06-20 — FIX-3: the CP-MoE 4D-mask expand crash (job 930793) — TRUE root cause + fix

## Recap of the crash (job 930793, run dir rl_30b_a3b_32k_cp2_5, CANCELLED)
Cleared all infra, reached **gs1**, then the gs1 TRAINING forward crashed at
`fully_async_trainer.py:493` / `model_wrapper.py:731`:
`Sharding propagation failed for aten.expand.default(Spec(bf16[4,1,24440,12220](R)),
[4,32,24440,24440]) on DeviceMesh((cp=2))`. (kv=12220=24440/2 → CP-sharded;
q=24440 full; head dim 1→32.)

## The FIX-1/FIX-2/initial-FIX-3 mask theory was WRONG for the SIF's transformers
The prior diagnosis (and the #232 task framing) assumed HF's Qwen3-MoE forward
BUILDS the 4D causal bias (via `create_causal_mask` → `find_packed_sequence_indices`
never returning None for monotonic positions). **That is true for transformers
4.57 (the laptop env) but NOT for the SIF.** Verified in `_cp_fixb3.sif`:
- transformers = **5.10.1**, torch = **2.11.0+cu130**.
- `find_packed_sequence_indices(monotonic)` → **returns None** (5.10.1 added the
  None-when-unpacked behavior the 4.57 source only wished for).
- `create_causal_mask(qwen3_moe cfg, attention_mask=None, monotonic pos)` with
  `_attn_implementation=sdpa` → **returns None** (no 4D mask). Only the `eager`
  path returns a bf16 `[B,1,S,S]` mask.
- A real `Qwen3MoeForCausalLM(attn_implementation="sdpa")` forward with
  `attention_mask=None` confirms `create_causal_mask` returns None and HF's
  `sdpa_attention_forward` then calls `F.scaled_dot_product_attention(...,
  attn_mask=None, is_causal=True)`.

So on the SIF, the existing FIX-1/FIX-2 path (`attention_mask=None` + monotonic
positions) ALREADY makes HF take the no-4D-mask `is_causal=True` SDPA route. HF
does not build the 4D bias. The first FIX-3 (commit `5471918`, monkeypatching
`find_packed_sequence_indices`) is INERT on 5.10.1 and was superseded.

## (a) TRUE root cause — torch's context-parallel SDPA backend builds the bias
The job-930793 **Python** traceback (recovered from the `.out`) is unambiguous:
```
model_wrapper.py:731  output = self.model(sequences_fwd, attention_mask=None, position_ids=cp_position_ids)
modeling_qwen3_moe.py:181  attn_output, attn_weights = attention_interface(...)
integrations/sdpa_attention.py:92  torch.nn.functional.scaled_dot_product_attention(..., attn_mask=None, is_causal=True)
torch/.../_context_parallel/_attention.py:966  outputs = target_fn(*args, **kwargs)   # CP monkeypatch wrapper
torch/.../tensor/_dispatch.py:251  _propagate_op_sharding_dispatch_slow_path → aten.expand bf16[4,1,24440,12220]→[4,32,24440,24440]
```
i.e. the `attn_mask` into SDPA is **None** (our path is correct). The 4D
`bf16[B,1,S_q,S_kv]` bias is **materialized by torch's CP SDPA backend**, not by
HF. With q/k/v CP-sharded on the seq dim (kv→S/cp) and `is_causal=True`, torch CP
routed the op to the **memory-efficient / cuDNN ring-attention backend**
(`_scaled_dot_product_ring_efficient_attention`), which constructs an explicit
`[B,1,S_q,S_kv/cp]` causal bias and `aten.expand`s it to all heads + full kv
`[B,H,S_q,S_kv]` — DTensor sharding-prop rejects the `S_kv/cp(12220)→S_kv(24440)`
expand. The FLASH ring backend (`_scaled_dot_product_ring_flash_attention`)
consumes `is_causal` natively and never builds a 4D bias, so it has no such
expand — but torch did not pick it by default for this (seq, dtype, head_dim)
combo on GH200. This is a **torch-internal CP-SDPA-backend** issue, NOT an HF
mask the wrapper builds — a genuinely distinct (4th) CP-shape cause from FIX-1/2.

Why the genbuf/logprob forward was "fine": it is the SAME structural path, but
the crash only manifests once a CP forward actually runs the SDPA op that gets
routed to the efficient/cuDNN backend with CP-sharded kv at full seq length (the
training micro-batch forward at gs1). Short/earlier forwards can dodge backend
selection.

## (a, cont.) The root fix (file:line + idiom)
`skyrl-train/skyrl_train/model_wrapper.py`:
- New `@contextlib.contextmanager _cp_force_flash_sdpa()` (~L120): pins
  `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION], set_priority=True)`
  for the wrapped forward; guarded no-op if the `sdpa_kernel` API is absent.
- Wrapped BOTH CP forward branches in `with _cp_force_flash_sdpa():`
  — policy `HFModelWrapper.forward` (the `elif cp_size > 1:` block, ~L753) and
  critic `_get_critic_model` (the `elif cp_size > 1:` block, ~L1189) — covering
  both the dense dict-mask path and the MoE `attention_mask=None` path (both
  route through torch CP SDPA and both benefit). CP1 / non-CP forwards never enter
  the context → byte-unchanged. The MoE branch keeps the FIX-2 monotonic
  `cp_position_ids` (still required so `create_causal_mask` returns None / so the
  is_causal route is taken; recompute-safe).
- In-SIF verified: the context pins `FLASH_ATTENTION` (`_cur_sdpa_kernel_backends()`
  → `['FLASH_ATTENTION']`). `ast.parse` clean. Dense/CP1 untouched.

## (a, cont.) Commits pushed/pulled (marin `penfever/working`)
- `5471918` — first FIX-3 (find_packed_sequence_indices monkeypatch; correct for
  transformers 4.57 but **inert on the SIF's 5.10.1**). Superseded.
- **`bee47ca`** — FIX-3 v2 (pin FLASH ring SDPA backend). THE fix. Pushed to
  marin `penfever/working`; `git pull` on Jupiter SkyRL clone fast-forwarded
  `0554dae→…→bee47ca`, HEAD = `bee47ca` (editable install → live).

## (b) FAST smoke validation
- Config `extra/6node_qwen3_30b_a3b_32k_cp2_SMOKE.yaml`, `_cp_fixb3.sif`,
  max_steps=3, 6 nodes, `--reservation reformo`, db_reg false, detached tmux
  `cp2smoke`. **Smoke jobid 932121** (exp dir `rl_30b_a3b_32k_cp2_SMOKE_4`),
  RUNNING from 10:43. SkyRL clone confirmed at `bee47ca`, sbatch confirmed
  `_cp_fixb3.sif`. The smoke uses the FULL 32k seq config (only batch/episode
  counts shrunk), so its training forward exercises the SAME CP-sharded SDPA path
  at the same seq regime → it genuinely covers the bug. MONITORING for gs≥2 with
  no `aten.expand`/sharding error. (Verdict appended below once it lands.)

## (c) real cp2 relaunch
PENDING smoke result.

## (d) escape-hatch note
The true cause turned out to be torch-CP-internal SDPA backend selection, not the
HF-mask theory the #232 task was built on. The FLASH-backend pin is a clean
wrapper-level fix (no transformers-internal surgery), so the escape hatch was NOT
invoked — BUT this WAS a distinct 4th CP-shape cause, so per the guardrail: if the
smoke shows the FLASH pin does not hold (e.g. flash kernel unavailable for these
q/k/v on GH200, or a NEW CP-shape error surfaces), STOP and recommend a holistic
FSDP2-CP-MoE attention-backend review rather than looping.

---

# FIX-3 (FLASH-backend pin, `bee47ca`) smoke re-validation — 2026-06-20

## Batch-config fix (the 932121 init failure was NOT the CP fix)
- Smoke 932121 FAILED at init (~4.5min) on `fully_async_trainer.py:313`:
  `AssertionError: train_batch_size must equal policy_mini_batch_size for fully
  async training`. The SMOKE yaml had `train_batch_size: 16` / `policy_mini_batch_size: 8`
  (the earlier 16/8 fix only addressed the DP=16 mini-batch floor, not the
  fully-async equality constraint).
- **Fix:** `train_batch_size 16 -> 8`, keep `policy_mini_batch_size: 8`,
  `eval_batch_size: 8`. 8/8/8 is the SMALLEST pair satisfying BOTH constraints:
  - fully-async equality: `8 == 8` ✓
  - DP=16 floor: `policy_mini_batch_size_per_gpu = 8 * n_samples(2) // dp(16) = 1 > 0` ✓,
    `1 % micro_train(1) == 0` ✓; `train % mini = 8%8 == 0` ✓
    (policy_dp_size = policy_world_size(16) // sequence_parallel_size(1) = 16, FSDP2
    path; CP handled in fsdp_config, not counted in this DP formula).
  - 8 prompts × 2 samples = 16 episodes/step. CP G4 seq-divisibility untouched.
- Commit **`53e6e6bc`** (OpenThoughts-Agent, penfever/working), pushed; pulled on
  Jupiter (OTAgent HEAD `53e6e6bc`). YAML parses clean (yaml.safe_load OK).
- SkyRL clone on Jupiter `/e/scratch/.../OpenThoughts-Agent/SkyRL` confirmed at HEAD
  **`bee47ca`** (= origin/penfever/working HEAD) → FLASH-backend FIX-3 v2 live on the
  cluster PYTHONPATH (editable install). The fix survives Jupiter shutdown regardless
  (committed + pushed).

## Relaunch attempt 1 — job 932194 (FAILED, 3rd config assert, NOT the CP fix)
- Smoke **932194**, exp dir `rl_30b_a3b_32k_cp2_SMOKE_5`, detached tmux `cp2smoke`,
  6 nodes (jpbo-105-[17-19,24-25,28]), `_cp_fixb3.sif`. Ray cluster formed fine,
  PYTHONPATH carried the SkyRL clone (bee47ca) + SIF confirmed `_cp_fixb3.sif`.
- FAILED at init (~2min) on `validate_batch_sizes` (utils.py:388):
  `AssertionError: train_batch_size (8) should be >= lcm of enabled DP sizes:
  policy_dp_size=16, lcm_dp_size=16`. A THIRD batch constraint the 8/8 fix missed
  (use_ref_model=False since use_kl_loss=false → lcm = policy_dp_size = 16).
  Still NOT the CP fix — pure config bringup. Cancelled.

## Final batch fix — 16/16/16 (clears all three coupled constraints)
- `train_batch_size 8 -> 16`, `policy_mini_batch_size 8 -> 16`, `eval_batch_size 16`.
  16 is the UNIQUE smallest value satisfying: (1) train==mini (fully-async),
  (2) train >= lcm_dp_size = 16, (3) mini*2//16 = 2 > 0 / train%mini = 0.
  Commit **`b5ce5812`** (OpenThoughts-Agent), pushed; pulled on Jupiter
  (HEAD `b5ce5812`). yaml.safe_load OK.

## Relaunch attempt 2 — job 932229 (the FIX-3 validation run)
- Smoke **932229**, exp dir `rl_30b_a3b_32k_cp2_SMOKE_6`, detached tmux `cp2smoke`,
  6 nodes (jpbo-105-[17-19,24-25,28]), `_cp_fixb3.sif`, 32k full-seq config. Log:
  `/e/data1/datasets/playground/ot-baf/rl_30b_a3b_32k_cp2_SMOKE_6/logs/rl_30b_a3b_32k_cp2_SMOKE_932229.out`.
  This is the ONE authorized relaunch after the trivial-config fix.

## VERDICT
PENDING — monitoring 932229 to gs≥2.
