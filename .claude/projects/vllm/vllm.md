# vLLM fork — dependency overview

Our **`mlfoundations/vllm`** fork: the OpenAI-compatible inference engine for RL rollouts, datagen, and
eval (spawned by `hpc/vllm_utils.py`). Written 2026-06-14 from notes + the local fork
(`/Users/benjaminfeuer/Documents/vllm`). Two divergences from upstream carry the project: **R3 routed-experts
capture** and the in-progress **DCP GQA-LSE fix**.

> **vLLM is our fork (`mlfoundations/vllm`, own upstream) — the local clone is ground truth.** Unlike
> Harbor/SkyRL/OT-Agent (editable + `git pull`), vLLM is **compiled**, so it's **built from source on each
> cluster (per-arch) from the committed fork**, or **baked into a SIF** from a committed commit. Edit the
> fork locally → commit → push → build on the cluster from that commit. **Never** rsync working-tree edits
> or hand-patch a cluster (no patch-by-rsync). Every cluster keeps at least one env with our fork built for
> it; some envs may run **vanilla** vLLM, which is fine. Version-bump the fork only when necessary.

> ### ⭐ CANONICAL BRANCH = `penfever/working` (set 2026-06-16)
> The single branch the clusters' builds/SIFs should track — the vLLM analogue of OT-Agent's
> `penfever/working`. It is the union of the prior mainline **`v2-migration`** (0.20.2rc0 / torch-2.11 + R3)
> and the **`feuer/dcp-gqa-lse-fix`** line (fp32 DCP combine fix `5d7319dd1` + DCP>1 R3 guard-lift
> `17c7c70a5` + env-gated DCP debug instrumentation, all off by default). `v2-migration` is a strict
> ancestor, so `penfever/working` contains everything. New fork work branches from here and merges back here;
> `v2-migration`/`feuer/dcp-gqa-lse-fix` are retained for history but are no longer the integration target.
> **Cluster-deploy state:** the prod SIF `skyrl_megatron_vllm0202rc0_r3.sif` is still base-`v2-migration`
> (no DCP fix) — rebuilding/baking it from `penfever/working` is tracked separately (the DCP fix only
> activates for DCP>1 runs, so non-DCP production is unaffected until then).

---

## Version lines we run

| Line | torch | Runtime | Notes |
|---|---|---|---|
| **vLLM 0.16.0** | 2.9 | RL venv + `*_r3baked.sif` (Jupiter) | the dense-RL + MoE/80B stack; carries the R3 patch |
| **vLLM 0.20.2rc0** | 2.11 | `skyrl_megatron_vllm0202rc0_r3.sif` + otagent | the new SIF; R3 upstreamed; getting the DCP fix |

- **The router patch is commit-sensitive on the 0.16 line:** `084aa19f0` is the newest torch-2.9.1-pinned fork commit that still carries the `routed_experts` **RL-emission** path — later commits bump torch (2.10→2.11). The older torch-2.9.1 bump predates the patch (has routed_experts only in upstream MoE infra, NOT the RL emission). Verify a build by grepping `gpu_model_runner`/`scheduler`/`output_processor` for the emission path, not just `vllm.__version__` (which reports `dev`/`0.1.dev…`). See `.claude/ops/jupiter/ENVIRONMENT_MAP.md` §0 (torch is the reliable discriminator).

---

## R3 routed-experts capture (the MoE router-replay transport)

Lets RL replay MoE routing: vLLM serializes which experts each token routed to, over `/chat/completions`.

- **Flag:** `enable_return_routed_experts` (engine `--return-routed-experts`); default off.
- **Protocol:** a top-level choice field `routed_experts`, shape `[gen_len, num_layers, top_k]` (int) — harvested like `token_ids` (litellm → `provider_specific_fields` → Harbor `RolloutDetail.extra["routed_experts"]` → SkyRL `extract_routed_experts_from_rollout_details` → `router_replay.py`/`moe.py`, which asserts `shape[-1]==top_k`).
- **Files:** `config/model.py` (flag), `model_executor/layers/fused_moe/{routed_experts_capturer.py,layer.py}` (GPU capture buffer + D2H copy), `v1/worker/gpu_model_runner.py`, `v1/core/sched/scheduler.py`, `v1/engine/output_processor.py`, `entrypoints/.../chat_completion/serving.py` + `protocol.py` (serialization), `outputs.py`.
- **Qwen3-Next gotcha:** the Ray **Compiled-DAG** backend deadlocks on the hybrid arch when capture is on → run with the **mp executor backend** (`generator.inference_engine_mp_backend: true`), validated clean. Plus an undersized hybrid-kv-buffer fix + defensive clip (`gmr_fix`/`scheduler_fix`/`capturer_fix` single-file binds). Full detail in `.claude/projects/marinskyrl/marinskyrl.md` / `.claude/ops/jupiter/ENVIRONMENT_MAP.md`.
- **Status:** RESOLVED on the existing prod SIF (no rebuild) — only `enable_return_routed_experts=False` ever blocked it. The FSDP2 router-replay hook exists and ran a full GRPO backprop step on the 80B (do NOT repeat the "Megatron-only, no FSDP2 replay" claim).

---

## DCP GQA-LSE fix (LANDED 2026-06-16 — on `penfever/working`, commit `5d7319dd1`)

Decode-Context-Parallel shards the decode KV cache across ranks to cut KV memory; under GQA the multi-rank
attention-output + log-sum-exp (LSE) recombination diverged from `dcp=1`.

- **Root cause = precision, not math/indexing.** The kernel `_correct_attn_cp_out_kernel` and head-slot
  indexing were correct. The AG+RS combine (`cp_lse_ag_out_rs`) did its per-rank rescale **and** the
  cross-rank `reduce_scatter` SUM in **bf16**, then re-quantized at the `merge_attn_states` boundary — the
  only DCP combine not accumulating in fp32 (the A2A sibling `_dcp_a2a_unpack_combine_kernel` already used an
  fp32 register). Under GQA the per-shard context partials are close in magnitude, so the ~3–4e-2 loss flips
  the (e.g. 128-expert top-8) router and then greedy tokens vs `dcp=1`.
- **Fix:** `v1/attention/ops/common.py` — accumulate rescale + cross-rank reduce in **fp32** in both
  `cp_lse_ag_out_rs` and `cp_lse_ag_out_ar`; add opt-in `out_fp32` (default keeps the bf16 return contract for
  FlashInfer/MLA callers). `v1/attention/backends/flash_attn.py` — `_forward_with_dcp` requests `out_fp32=True`
  (AG+RS only) and runs the context+self `merge_attn_states` in fp32, downcasting **once** at the final attn
  output (matching `dcp=1`'s single fp32-accumulated FA call). Env-gated debug instrumentation left intact (off).
- **Validated** (Qwen3-Coder-30B-A3B, 2-node tp=8, dcp=1 vs dcp=2, greedy temp=0, jobs 905658→905677→905726):
  token mismatch **24.31% → 6.94%**, prompts identical **3/6 → 5/6**.
- **Known floor (accepted):** strict routed-expert bit-exactness is **architecturally impossible** on the FA
  backend — `dcp=2` emits a separate **bf16** context partial (FA rejects an fp32 `out` with bf16 q/k/v),
  whereas `dcp=1` folds context+self into one fp32 FA call. Residual is provably bf16 tie-noise: ~99% of
  routed-expert disagreements are a single top-k expert swapped at a routing boundary. **Decision 2026-06-16:
  proceed with DCP+R3 at this parity** (same magnitude as existing bf16 rollout nondeterminism; TIS already
  corrects small train/inference routing mismatch).
- **Next:** the 30B-A3B long-ctx RL (task #232) uses DCP+R3 on this branch; resume MarinSkyRL rollout-DCP
  (task #222) — long-ctx OOM→OK. Cluster SIF must be rebuilt/baked from `penfever/working` before a DCP>1 run
  (the running bind-mount smokes are validation-only, not a production deploy).

---

## Build + branches

- **From-source build env:** `SETUPTOOLS_SCM_PRETEND_VERSION=<ver>` (required when building from a source tree without `.git` — setuptools-scm can't derive `_version.py`), `MAX_JOBS=<N>`, `TORCH_CUDA_ARCH_LIST="9.0"` (GH200/H100) / `8.0` (A100), built against the SIF's own torch for ABI match (~60–75 min compile). Full recipe + the GCC/PATH scrubbing gotchas are in `.claude/ops/jupiter/ENVIRONMENT_MAP.md` §2c (the `skyrl_megatron_vllm.sif` build notes).
- **Branches on the fork:** **`penfever/working` = CANONICAL** (integration target; v2-migration + DCP fp32 fix — see the starred banner up top). `v2-migration` (0.20.2rc0/torch-2.11 mainline + R3; now a strict ancestor of canonical), `feuer/dcp-gqa-lse-fix` (the DCP fix line, merged into canonical; retained for history), plus older debug branches (`penfever-debug-layer-split-v0.16.0`, `dp1-debug-instrumentation-*`). Push fork work to `penfever/working`; build clusters from it.
- **0.20.2rc0 SIF gotchas** (run-time, from the env map): set `VLLM_USE_FLASHINFER_SAMPLER=0` (SIF has no flashinfer), `LIBRARY_PATH=/.singularity.d/libs` for tp>1 Triton linking, and `VLLM_ATTENTION_BACKEND` is ignored on 0.20.2rc0.
