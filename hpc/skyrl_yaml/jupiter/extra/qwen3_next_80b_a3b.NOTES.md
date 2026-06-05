# Qwen3-Next-80B-A3B-Instruct RL config — design rationale & risk register

Companion to `qwen3_next_80b_a3b.yaml`. Read this before launching.

## Target model facts (from config.json + model card, verified 2026-06-04)

| field | value |
|---|---|
| `model_type` / arch | `qwen3_next` / `Qwen3NextForCausalLM` |
| total / activated params | 80B total / ~3B activated (79B non-embedding) |
| layers | 48: `12 × [3×(GatedDeltaNet→MoE) + 1×(GatedAttn→MoE)]` = **36 linear-attn + 12 full-attn** |
| `full_attention_interval` | 4 (every 4th layer is full attention) |
| MoE | `num_experts: 512`, `num_experts_per_tok: 10`, `moe_intermediate_size: 512`, 1 shared expert (`shared_expert_intermediate_size: 512`), `decoder_sparse_step: 1` |
| hidden / heads | `hidden_size: 2048`, `num_attention_heads: 16`, `num_key_value_heads: 2` (GQA), `head_dim: 256` |
| GatedDeltaNet | 16 QK heads / 32 V heads, head_dim 128, `linear_conv_kernel_dim: 4` |
| rope | `rope_theta: 1e7`, `partial_rotary_factor: 0.25` (RoPE dim 64), `rope_scaling: null` |
| context | native **262144**, YaRN-extendable to ~1.01M |
| vocab / dtype | 151936 / bf16 |

## Stack-version gate (the launch blocker to verify first)

- **vLLM**: model card requires `>=0.10.2`. SkyRL `skyrl-train/pyproject.toml` pins `vllm==0.11.0` (the `vllm` extra) — **satisfies the floor**. The `mcore`/megatron extra pins `0.10.1.1`, which is BELOW the floor — do NOT use the megatron path for this model.
- **transformers**: `qwen3_next` landed in transformers 4.57.0 (config says `transformers_version: 4.57.0.dev0`). The local `otagent` env has 4.57.3 (OK), but **the RL runtime is the cluster `envs/rl` venv, NOT otagent** (per project memory). Must confirm `envs/rl` has transformers>=4.57.0 AND vllm>=0.10.2. SkyRL's FSDP worker loads the policy via `AutoConfig.from_pretrained(..., trust_remote_code=True)` (`workers/fsdp/fsdp_worker.py:121/360/431`), so an old transformers in the training venv will fail to even build the policy model.
- Optional accel: `flash-linear-attention` + `causal-conv1d` speed up the GatedDeltaNet layers. vLLM 0.11 has its own Qwen3-Next kernels; these packages are an optimization, not a hard dependency. Confirm they're present if generation throughput is poor.

## Sizing decisions

### Node count: 16 nodes / 64 GPUs (`colocate_all: false`)
- Generation: **8 vLLM engines × TP=4 = 32 GPUs = 8 nodes**.
- Policy+Ref FSDP: **8 nodes × 4 = 32 GPUs** (policy and ref share the devices, time-sharing exactly like 56GPU_base shares 2 nodes for the 8B policy/ref).
- Total = 64 GPUs = 16 nodes. Pass `--num_nodes 16`.

### FSDP memory arithmetic (why 32 policy GPUs + mandatory cpu_offload)
Per-param state for full-fidelity Adam in bf16-mixed:
```
bf16 weights : 80e9 × 2 = 160 GB
fp32 grads   : 80e9 × 4 = 320 GB
Adam m,v     : 80e9 × 8 = 640 GB
fp32 master  : 80e9 × 4 = 320 GB
            TOTAL       ≈ 1440 GB sharded state
```
- Over 32 GPUs → ~45 GB/GPU persistent. GH200 usable ≈ 85-90 GB after CUDA/NCCL reserve → ~40 GB left for activations. Tight but workable, and `cpu_offload: true` moves the params+optimizer into the node's 480 GB coherent LPDDR (NVLink-C2C), leaving HBM almost entirely for the active shard + activations. **cpu_offload is required, not optional**, at this scale.
- At only 8 policy GPUs the working-set + activation peaks are too tight even offloaded; 32 GPUs (`fsdp_size: 32`) gives headroom and cuts step time via wider sharding.

### vLLM memory
- TP=4 → 80 GB bf16 weights / 4 = 20 GB/GPU. At `gpu_memory_utilization: 0.75` on 94 GB, ~50 GB/GPU remains for KV + GDN recurrent state.
- The 36 GatedDeltaNet layers use **fixed-size recurrent/conv state** (not length-growing KV); only the **12 full-attention layers** scale KV with sequence length. So KV pressure at 32k is roughly a quarter of a 48-layer dense model — `max_model_len: 32768` is very safe at TP=4, and 0.75 util is conservative.

### TP=4, EP=1
- TP=4 is the model-card recommendation and divides 16 attn heads (4/GPU), 512 experts (128/GPU), and the GDN heads cleanly.
- Expert-parallel left at 1: with TP=4 the experts are already 128/GPU. SkyRL wires EP straight into vLLM (`enable_expert_parallel = ep_size > 1`, `ray_wrapped_inference_engine.py:228`), so it can be enabled later via `generator.inference_engine_expert_parallel_size`, but EP+TP on this MoE in vLLM 0.11 is unvalidated in our stack — start with plain TP.

### RL/throughput knobs (vs 56GPU_base 8B baseline)
- `train/mini/eval batch = 64`, `micro = 1/1`: the 80B sharded all-gather dominates, not batch size; micro 1/1 matches the 32B configs.
- `n_concurrent_trials: 384`, `num_parallel_generation_workers: 192` (~½): scaled to the smaller engine count (8 vs 48). Over-subscribing Daytona with few engines just piles resident trial state in the driver → the host-OOM/FD-exhaustion failure mode documented in 56GPU_base.
- `lr: 6e-6` (down from 8e-6): larger + sparse-MoE models are more update-sensitive; reduce collapse risk, tune up if grad_norm/log-ratio diagnostics show headroom.
- `max_grad_norm: 0.9`, `eps_clip_high: 0.05`, `rloo_n`: unchanged collapse guardrails from production.
- `max_generate_length: 4096`, `max_model_len: 32768`: production a3 values; no YaRN (native 256k >> our needs).
- `ckpt_interval: 1`: 80B checkpoints are expensive to redo; save every step so chain-restart loses ≤1 step.
- `rollout.fanout.enabled: true`: current SkyRL default; libuv asyncio fix already in coordinator `__init__` (SkyRL `2ab513a6`).

## OPEN RISKS / UNKNOWNS — confirm before launch

1. **[BIGGEST] Qwen3-Next FSDP-training support in SkyRL is unproven.** Serving (vLLM) is supported at 0.11, but RL *training* requires the FSDP2 worker to wrap, shard, forward AND backward through the GatedDeltaNet linear-attention + 512-expert MoE under `trust_remote_code`. We have never trained a `qwen3_next` policy here. The GatedDeltaNet recurrent kernels and the MoE router may not have backward kernels that play nicely with FSDP2 activation checkpointing / cpu_offload. **This needs a 1-step smoke test on ≤2 nodes before any real run.** It may need a transformers bump beyond 4.57 or fail outright.
2. **MoE router load-balancing / aux loss.** `router_aux_loss_coef: 0.001` is in the config but SkyRL's PPO loss path does not add the MoE auxiliary load-balancing loss. Under RL the router may drift toward expert collapse with no aux-loss counterweight. Watch for degenerate routing / dead experts. (No knob for this in the current SkyRL config — would be a code change.)
3. **Weight sync over NCCL for 80B + MoE.** SkyRL's `nccl` weight-sync streams FSDP-sharded weights to the vLLM engines each step. The MoE expert tensors are many small shards; the sync may be slow or hit the same fused-weight path issues seen with other MoEs (cf. the MiniMax/GLM MNNVL fused-allreduce note in project memory). `fuse_weights` is left default (false); FP8 is not used here.
4. **Chat template.** Qwen3-Next is thinking-by-default and ships its own template with a specific tool-call/thinking format. The `qwen3_thinking_acc.jinja2` used for Qwen3-8B is **not** confirmed compatible — the config leaves the custom template commented out so the model's bundled template is used. Verify interleaved-thinking history is preserved before adding a custom path.
5. **Placement override semantics.** The YAML hardcodes `policy_num_nodes/ref_num_nodes: 8`, which overrides the launcher's `--num_nodes`-based derivation (it only derives when null). You still pass `--num_nodes 16` for the SLURM allocation, but the policy footprint is fixed at 8 nodes here and the other 8 nodes serve the 8 TP=4 engines. Confirm the launcher's engine-count default (`num_nodes*gpus_per_node//tp`) doesn't fight the hardcoded `num_inference_engines: 8` — it won't, because `num_inference_engines` is set explicitly (derivation only runs when null), but sanity-check the generated Hydra config with `--dry_run`.
6. **Daytona RL concurrency cap.** Project rule: ≤6 RUNNING RL jobs per cluster. This is a single 16-node job, fine on its own, but it's a large allocation — confirm queue availability on Jupiter's 48 nodes.
7. **Pre-download.** Jupiter compute nodes have no internet. The 80B model (~160 GB) must be `snapshot_download`-ed to `$HF_HUB_CACHE` on the login node before submit (the launcher's `pre_download_dataset` path). Budget disk + time for a 160 GB pull.

## Recommended first action
`--dry_run` the launch to inspect the generated Hydra config, then run a **1-step smoke test on 2-4 nodes** (override `trainer.max_steps=1`, tiny `n_concurrent_trials`) to prove the FSDP policy builds, forwards, backwards, and weight-syncs to vLLM for `qwen3_next` BEFORE committing the full 16-node run.

---
## UPDATE 2026-06-04 — GatedDeltaNet kernels installed (risk #5 partially closed)
Installed into `envs/rl` (prebuilt aarch64 wheels, no source compile, INSTALL_EXIT=0):
`flash-linear-attention==0.5.0`, `fla-core==0.5.0`, `causal-conv1d==1.6.2.post1` (+ wheel 0.47.0).
Runtime pins UNCHANGED (torch 2.9.0+cu130 / transformers 4.57.6 / vllm 0.16.0 / flash_attn 2.8.3; Qwen3NextForCausalLM imports). GPU bf16 smoke: `causal_conv1d_fn` + `chunk_gated_delta_rule` both run, finite. **`is_fast_path_available == True`** → transformers now uses the FUSED GatedDeltaNet path (no slow torch fallback).
Backup for rollback: `/e/scratch/jureap59/feuer1/env_backups/rl_backup_20260604.tar.gz` (4.7 GB) + `rl_freeze_20260604.txt`.
**STILL OPEN (risk #1):** FSDP2 *backward* through GatedDeltaNet + 512-expert MoE under cpu_offload+activation-ckpt is unproven — the 2–4 node 1-step smoke test is still required before the full 16-node run.
