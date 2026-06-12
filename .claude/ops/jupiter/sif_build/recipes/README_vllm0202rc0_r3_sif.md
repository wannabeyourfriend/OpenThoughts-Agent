# SIF recipe: `skyrl_megatron_vllm0202rc0_r3.sif`

Durable reconstruction recipe for the (previously purged) #208/#211 "shelf asset":
a SkyRL + Megatron + TE SIF whose vLLM is the **penfever fork @ v0.20.2rc0 with
native R3 routed-experts capture**, plus Gemma4 / Qwen3-MoE / Qwen3-Next model
support, built for Jupiter GH200 (aarch64, CUDA 13.0).

This file + `build_vllm0202rc0_r3_sif.sbatch` ARE the recipe — committed to git so
the SIF is reproducible after the next scratch purge. (The vLLM fork *source* is
NOT in this repo; it is rsync'd to the cluster per the project's vLLM-sync rule —
see "vLLM fork source" below for the exact commit to rsync.)

## Output
`/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif`

## Ingredients

| Ingredient | Identity | Notes |
|---|---|---|
| **Base SIF** | `containers/skyrl_megatron.sif` | NGC 25.09 base, **NO vLLM**. torch `2.9.0a0+50eac811a6.nv25.09` (CUDA 13.0, aarch64), pytorch-triton `3.4.0+gitc817b9b6`, Megatron-core `0.14.0`, megatron-bridge `0.1.0rc4`, TE 2.7, apex, `flash_attn 2.7.4.post1`, transformers `5.10.1` (gemma4/gemma4_text/qwen3_next/qwen3_moe all present), SkyRL editable at `/opt/SkyRL` (`penfever/SkyRL@2ab513a6`), nvcc CUDA 13.0. |
| **vLLM fork source** | `/Users/benjaminfeuer/Documents/vllm`, branch `v2-migration`, HEAD **`1948bebd1968688f2eac8f30ecc1e418df7118b5`** (`git describe` = `v0.20.2rc0-305-g1948bebd1`, 2026-05-21) | Built **from source against the in-SIF torch 2.9** via `use_existing_torch.py` (the fork pins `torch==2.11.0` in `requirements/cuda.txt`; we strip that pin to keep the NGC torch the whole Megatron/TE/apex/flash_attn/flashinfer stack is built on). R3 is **native** — `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` + `routed_experts` emission in all four `entrypoints/openai/{chat_completion,completion}/{serving,protocol}.py` files (capture rail landed via PR #39917, 2026-05-07). gemma4 models present (`gemma4.py`, `gemma4_mm.py`); registry has `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`. **NO separate R3 patch is applied** — unlike the 0.16 SIF which needed `vllm_routed_experts_http_serialization.patch` via `vllm_http_overlay.img`, the 0.20.2rc0 fork carries it natively. |
| **GDN overlay** | `containers/fla_tilelang_overlay.img` | tilelang `0.1.8`, FlashQLA (`flash_qla 0.1.0+6ef4858`, QwenLM git `6ef4858`), apache-tvm-ffi `0.1.9`, fla `0.5.0` (masked/broken; FlashQLA is self-contained). Merged via `debugfs rdump /upper` (no fuse, no root), same as `bake_r3_sif.sbatch`. Provides fused GatedDeltaNet fwd+bwd for Qwen3-Next (Stage-8 validated, 4.8–27× speedup). |

## Why build from source (not a prebuilt wheel)
There is a prebuilt `vllm-0.20.2+cu130torch2.11-cp312-...whl` at
`/e/data1/datasets/playground/ot-baf/wheels/`, but it is (a) `0.20.2` not `rc0`,
and (b) built against **torch 2.11**, which mismatches the SIF's torch 2.9 and
would force a torch bump that shatters Megatron-core/TE/apex/flash_attn/flashinfer
(no NGC 2.11 aarch64 wheel exists). So we compile the fork from source against the
in-SIF NGC torch 2.9 — exactly what #208/#211 did.

## Build steps (encoded in `build_vllm0202rc0_r3_sif.sbatch`)
1. `apptainer build --sandbox` from `skyrl_megatron.sif`.
2. Stage fork source into `$SANDBOX/opt/vllm_build`.
3. Inside sandbox (`apptainer exec --writable`): `python use_existing_torch.py`
   (strips torch pins) → `pip install --no-build-isolation -v -e .` (compiles
   CUDA/C++ kernels with the SIF's nvcc; `TORCH_CUDA_ARCH_LIST=9.0+PTX` for GH200,
   `MAX_JOBS=48`). Editable install — source stays at `/opt/vllm_build`.
4. Merge `fla_tilelang_overlay.img` via `debugfs rdump`.
5. `apptainer build` the final SIF.
6. Validate (see below).

## How to run (on Jupiter)
```bash
# 1) rsync the fork source to the cluster (per project vLLM-sync rule).
#    Pin the exact commit first:
cd /Users/benjaminfeuer/Documents/vllm
git checkout v2-migration   # HEAD must be 1948bebd1...
git rev-parse HEAD > .vllm_commit
rsync -az --delete \
  --exclude '.git' --exclude 'build' --exclude '*.so' --exclude '.deps' \
  -e "ssh -i ~/.ssh/id_ed25519_jsc -o AddressFamily=inet" \
  /Users/benjaminfeuer/Documents/vllm/ \
  feuer1@login02.jupiter.fz-juelich.de:/e/scratch/jureap59/feuer1/sif_build/vllm_src/

# 2) place this sbatch on the cluster and submit:
ssh Jupiter 'mkdir -p /e/scratch/jureap59/feuer1/sif_build/logs'
scp sif_build/recipes/build_vllm0202rc0_r3_sif.sbatch \
  Jupiter:/e/scratch/jureap59/feuer1/sif_build/
ssh Jupiter 'sbatch /e/scratch/jureap59/feuer1/sif_build/build_vllm0202rc0_r3_sif.sbatch'
```

## Acceptance / validation (asserted in step 5 of the sbatch)
- `vllm.__version__` == `0.20.2rc0` (dev tree may render `0.20.2rc0.devN+g<sha>`).
- `ModelRegistry.get_supported_archs()` contains `Gemma4ForConditionalGeneration`,
  `Gemma4ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`.
- `routed_experts` present in the four chat/completion serving+protocol files and
  `routed_experts_capturer.py` exists (native R3 capture).
- transformers `gemma4`/`gemma4_text` in `CONFIG_MAPPING_NAMES` (≥ 5.10.1).
- `import skyrl_train, skyrl_gym` OK (training stack intact).
- GDN overlay: `tilelang 0.1.8`, `apache-tvm-ffi 0.1.9`, `flash_qla 0.1.0+6ef4858`.

## Notes / gotchas
- The base SIF already has nvcc (CUDA 13.0) and the full training stack; only vLLM
  and the GDN overlay are added.
- `use_existing_torch.py` MUST run before `pip install` or pip will try to pull
  torch 2.11 and break the stack.
- Build scratch lives on GPFS (`/e/scratch/.../sif_build/job_<id>`); node-local
  `/tmp` (96G) is too small for sandbox + extracts + SIF.
- vLLM kernel compile is the long pole (~1–2h on 48 cores); 5h wall is generous.
- Do NOT perturb triton/torch/flash_attn/flashinfer in the sandbox — vLLM is built
  against the exact in-SIF versions.
