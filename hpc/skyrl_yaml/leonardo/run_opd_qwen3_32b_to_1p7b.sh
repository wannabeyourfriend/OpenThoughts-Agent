#!/bin/bash
# MarinSkyRL NON-AGENTIC On-Policy Distillation (OPD, teacher-logits) canary/full
# for Leonardo (CINECA) — teacher Qwen3-32B -> student Qwen3-1.7B.
#
# WHY non-agentic: the agentic OPD entrypoint
# (examples.terminal_bench.entrypoints.main_tbench_opd_logits) drives Harbor ->
# Daytona cloud sandboxes, which require external internet. Leonardo compute
# nodes have NO internet, so that path is INFEASIBLE here. This script uses the
# NON-AGENTIC sibling entrypoint
#   examples.on_policy_distillation_logits.main_on_policy_distill_logits
# which runs exactly like the gsm8k GRPO canary (local parquet prompts + the
# gsm8k env) but swaps the environment reward for teacher-logit KL. No Harbor /
# Daytona / terminal_bench / proxyserver. The teacher KL replaces the env reward
# (advantage_estimator=no_op + use_kl_in_reward=true), so the gsm8k rule reward
# is unused — gsm8k is just the on-policy prompt source.
#
# Model note: the directive asked for "student Qwen3-1.5B" but Qwen3 has NO 1.5B
# dense variant (sizes: 0.6B / 1.7B / 4B / 8B / 14B / 32B). Qwen3-1.7B is the
# nearest size and shares Qwen3-32B's tokenizer (identical vocab), so the
# teacher<->student retokenization is a no-op (the cleanest distillation case).
#
# Invoked INSIDE marinskyrl_sandbox via singularity exec from the accompanying
# sbatch (sbatch_opd_qwen3.sh). Sized via env knobs so the same script serves
# both the SMOKE test and the FULL run; the sbatch sets the knobs.
#
# Required env (exported by the sbatch):
#   DATA_DIR      -> dir with train.parquet + validation.parquet (local, gsm8k)
#   STUDENT_MODEL -> local/offline HF id for the student (Qwen/Qwen3-1.7B)
#   TEACHER_MODEL -> local/offline HF id for the teacher (Qwen/Qwen3-32B)
#   NUM_GPUS      -> GPUs per node (4 on Leonardo)
#   CKPT_DIR      -> writable ckpt dir on scratch
#   VENV_PY       -> external uv venv python
# Sizing knobs (sbatch sets these; defaults here = smoke):
#   TRAIN_BATCH_SIZE MINI_BATCH_SIZE N_SAMPLES MICRO_FWD MICRO_TRN
#   MAX_PROMPT_LEN MAX_GEN_LEN MAX_STEPS EPOCHS TOPK
#   STUDENT_NUM_ENGINES STUDENT_TP POLICY_NODES
#   TEACHER_NUM_ENGINES TEACHER_TP GPU_MEM_UTIL TEACHER_GPU_MEM_UTIL
set -x

: "${DATA_DIR:?set DATA_DIR}"
: "${STUDENT_MODEL:?set STUDENT_MODEL}"
: "${TEACHER_MODEL:?set TEACHER_MODEL}"
: "${NUM_GPUS:=4}"
: "${CKPT_DIR:?set CKPT_DIR}"
: "${LOGGER:=console}"
: "${VENV_PY:=python}"

# --- Sizing defaults = SMOKE (fast single-step OPD inside debug window) ---
: "${TRAIN_BATCH_SIZE:=16}"
: "${MINI_BATCH_SIZE:=16}"
: "${N_SAMPLES:=4}"
: "${MICRO_FWD:=2}"
: "${MICRO_TRN:=1}"
: "${MAX_PROMPT_LEN:=512}"
: "${MAX_GEN_LEN:=512}"
: "${MAX_STEPS:=2}"
: "${EPOCHS:=1}"
: "${TOPK:=64}"

# --- Placement / engine knobs ---
# Student is colocated (policy FSDP <-> vLLM gen sleep-swap) on POLICY_NODES*NUM_GPUS GPUs.
# Teacher gets its OWN Ray placement group (PACK) and lands on the remaining free GPUs.
: "${POLICY_NODES:=1}"               # nodes for the (colocated) student
: "${STUDENT_NUM_ENGINES:=4}"        # student vLLM engines (TP1) -> 1 per GPU on node0
: "${STUDENT_TP:=1}"
: "${TEACHER_NUM_ENGINES:=1}"
: "${TEACHER_TP:=2}"                 # Qwen3-32B bf16 ~64GB -> needs TP>=2 on A100-64GB
: "${GPU_MEM_UTIL:=0.80}"            # student engines
: "${TEACHER_GPU_MEM_UTIL:=0.85}"

"$VENV_PY" -m examples.on_policy_distillation_logits.main_on_policy_distill_logits \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="no_op" \
  trainer.algorithm.policy_loss_type="importance_sampling" \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=true \
  trainer.policy.model.path="$STUDENT_MODEL" \
  trainer.ref.model.path="$TEACHER_MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=$POLICY_NODES \
  trainer.placement.ref_num_nodes=$POLICY_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$STUDENT_NUM_ENGINES \
  generator.inference_engine_tensor_parallel_size=$STUDENT_TP \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  generator.gpu_memory_utilization=$GPU_MEM_UTIL \
  generator.enforce_eager=true \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.sampling_params.max_generate_length=$MAX_GEN_LEN \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  teacher.model_path="$TEACHER_MODEL" \
  teacher.num_inference_engines=$TEACHER_NUM_ENGINES \
  teacher.inference_engine_tensor_parallel_size=$TEACHER_TP \
  teacher.top_k_logprobs=$TOPK \
  teacher.gpu_memory_utilization=$TEACHER_GPU_MEM_UTIL \
  teacher.enforce_eager=true \
  +teacher.engine_init_kwargs.max_model_len=$((MAX_PROMPT_LEN + MAX_GEN_LEN)) \
  environment.env_class=gsm8k \
  trainer.epochs=$EPOCHS \
  trainer.max_steps=$MAX_STEPS \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FWD \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRN \
  trainer.max_prompt_length=$MAX_PROMPT_LEN \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.ckpt_interval=999999 \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.logger="$LOGGER" \
  trainer.project_name="opd_qwen3" \
  trainer.run_name="leonardo_opd_q3-32b_to_1p7b" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$CKPT_DIR/exports" \
  "$@"
