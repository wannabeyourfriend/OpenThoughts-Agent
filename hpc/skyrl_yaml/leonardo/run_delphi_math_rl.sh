#!/bin/bash
# Delphi RL scaling-laws (marin #6279) — MATH-500 GRPO, the GOLD hparams.
# STARTER TEMPLATE. Mirrors hpc/skyrl_yaml/leonardo/run_math_grid.sh but bakes in
# the gold config selected in main_rl_evals/RL_CONVENTION.md §3 (from the gsm8k
# accuracy + throughput grids, clamped to the Delphi 4k-context ceiling).
#
# Runs via skyrl_train.entrypoints.main_base inside the marinskyrl sandbox
# (singularity), fully offline, no Harbor/Daytona. Invoked by sbatch_delphi_math_rl.sh.
#
# Reward/contract: env_class=aime = the Minerva/boxed `Answer: \boxed{}` verifier on
# data built by hpc/skyrl_yaml/leonardo/math_dataset.py — the SAME grader the held-out
# MATH500 eval uses (EVAL_CONVENTION.md). Do NOT change the env or the parser.
#
# Env (from sbatch): DATA_DIR MODEL_PATH NUM_GPUS CKPT_DIR RUN_NAME [LOGGER VENV_PY]
set -x

: "${DATA_DIR:?set DATA_DIR (=\$WORK/data/math, built by math_dataset.py)}"
: "${MODEL_PATH:?set MODEL_PATH (the starting checkpoint: SFT repo, midtrained, or base)}"
: "${NUM_GPUS:=4}"
: "${CKPT_DIR:?set CKPT_DIR (fresh per-cell)}"
: "${RUN_NAME:=delphi_math_rl}"
: "${LOGGER:=console}"                 # offline → console (WANDB_MODE=offline)
: "${INFERENCE_BACKEND:=vllm}"
: "${VENV_PY:=python}"

# --- GOLD knobs (RL_CONVENTION.md §3); override only for a FLAGGED fallback cell ---
: "${LR:=1.0e-5}"                      # GRPO knee (accuracy_grid winner)
: "${N_SAMPLES:=8}"                    # GRPO group size (efficiency winner)
: "${ENTROPY_COEF:=0.01}"              # anti-collapse stabilizer (combo_acc_stab)
: "${MAX_GEN_LEN:=3584}"              # 4k ceiling − 512 prompt; == held-out eval max_tokens
: "${MAX_PROMPT_LEN:=512}"
: "${TEMPERATURE:=0.7}"                # == held-out eval temp (rollout ≡ eval)
: "${TOP_P:=1.0}"
: "${TBS:=64}"                         # long-CoT batch (memory-safe at gen3584)
: "${EPOCHS:=20}"                      # N=500 subsample ÷ tbs64 ≈ 7.8 steps/epoch → 20 ep > 100 steps (§2.2)
: "${MAX_STEPS:=100}"                  # the issue's first RL pass (the binding cap)
: "${MICRO_FWD:=4}"
: "${MICRO_TRAIN:=2}"                  # lower to 1 if a larger ckpt OOMs
: "${ENV_CLASS:=aime}"                 # aime = boxed/answer-match (D1/D3/D4); ifeval (D2) NOT yet wired

# 4 engines × TP1 (colocated): throughput-grid layout winner AND sidesteps the
# vLLM num_heads % TP constraint for every small Delphi checkpoint.
TP=1
ENGINES=$(( NUM_GPUS / TP ))

"$VENV_PY" -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP \
  trainer.epochs=$EPOCHS \
  trainer.max_steps=$MAX_STEPS \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TBS \
  trainer.policy_mini_batch_size=$TBS \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FWD \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN \
  trainer.ckpt_interval=20 \
  trainer.max_prompt_length=$MAX_PROMPT_LEN \
  generator.sampling_params.max_generate_length=$MAX_GEN_LEN \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.max_num_batched_tokens=8192 \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_entropy_loss=true \
  trainer.algorithm.entropy_loss_coef=$ENTROPY_COEF \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.enforce_eager=false \
  environment.env_class=$ENV_CLASS \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.gpu_memory_utilization=0.85 \
  generator.vllm_stats_interval=1 \
  trainer.logger="$LOGGER" \
  trainer.project_name="delphi_rl_scaling_6279" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$CKPT_DIR/exports" \
  "$@"
