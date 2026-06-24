#!/bin/bash
#SBATCH --job-name=opd_q3_smoke
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# MarinSkyRL NON-AGENTIC On-Policy Distillation (teacher-logits) on Leonardo.
# Teacher Qwen3-32B (vLLM, TP2) -> student Qwen3-1.7B (FSDP2, colocated vLLM gen).
# Brings up a Ray cluster across N nodes (4x A100-64GB each) inside the
# singularity sandbox, then runs the OPD trainer on the head. Identical
# bring-up to sbatch_gsm8k_grid_multinode.sh; only the trainer script
# (run_opd_qwen3_32b_to_1p7b.sh) + the model/sizing env knobs differ.
#
# GPU plan (2 nodes / 8 GPUs):
#   - Student colocated (policy FSDP <-> vLLM gen sleep-swap) on node0's 4 GPUs
#     (POLICY_NODES=1, STUDENT_NUM_ENGINES=4 TP1).
#   - Teacher Qwen3-32B TP2 (1 engine) -> own Ray PACK PG -> 2 GPUs on node1.
#   - 2 GPUs on node1 left free (headroom).
#
# Submit:
#   SMOKE (defaults below):  sbatch sbatch_opd_qwen3.sh
#   FULL  (override knobs):  sbatch --job-name=opd_q3_full --time=08:00:00 sbatch_opd_qwen3.sh \
#       MAX_STEPS=60 EPOCHS=2 TRAIN_BATCH_SIZE=64 MINI_BATCH_SIZE=64 \
#       N_SAMPLES=8 MAX_GEN_LEN=1024 TOPK=128
#   (trailing KEY=VAL are exported as sizing env; trailing trainer.X=Y hydra
#    overrides are passed through to the trainer too — see arg split below.)
set -euxo pipefail

WORK=/leonardo_work/AIFAC_5C0_290/bfeuer00
SF=/leonardo_scratch/fast/AIFAC_5C0_290/bfeuer00
SANDBOX=$SF/marinskyrl_sandbox
VENV=$SF/marin_venv
MARIN=$WORK/code/MarinSkyRL/skyrl-train
CFG=$WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo

# ---- Models (resolved from offline HF cache under $HF_HUB_CACHE) ----
export STUDENT_MODEL=Qwen/Qwen3-1.7B
export TEACHER_MODEL=Qwen/Qwen3-32B
export NUM_GPUS=4                                  # GPUs PER NODE

# ---- Sizing / placement defaults = SMOKE (overridable via trailing KEY=VAL) ----
export TRAIN_BATCH_SIZE=16
export MINI_BATCH_SIZE=16
export N_SAMPLES=4
export MICRO_FWD=2
export MICRO_TRN=1
export MAX_PROMPT_LEN=512
export MAX_GEN_LEN=512
export MAX_STEPS=2
export EPOCHS=1
export TOPK=64
export POLICY_NODES=1
export STUDENT_NUM_ENGINES=4
export STUDENT_TP=1
export TEACHER_NUM_ENGINES=1
export TEACHER_TP=2
export GPU_MEM_UTIL=0.80
export TEACHER_GPU_MEM_UTIL=0.85

# Split trailing args: KEY=VAL (no dot) -> exported sizing env; everything else
# (e.g. trainer.X=Y / generator.X=Y / +teacher.X=Y) -> hydra passthrough to trainer.
HYDRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == *=* && "$arg" != *.*=* ]]; then
        export "$arg"
    else
        HYDRA_ARGS+=("$arg")
    fi
done

# Fresh ckpt dir per job (no cross-job resume).
export CKPT_DIR=$SF/opd_ckpts/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
# Empty-variable / unsafe-path guard: NEVER `rm -rf ""` or `rm -rf /`.
if [ -z "$CKPT_DIR" ] || [ "$CKPT_DIR" = "/" ] || [ "${CKPT_DIR#$SF/}" = "$CKPT_DIR" ]; then
  echo "FATAL: refusing to rm CKPT_DIR='$CKPT_DIR' (empty or outside $SF/)." >&2; exit 1
fi
rm -rf "$CKPT_DIR"

# Offline / cache env (compute nodes have no internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=$WORK/data/hub
export HF_HUB_CACHE=$WORK/data/hub
export WANDB_MODE=offline
export VLLM_CACHE_ROOT=$SF/vllm_cache
export TRITON_CACHE_DIR=$SF/vllm_cache/triton
export FLASHINFER_WORKSPACE_BASE=$SF/vllm_cache/flashinfer
export CONTAINER_HOME=$SF/canary_home
export CONDA_BIN=$WORK/miniforge3/envs/otagent/bin
export RAY_USAGE_STATS_ENABLED=0
export VENV_PY=$VENV/bin/python
export DATA_DIR=$WORK/data/gsm8k
mkdir -p "$CKPT_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$CONTAINER_HOME"

# Short Ray temp dir (AF_UNIX socket path <=107 bytes; Lustre root too long).
export RAY_TMP=/tmp/r$SLURM_JOB_ID

# Pin cross-node fabric to routable IB NIC (ib0); eno* not routable between nodes.
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export RAY_PORT=6479

# ---- Discover head node + its IB ip ----
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
NNODES=${#NODES[@]}
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
    bash -c "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")
HEAD_IP=$(echo "$HEAD_IP" | tr -d '[:space:]')
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
echo "MN_BRINGUP: NNODES=$NNODES HEAD_NODE=$HEAD_NODE HEAD_IP=$HEAD_IP RAY_ADDRESS=$RAY_ADDRESS NODES=${NODES[*]}"

# Common singularity env string (shared by head / worker / trainer execs).
SING_ENV="HOME=$CONTAINER_HOME,PATH=$CONDA_BIN:/usr/local/bin:/usr/bin:/bin,CC=$CONDA_BIN/gcc,CXX=$CONDA_BIN/g++"
SING_ENV="$SING_ENV,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE"
SING_ENV="$SING_ENV,WANDB_MODE=offline,VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT,TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
SING_ENV="$SING_ENV,FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE,RAY_USAGE_STATS_ENABLED=0"
SING_ENV="$SING_ENV,NCCL_SOCKET_IFNAME=ib0,GLOO_SOCKET_IFNAME=ib0,RAY_ADDRESS=$RAY_ADDRESS"
SING_ENV="$SING_ENV,DATA_DIR=$DATA_DIR,STUDENT_MODEL=$STUDENT_MODEL,TEACHER_MODEL=$TEACHER_MODEL,NUM_GPUS=$NUM_GPUS,CKPT_DIR=$CKPT_DIR,LOGGER=console,VENV_PY=$VENV_PY"
# Sizing / placement knobs into the container env.
SING_ENV="$SING_ENV,TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE,MINI_BATCH_SIZE=$MINI_BATCH_SIZE,N_SAMPLES=$N_SAMPLES,MICRO_FWD=$MICRO_FWD,MICRO_TRN=$MICRO_TRN"
SING_ENV="$SING_ENV,MAX_PROMPT_LEN=$MAX_PROMPT_LEN,MAX_GEN_LEN=$MAX_GEN_LEN,MAX_STEPS=$MAX_STEPS,EPOCHS=$EPOCHS,TOPK=$TOPK"
SING_ENV="$SING_ENV,POLICY_NODES=$POLICY_NODES,STUDENT_NUM_ENGINES=$STUDENT_NUM_ENGINES,STUDENT_TP=$STUDENT_TP"
SING_ENV="$SING_ENV,TEACHER_NUM_ENGINES=$TEACHER_NUM_ENGINES,TEACHER_TP=$TEACHER_TP,GPU_MEM_UTIL=$GPU_MEM_UTIL,TEACHER_GPU_MEM_UTIL=$TEACHER_GPU_MEM_UTIL"

SING_BIND="/leonardo_work:/leonardo_work,/leonardo_scratch:/leonardo_scratch"

nvidia-smi || true

# ---- Start Ray HEAD on node 0 ----
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
    --output="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rayhead.out" \
    singularity exec --nv --no-home \
        --bind "$SING_BIND" --pwd "$MARIN" \
        --env "$SING_ENV" \
        "$SANDBOX" \
        "$VENV_PY" -m ray.scripts.scripts start --head \
            --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
            --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$NUM_GPUS" \
            --temp-dir="$RAY_TMP" --disable-usage-stats --block &
RAY_HEAD_SRUN_PID=$!

wait_for_ray() {
    for _ in $(seq 1 60); do
        if srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
            singularity exec --nv --no-home --bind "$SING_BIND" --pwd "$MARIN" \
              --env "$SING_ENV" "$SANDBOX" \
              "$VENV_PY" -m ray.scripts.scripts status --address "$RAY_ADDRESS" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "MN_BRINGUP: FATAL ray head at $RAY_ADDRESS not ready after 300s" >&2
    return 1
}
wait_for_ray || { echo "RAY_HEAD_FAIL"; scancel "$SLURM_JOB_ID"; exit 1; }
echo "MN_BRINGUP: ray head READY at $RAY_ADDRESS"

# ---- Start Ray WORKERS on nodes 1..N-1 ----
WORKER_PIDS=()
for ((i=1; i<NNODES; i++)); do
    WN=${NODES[$i]}
    echo "MN_BRINGUP: starting worker $i on $WN -> $RAY_ADDRESS"
    srun --nodes=1 --ntasks=1 -w "$WN" --overlap \
        --output="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rayw${i}.out" \
        singularity exec --nv --no-home \
            --bind "$SING_BIND" --pwd "$MARIN" \
            --env "$SING_ENV" \
            "$SANDBOX" \
            "$VENV_PY" -m ray.scripts.scripts start --address "$RAY_ADDRESS" \
                --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$NUM_GPUS" \
                --disable-usage-stats --block &
    WORKER_PIDS+=($!)
    sleep 5
done

# Wait for all expected GPUs to register.
EXPECT_GPUS=$((NNODES * NUM_GPUS))
for _ in $(seq 1 60); do
    NGPU=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
        singularity exec --nv --no-home --bind "$SING_BIND" --pwd "$MARIN" \
          --env "$SING_ENV" "$SANDBOX" \
          "$VENV_PY" -c "import ray; ray.init(address='$RAY_ADDRESS'); print(int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null | tail -1)
    echo "MN_BRINGUP: cluster GPUs=$NGPU / expected=$EXPECT_GPUS"
    [ "$NGPU" = "$EXPECT_GPUS" ] && break
    sleep 5
done
[ "$NGPU" = "$EXPECT_GPUS" ] || { echo "RAY_WORKER_JOIN_FAIL: got $NGPU expected $EXPECT_GPUS"; }

# ---- Run the OPD trainer on the HEAD (joins existing Ray cluster via RAY_ADDRESS) ----
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
    --output="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
    singularity exec --nv --no-home \
        --bind "$SING_BIND" --pwd "$MARIN" \
        --env "$SING_ENV" \
        "$SANDBOX" bash "$CFG/run_opd_qwen3_32b_to_1p7b.sh" "${HYDRA_ARGS[@]}" \
        trainer.resume_mode=null trainer.ckpt_path="$CKPT_DIR" trainer.export_path="$CKPT_DIR/exports"
TRAIN_RC=$?
echo "OPD_EXIT=$TRAIN_RC"

# ---- Teardown ----
for ((i=0; i<NNODES; i++)); do
    srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" --overlap \
        singularity exec --nv --no-home --bind "$SING_BIND" --pwd "$MARIN" \
          --env "$SING_ENV" "$SANDBOX" \
          "$VENV_PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
done
kill "$RAY_HEAD_SRUN_PID" "${WORKER_PIDS[@]}" 2>/dev/null || true
rm -rf "$RAY_TMP" || true
exit $TRAIN_RC
