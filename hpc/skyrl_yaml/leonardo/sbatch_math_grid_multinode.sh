#!/bin/bash
#SBATCH --job-name=marinskyrl_math_grid_mn
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# MarinSkyRL NON-AGENTIC MATH (long-CoT) GRPO grid — MULTI-NODE BIG-MODEL variant.
# EXTENSION 2 / AXIS A: run the models that OOM single-node (Qwen3-32B dense, and
# the Qwen3-30B-A3B MoE) ACROSS N nodes (4x A100-64GB each), bringing up a Ray
# cluster INSIDE the singularity sandbox and launching the trainer on the head
# with multi-node FSDP placement. Reuses run_math_grid.sh (MATH/aime env,
# tbs64 x n4, cudagraph ON, max_generate 4096) -- only the model + node count +
# (for MoE) the EP/grouped-GEMM flags change.
#
# Per-cell env overrides (via --export): MODEL_PATH, NUM_GPUS (per node).
# Submit with --nodes=N --job-name=mathmn_<cell> + trailing trainer flags. E.g.
#   Qwen3-32B dense, 2 nodes / 8 GPU, 8xTP1 vLLM, FSDP across both nodes:
#   sbatch --nodes=2 --job-name=mathmn_32b_nd2 \
#       --export=ALL,MODEL_PATH=Qwen/Qwen3-32B \
#       sbatch_math_grid_multinode.sh \
#       generator.num_inference_engines=8 generator.inference_engine_tensor_parallel_size=1 \
#       trainer.placement.policy_num_gpus_per_node=4 trainer.placement.policy_num_nodes=2 \
#       trainer.placement.ref_num_gpus_per_node=4 trainer.placement.ref_num_nodes=2 \
#       trainer.placement.critic_num_gpus_per_node=4 trainer.placement.critic_num_nodes=2 \
#       trainer.placement.colocate_all=true trainer.max_steps=25
#   Qwen3-30B-A3B MoE adds (proven combo, e2e test test_e2e_moe_rl_step):
#       trainer.policy.fsdp_config.moe_grouped_gemm=true \
#       trainer.policy.fsdp_config.moe_router_replay=true \
#       trainer.policy.fsdp_config.expert_model_parallel_size=<EP> \
#       trainer.ref.fsdp_config.moe_grouped_gemm=true trainer.ref.fsdp_config.moe_router_replay=true \
#       trainer.ref.fsdp_config.expert_model_parallel_size=<EP> \
#       trainer.gradient_checkpointing=false +generator.engine_init_kwargs.enable_expert_parallel=true
#
# Container / env approach is identical to the 1-node grid launcher
# (singularity exec --nv --no-home, external uv venv at $SF/marin_venv, fully
# offline). The ONLY additions are the SLURM multi-node Ray bring-up: a Ray
# head started inside the container on node 0, workers on nodes 1..N-1 attaching
# to it, then the trainer launched on the head with RAY_ADDRESS set so SkyRL's
# ray.init() joins the existing cluster instead of spawning a local one.
set -euxo pipefail

WORK=/leonardo_work/AIFAC_5C0_290/bfeuer00
SF=/leonardo_scratch/fast/AIFAC_5C0_290/bfeuer00
SANDBOX=$SF/marinskyrl_sandbox
VENV=$SF/marin_venv
MARIN=$WORK/code/MarinSkyRL/skyrl-train
CFG=$WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo

export DATA_DIR=${DATA_DIR:-$WORK/data/math}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-32B}    # resolved from offline HF cache; override per cell
export NUM_GPUS=${NUM_GPUS:-4}                      # GPUs PER NODE (Leonardo A100x4)

# Per-cell FRESH ckpt dir (cleared) so no cross-cell / cross-wave resume can
# ever skip training. Cell name derived from --job-name (mathmn_<cell>).
GRID_CELL="${SLURM_JOB_NAME#mathmn_}"
[ -z "$GRID_CELL" -o "$GRID_CELL" = "$SLURM_JOB_NAME" ] && GRID_CELL="${SLURM_JOB_ID:-cell}"
export CKPT_DIR=$SF/grid_ckpts/mn_$GRID_CELL
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
mkdir -p "$CKPT_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$CONTAINER_HOME"

# Per-job Ray temp dir. MUST be SHORT: Ray builds an AF_UNIX socket path
# <temp-dir>/session_<ts>/sockets/plasma_store which cannot exceed 107 bytes.
# The Lustre scratch root is already ~55 chars, so a temp-dir there overflows.
# Use a short /tmp path (node-local, writable in-container; the 1-node grid
# used Ray's default /tmp/ray). /tmp is per-node and the job owns the node.
export RAY_TMP=/tmp/r$SLURM_JOB_ID

# NCCL / Ray cross-node fabric: pin to the routable InfiniBand NIC (ib0). On
# Leonardo the eno* NICs are not routable between compute nodes; without this
# the Ray GCS connection and the NCCL weight-sync communicator pick the wrong
# interface and the run hangs at cluster join / first collective.
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export RAY_PORT=6479

# ---- Discover head node + its IB ip ----
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
NNODES=${#NODES[@]}
HEAD_NODE=${NODES[0]}
# Resolve the head node's ib0 address (route-correct, not the eno* mgmt addr).
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
SING_ENV="$SING_ENV,DATA_DIR=$DATA_DIR,MODEL_PATH=$MODEL_PATH,NUM_GPUS=$NUM_GPUS,TP=${TP:-1},CKPT_DIR=$CKPT_DIR,LOGGER=console,VENV_PY=$VENV_PY"

SING_BIND="/leonardo_work:/leonardo_work,/leonardo_scratch:/leonardo_scratch"

nvidia-smi || true

# ---- Start Ray HEAD on node 0 (inside container, blocking, backgrounded) ----
# `ray start --block` keeps the head process alive for the duration of the job;
# we background the srun and poll for readiness before launching workers.
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

# Wait for the head GCS to come up (poll `ray status` from inside the container).
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

# ---- Start Ray WORKERS on nodes 1..N-1 (inside container, blocking, bg) ----
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

# Wait until the cluster reports all expected GPUs (NNODES * NUM_GPUS), so the
# trainer's placement group (which needs every node) can be satisfied.
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

# ---- Run the trainer on the HEAD (joins the existing Ray cluster via RAY_ADDRESS) ----
# run_math_grid.sh emits the MATH/aime GRPO base invocation (tbs64 x n4,
# max_generate 4096, cudagraph ON); trailing "$@" = per-cell big-model flags
# (multi-node placement + engine count + EP/grouped-GEMM for MoE).
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
    --output="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
    singularity exec --nv --no-home \
        --bind "$SING_BIND" --pwd "$MARIN" \
        --env "$SING_ENV" \
        "$SANDBOX" bash "$CFG/run_math_grid.sh" "$@" \
        trainer.resume_mode=null trainer.ckpt_path="$CKPT_DIR" trainer.export_path="$CKPT_DIR/exports"
TRAIN_RC=$?
echo "CANARY_EXIT=$TRAIN_RC"

# ---- Teardown: stop Ray everywhere, kill backgrounded bring-up sruns ----
for ((i=0; i<NNODES; i++)); do
    srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" --overlap \
        singularity exec --nv --no-home --bind "$SING_BIND" --pwd "$MARIN" \
          --env "$SING_ENV" "$SANDBOX" \
          "$VENV_PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
done
kill "$RAY_HEAD_SRUN_PID" "${WORKER_PIDS[@]}" 2>/dev/null || true
rm -rf "$RAY_TMP" || true
exit $TRAIN_RC
