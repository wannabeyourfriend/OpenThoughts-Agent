#!/bin/bash
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --job-name=passatk
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft/logs/%x-%A_%a.log
#
# pass@k grid (Delphi #6279) — ONE array task = ONE (model, task) full N=128 pass@k eval.
# Reads its grid line from $GRID (default passatk-sft/grid.txt), line number = SLURM_ARRAY_TASK_ID+1.
#   grid.txt columns:  <MODEL_REPO> <RUN> <TYPE> <TASK> [TP]
#     TYPE = base               -> NO chat template, 4k clamp (max_model_len=4096, max_gen_toks=3584), TP per row
#     TYPE = qwen3              -> chat template ON, long ctx (max_model_len=32768, max_gen_toks=30720), TP=4
#     TYPE = qwen3-base         -> NO chat template (raw completion), long ctx 32768/30720, TP=4  (Qwen3 dense BASE)
#     TYPE = qwen3-moe-base     -> NO chat template (raw completion), long ctx 32768/30720, TP=4  (Qwen3-30B-A3B base MoE)
#     TYPE = qwen3-moe-thinking -> chat template ON (thinking default), long ctx 32768/30720, TP=4  (Qwen3-30B-A3B-Thinking MoE)
#     TASK = MATH500 | AIME24 | gsm8k
#   (qwen3-moe-* are qwen3_moe arch; vLLM TP=4 shards experts within TP — no --enable-expert-parallel needed.)
# Mechanism: passatk_driver.py — single vLLM engine, SamplingParams(n=128), grader-faithful,
# unbiased pass@k estimator. Idempotent: skips if <OUT>/passatk_results.json exists.
#
# Submit (concurrency capped at 8):
#   sbatch --array=0-$(($(wc -l < grid.txt)-1))%8 passatk_eval.sbatch
set -euo pipefail

GRID="${GRID:-/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft/grid.txt}"
DRIVER=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft/passatk_driver.py
PASSATK_ROOT=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft

source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate evalchemy-marin
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy-marin

source ~/secrets.env
export HF_HOME=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_CACHE=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

# writable caches ($HOME read-only on Leonardo — §3.0)
CACHE_ROOT=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/.cache
mkdir -p "$CACHE_ROOT"/{flashinfer,triton,inductor,vllm,xdg,hf_local}
export HOME="$CACHE_ROOT/hf_local"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
mkdir -p "$HOME/.cache/flashinfer"

# ----- pick the grid line for this array task -----
LINE_NO=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
LINE=$(sed -n "${LINE_NO}p" "$GRID")
[ -z "$LINE" ] && { echo "ERROR: empty grid line $LINE_NO in $GRID"; exit 2; }
read -r MODEL_REPO RUN TYPE TASK EXTRA_TP <<< "$LINE"
echo ">>> PASSATK_JOBMAP arrayjob=${SLURM_ARRAY_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-NA} repo=$MODEL_REPO run=$RUN type=$TYPE benchmark=$TASK"

OUT="${PASSATK_ROOT}/${TASK}/${RUN}"
mkdir -p "$OUT"
{ echo "ARRAYJOB=${SLURM_ARRAY_JOB_ID:-NA}"; echo "ARRAYTASK=${SLURM_ARRAY_TASK_ID:-NA}";
  echo "MODEL_REPO=$MODEL_REPO"; echo "RUN=$RUN"; echo "TYPE=$TYPE"; echo "BENCHMARK=$TASK"; } > "$OUT/meta.env"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="passatk-${RUN}-${TASK}" 2>/dev/null || true
fi

# ----- per-type knobs -----
# VLLM_FLAGS: extra engine knobs passed to the driver. EMPTY for dense cells (preserve prior
# gpu_memory_utilization=0.9 / cudagraph behavior — don't degrade them). The Qwen3-30B-A3B MoE
# cells OOM during the n=128 sampler warmup (256 dummy requests) at TP4 on one 4xGH200 node, so
# the MoE types get a dedicated OOM-headroom override: lower gpu_memory_utilization + max_num_seqs
# and enforce_eager (frees the cudagraph-capture pool that was consuming the last ~few-hundred MiB).
VLLM_FLAGS=""
case "$TYPE" in
  base)
    CHAT_FLAG=""
    MAX_MODEL_LEN=4096
    MAX_GEN_TOKS=3584
    TP="${EXTRA_TP:-2}"   # per-row TP (divisor of head count); default 2
    ;;
  qwen3)
    CHAT_FLAG="--apply-chat-template"
    MAX_MODEL_LEN=32768
    MAX_GEN_TOKS=30720
    TP="${EXTRA_TP:-4}"
    ;;
  qwen3-base)
    # Qwen3 dense BASE: raw completion (NO chat template), clean rope -> 32k ctx.
    CHAT_FLAG=""
    MAX_MODEL_LEN=32768
    MAX_GEN_TOKS=30720
    TP="${EXTRA_TP:-4}"
    ;;
  qwen3-moe-base)
    # Qwen3-30B-A3B base MoE: raw completion (NO chat template), clean rope -> 32k ctx.
    CHAT_FLAG=""
    MAX_MODEL_LEN=32768
    MAX_GEN_TOKS=30720
    TP="${EXTRA_TP:-4}"
    VLLM_FLAGS="--gpu-mem-util 0.80 --max-num-seqs 8 --enforce-eager"   # n=128 sampler-warmup OOM headroom
    ;;
  qwen3-moe-thinking)
    # Qwen3-30B-A3B-Thinking-2507: own ChatML template, thinking default-on (enable_thinking UNSET), 32k ctx.
    CHAT_FLAG="--apply-chat-template"
    MAX_MODEL_LEN=32768
    MAX_GEN_TOKS=30720
    TP="${EXTRA_TP:-4}"
    VLLM_FLAGS="--gpu-mem-util 0.80 --max-num-seqs 8 --enforce-eager"   # n=128 sampler-warmup OOM headroom
    ;;
  *) echo "ERROR: TYPE must be base|qwen3|qwen3-base|qwen3-moe-base|qwen3-moe-thinking (got '$TYPE')"; exit 2 ;;
esac

echo "RUN=$RUN TASK=$TASK TYPE=$TYPE TP=$TP MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_GEN_TOKS=$MAX_GEN_TOKS CHAT_FLAG='$CHAT_FLAG' VLLM_FLAGS='$VLLM_FLAGS' OUT=$OUT"

python "$DRIVER" \
  --task "$TASK" \
  --model-repo "$MODEL_REPO" \
  --out "$OUT" \
  --evalchemy-root /leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy-marin \
  --n 128 \
  --tp "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-gen-toks "$MAX_GEN_TOKS" \
  --temperature 0.7 --top-p 1.0 --seed 42 \
  $CHAT_FLAG $VLLM_FLAGS

echo "DONE: $RUN / $TASK -> $OUT"
