#!/bin/bash
#SBATCH -p gh
#SBATCH --time=23:59:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --account CCR24067
#SBATCH --output=eval/tacc/logs/%x_%j.out
#SBATCH --job-name=data

# ==============================================================================
# Unified Eval Harbor — TACC Vista (GH200 96GB, aarch64, sm_90)
#
# Starts vLLM (the from-source fork wheel in the otagent env), runs Harbor eval
# against Daytona sandboxes, checks errors, uploads results.
# Adapted from eval/leonardo/eval_harbor.sbatch.
#
# KEY DIFFERENCES vs Leonardo (all because Vista compute nodes have FULL DIRECT
# internet egress — verified job 787310):
#   - NO proxychains / SSH SOCKS5 tunnel / step-ca cert. Daytona API + HF are
#     reached directly. No `proxied()` wrapper — every command runs unwrapped.
#   - NO model pre-download step (compute node downloads from HF directly).
#   - NO --gres / --mem (Vista doesn't gres-track GPUs + misreports RealMemory);
#     the node is allocated whole. TP from the per-model config (TP=1 default for
#     the 32B family — fits the 96GB GH200).
#   - aarch64: load the build toolchain (gcc/13.2.0 cuda/12.8) that the fork vLLM
#     wheel was built against, and put conda's libstdc++ ahead on LD_LIBRARY_PATH.
#
# Positional args:
#   $1 = MODEL (HF model name)
#   $2 = REPO_ID (HF dataset repo, e.g. DCAgent2/terminal_bench_2)
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG_ARG (optional, override run tag)
#
# Env vars from listener: EVAL_N_CONCURRENT, EVAL_GPU_MEMORY_UTIL,
#   EVAL_DAYTONA_THRESHOLD, EVAL_CONFIG_YAML, EVAL_AGENT_KWARGS,
#   EVAL_AGENT_PARSER, EVAL_DB_JOB_ID, EVAL_VLLM_* (serve overrides).
# ==============================================================================

set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# --- Parse positional args ---
MODEL="${1:-Qwen/Qwen3-32B}"
REPO_ID="${2:-DCAgent2/terminal_bench_2}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"

# --- Read env vars from listener (with defaults) ---
N_CONCURRENT="${EVAL_N_CONCURRENT:-32}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.90}"
DAYTONA_ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-999999}"
CONFIG_YAML="${EVAL_CONFIG_YAML:-dcagent_eval_config_no_override.yaml}"
AGENT_KWARGS="${EVAL_AGENT_KWARGS:-}"
AGENT_PARSER="${EVAL_AGENT_PARSER:-}"
DB_JOB_ID="${EVAL_DB_JOB_ID:-}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
if [[ "$REPO_ID" == /* ]]; then
    SAFE_REPO=$(basename "$REPO_ID")
else
    SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')
fi

SHORT_MODEL=$(echo "$MODEL" | awk -F'/' '{print $NF}')
scontrol update JobId=$SLURM_JOB_ID Name="eval_${SHORT_MODEL}" 2>/dev/null || true

echo "=============================================="
echo "TACC Vista Eval Harbor (GH200 aarch64)"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo "N concurrent: $N_CONCURRENT"
echo "GPU memory util: $GPU_MEMORY_UTIL"
echo "Config YAML: $CONFIG_YAML"
echo "Agent kwargs: ${AGENT_KWARGS:-<none>}"
echo "Agent parser: ${AGENT_PARSER:-<default>}"
echo "Daytona threshold: $DAYTONA_ERROR_THRESHOLD"
echo "DB Job ID (pending): ${DB_JOB_ID:-<none>}"
echo "=============================================="

# ==============================================================================
# TACC Vista Environment Setup
# ==============================================================================
SCRATCH="${SCRATCH:-/scratch/10635/penfever}"
DCFT="${EVAL_PROJECT_ROOT:-${SCRATCH}/OpenThoughts-Agent}"

# Resolve the harbor config path from CONFIG_YAML (mirrors Leonardo/Jupiter:
# hpc/harbor_yaml/eval/configs/ -> eval/tacc/ -> eval/MBZ/).
if [[ "$CONFIG_YAML" == /* ]]; then
    HARBOR_CONFIG="$CONFIG_YAML"
elif [ -f "$DCFT/hpc/harbor_yaml/eval/configs/$CONFIG_YAML" ]; then
    HARBOR_CONFIG="$DCFT/hpc/harbor_yaml/eval/configs/$CONFIG_YAML"
elif [ -f "$DCFT/eval/tacc/$CONFIG_YAML" ]; then
    HARBOR_CONFIG="$DCFT/eval/tacc/$CONFIG_YAML"
elif [ -f "$DCFT/eval/MBZ/$CONFIG_YAML" ]; then
    HARBOR_CONFIG="$DCFT/eval/MBZ/$CONFIG_YAML"
else
    echo "ERROR: Harbor config not found: $CONFIG_YAML"
    exit 1
fi
echo "Resolved harbor config: $HARBOR_CONFIG"

unset PYTHONPATH

# --- aarch64 toolchain (matches the fork vLLM wheel build: gcc/13.2.0 cuda/12.8) ---
module purge 2>/dev/null || true
module load gcc/13.2.0 cuda/12.8 2>/dev/null || true

# --- Conda activation: otagent (harbor + fork vLLM + DB client + uploader) ---
source "${SCRATCH}/miniconda3/etc/profile.d/conda.sh"
conda activate otagent
echo "Conda env: otagent (prefix $CONDA_PREFIX)"

# Prepend conda's lib so its libstdc++ wins over the system/module versions.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Source TACC environment + secrets
source "$DCFT/hpc/dotenv/tacc.env" 2>/dev/null || true
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-${SCRATCH}/keys.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

# --- Daytona API key (3:1 weighted: 3/4 org2, 1/4 org1) — same orgs as Leonardo ---
# Keys come from the secrets env (sourced above as $DC_AGENT_SECRET_ENV) — NEVER hardcode.
# The two eval orgs are DAYTONA_API_KEY (org1) + DAYTONA_DATA_API_KEY (org2).
DAYTONA_KEY_ORG1="${DAYTONA_API_KEY:?DAYTONA_API_KEY unset — source the secrets env ($DC_AGENT_SECRET_ENV) before launch}"
DAYTONA_KEY_ORG2="${DAYTONA_DATA_API_KEY:?DAYTONA_DATA_API_KEY unset — source the secrets env ($DC_AGENT_SECRET_ENV) before launch}"
if (( RANDOM % 4 == 0 )); then
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG1"
    echo "Selected Daytona org: org1 (${DAYTONA_API_KEY:0:12}...)"
else
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG2"
    echo "Selected Daytona org: org2 (${DAYTONA_API_KEY:0:12}...)"
fi

# Daytona timeout/retry settings
export DAYTONA_MAX_RETRIES=5
export DAYTONA_RETRY_DELAY=30
export DAYTONA_BACKOFF_FACTOR=2
export DAYTONA_TIMEOUT=1800
export AIOHTTP_CLIENT_TIMEOUT=900
export AIOHTTP_CONNECTOR_TIMEOUT=900
export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
export AIOHTTP_TOTAL_TIMEOUT=1800

# --- Caches (home quota is tiny on Vista; use scratch) ---
export VLLM_CACHE_ROOT="${SCRATCH}/vllm"
export VLLM_CONFIG_ROOT="${SCRATCH}/vllm"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_${SLURM_JOB_ID}"
# FlashInfer: PERSISTENT cache on scratch (NOT /tmp — MoE kernel JIT compilation
# takes 20-40 min for 182 kernels and MUST persist across jobs).  The first job
# with a new model compiles; subsequent jobs hit the cache instantly.
export FLASHINFER_WORKSPACE_BASE="${SCRATCH}/flashinfer_cache"
# Limit ninja parallelism during flashinfer JIT — unlimited (72 cores) OOM-kills
# nvcc on MoE kernel compilation. 8 parallel × ~6GB/nvcc = ~48GB, safe on 480GB.
export MAX_JOBS=8
export TORCH_EXTENSIONS_DIR="${SCRATCH}/torch_extensions"
export TORCH_HOME="${SCRATCH}/torch_home"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torchinductor"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${SCRATCH}/hub}"
# Pin NETRC to dodge the aiohttp Path.home() thread-unsafety segfault under
# high harbor concurrency (see Leonardo sbatch rationale).
export NETRC="${SCRATCH}/.netrc_eval_placeholder"
touch "$NETRC" 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=0   # Vista compute nodes have direct egress

# PYTHONPATH: harbor src + project root
HARBOR_SRC="${EVAL_HARBOR_SRC:-${SCRATCH}/harbor/src}"
export PYTHONPATH="${HARBOR_SRC}:${DCFT}:${PYTHONPATH:-}"

PYTHON_BIN="$(which python)"
SERVE_PYTHON_BIN="$PYTHON_BIN"   # single otagent env serves everything on Vista
echo "Python: $($PYTHON_BIN --version)"
echo "PYTHONPATH: $PYTHONPATH"

# ==============================================================================
# Cleanup trap
# ==============================================================================
cleanup() {
    echo "Cleaning up..."
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    echo "Cleanup done."
}
trap cleanup EXIT

# ==============================================================================
# Construct run tag and unified run directory (BEFORE vLLM start)
# ==============================================================================
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:-${SCRATCH}/eval_jobs}"

if [ -n "$RUN_TAG_ARG" ]; then
    RUN_TAG="$RUN_TAG_ARG"
else
    RUN_TAG="eval-${SAFE_MODEL}_${SAFE_REPO}"
fi
RUN_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

mkdir -p "$DCFT/eval/tacc/logs"
SLURM_LOG="${DCFT}/eval/tacc/logs/data_${SLURM_JOB_ID}.out"
ln -sf "$SLURM_LOG" "$RUN_DIR/slurm.log" 2>/dev/null || true

mkdir -p "${DCFT}/trace_jobs"
rm -rf "${DCFT}/trace_jobs/${RUN_TAG}" 2>/dev/null || true
ln -sfn "$RUN_DIR" "${DCFT}/trace_jobs/${RUN_TAG}"

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

# ==============================================================================
# Start vLLM Server (single node, TP from per-model config; default TP=1)
# ==============================================================================
export HF_HUB_OFFLINE=0

echo "Starting vLLM server for model: $MODEL"
source "$DCFT/eval/build_vllm_cmd.sh"
build_vllm_cmd "$SERVE_PYTHON_BIN" "$MODEL" "$GPU_MEMORY_UTIL"

# ==============================================================================
# model_info for hosted_vllm (harbor #339 hard-requires it; see Jupiter sbatch).
# ==============================================================================
MI_MAX_MODEL_LEN="${EVAL_VLLM_MAX_MODEL_LEN:-32768}"
MI_MAX_OUTPUT="${EVAL_MAX_OUTPUT_TOKENS:-16384}"
if [ "$MI_MAX_OUTPUT" -ge "$MI_MAX_MODEL_LEN" ] 2>/dev/null; then
    MI_MAX_OUTPUT=$(( MI_MAX_MODEL_LEN / 2 ))
fi
MODEL_INFO_JSON="{\"max_input_tokens\":${MI_MAX_MODEL_LEN},\"max_output_tokens\":${MI_MAX_OUTPUT},\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
echo "[model_info] $MODEL_INFO_JSON"

# Vista does NOT gres-track GPUs and the node is allocated whole, so the serve
# srun must NOT pass --gres (it would be rejected as "Invalid gres"). For a
# single-node (TP*DP <= 1 node) serve, run vLLM directly in this allocation.
"${VLLM_CMD[@]}" > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# Health check loop
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
VLLM_HEALTH_URL="http://${MASTER_ADDR}:8000/v1/models"
echo "Health check URL: $VLLM_HEALTH_URL"
MAX_RETRIES=30
RETRY_INTERVAL=100
for i in $(seq 1 $MAX_RETRIES); do
    if $PYTHON_BIN -c "
import urllib.request, sys
try:
    urllib.request.urlopen('$VLLM_HEALTH_URL', timeout=10)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "vLLM server is ready"
        break
    fi
    echo "Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        echo "--- Diagnostic ---"
        $PYTHON_BIN -c "
import socket
try:
    s = socket.create_connection(('$MASTER_ADDR', 8000), timeout=5); s.close()
    print('TCP connect to port 8000: OK')
except Exception as e:
    print(f'TCP connect to port 8000: FAILED ({e})')
" 2>&1 || true
        tail -80 "$RUN_DIR/vllm.log" 2>/dev/null || true
        exit 1
    fi
done

# ==============================================================================
# Download / locate dataset (direct HF access — compute nodes have egress)
# ==============================================================================
if [[ "$REPO_ID" == /* ]]; then
    echo "Using local dataset path: $REPO_ID"
    DATASET_PATH="$REPO_ID"
    if [ ! -d "$DATASET_PATH" ]; then
        echo "ERROR: Local dataset path does not exist: $DATASET_PATH"
        exit 1
    fi
    TASK_COUNT=$(ls -d "$DATASET_PATH"/*/instruction.md 2>/dev/null | wc -l)
    echo "[DEBUG] Found $TASK_COUNT tasks with instruction.md"
else
    echo "Downloading/locating dataset: $REPO_ID"
    DATASET_LOCAL_DIR="${SCRATCH}/data/datasets/${SAFE_REPO}"

    if [ -d "$DATASET_LOCAL_DIR" ]; then
        TASK_COUNT=$(ls -d "$DATASET_LOCAL_DIR"/*/instruction.md 2>/dev/null | wc -l)
    fi
    if [ "${TASK_COUNT:-0}" -eq 0 ]; then
        HF_CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
        HF_REPO_CACHE_NAME="datasets--$(echo "$REPO_ID" | sed 's/\//--/g')"
        HF_DATASET_CACHE=$(find "$HF_CACHE_DIR" -maxdepth 3 -type d -name "snapshots" -path "*${HF_REPO_CACHE_NAME}*" 2>/dev/null | head -1)
        if [ -n "$HF_DATASET_CACHE" ]; then
            HF_SNAPSHOT=$(ls -dt "$HF_DATASET_CACHE"/*/ 2>/dev/null | head -1)
            if [ -n "$HF_SNAPSHOT" ]; then
                HF_TASK_COUNT=$(ls -d "$HF_SNAPSHOT"/*/instruction.md 2>/dev/null | wc -l)
                if [ "$HF_TASK_COUNT" -gt 0 ]; then
                    TASK_COUNT=$HF_TASK_COUNT
                    DATASET_LOCAL_DIR="$HF_SNAPSHOT"
                fi
            fi
        fi
    fi

    if [ "${TASK_COUNT:-0}" -gt 0 ]; then
        echo "Using pre-downloaded dataset: $DATASET_LOCAL_DIR ($TASK_COUNT tasks)"
        DATASET_PATH="$DATASET_LOCAL_DIR"
    else
        echo "Dataset not local, downloading directly..."
        DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
        $PYTHON_BIN "$DCFT/eval/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
        DOWNLOAD_EXIT=$?
        cat "$DOWNLOAD_LOG"
        DATASET_PATH=$(grep DATASET_PATH "$DOWNLOAD_LOG" | tail -n 1 | cut -d'=' -f2)
        rm -f "$DOWNLOAD_LOG"
        if [ $DOWNLOAD_EXIT -ne 0 ] || [ -z "${DATASET_PATH:-}" ]; then
            echo "ERROR: Failed to get dataset path (exit code: $DOWNLOAD_EXIT)"
            exit 1
        fi
    fi
fi
echo "Using dataset path: $DATASET_PATH"

# --- Canonical benchmark name for DB uploads ---
declare -A BENCHMARK_NAME_MAP=(
    ["DCAgent2_aider_polyglot"]="aider_polyglot"
    ["DCAgent_dev_set_v2"]="dev_set_v2"
    ["DCAgent_dev_set_71_tasks"]="dev_set_71_tasks"
    ["DCAgent2_terminal_bench_2"]="terminal_bench_2"
    ["DCAgent_swebench_verified_eval_set"]="swebench-verified-random-100-folders"
    ["DCAgent2_bfcl-parity"]="bfcl-parity"
)
BENCHMARK_NAME="${BENCHMARK_NAME_MAP[$SAFE_REPO]:-$SAFE_REPO}"
export BENCHMARK_NAME
echo "Benchmark name: $BENCHMARK_NAME"

# ==============================================================================
# Update DB: Started
# ==============================================================================
echo "Creating/updating DB job entry..."
export MODEL REPO_ID RUN_TAG SLURM_JOB_ID BENCHMARK_NAME
HARBOR_VERSION=$($PYTHON_BIN -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

$PYTHON_BIN - <<'PY' || true
import os, sys
from database.unified_db.utils import create_job_entry_started
result = create_job_entry_started(
    model_hf_name=os.environ["MODEL"],
    benchmark_hf_name=os.environ["REPO_ID"],
    job_name=os.environ["RUN_TAG"],
    username=os.environ.get("USER", "tacc"),
    slurm_job_id=os.environ["SLURM_JOB_ID"],
    harbor_package_version=os.environ.get("HARBOR_VERSION", "unknown"),
    agent_name="terminus-2",
    config={"agent": "terminus-2", "env": "daytona"},
    n_trials=128,
    n_rep_eval=3,
)
if not result.get("success"):
    print(f"WARNING: DB create failed: {result.get('error')}", file=sys.stderr)
    sys.exit(0)
print(f"DB job created with ID: {result['job']['id']}")
PY

DB_JOB_ID=$($PYTHON_BIN - <<'PY' || true
import os
from database.unified_db.utils import get_latest_job_for_model_benchmark
try:
    result = get_latest_job_for_model_benchmark(os.environ["MODEL"], os.environ["REPO_ID"])
    if result and result.get("id"):
        print(result["id"])
except Exception as e:
    import sys; print(f"WARNING: DB lookup failed: {e}", file=sys.stderr)
PY
)
[ -z "${DB_JOB_ID:-}" ] && echo "WARNING: Failed to get DB job ID" || echo "DB job entry: $DB_JOB_ID"

# ==============================================================================
# Run Harbor Eval (direct Daytona access — no proxychains)
# ==============================================================================
set +e

EXTRA_HARBOR_ARGS=""
if [ -n "${EVAL_SNAPSHOT_NAME:-}" ]; then
    EXTRA_HARBOR_ARGS="--environment-kwarg snapshot_template_name=$EVAL_SNAPSHOT_NAME --no-force-build"
fi
if [ -n "${EVAL_TIMEOUT_MULTIPLIER:-}" ]; then
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --timeout-multiplier $EVAL_TIMEOUT_MULTIPLIER"
fi
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ]; then
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --override-memory-mb $EVAL_OVERRIDE_MEMORY_MB"
fi
# EVAL_AGENT_KWARGS is newline-separated; each entry is a space-free key=value
# (e.g. the live nested extra_body thinking form — no enable_thinking flag), so
# it stays one token under the unquoted $EXTRA_HARBOR_ARGS word-split.
if [ -n "$AGENT_KWARGS" ]; then
    while IFS= read -r _kw; do
        [ -z "$_kw" ] && continue
        EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --agent-kwarg $_kw"
    done <<< "$AGENT_KWARGS"
fi
if [ -n "$AGENT_PARSER" ]; then
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --agent-kwarg parser=$AGENT_PARSER"
fi

echo "[DEBUG] EXTRA_HARBOR_ARGS=$EXTRA_HARBOR_ARGS"

if [ -d "$RUN_DIR" ] && [ -f "$RUN_DIR/config.json" ]; then
    echo "Found existing job dir, resuming: $RUN_DIR"
    harbor jobs resume \
        -p "$RUN_DIR" \
        --filter-error-type EnvironmentStartTimeoutError \
        --filter-error-type DaytonaError \
        --filter-error-type DaytonaRateLimitError
else
    echo "Starting new job"
    harbor jobs start \
        -p "$DATASET_PATH" \
        --n-concurrent "$N_CONCURRENT" \
        --agent terminus-2 \
        --model "hosted_vllm/$MODEL" \
        --env "daytona" \
        --agent-kwarg "api_base=http://${MASTER_ADDR}:8000/v1" \
        --agent-kwarg "key=fake_key" \
        --agent-kwarg "model_info=$MODEL_INFO_JSON" \
        --n-attempts 3 \
        --job-name "$RUN_TAG" \
        --jobs-dir "$EVAL_JOBS_DIR" \
        --config "$HARBOR_CONFIG" \
        $EXTRA_HARBOR_ARGS
fi
SB_EXIT=$?
set -e

# ==============================================================================
# Save meta.env
# ==============================================================================
mkdir -p "$RUN_DIR"
{
    echo "MODEL=$MODEL"
    echo "REPO_ID=$REPO_ID"
    echo "TIMESTAMP=$TIMESTAMP"
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "DB_JOB_ID=${DB_JOB_ID:-}"
    echo "BENCHMARK_ID=${BENCHMARK_ID:-}"
    echo "N_CONCURRENT=$N_CONCURRENT"
    echo "GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
} > "$RUN_DIR/meta.env"

TRIAL_COUNT=$(find "$RUN_DIR" -maxdepth 1 -type d -name '*__*' 2>/dev/null | wc -l)
echo "Found $TRIAL_COUNT trial dirs in $RUN_DIR"

if [ ${SB_EXIT:-0} -ne 0 ]; then
    echo "Harbor eval exited with non-zero status: ${SB_EXIT}. Skipping upload."
    exit ${SB_EXIT}
fi
if [ ! -d "$RUN_DIR" ]; then
    echo "Expected run directory not found: $RUN_DIR"
    exit 2
fi

# ==============================================================================
# Check for DaytonaErrors before upload
# ==============================================================================
RESULT_FILE="$RUN_DIR/result.json"
if [ -f "$RESULT_FILE" ]; then
    DAYTONA_COUNT=$($PYTHON_BIN -c "
import json, sys
try:
    with open('$RESULT_FILE') as f: data = json.load(f)
    total = 0
    if 'stats' in data and 'evals' in data['stats']:
        for k, v in data['stats']['evals'].items():
            es = v.get('exception_stats', {})
            de = es.get('DaytonaError')
            if isinstance(de, list): total += len(de)
    print(total)
except Exception as e:
    print(f'Error parsing result.json: {e}', file=sys.stderr); print('0')
" 2>&1 | tail -n 1)
    echo "DaytonaError count: ${DAYTONA_COUNT}"
    if [ "${DAYTONA_COUNT:-0}" -gt "$DAYTONA_ERROR_THRESHOLD" ]; then
        echo "Job has ${DAYTONA_COUNT} DaytonaErrors (> ${DAYTONA_ERROR_THRESHOLD}), skipping upload"
        exit 0
    fi
else
    echo "Warning: result.json not found, continuing with upload"
fi

# ==============================================================================
# Upload results to DB + HF (direct — no proxychains)
# ==============================================================================
export HF_HUB_OFFLINE=0
export UPLOAD_DIR="$RUN_DIR"
export RUN_DIR
export UPLOAD_USERNAME="${UPLOAD_USERNAME:-$USER}"
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG

UPLOAD_LOG="$RUN_DIR/upload.log"
echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"

$PYTHON_BIN - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
import os, re, hashlib
from database.unified_db.utils import upload_eval_results

def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    def collapse(s):
        prev = None
        while s != prev:
            prev = s; s = s.replace("--", "-").replace("..", ".")
        return s
    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name); name = collapse(name).strip("-.")
    if not name: name = "repo"
    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest)); base = name[:keep].rstrip("-.") or "r"
        name = collapse(f"{base}{digest}").strip("-.")
    name = collapse(name).strip("-.")
    if name[0] in "-.": name = "r" + name[1:]
    if name[-1] in "-.": name = name[:-1] + "0"
    return f"{org}/{name}" if org else name

run_dir = os.environ["RUN_DIR"]; run_tag = os.environ["RUN_TAG"]
username = os.environ.get("UPLOAD_USERNAME", os.environ.get("USER", "tacc"))
error_mode = os.environ.get("UPLOAD_MODE", "skip_on_error")
hf_repo_id = sanitize_hf_repo_id(f"DCAgent2/{run_tag}")
hf_token = os.environ["HF_TOKEN"]
dataset_hf = os.environ.get("REPO_ID", "")
benchmark_name = os.environ.get("BENCHMARK_NAME", "") or (dataset_hf.split("/")[-1] if "/" in dataset_hf else dataset_hf)
benchmark_version_hash = hashlib.sha256(benchmark_name.encode()).hexdigest()
print(f"[uploader] upload_eval_results(path={run_dir!r}, hf_repo_id={hf_repo_id!r})")
upload_eval_results(
    os.environ.get("UPLOAD_DIR", run_dir),
    username=username, error_mode=error_mode, hf_token=hf_token,
    hf_repo_id=hf_repo_id, register_benchmark=True,
    benchmark_name=benchmark_name, benchmark_version_hash=benchmark_version_hash,
)
print("[uploader] done.")
PY
UPLOAD_EXIT=${PIPESTATUS[0]}
[ $UPLOAD_EXIT -ne 0 ] && { echo "Upload failed with exit code: $UPLOAD_EXIT"; exit $UPLOAD_EXIT; }

echo "=============================================="
echo "Eval and upload finished successfully."
echo "=============================================="
