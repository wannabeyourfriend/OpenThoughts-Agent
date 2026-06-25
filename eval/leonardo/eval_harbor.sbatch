#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account AIFAC_5C0_290
#SBATCH --output=eval/leonardo/logs/%j.out
#SBATCH --job-name=eval

# ==============================================================================
# Unified Eval Harbor — Leonardo Cluster (CINECA A100-64GB)
#
# Starts vLLM, runs Harbor eval, checks errors, uploads results.
# Adapted from the canonical eval/jupiter/eval_harbor.sbatch for Leonardo x86_64.
#
# Positional args:
#   $1 = MODEL (HF model name, e.g. mlfoundations-dev/some_model)
#   $2 = REPO_ID (HF dataset repo, e.g. DCAgent/dev_set_v2)
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG_ARG (optional, override run tag)
#
# Env vars from listener:
#   EVAL_N_CONCURRENT       (default: 64)
#   EVAL_GPU_MEMORY_UTIL    (default: 0.85 — Leonardo A100 64GB OOMs at 0.90+)
#   EVAL_DAYTONA_THRESHOLD  (default: 3)
#   EVAL_CONFIG_YAML        (default: dcagent_eval_config_no_override.yaml; resolved from hpc/harbor_yaml/eval/configs/)
#   EVAL_ENABLE_THINKING    (default: false)   EVAL_AGENT_PARSER (default: "")
#   EVAL_DB_JOB_ID          (optional, pre-created Pending job ID)
# ==============================================================================

set -eo pipefail
ulimit -c 0  # Disable core dumps
ulimit -n 65536 2>/dev/null || true

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# --- Parse positional args ---
MODEL="${1:-mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1}"
REPO_ID="${2:-DCAgent2/terminal_bench_2}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"

# --- Read env vars from listener (with defaults) ---
N_CONCURRENT="${EVAL_N_CONCURRENT:-64}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.85}"
DAYTONA_ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-999999}"
# Harbor config is listener/preset-driven via EVAL_CONFIG_YAML (resolved below,
# after DCFT is set) — NOT a custom per-cluster file. The tb2/v2/swebench presets
# forward dcagent_eval_config_no_override.yaml so harbor inherits each task's
# declared (per-task) sandbox sizes. Thinking + agent parser are listener-driven
# too (EVAL_ENABLE_THINKING / EVAL_AGENT_PARSER, mirroring the canonical
# eval/jupiter/eval_harbor.sbatch), so no per-cluster config carries them.
CONFIG_YAML="${EVAL_CONFIG_YAML:-dcagent_eval_config_no_override.yaml}"
ENABLE_THINKING="${EVAL_ENABLE_THINKING:-false}"
AGENT_PARSER="${EVAL_AGENT_PARSER:-}"
DB_JOB_ID="${EVAL_DB_JOB_ID:-}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
if [[ "$REPO_ID" == /* ]]; then
    SAFE_REPO=$(basename "$REPO_ID")
else
    SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')
fi

# Update SLURM job name to include model short name (default "%j" is just job ID)
SHORT_MODEL=$(echo "$MODEL" | awk -F'/' '{print $NF}')
scontrol update JobId=$SLURM_JOB_ID Name="eval_${SHORT_MODEL}" 2>/dev/null || true

echo "=============================================="
echo "Leonardo Eval Harbor"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo "N concurrent: $N_CONCURRENT"
echo "GPU memory util: $GPU_MEMORY_UTIL"
echo "Config YAML: $CONFIG_YAML"
echo "Thinking: $ENABLE_THINKING"
echo "Agent parser: ${AGENT_PARSER:-<default>}"
echo "Daytona threshold: $DAYTONA_ERROR_THRESHOLD"
echo "DB Job ID (pending): ${DB_JOB_ID:-<none>}"
echo "=============================================="

# ==============================================================================
# Leonardo Environment Setup
# ==============================================================================

# Hardcode WORK — Leonardo sets WORK=/leonardo_work/AIFAC_5C0_290 system-wide
# which is the project dir, not the user dir. Override unconditionally.
WORK="/leonardo_work/AIFAC_5C0_290/bfeuer00"
DCFT="${WORK}/code/OpenThoughts-Agent"

# Resolve the harbor config path from CONFIG_YAML (mirrors the canonical
# eval/jupiter/eval_harbor.sbatch resolution: hpc/harbor_yaml/eval/configs/ -> eval/leonardo/ -> eval/MBZ/).
# Presets forward dcagent_eval_config_no_override.yaml (in hpc/harbor_yaml/eval/configs/).
if [[ "$CONFIG_YAML" == /* ]]; then
    HARBOR_CONFIG="$CONFIG_YAML"
elif [ -f "$DCFT/hpc/harbor_yaml/eval/configs/$CONFIG_YAML" ]; then
    HARBOR_CONFIG="$DCFT/hpc/harbor_yaml/eval/configs/$CONFIG_YAML"
elif [ -f "$DCFT/eval/leonardo/$CONFIG_YAML" ]; then
    HARBOR_CONFIG="$DCFT/eval/leonardo/$CONFIG_YAML"
elif [ -f "$DCFT/eval/MBZ/$CONFIG_YAML" ]; then
    HARBOR_CONFIG="$DCFT/eval/MBZ/$CONFIG_YAML"
else
    echo "ERROR: Harbor config not found: $CONFIG_YAML (looked in hpc/harbor_yaml/eval/configs, eval/leonardo, eval/MBZ)"
    exit 1
fi
echo "Resolved harbor config: $HARBOR_CONFIG"

# Reset PYTHONPATH
unset PYTHONPATH

# Conda activation (x86_64 — no LD_LOADER wrapper needed).
# The MAIN shell always runs in `otagent` — that is the env with harbor, the
# eval orchestrator/CLI, proxychains, the DB client and the uploader. A per-model
# `conda_env:` override (listener -> EVAL_CONDA_ENV) selects a DIFFERENT env for
# the vLLM SERVE ONLY (e.g. eval-qwen35 for qwen3_5 tmax-9b/27b — the only env
# with the qwen3_5 vLLM impl; ENVIRONMENT_MAP §2f). Activating it for the whole
# job would break harbor (`PackageNotFoundError: harbor`), so we scope it to the
# serve subprocess via SERVE_PYTHON_BIN + SERVE_ENV_PREFIX below.
source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate otagent
echo "Conda env (main/harbor): otagent (prefix $CONDA_PREFIX)"

# All build deps (CUDA, libstdc++, etc.) managed via conda — no module loads.
# Prepend conda's lib to ensure its libstdc++ wins over any system/module versions.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Per-model SERVE env selection -------------------------------------------
# Default: serve from the same otagent env (SERVE_PYTHON_BIN set after PYTHON_BIN
# resolves below). eval-qwen35: point the serve at that env's python AND carry
# its ENVIRONMENT_MAP §2f load-bearing env vars (without them the fork-vLLM serve
# crashes): conda gcc-14 libstdc++ ahead on LD path for CXXABI_1.3.15 (bundled
# vllm_flash_attn), VLLM_USE_FLASHINFER_SAMPLER=0 (else flashinfer sampler JIT
# link fails `-lcuda`), the libcuda stub on LIBRARY_PATH, FLASH_ATTN backend.
# These apply ONLY to the serve srun (env prefix), NOT the otagent main shell.
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-otagent}"
SERVE_CONDA_PREFIX="${WORK}/miniforge3/envs/${EVAL_CONDA_ENV}"
SERVE_ENV_PREFIX=()
if [ "$EVAL_CONDA_ENV" = "eval-qwen35" ]; then
    SERVE_ENV_PREFIX=( env \
        "LD_LIBRARY_PATH=${SERVE_CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
        "VLLM_USE_FLASHINFER_SAMPLER=0" \
        "LIBRARY_PATH=${SERVE_CONDA_PREFIX}/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" \
        "VLLM_ATTENTION_BACKEND=FLASH_ATTN" )
    echo "[eval-qwen35] serve env: python=${SERVE_CONDA_PREFIX}/bin/python VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN"
fi

# Strip inherited LD_PRELOAD (proxychains from login node).
# The proxied() helper re-adds it only for commands that need external access.
unset LD_PRELOAD 2>/dev/null || true

# Source Leonardo environment
source "$DCFT/hpc/dotenv/leonardo.env"

# Source secrets
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/secrets.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

# --- Daytona API key for evals (3:1 weighted: 3/4 org2, 1/4 org1) ---
DAYTONA_KEY_ORG1="dtn_17868a1955b56a52cb367af6dd3c6e93ee531b2073df801784273435c0e0fc6c"
DAYTONA_KEY_ORG2="dtn_ecfb759d2abcbf0f044413182677b3cc7624af495cb48409e8873b5283bda313"
if (( RANDOM % 4 == 0 )); then
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG1"
    echo "Selected Daytona org: org1 (${DAYTONA_API_KEY:0:12}...)"
else
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG2"
    echo "Selected Daytona org: org2 (${DAYTONA_API_KEY:0:12}...)"
fi

# --- vLLM / Triton / torch cache dirs (home quota is small; use WORK) ---
export VLLM_CACHE_ROOT="${WORK}/vllm_cache"
export VLLM_CONFIG_ROOT="${WORK}/vllm_cache"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_${SLURM_JOB_ID}"
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache_${USER}_${SLURM_JOB_ID}"
export UV_CACHE_DIR="${SCRATCH_FAST}/uv_cache_dir"
export TORCH_EXTENSIONS_DIR="${WORK}/torch_extensions"
export TORCH_HOME="${WORK}/torch_home"
export XDG_CACHE_HOME="${WORK}/.cache"
export TORCHINDUCTOR_CACHE_DIR="${WORK}/.cache/torchinductor"
export HOME="${WORK}"
# --- Pin NETRC to avoid the aiohttp Path.home() segfault (Leonardo, 2026-06-23) ---
# Under high harbor concurrency (the v2 preset's --n-concurrent 128) many aiohttp
# client sessions concurrently call _get_netrc_auth -> netrc_from_env() ->
# Path.home().expanduser()/stat on GPFS. That native path is not thread-safe at
# scale and SIGSEGVs the harbor orchestrator ~9 min in (jobs 47674124 a1-codeelo/v2,
# 47675130 a1-codeforces/v2 — Fatal Python error: Segmentation fault, current thread
# in aiohttp/helpers.py:210 netrc_from_env -> pathlib home() -> posixpath expanduser).
# Setting NETRC makes netrc_from_env() take the `netrc_env is not None` branch and
# skip Path.home() entirely (see aiohttp/helpers.py). The file is intentionally a
# no-op placeholder (we never use .netrc auth — model API auth is via headers).
export NETRC="${WORK}/.netrc_eval_placeholder"
touch "$NETRC" 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=0  # Login nodes pre-downloaded; set to 1 after download

# PYTHONPATH: harbor src + project root
HARBOR_SRC="${WORK}/code/harbor/src"
export PYTHONPATH="${HARBOR_SRC}:${DCFT}:${PYTHONPATH:-}"

PYTHON_BIN="$(which python)"   # otagent — harbor / DB / uploads / pre-download
echo "Python: $($PYTHON_BIN --version)"
echo "PYTHONPATH: $PYTHONPATH"

# vLLM SERVE interpreter: the per-model conda env's python (eval-qwen35 for
# qwen3_5), or otagent's python when no override. Only the serve uses this.
if [ "$EVAL_CONDA_ENV" != "otagent" ] && [ -x "${SERVE_CONDA_PREFIX}/bin/python" ]; then
    SERVE_PYTHON_BIN="${SERVE_CONDA_PREFIX}/bin/python"
else
    SERVE_PYTHON_BIN="$PYTHON_BIN"
fi
echo "Serve python ($EVAL_CONDA_ENV): $SERVE_PYTHON_BIN"

# ==============================================================================
# Proxy Setup: SSH tunnel from compute node → login node
#
# Leonardo compute nodes have no internet. We create a SOCKS5 proxy via
# SSH dynamic port forwarding to a login node (which has internet).
#
# Prerequisites (cert-based key, synced from local machine):
#   1. step ssh certificate 'bfeuer00' --provisioner cineca-hpc ~/.ssh/leonardo_daytona --no-password --insecure
#   2. rsync keys to Leonardo (see start_proxy_tunnel.sh)
#   3. Set SSH_KEY env var or ensure key is at ~/.ssh/leonardo_daytona
# ==============================================================================

# Add proxychains to PATH so start_proxy_tunnel.sh (and all subprocesses) can find it
export PATH="${WORK}/proxychains/bin:${PATH}"
PROXYCHAINS_BIN="$(which proxychains4 2>/dev/null || echo "${WORK}/proxychains/bin/proxychains4")"

# SSH key for intra-cluster tunnel (cert-based, no passphrase)
export SSH_KEY="${SSH_KEY:-/leonardo/home/userexternal/bfeuer00/.ssh/leonardo_daytona}"

echo "[proxy] Setting up SSH tunnel proxy..."
# stdout = proxychains4 command; stderr = diagnostic messages (shown to log)
CMD_PREFIX=$(bash "$DCFT/eval/leonardo/start_proxy_tunnel.sh")

if [[ "$CMD_PREFIX" == proxychains4* ]]; then
    echo "[proxy] Tunnel established. CMD_PREFIX=$CMD_PREFIX"

    # Source the exported env vars from the tunnel script
    # (PROXYCHAINS_CONF_FILE was set in the subshell; re-extract from CMD_PREFIX)
    PROXYCHAINS_CONF_FILE=$(echo "$CMD_PREFIX" | awk '{print $NF}')
    export PROXYCHAINS_CONF_FILE

    # Daytona timeout/retry settings
    export DAYTONA_MAX_RETRIES=5
    export DAYTONA_RETRY_DELAY=30
    export DAYTONA_BACKOFF_FACTOR=2
    export DAYTONA_TIMEOUT=1800
    export AIOHTTP_CLIENT_TIMEOUT=900
    export AIOHTTP_CONNECTOR_TIMEOUT=900
    export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
    export AIOHTTP_TOTAL_TIMEOUT=1800

    # Disable SSL verification (cert chain issues through tunnel)
    export PYTHONHTTPSVERIFY=0
    unset SSL_CERT_FILE
    unset CURL_CA_BUNDLE
    unset REQUESTS_CA_BUNDLE
    unset SSL_CERT_DIR

    # Test connectivity
    if $CMD_PREFIX curl -s --connect-timeout 10 https://huggingface.co -o /dev/null 2>/dev/null; then
        echo "[proxy] Connectivity test passed (huggingface.co reachable)"
    else
        echo "[proxy] WARNING: Connectivity test failed (tunnel may still work for Daytona)"
    fi
else
    echo "[proxy] WARNING: SSH tunnel setup failed. External connectivity will not work."
    echo "[proxy] CMD_PREFIX output was: $CMD_PREFIX"
    CMD_PREFIX=""
fi

# Helper: run command through proxychains
proxied() {
    if [ -n "${CMD_PREFIX:-}" ]; then
        $CMD_PREFIX "$@"
    else
        "$@"
    fi
}

# ==============================================================================
# Cleanup trap
# ==============================================================================
cleanup() {
    echo "Cleaning up..."
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    # Kill SSH tunnel (all ssh -D processes for this job)
    pkill -f "ssh.*-D.*${TUNNEL_PORT:-27003}" 2>/dev/null || true
    # Kill ssh-agent (env vars saved by start_proxy_tunnel.sh)
    _AGENT_ENV="/tmp/.ssh_agent_env_${SLURM_JOB_ID:-$$}"
    if [ -f "$_AGENT_ENV" ]; then
        source "$_AGENT_ENV"
        kill "$SSH_AGENT_PID" 2>/dev/null || true
        rm -f "$_AGENT_ENV"
    fi
    rm -f "${PROXYCHAINS_CONF_FILE:-}" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# ==============================================================================
# Construct run tag and unified run directory (BEFORE vLLM start)
# ==============================================================================
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:-${WORK}/eval_jobs}"

if [ -n "$RUN_TAG_ARG" ]; then
    RUN_TAG="$RUN_TAG_ARG"
else
    RUN_TAG="eval-${SAFE_MODEL}_${SAFE_REPO}"
fi
RUN_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

# Symlink SLURM log into run dir for consolidated access
mkdir -p eval/leonardo/logs
SLURM_LOG="${DCFT}/eval/leonardo/logs/${SLURM_JOB_ID}.out"
ln -sf "$SLURM_LOG" "$RUN_DIR/slurm.log" 2>/dev/null || true

# Ensure Harbor writes directly to our unified run dir via symlink
mkdir -p "${DCFT}/trace_jobs"
rm -rf "${DCFT}/trace_jobs/${RUN_TAG}" 2>/dev/null || true
ln -sfn "$RUN_DIR" "${DCFT}/trace_jobs/${RUN_TAG}"

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

# ==============================================================================
# Start vLLM Server (single node, TP=4 for 4 A100-64GB GPUs)
# ==============================================================================

# Pre-download model via proxychains (compute nodes have no direct internet)
echo "Pre-downloading model: $MODEL"
proxied $PYTHON_BIN -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ.get('HF_HUB_CACHE')
path = snapshot_download('$MODEL', cache_dir=cache)
print(f'Model cached at: {path}')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Model pre-download failed for $MODEL. Exiting."
    exit 1
fi

export HF_HUB_OFFLINE=1  # Prevent further network access during vLLM startup

echo "Starting vLLM server for model: $MODEL"
source "$DCFT/eval/build_vllm_cmd.sh"
build_vllm_cmd "$SERVE_PYTHON_BIN" "$MODEL" "$GPU_MEMORY_UTIL"

# ==============================================================================
# Build model_info for hosted_vllm (harbor #339 hard-requires it; see Jupiter
# sbatch for the full rationale). Derived from served context + output budget;
# costs zeroed (self-hosted). Passed below as an --agent-kwarg.
# ==============================================================================
MI_MAX_MODEL_LEN="${EVAL_VLLM_MAX_MODEL_LEN:-32768}"
MI_MAX_OUTPUT="${EVAL_MAX_OUTPUT_TOKENS:-16384}"
if [ "$MI_MAX_OUTPUT" -ge "$MI_MAX_MODEL_LEN" ] 2>/dev/null; then
    MI_MAX_OUTPUT=$(( MI_MAX_MODEL_LEN / 2 ))
fi
MODEL_INFO_JSON="{\"max_input_tokens\":${MI_MAX_MODEL_LEN},\"max_output_tokens\":${MI_MAX_OUTPUT},\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
echo "[model_info] $MODEL_INFO_JSON"

# Serve under the per-model env prefix (eval-qwen35 §2f vars; empty for otagent).
#
# --- GRES sizing for the serve srun (FIX 2026-06-24) ---
# The job-level allocation is set by the listener at submit time to TP×DP GPUs
# (eval/unified_eval_listener.py: `--gres gpu:{total_gpus}`), which OVERRIDES the
# stale `#SBATCH --gres=gpu:4` default in this script's header. For a TP=1 8B
# campaign model the job therefore holds only 1 GPU, but a hardcoded
# `srun --overlap --gres=gpu:4` then requests 4 → "Invalid generic resource (gres)
# specification" → vLLM never starts → the leg aborts (~52min, exit 1:0). Derive
# the serve GPU count from the TP×DP the listener already passed (falling back to
# the live SLURM allocation, then 4) so the inner srun matches the actual job
# allocation for BOTH TP=1 (1 GPU) and TP=4 (4 GPU, 32B) evals.
SERVE_TP="${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-}"
SERVE_DP="${EVAL_VLLM_DATA_PARALLEL_SIZE:-1}"
if [ -n "$SERVE_TP" ]; then
    SERVE_GPU_COUNT=$(( SERVE_TP * SERVE_DP ))
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    SERVE_GPU_COUNT="$SLURM_GPUS_ON_NODE"
elif [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then
    SERVE_GPU_COUNT="$SLURM_GPUS_PER_NODE"
else
    SERVE_GPU_COUNT=4
fi
echo "[serve] gres=gpu:${SERVE_GPU_COUNT} (TP=${SERVE_TP:-<unset>} DP=${SERVE_DP} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>})"

srun --overlap --gres=gpu:${SERVE_GPU_COUNT} "${SERVE_ENV_PREFIX[@]}" "${VLLM_CMD[@]}" \
    > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# Health check loop (Python-based to avoid curl/proxychains issues)
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
VLLM_HEALTH_URL="http://${MASTER_ADDR}:8000/v1/models"
echo "Health check URL: $VLLM_HEALTH_URL"
echo "LD_PRELOAD=${LD_PRELOAD:-<unset>}"
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
        # Print diagnostic info
        echo "--- Diagnostic: trying direct connection ---"
        $PYTHON_BIN -c "
import urllib.request, socket
print(f'MASTER_ADDR=$MASTER_ADDR')
try:
    s = socket.create_connection(('$MASTER_ADDR', 8000), timeout=5)
    s.close()
    print('TCP connect to port 8000: OK')
except Exception as e:
    print(f'TCP connect to port 8000: FAILED ({e})')
try:
    resp = urllib.request.urlopen('$VLLM_HEALTH_URL', timeout=10)
    print(f'HTTP response: {resp.status}')
except Exception as e:
    print(f'HTTP request failed: {e}')
" 2>&1 || true
        tail -50 "$RUN_DIR/vllm.log" 2>/dev/null || true
        exit 1
    fi
done

# ==============================================================================
# Download dataset (via proxychains for HF access)
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
    DATASET_LOCAL_DIR="${WORK}/data/datasets/${SAFE_REPO}"
    echo "[DEBUG] DATASET_LOCAL_DIR=$DATASET_LOCAL_DIR"

    # Check if dataset was pre-downloaded on login node (compute nodes have no internet)
    # First check the custom local dir, then check the HF cache
    if [ -d "$DATASET_LOCAL_DIR" ]; then
        TASK_COUNT=$(ls -d "$DATASET_LOCAL_DIR"/*/instruction.md 2>/dev/null | wc -l)
        echo "[DEBUG] Found $TASK_COUNT tasks with instruction.md in local dir"
    fi

    # Also check HF cache (listener pre-downloads datasets there via snapshot_download)
    # HF cache uses "datasets--<org>--<name>" format (double-dash separated)
    if [ "${TASK_COUNT:-0}" -eq 0 ]; then
        HF_CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
        HF_REPO_CACHE_NAME="datasets--$(echo "$REPO_ID" | sed 's/\//--/g')"
        HF_DATASET_CACHE=$(find "$HF_CACHE_DIR" -maxdepth 3 -type d -name "snapshots" -path "*${HF_REPO_CACHE_NAME}*" 2>/dev/null | head -1)
        if [ -n "$HF_DATASET_CACHE" ]; then
            HF_SNAPSHOT=$(ls -dt "$HF_DATASET_CACHE"/*/ 2>/dev/null | head -1)
            if [ -n "$HF_SNAPSHOT" ]; then
                HF_TASK_COUNT=$(ls -d "$HF_SNAPSHOT"/*/instruction.md 2>/dev/null | wc -l)
                echo "[DEBUG] Found $HF_TASK_COUNT tasks in HF cache: $HF_SNAPSHOT"
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
        echo "Dataset not pre-downloaded, attempting proxied download..."
        DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
        proxied $PYTHON_BIN "$DCFT/eval/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
        DOWNLOAD_EXIT=$?
        cat "$DOWNLOAD_LOG"
        DATASET_PATH=$(grep DATASET_PATH "$DOWNLOAD_LOG" | tail -n 1 | cut -d'=' -f2)
        rm -f "$DOWNLOAD_LOG"
        if [ $DOWNLOAD_EXIT -ne 0 ] || [ -z "${DATASET_PATH:-}" ]; then
            echo "ERROR: Failed to get dataset path (exit code: $DOWNLOAD_EXIT)"
            echo "Hint: pre-download the dataset on the login node first:"
            echo "  python eval/snapshot_download.py $REPO_ID --local-dir $DATASET_LOCAL_DIR"
            exit 1
        fi
    fi
fi
echo "Using dataset path: $DATASET_PATH"

# --- Compute canonical benchmark name for DB uploads ---
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
# Update DB: Started (using database/unified_db)
# ==============================================================================
echo "Creating/updating DB job entry..."
export MODEL REPO_ID RUN_TAG SLURM_JOB_ID BENCHMARK_NAME

HARBOR_VERSION=$($PYTHON_BIN -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

proxied $PYTHON_BIN - <<'PY' || true
import os, sys

from database.unified_db.utils import create_job_entry_started

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]
run_tag = os.environ["RUN_TAG"]
slurm_job_id = os.environ["SLURM_JOB_ID"]
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")

result = create_job_entry_started(
    model_hf_name=model_hf,
    benchmark_hf_name=dataset_hf,
    job_name=run_tag,
    username=os.environ.get("USER", "leonardo"),
    slurm_job_id=slurm_job_id,
    harbor_package_version=harbor_version,
    agent_name="terminus-2",
    config={"agent": "terminus-2", "env": "daytona"},
    n_trials=128,
    n_rep_eval=3
)

if not result.get("success"):
    print(f"WARNING: DB create failed: {result.get('error')}", file=sys.stderr)
    sys.exit(0)

db_job_id = result["job"]["id"]
print(f"DB job created with ID: {db_job_id}")
PY

# Get the DB job ID
DB_JOB_ID=$(proxied $PYTHON_BIN - <<'PY' || true
import os, sys

from database.unified_db.utils import get_latest_job_for_model_benchmark

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]

try:
    result = get_latest_job_for_model_benchmark(model_hf, dataset_hf)
    if result and result.get("id"):
        print(result["id"])
except Exception as e:
    print(f"WARNING: DB lookup failed: {e}", file=sys.stderr)
PY
)

if [ -z "${DB_JOB_ID:-}" ]; then
    echo "WARNING: Failed to get DB job ID"
else
    echo "DB job entry: $DB_JOB_ID"
fi

# ==============================================================================
# Run Harbor Eval (via proxychains for Daytona access)
# ==============================================================================
set +e

EXTRA_HARBOR_ARGS=""
if [ -n "${EVAL_SNAPSHOT_NAME:-}" ]; then
    echo "Using Daytona snapshot: $EVAL_SNAPSHOT_NAME"
    EXTRA_HARBOR_ARGS="--environment-kwarg snapshot_template_name=$EVAL_SNAPSHOT_NAME --no-force-build"
fi
if [ -n "${EVAL_TIMEOUT_MULTIPLIER:-}" ]; then
    echo "Using timeout multiplier: $EVAL_TIMEOUT_MULTIPLIER"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --timeout-multiplier $EVAL_TIMEOUT_MULTIPLIER"
fi
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ]; then
    echo "Using memory override: ${EVAL_OVERRIDE_MEMORY_MB}MB"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --override-memory-mb $EVAL_OVERRIDE_MEMORY_MB"
fi
# Thinking + agent parser are listener-driven (mirror eval/jupiter/eval_harbor.sbatch),
# replacing what the deprecated custom Leonardo config used to hardcode.
if [ "$ENABLE_THINKING" = "true" ]; then
    echo "Enabling thinking (--agent-kwarg enable_thinking=true)"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --agent-kwarg enable_thinking=true"
fi
if [ -n "$AGENT_PARSER" ]; then
    echo "Using agent parser: $AGENT_PARSER"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --agent-kwarg parser=$AGENT_PARSER"
fi

echo "[DEBUG] EXTRA_HARBOR_ARGS=$EXTRA_HARBOR_ARGS"
echo "[DEBUG] Full harbor command:"
echo "  harbor jobs start -p $DATASET_PATH --n-concurrent $N_CONCURRENT --agent terminus-2 --model hosted_vllm/$MODEL --env daytona ... $EXTRA_HARBOR_ARGS"

# Check if a previous job dir exists for this run tag (for resume)
if [ -d "$RUN_DIR" ] && [ -f "$RUN_DIR/config.json" ]; then
    echo "Found existing job dir, resuming: $RUN_DIR"
    proxied harbor jobs resume \
        -p "$RUN_DIR" \
        --filter-error-type EnvironmentStartTimeoutError \
        --filter-error-type DaytonaError \
        --filter-error-type DaytonaRateLimitError
else
    echo "Starting new job"
    proxied harbor jobs start \
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

# Harbor writes trials, result.json, and config.json directly to RUN_DIR
# (via trace_jobs symlink → RUN_DIR)
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
ERROR_LOG="$RUN_DIR/daytona_errors.log"

if [ -f "$RESULT_FILE" ]; then
    echo "Checking for DaytonaErrors in $RESULT_FILE..."

    DAYTONA_COUNT=$($PYTHON_BIN -c "
import json, sys
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)

    total_errors = 0
    if 'stats' in data and 'evals' in data['stats']:
        for eval_key, eval_data in data['stats']['evals'].items():
            if 'exception_stats' in eval_data and 'DaytonaError' in eval_data['exception_stats']:
                daytona_errors = eval_data['exception_stats']['DaytonaError']
                if isinstance(daytona_errors, list):
                    total_errors += len(daytona_errors)
    print(total_errors)
except Exception as e:
    print(f'Error parsing result.json: {e}', file=sys.stderr)
    print('0')
" 2>&1 | tail -n 1)

    echo "DaytonaError count: ${DAYTONA_COUNT}"

    if [ "${DAYTONA_COUNT:-0}" -gt "$DAYTONA_ERROR_THRESHOLD" ]; then
        echo "Job has ${DAYTONA_COUNT} DaytonaErrors (> ${DAYTONA_ERROR_THRESHOLD}), skipping upload"
        echo "Job completed but not uploaded due to excessive DaytonaErrors"
        exit 0
    fi
else
    echo "Warning: result.json not found, continuing with upload"
fi

# ==============================================================================
# Upload results to DB (via proxychains for HF upload)
# ==============================================================================
# Re-enable HF Hub access for uploads (was set to offline for vLLM startup)
export HF_HUB_OFFLINE=0

# All data (trials, result.json, config.json, meta.env) is in RUN_DIR
export UPLOAD_DIR="$RUN_DIR"
export RUN_DIR
export UPLOAD_USERNAME="${UPLOAD_USERNAME:-$USER}"
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG

UPLOAD_LOG="$RUN_DIR/upload.log"

echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"
echo "Using username=${UPLOAD_USERNAME}, mode=${UPLOAD_MODE}" | tee -a "$UPLOAD_LOG"

proxied $PYTHON_BIN - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
import os, sys, re, hashlib

from database.unified_db.utils import upload_eval_results


def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    def collapse(s: str) -> str:
        prev = None
        while s != prev:
            prev = s
            s = s.replace("--", "-").replace("..", ".")
        return s

    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = collapse(name).strip("-.")
    if not name:
        name = "repo"

    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest))
        base = name[:keep].rstrip("-.")
        if not base:
            base = "r"
        name = f"{base}{digest}"
        name = collapse(name).strip("-.")

    name = collapse(name).strip("-.")
    if name[0] in "-.":
        name = "r" + name[1:]
    if name[-1] in "-.":
        name = name[:-1] + "0"

    return f"{org}/{name}" if org else name


upload_dir = os.environ.get("UPLOAD_DIR", os.environ["RUN_DIR"])
run_dir = os.environ["RUN_DIR"]
run_tag = os.environ["RUN_TAG"]
username = os.environ.get("UPLOAD_USERNAME", os.environ.get("USER", "leonardo"))
error_mode = os.environ.get("UPLOAD_MODE", "skip_on_error")
hf_repo_id = sanitize_hf_repo_id(f"DCAgent2/{run_tag}")
hf_token = os.environ["HF_TOKEN"]

print(f"[uploader] upload_eval_results(path={run_dir!r}, username={username!r}, "
      f"error_mode={error_mode!r}, hf_repo_id={hf_repo_id!r})")
dataset_hf = os.environ.get("REPO_ID", "")
benchmark_name = os.environ.get("BENCHMARK_NAME", "")
if not benchmark_name:
    benchmark_name = dataset_hf.split("/")[-1] if "/" in dataset_hf else dataset_hf

import hashlib
benchmark_version_hash = hashlib.sha256(benchmark_name.encode()).hexdigest()
print(f"[uploader] benchmark_name={benchmark_name!r}, version_hash={benchmark_version_hash[:16]}...")

print(f"[uploader] Using upload dir: {upload_dir}")
upload_eval_results(
    upload_dir,
    username=username,
    error_mode=error_mode,
    hf_token=hf_token,
    hf_repo_id=hf_repo_id,
    register_benchmark=True,
    benchmark_name=benchmark_name,
    benchmark_version_hash=benchmark_version_hash,
)
print("[uploader] done.")
PY
UPLOAD_EXIT=${PIPESTATUS[0]}

if [ $UPLOAD_EXIT -ne 0 ]; then
    echo "Upload failed with exit code: $UPLOAD_EXIT"
    exit $UPLOAD_EXIT
fi

echo "=============================================="
echo "Eval and upload finished successfully."
echo "=============================================="
