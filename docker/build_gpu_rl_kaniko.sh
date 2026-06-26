#!/usr/bin/env bash
# build_gpu_rl_kaniko.sh — in-cluster kaniko build of the gpu-rl image.
#
# Runs INSIDE an iris job whose task-image is docker.io/library/ubuntu:22.04
# (kaniko's executor image is distroless / has no bash, so it cannot be the task
# image directly). We crane-export the kaniko executor rootfs over / and run
# /kaniko/executor. See .claude/skills/build-gpu-rl-image-iris/SKILL.md.
#
# Required env (passed by the iris launch as -e):
#   DOCKER_USER_ID  ghcr user (penfever)
#   DOCKER_TOKEN    a GitHub PAT with write:packages (NOT the Docker Hub dckr_pat_)
#   GITSHA          OT-Agent commit sha for the immutable :gpu-rl-<gitsha> tag
set -euxo pipefail

: "${DOCKER_USER_ID:?}"
: "${DOCKER_TOKEN:?}"
: "${GITSHA:?}"

# WHEEL_SOURCE selects how the vLLM-fork + flash-attn wheels enter the rl stage:
#   prebuilt-wheelhouse (FAST): COPY docker/wheelhouse/*.whl + uv pip install — NO
#     nvcc. Requires the wheels to be present in the /app bundle; since
#     docker/wheelhouse/* is gitignored, force-stage them first (git add -f) so the
#     iris bundle (git ls-files --cached) includes them.
#   wheel-builder (SLOW, default): compile both wheels inline with nvcc (~3 h). Use
#     only when no prebuilt wheels can be force-staged.
WHEEL_SOURCE="${WHEEL_SOURCE:-wheel-builder}"

CACHE_REPO=ghcr.io/open-thoughts/openthoughts-agent/cache
DEST_FLOATING=ghcr.io/open-thoughts/openthoughts-agent:gpu-rl
DEST_PINNED=ghcr.io/open-thoughts/openthoughts-agent:gpu-rl-${GITSHA}

# --- 1. fetch crane (static binary) ---
apt-get update -y && apt-get install -y --no-install-recommends ca-certificates curl tar
cd /tmp
CRANE_VER=v0.20.2
curl -fsSL "https://github.com/google/go-containerregistry/releases/download/${CRANE_VER}/go-containerregistry_Linux_x86_64.tar.gz" -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane

# --- 2. crane-export the kaniko executor rootfs over / ---
# Overlays kaniko's /kaniko/... onto the ubuntu rootfs.
crane export gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true

# --- 3. write the ghcr auth config AFTER the overlay (kaniko clobbers /kaniko otherwise) ---
export DOCKER_CONFIG=/kaniko/.docker
mkdir -p "$DOCKER_CONFIG"
AUTH=$(printf '%s:%s' "$DOCKER_USER_ID" "$DOCKER_TOKEN" | base64 | tr -d '\n')
cat > "$DOCKER_CONFIG/config.json" <<EOF
{"auths":{"ghcr.io":{"auth":"${AUTH}"}}}
EOF

# --- 3.5. populate docker/wheelhouse/ for the prebuilt-wheelhouse (no-nvcc) path ---
# The iris /app bundle has a 25 MB cap, so the ~900 MB prebuilt wheels CANNOT ride
# it. Instead fetch them from the public HF dataset mirror into the build context
# (/app/docker/wheelhouse/) so kaniko's `COPY docker/wheelhouse/` (rl stage) finds
# them => ZERO nvcc. Skip if the wheels are already present (e.g. a real x86 host
# that ran build_wheels.sh). Mirror: laion/gpu-rl-build-wheels (vLLM-fork +
# flash-attn wheels are open-source; public, no token needed for download).
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ]; then
  WH=/app/docker/wheelhouse
  mkdir -p "$WH"
  HF_BASE="https://huggingface.co/datasets/laion/gpu-rl-build-wheels/resolve/main"
  FLASH_WHL=flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
  VLLM_WHL=vllm-0.1.dev16611+g76259c63a.d20260625.cu128-cp312-cp312-linux_x86_64.whl
  for f in "$FLASH_WHL" "$VLLM_WHL" MANIFEST; do
    if [ ! -s "$WH/$f" ]; then
      echo "fetching wheelhouse artifact: $f"
      curl -fSL --retry 5 --retry-delay 5 "$HF_BASE/$(printf '%s' "$f" | sed 's/+/%2B/g')" -o "$WH/$f"
    fi
  done
  echo "=== wheelhouse contents ==="; ls -la "$WH"
  # fail fast if a wheel is missing/empty -> do NOT silently fall through to a compile
  test -s "$WH/$FLASH_WHL" && test -s "$WH/$VLLM_WHL" || { echo "FATAL: wheelhouse not populated"; exit 1; }
fi

# --- 4. run kaniko ---
# --skip-unused-stages is LOAD-BEARING for the prebuilt-wheelhouse path: kaniko
# builds EVERY Dockerfile stage by default (unlike BuildKit), so without it the
# `wheel-builder` (nvcc) stage compiles even when the rl stage takes its wheels
# from prebuilt-wheelhouse. With it, the unreferenced wheel-builder stage is
# pruned => ZERO nvcc on the prebuilt-wheelhouse path.
exec /kaniko/executor \
  --context dir:///app \
  --dockerfile docker/Dockerfile.gpu-rl \
  --build-arg WHEEL_SOURCE="$WHEEL_SOURCE" \
  --skip-unused-stages \
  --single-snapshot \
  --compressed-caching=false \
  --cache=true \
  --cache-repo="${CACHE_REPO}" \
  --destination "${DEST_FLOATING}" \
  --destination "${DEST_PINNED}"
