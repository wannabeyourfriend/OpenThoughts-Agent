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

# --- 4. run kaniko ---
exec /kaniko/executor \
  --context dir:///app \
  --dockerfile docker/Dockerfile.gpu-rl \
  --build-arg WHEEL_SOURCE=wheel-builder \
  --single-snapshot \
  --compressed-caching=false \
  --cache=true \
  --cache-repo="${CACHE_REPO}" \
  --destination "${DEST_FLOATING}" \
  --destination "${DEST_PINNED}"
