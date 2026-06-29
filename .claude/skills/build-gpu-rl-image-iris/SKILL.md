---
name: build-gpu-rl-image-iris
description: >-
  Build the gpu-rl Docker image (`ghcr.io/open-thoughts/openthoughts-agent:gpu-rl`) — the RL runtime for
  CoreWeave H100 (torch 2.11 + the from-source vLLM fork + flash-attn 2.8.3 + MarinSkyRL/torchtitan EP) — IN
  the CoreWeave `cw-us-east-02a` cluster, AS AN IRIS JOB USING KANIKO. Covers WHY kaniko not buildkit (the
  cluster denies CAP_SYS_ADMIN/bind-mounts + gVisor), the exact crane-export-over-ubuntu recipe, the
  load-bearing resource/flag settings (512GB node, `--single-snapshot`, `--cache`, ghcr GitHub-PAT creds), the
  Dockerfile gotchas baked into the build (uv `--index-strategy unsafe-best-match`, `python -m pip wheel`,
  torchtitan needs `tyro`), the build-asserts-as-validation, capturing the digest + bumping
  `DEFAULT_RL_DOCKER_IMAGE`, monitoring the build job, and WHEN a rebuild is actually required vs a runtime
  `--skyrl-ref` checkout. Use when asked to build / rebuild / push the gpu-rl image, after bumping the vLLM-fork
  commit / flash-attn / torch-CUDA / a baked dep. The Mac CANNOT build it (arm64 + needs linux/amd64 + ~512GB).
  Reference: docker/Dockerfile.gpu-rl, docker/build_wheels.sh, docker/README.gpu-rl-wheelcache.md,
  .claude/ops/iris/coreweave_gpu_ops.md, rl/cloud/launch_rl_iris.py.
---

# build-gpu-rl-image-iris

> **📍 Iris orientation — read first.** Before acting on anything in this skill, read the Iris **tools
> catalog** (`.claude/ops/iris/iris_tools.md`) and the Iris **ops directory** (`.claude/ops/iris/` — the
> CoreWeave GPU particulars in `coreweave_gpu_ops.md`, the TPU `marin` particulars in `iris_job_lifecycle.md`).
> They carry the binding access/preamble/gotchas and the helper-script inventory the steps below rely on.

> **⚠ Local clone = ground truth (CLAUDE.md §Always).** ALL Dockerfile / build-script edits go in the local
> Mac checkout on `penfever/working` → commit → (push). **The iris job bundles the LOCAL workspace to `/app`**
> via `git ls-files --cached --others --exclude-standard` (respects `.gitignore`; reads WORKING-TREE content,
> so uncommitted *tracked* edits ARE included — you do NOT have to commit/push before a build), so a local edit
> takes effect on the next build immediately. Never hand-edit on a remote / leave divergent state / patch-by-rsync.

The **gpu-rl image** is the CoreWeave RL runtime — `linux/amd64`, **from-source CUDA**: the vLLM fork (cutlass
GEMM kernels) + flash-attn 2.8.3 (`flash_attn_2_cuda`) compiled with `nvcc` against torch 2.11.0+cu128 / cp312
/ x86_64, plus MarinSkyRL editable + torchtitan EP + harbor. It is consumed by `rl-agentic-launch-iris`
(digest-pinned in `rl/cloud/launch_rl_iris.py:DEFAULT_RL_DOCKER_IMAGE`). This skill is the **how-to to BUILD +
PUSH it**. Cluster access/hardware particulars defer to **`.claude/ops/iris/coreweave_gpu_ops.md`** (kubeconfig,
the otagent iris binary, node shape); the Dockerfile internals are documented in
**`docker/README.gpu-rl-wheelcache.md`** + the `Dockerfile.gpu-rl` header comments.

> **Ground truth.** This skill is the durable capture of the **v7 build** that succeeded on `cw-us-east-02a`
> (the prior `agent_logs` build session). Every WHY/gotcha below was paid for in v5/v6 — apply them up front.

## 0. Where the build runs, and why it can't run elsewhere

- **The Mac CANNOT build it.** It's arm64; the image is `linux/amd64`-only (CoreWeave H100 + the x86 CUDA
  build). Under QEMU + Docker Desktop the amd64 nvcc compiles are impractically slow and RAM-bound
  (`MAX_JOBS=8 × ~5 GB/job ≈ 40 GB` just for the compile; wheel-packaging needs much more — see §4). Don't.
- **There is NO in-cluster image-build primitive in iris.** `iris build` = LOCAL `docker buildx` + `docker
  login` (and marin `cluster start` likewise builds locally). So to build amd64 in the cluster you run the
  build **AS AN IRIS JOB** — a CPU-only iris task that does the Docker build itself.
- **The build needs no GPU** (nvcc compiles on a CPU builder). It's a CPU/memory/disk job.

## 1. Mechanism: KANIKO, not BuildKit (decisive)

The build job uses **kaniko** (`gcr.io/kaniko-project/executor`), NOT buildkit/`buildctl`:

- **BuildKit FAILS on this cluster.** `buildkitd`'s executor needs **bind mounts / `CAP_SYS_ADMIN`**. The
  cluster DENIES that: `--container-profile container_profile_privileged` is silently resolved to **DEFAULT
  caps** for a non-admin key (privileged "requires admin" — `service.py authorize(SET_CONTAINER_PROFILE)`; our
  key isn't admin), so the pod gets only `SYS_PTRACE`, no privileged. The nodes also run **gVisor**
  (`RuntimeClass gvisor`, label `node.coreweave.cloud/gvisor=true`). Every buildkit attempt died
  `failed to mount …: permission denied` (both native + overlayfs snapshotters).
- **KANIKO snapshots its OWN container rootfs** — no bind mounts, no privileged, runs as root under these
  restrictions. This is the standard k8s no-daemon build answer, and it WORKS here.

## 2. How the iris job is shaped (the crane-export trick)

- The iris pod command is a **HARDCODED `["bash","-lc", script]`** — you can't set an arbitrary entrypoint, and
  **kaniko's image is distroless (no bash)**, so kaniko CANNOT be the task image directly.
- So: use **`docker.io/library/ubuntu:22.04`** as the task image (it has bash), then inside the script
  **`crane export` the kaniko executor rootfs over `/`** and run `/kaniko/executor`. (crane is fetched in the
  script; the export overlays kaniko's `/kaniko/…` onto the ubuntu rootfs.)
- **Context = the iris-synced `/app` bundle** (the OT-Agent repo). The build's Dockerfile is
  `docker/Dockerfile.gpu-rl`; context is the repo root.
- **`WHEEL_SOURCE`: prefer the FAST `prebuilt-wheelhouse` path; `wheel-builder` is the SLOW fallback.**
  The `rl` stage's default `prebuilt-wheelhouse` COPYs `docker/wheelhouse/*.whl` and `uv pip install`s them
  → **ZERO nvcc** (build = minutes, dominated by the rl-stage dep installs). `wheel-builder` instead compiles
  the vLLM-fork + flash-attn wheels inline (the ~3 h nvcc cost; §4).
  - **The wheels do NOT ride the iris bundle.** `docker/wheelhouse/` is gitignored, AND even force-staged
    (`git add -f`) the ~900 MB of wheels blow the **hard 25 MB bundle cap** (`iris/.../bundle.py`
    `MAX_BUNDLE_SIZE_BYTES`; no per-file skip, no override). So force-staging does NOT get them to `/app`.
  - **FAST PATH (no nvcc) — fetch the prebuilt wheels in-build.** If you have the compiled wheels (e.g.
    extracted from a prior image, or from `build_wheels.sh` on a real x86 host), upload them to a remote the
    build pod can reach and have the kaniko build script fetch them into `/app/docker/wheelhouse/` BEFORE
    `kaniko/executor` runs (kaniko reads the context dir live, so `COPY docker/wheelhouse/` then finds them).
    `build_gpu_rl_kaniko.sh` does exactly this (step 3.5) for `WHEEL_SOURCE=prebuilt-wheelhouse`: curl the 2
    wheels + MANIFEST from the public HF dataset `laion/gpu-rl-build-wheels` (vLLM-fork + flash-attn are
    open-source) and fail-fast if a wheel is missing (never silently fall back to a compile). Launch with
    `-e WHEEL_SOURCE=prebuilt-wheelhouse`.
    - **`--skip-unused-stages` is LOAD-BEARING on this path.** kaniko builds EVERY Dockerfile stage by default
      (unlike BuildKit, which prunes), so WITHOUT this flag the `wheel-builder` (nvcc) stage compiles even when
      the `rl` stage takes `--from=wheels` = `prebuilt-wheelhouse`. WITH it, the unreferenced `wheel-builder`
      stage is pruned → ZERO nvcc. `build_gpu_rl_kaniko.sh` passes it. (Symptom if missing: "Building wheel for
      flash-attn … still running" despite `WHEEL_SOURCE=prebuilt-wheelhouse`.) **The MANIFEST pins MUST match the Dockerfile's
    `{VLLM_FORK_COMMIT, FLASH_ATTN_VERSION, TORCH_VERSION, TORCH_CUDA_ARCH_LIST, cu128, cp312}`** — a
    SKYRL_COMMIT-only bump does not touch the wheels, so the same wheels stay ABI-correct.
  - **SLOW FALLBACK — `wheel-builder`** (`-e WHEEL_SOURCE=wheel-builder`, or the script default): use ONLY when
    no prebuilt wheels are available to fetch. Pays the ~3 h nvcc compile.
  - **`--cache` is irrelevant on the prebuilt-wheelhouse path** (there is no compile RUN to cache). NOTE the
    cache repo holds only ONE `--single-snapshot` final-snapshot layer, so on the `wheel-builder` path `--cache`
    does NOT reuse the nvcc layers either — `--single-snapshot` and per-layer cache reuse are mutually
    defeating. Don't expect a cache HIT to save the 3 h; use the prebuilt-wheelhouse fast path instead.
- **`--destination ghcr.io/open-thoughts/openthoughts-agent:gpu-rl` (+ an immutable `:gpu-rl-<gitsha>` tag).**
  kaniko's `--destination` push is the TERMINAL step → a successful job = a pushed image (§5).

### ghcr push creds (the load-bearing gotcha)
- **`secrets.env DOCKER_TOKEN` is a Docker Hub PAT (`dckr_pat_…`) — WRONG for ghcr.io.** Use the **GitHub
  PAT**: `GHCR_TOKEN=$(gh auth token)` (user `penfever`, scope `write:packages`).
- kaniko reads `$DOCKER_CONFIG/config.json`. **WRITE that config AFTER the crane-export overlay** — kaniko's
  image ships its own `/kaniko` dir and clobbers a config written before the overlay — and
  `export DOCKER_CONFIG=/kaniko/.docker`. Contents:
  `{"auths":{"ghcr.io":{"auth":"<base64 of penfever:$GHCR_TOKEN>"}}}`.

## 3. The launch command (verbatim shape)

```bash
source /Users/benjaminfeuer/Documents/secrets.env
export KUBECONFIG=~/.kube/coreweave-iris-gpu                       # HARD prereq (see ops doc)
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris        # the cw-capable (otagent-env) iris binary
GHCR_TOKEN=$(gh auth token)                                       # GitHub PAT, NOT the Docker Hub DOCKER_TOKEN
B64=$(base64 -i build_gpu_rl_kaniko.sh | tr -d '\n')              # the kaniko build script, base64'd in

$IRIS --cluster=cw-us-east-02a job run \
  --task-image docker.io/library/ubuntu:22.04 --no-sync --enable-extra-resources \
  --cpu 48 --memory 512GB --disk 400GB \
  --job-name gpurl-kaniko-<gitsha> \
  --max-retries 0 --timeout 18000 \
  -e DOCKER_USER_ID penfever -e DOCKER_TOKEN "$GHCR_TOKEN" -e BUILD_B64 "$B64" --no-wait \
  -- bash -lc 'echo "$BUILD_B64" | base64 -d > /tmp/build.sh && exec bash /tmp/build.sh'
```
The build script (`build_gpu_rl_kaniko.sh`) does, in order: fetch crane → `crane export` the kaniko executor
over `/` → write `$DOCKER_CONFIG/config.json` with the ghcr `penfever:$GHCR_TOKEN` auth → run
`/kaniko/executor --context dir:///app --dockerfile docker/Dockerfile.gpu-rl --build-arg
WHEEL_SOURCE=wheel-builder --single-snapshot --cache=true --cache-repo=ghcr.io/open-thoughts/openthoughts-agent/cache
--destination ghcr.io/open-thoughts/openthoughts-agent:gpu-rl --destination …:gpu-rl-<gitsha>`. Keep the script
in the repo (it ships in the `/app` bundle); `--no-sync` here only means the *kaniko* job's own sync — the
context is still the synced repo.

## 4. The critical resource + kaniko flag settings (paid for in v5/v6)

- **512 GB node — NOT 256 GB.** v5 ran `cpu32 / mem256GB / disk250GB` and **DIED SILENTLY** (pod reaped, no
  error log) right after the ~3 h compile, at the **wheel-packaging / snapshot** step → OOM (256 GB is not
  enough to package the wheels + snapshot the rootfs). v6+ uses **`--cpu 48 --memory 512GB --disk 400GB`**.
  (`--cpu 48` not 64 is the same daemonset-overhead gang-fit rule as launch — see the ops doc; for a 1-node
  build it's less critical, but keep 48.)
- **`--single-snapshot`** — kaniko snapshots the full rootfs ONCE at the end instead of per-RUN-layer. This is
  what keeps the wheel-packaging step within the memory budget (per-layer snapshots of a 21 GB CUDA image
  blow up). Pair with `--compressed-caching=false` if memory is still tight.
- **`--cache=true --cache-repo=ghcr.io/open-thoughts/openthoughts-agent/cache`** — the cache repo is NOW
  POPULATED (from the v7 build), so a future rebuild that doesn't change the wheel cache-key REUSES the ~3 h
  nvcc-compile layers. **Without `--cache` every attempt recompiles vLLM+flash-attn from scratch (~3 h).** Always
  pass it.
- **Build time ≈ 3 h**: flash-attn ~30–40 min, vLLM cutlass (367 CUDA objects) ~90 min, then wheel packaging +
  the rl-stage install + the build asserts + the push. `--timeout 18000` (5 h) gives headroom.

## 5. The Dockerfile gotchas baked into the build (would fail on ANY host)

These are FIXED in `docker/Dockerfile.gpu-rl` already — listed so you recognize them if a regression
reintroduces one:

1. **uv first-index guard.** `uv pip install --extra-index-url <pytorch-cu128> … "setuptools>=77"` makes
   setuptools UNSATISFIABLE (uv pins to the first index). FIX (in the Dockerfile): add
   **`--index-strategy unsafe-best-match`** to every `uv pip install` that mixes the pytorch index with PyPI.
2. **`uv pip wheel` does not exist.** uv has NO `wheel` subcommand. The `wheel-builder` stage uses
   **`python -m pip wheel … --no-cache-dir --no-build-isolation`** (with `pip` installed into the wheel venv),
   NOT `uv pip wheel`.
3. **torchtitan needs `tyro` (step 4a).** MarinSkyRL's `apply_ep` imports
   `torchtitan.distributed.expert_parallel.ExpertParallel` for EP>1 (the 30B-A3B MoE arms). torchtitan is
   installed **`--no-deps`** (so its heavy/unrelated deps can't clobber the carefully-pinned RL venv) — BUT
   `torchtitan/__init__.py` is EAGER (`→ quantization/float8.py → config/manager.py → import tyro`), and `tyro`
   is NOT a transitive of torch/transformers, so a bare `--no-deps torchtitan` leaves it ABSENT → the
   `ExpertParallel` import-assert dies `ModuleNotFoundError: No module named 'tyro'`. FIX:
   `uv pip install "tyro"` (WITH its light pure-python deps) BEFORE
   `uv pip install --no-deps "torchtitan @ git+…@a1fdd7e"`.

## 6. Build-asserts ARE the validation (a successful build = a working EP>1 stack)

The `rl` stage RUNs hard import asserts at build time. Because kaniko fails the build if any RUN exits non-zero,
**a SUCCESSFUL build PROVES the stack works**, and the `--destination` push is the terminal step, so
**success = pushed**:
- `import flash_attn, flash_attn_2_cuda` — the CUDA extension EXISTS (from the compiled wheel; this is exactly
  what the old `SKIP_CUDA_BUILD` image lacked).
- `import torch, vllm, skyrl_train, flash_attn, flash_attn_2_cuda`.
- `from torchtitan.distributed.expert_parallel import ExpertParallel` — the **EP>1 MoE unblock**. If this prints
  `… import OK`, `apply_ep` resolves `ExpertParallel` and the CoreWeave EP=8 RL jobs can launch.

You do NOT need a separate post-build smoke for these — they're inside the build.

## 7. Monitoring the build job

> **⚠ State-poll, NEVER a log-string watch** (same rule as the launch skill). Poll the authoritative iris
> lifecycle state; a clean kill / OOM-reap emits no terminal log line.

```bash
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python   # otagent env: ships iris + a WORKING kubernetes
export KUBECONFIG=~/.kube/coreweave-iris-gpu                 # HARD prereq for every cw call
$PY scripts/iris/watch_job_state.py /benjaminfeuer/gpurl-kaniko-<gitsha> --once --json     # authoritative state now
$PY scripts/iris/watch_job_state.py /benjaminfeuer/gpurl-kaniko-<gitsha> --interval 60      # watch until terminal
# richest single call (state + error + exit + finished_at):
$IRIS --cluster=cw-us-east-02a job summary /benjaminfeuer/gpurl-kaniko-<gitsha> --json
# full log by time-window (NOT --tail, which under-samples): --since-ms <submitted_at_ms> --no-tail
$IRIS --cluster=cw-us-east-02a job logs /benjaminfeuer/gpurl-kaniko-<gitsha> --since-ms <submit_ms> --max-lines 500000 --no-tail
```
- **KUBECONFIG prereq**: `~/.kube/coreweave-iris-gpu` MUST be exported in the same shell (the default kubeconfig
  points at a different cluster → misleading "0 pods / not found"). See `.claude/ops/iris/coreweave_gpu_ops.md`.
- **otagent iris binary**: use `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris` (the bare
  `marin/.venv/bin/iris` has a broken `kubernetes` import and CANNOT drive cw). All `iris`/`kubectl` calls
  **synchronous** — never background them.

## 8. Capture the digest + bump the launcher (the terminal step)

After the build SUCCEEDS (state SUCCEEDED, push done), capture the immutable digest and pin it:

```bash
# any of these resolves the :gpu-rl-<gitsha> tag's sha256 (use the immutable tag, NEVER the floating :gpu-rl):
docker buildx imagetools inspect ghcr.io/open-thoughts/openthoughts-agent:gpu-rl-<gitsha>
crane manifest --platform linux/amd64 ghcr.io/open-thoughts/openthoughts-agent:gpu-rl-<gitsha>   # then digest it
# (or the ghcr bearer-token API exchange with $GHCR_TOKEN)
```
Then **bump `DEFAULT_RL_DOCKER_IMAGE` in `rl/cloud/launch_rl_iris.py`** to the new
`ghcr.io/open-thoughts/openthoughts-agent@sha256:<digest>` and update the provenance comment. **Pin the DIGEST,
never the floating `:gpu-rl` tag** (it stale-caches under `imagePullPolicy: IfNotPresent`). Last known-good
digest (v7): `sha256:2055412f56ac7ee4f69e24762e5a7b25ba6958ae020b7a266e02fa0cde8553fc`. Commit the launcher
bump locally.

## 9. WHEN a rebuild is actually required (vs a runtime checkout)

Rebuild the image ONLY for a change to something the image **BAKES / COMPILES**:
- **vLLM-fork commit** (`VLLM_FORK_COMMIT` ARG) — it's compiled from source; a runtime checkout can't replace
  a CUDA build.
- **flash-attn version** (`FLASH_ATTN_VERSION`) — compiled.
- **torch / CUDA** (`TORCH_VERSION` / `BASE_IMAGE` cu128) — the whole ABI/cache-key.
- **a baked dep change** (torchtitan/tyro/harbor/ray pins, the rl-stage install set).

Do **NOT** rebuild for a **MarinSkyRL source** change — that's editable at `/opt/skyrl` and picked up live at
launch via **`--skyrl-ref <commit>`** (a `git fetch && checkout`), no image rebuild. Likewise first-party
OT-Agent edits live on the next launch (the launcher syncs the local workspace to `/app`). The compiled
vLLM-fork is the only thing that genuinely forces a rebuild.

> If the build host WERE a real x86 box (not the cluster), the documented fast path is the wheel-cache:
> `docker/build_wheels.sh` (compile the wheels once → `docker/wheelhouse/`) then `docker/build_and_push.sh
> gpu-rl` with the default `WHEEL_SOURCE=prebuilt-wheelhouse` (no nvcc). On the Mac neither is usable (arm64);
> in the cluster the kaniko `--cache` repo is the equivalent durable win. See
> `docker/README.gpu-rl-wheelcache.md`.

## 10. Standing constraints

- **HF uploads default PUBLIC to `laion/`** — N/A here (this pushes a Docker image to ghcr, not a model).
- **Never `iris cluster restart`/stop/bounce** (kills every running job). The build job is its own iris job;
  `iris job kill /benjaminfeuer/gpurl-kaniko-<gitsha>` is job-scoped (with permission).
- **`--cache-repo` is shared** — don't delete the populated cache repo (it's the ~3 h-compile reuse win).

---

## Cross-reference
- **Consumes this image:** `rl-agentic-launch-iris` (the gpu-rl runtime; bump its `DEFAULT_RL_DOCKER_IMAGE`
  digest after a build).
- **Cluster access / hardware / iris-binary / KUBECONFIG:** `.claude/ops/iris/coreweave_gpu_ops.md`.
- **Dockerfile internals + the wheel cache (for a real x86 host):** `docker/README.gpu-rl-wheelcache.md`,
  `docker/Dockerfile.gpu-rl` (header + step 4a comments), `docker/build_wheels.sh`.
- **Code:** `rl/cloud/launch_rl_iris.py` (`DEFAULT_RL_DOCKER_IMAGE` digest), `scripts/iris/watch_job_state.py`
  (the monitor primitive).
