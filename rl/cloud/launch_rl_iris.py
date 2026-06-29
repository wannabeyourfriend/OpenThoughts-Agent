#!/usr/bin/env python3
"""Launch a MarinSkyRL RL training job on Marin's Iris GPU cluster (CoreWeave).

This is the GPU/Iris analog of ``rl/cloud/launch_rl_cloud.py`` (the SkyPilot RL
launcher). It combines:
  - the RL-job structure from ``launch_rl_cloud.py`` (gpu-rl venv, run_rl.py
    entrypoint, rl_config / model_path / train_data / overrides), and
  - the Iris SDK submission mechanics from ``eval/cloud/launch_eval_iris.py``
    (controller tunnel, IrisClient.submit, --secrets-env injection, --no-wait,
    job-name, max-retries, workspace source-sync to /app).

It does NOT subclass ``hpc.iris_launch_utils.IrisLauncher``: that base is
TPU-shaped (build_resources(tpu=...), build_tpu_alternatives, the uv-project
``/app/.venv`` bootstrap). The gpu-rl image is a conda-venv image
(/opt/openthoughts/envs/rl), and the target is GPU, so we drive the iris SDK's
GPU helpers (build_resources(gpu=...), gpu_device, the leafgroup-coscheduling
``resolve_multinode_defaults``) directly. Where it overlaps, the flag names and
secrets handling mirror the two templates exactly.

Multi-node / gang scheduling
----------------------------
Iris HAS a native gang mechanism for GPUs (verified via `iris job run --help`
and lib/iris/src/iris/cli/job.py):
  - ``--gpu H100x8`` requests a whole CoreWeave node (8 H100 + IB) per task.
  - ``--replicas N`` (the `--help` text: "Number of tasks for gang scheduling")
    requests N such tasks.
  - For GPUs with replicas>1, ``resolve_multinode_defaults`` returns
    ``CoschedulingConfig(group_by="leafgroup")`` — the H100/InfiniBand
    colocation level — so all N replicas are co-scheduled together on the same
    IB leaf fabric, all-or-nothing.
  - The cw-us-east-02a cluster config enables **Kueue gang admission**
    (``kueue.cluster_queue: iris-cq``, ``host_network: true`` for NCCL/IB), so
    the N-task gang is admitted atomically: either all N whole nodes are
    granted or the job queues — true exclusive, co-scheduled multi-node.

So this launcher requests ``--num-nodes N`` whole H100x8 nodes EXCLUSIVELY: one
iris task per node (``replicas=N``), each holding all 8 GPUs of its node (no
co-tenants), coscheduled by leafgroup. The RL topology (one cross-node Ray
cluster, NCCL over IB) is wired by an in-container controller
(``scripts/iris/start_rl_iris_controller.py``): rank 0 starts the Ray head and
publishes its IP to a shared rendezvous; ranks 1..N-1 join; then rank 0 runs the
SkyRL/MarinSkyRL driver (``run_rl.py --num_nodes N``) attached to that cluster.

Usage
-----
    source /Users/benjaminfeuer/Documents/secrets.env

    python -m rl.cloud.launch_rl_iris \
        --rl_config hpc/skyrl_yaml/iris/<config>.yaml \
        --model_path Qwen/Qwen3-8B \
        --train_data '["mlfoundations-dev/dataset"]' \
        --num-nodes 4 \
        --job-name my-rl-iris-run \
        --no-wait
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to sys.path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from hpc.launch_utils import PROJECT_ROOT
from hpc.iris_launch_utils import IrisLauncher  # reused only for the secrets-env loader

# Defaults for the CoreWeave H100 GPU cluster.
DEFAULT_CLUSTER = "cw-us-east-02a"
# Pin the RL image by IMMUTABLE DIGEST, not the floating ``:gpu-rl`` tag.
#
# WHY (the floating-tag stale-cache trap): the iris k8s backend always stamps
# the task pod with ``imagePullPolicy: IfNotPresent``
# (marin lib/iris .../backends/k8s/tasks.py) and we cannot override it from here.
# With a FLOATING tag, IfNotPresent means a node that already has *some* image
# under that tag name will NOT re-pull when the tag is later retagged to new
# bytes — so a node that cached an OLD ``:gpu-rl`` keeps running stale code
# (observed: a launcher run executed MarinSkyRL 4c668f4 with NO flash_attn_2_cuda
# while the freshly-retagged ``:gpu-rl`` pointed at the good build).
#
# A content-addressed ``@sha256:`` reference is self-verifying: IfNotPresent only
# treats the cache as a hit when the cached bytes hash to exactly this digest, so
# it always runs the intended image regardless of node cache state — sidestepping
# the stale-tag problem entirely without needing imagePullPolicy: Always.
#
# This digest == the immutable gitsha tag ``:gpu-rl-44c06ea8`` (OT-Agent commit
# 44c06ea8, "bump gpu-rl SKYRL_COMMIT 2d9feef -> 78d83a5"): flash_attn 2.8.3 +
# flash_attn_2_cuda present, /opt/skyrl baked at MarinSkyRL 78d83a5 — which ADDS
# the two fixes that deterministically crashed CoreWeave RL at build_models:
# 518179d (default norm_topk_prob=True for Qwen3.5/3.6 MoE) + 0b2b05b (retry around
# rank-0 HF weight-index resolution). Also still includes 2d9feef's trials_dir
# raw-str fix; harbor BAKED at 342729d5 (reward-zeroing trial.paths.trial_dir fix).
# When the gpu-rl image is rebuilt, bump this digest (use the immutable
# ``:gpu-rl-<gitsha>`` tag's digest, never the floating ``:gpu-rl``).
#
# This digest BAKES torchtitan a1fdd7e (+ tyro): the `ExpertParallel` import-assert
# (step-4a of Dockerfile.gpu-rl) PASSED in-build → the EP>1 MoE unblock is proven.
# The CoreWeave EP=8 RL jobs (30B-A3B 131k, 35B) no longer hit
# `ModuleNotFoundError: torchtitan`. Also baked: vLLM-fork 76259c63 + flash-attn
# 2.8.3 (flash_attn_2_cuda present) + MarinSkyRL 78d83a5 + harbor 342729d5.
# In-build asserts ran green: flash_attn_2_cuda OK (from cached wheel), torch
# 2.11.0+cu128 / vllm 0.1.dev16611+g76259c63a / skyrl_train import OK, torchtitan
# ExpertParallel import OK, baked MarinSkyRL HEAD == 78d83a5.
#
# BUILT IN-CLUSTER ON COREWEAVE (not the arm64 Mac): the image is amd64 + a
# from-source x86 CUDA build QEMU/Docker-Desktop can't do locally, and iris has NO
# in-cluster build primitive (`iris build` = LOCAL buildx). The build ran as an iris
# job with KANIKO (BuildKit needs CAP_SYS_ADMIN/bind-mounts the cluster denies —
# privileged is silently downgraded; nodes run gVisor). Context = the iris-synced
# /app bundle (cpu48/mem512GB/disk400GB). FAST no-nvcc PREBUILT-WHEELHOUSE path: the
# kaniko script (docker/build_gpu_rl_kaniko.sh) fetched the prebuilt vLLM-fork +
# flash-attn wheels (from laion/gpu-rl-build-wheels) into the context and ran with
# WHEEL_SOURCE=prebuilt-wheelhouse + --skip-unused-stages → ZERO nvcc (~minutes, not
# ~3h); the SKYRL_COMMIT-only bump did not change the wheel cache-key, so the wheels
# stayed ABI-correct. ghcr push via the GitHub PAT (`gh auth token`, write:packages),
# NOT the Docker-Hub DOCKER_TOKEN in secrets.env.
#
# Single-platform linux/amd64 manifest, 13 layers ~21.5 GB. The floating :gpu-rl
# tag resolves to the same digest. When the image is rebuilt, bump this digest
# (use the immutable :gpu-rl-<gitsha> tag's digest, never the floating :gpu-rl).
DEFAULT_RL_DOCKER_IMAGE = (
    "ghcr.io/open-thoughts/openthoughts-agent"
    # gpu-rl-00220aac — bumps baked harbor 342729d5 -> f7f51f13 (litellm Provider-List
    # /_turn_on_debug footer suppression + the TrialNotScoredError de-flatten in history),
    # and picks up MarinSkyRL tip 23709366 (the async-actor drain fix). vLLM-fork unchanged
    # (76259c63 == prebuilt-wheelhouse MANIFEST) so rebuilt via the fast prebuilt-wheelhouse
    # path (zero nvcc, ~22 min). Built 2026-06-29 (kaniko job gpurl-kaniko-00220aac).
    "@sha256:65b07cec09b015117271dc6a2b19bb657eb1b025f969b3476cea2288501838b6"
)
DEFAULT_GPU_VARIANT = "H100"
DEFAULT_GPUS_PER_NODE = 8           # gd-8xh100ib-i128 = 8x H100-80GB + IB
# These H100 nodes are requested WHOLE-NODE-EXCLUSIVE (no co-tenants) — so request ALL the
# node's allocatable resources; don't under-request (wasted capacity + a too-low --memory
# caused a container-cgroup OOM at FSDP weight-load on the 30B run). Node allocatable ≈ 128
# CPU / ~2014 GiB mem / 8 GPU.
#   - CPU 48 (NOT 64): ~64-68 of the 128 cores are persistent daemonset reservation, so a
#     request >~60 FAILS the single-IB-leaf gang admission (observed: 64 unplaceable, 48 admits).
#   - MEMORY 1400GB is the validated middle. It must clear TWO opposing footguns:
#     (a) too LOW (e.g. the old 512GB) → container-cgroup OOM at FSDP weight-load on an EP=8
#         + cpu_offload policy rank (peaks >512GB while the node sits <200GB used); and
#     (b) too HIGH (1800GB ≈ 1676 GiB) → sits so close to node-allocatable (~2014 GiB) that
#         after daemonset/persistent-reservation overhead a leafgroup gang (all-or-nothing,
#         one IB leaf) can't fit all pods → Kueue SchedulingGated stall (cost multiple
#         60-120min stalls overnight 2026-06-26, on a 1-GPU probe AND 8-node gangs).
#     1400GB admits the 8-node 131k EP8 gang cleanly AND does the full weight-load with no
#     cgroup-OOM. Lower toward the real need on an admission stall; NEVER raise toward 1800.
#     (1000-1200GB suffices for 2-node smokes.) See .claude/ops/iris/coreweave_gpu_ops.md.
#   - DISK defaults to "auto" = 80% of the node's live allocatable ephemeral-storage (~27.2 TiB
#     → ~21 TiB). WHY NOT the old 512GB: the long MoE training step's Ray object store spills to
#     /tmp (a metered emptyDir that counts against the --disk ephemeral-storage limit), growing
#     to >1.6 TB and EVICTING the pod (2026-06-28). Whole-node-exclusive gangs have NO co-tenants,
#     so reserving disk is pure waste — claim ~80%. (R2 object-spilling, the durable fix, is also
#     on; this headroom is belt-and-suspenders.) Pass --disk explicitly to override.
DEFAULT_CPU_PER_NODE = 48.0
DEFAULT_MEMORY_PER_NODE = "1400GB"
# --disk "auto" → DISK_FRACTION of the GPU node's live allocatable ephemeral-storage at launch
# (FALLBACK_DISK_GIB iff the node query fails). See _resolve_default_disk().
DEFAULT_DISK_PER_NODE = "auto"
DISK_FRACTION = 0.80
FALLBACK_DISK_GIB = 21800  # ~80% of the h100-8x ~27.2 TiB allocatable, used only if kubectl is unavailable
DEFAULT_PRIORITY = "interactive"


def _parse_quantity_to_gib(q: str) -> float:
    """Parse a k8s resource quantity (plain bytes, or Ki/Mi/Gi/Ti binary / k/M/G/T decimal suffix) to GiB."""
    q = q.strip()
    for suf, mult in (("Ki", 2**10), ("Mi", 2**20), ("Gi", 2**30), ("Ti", 2**40), ("Pi", 2**50)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    for suf, mult in (("k", 1e3), ("M", 1e6), ("G", 1e9), ("T", 1e12), ("P", 1e15)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    return float(q) / 2**30  # plain bytes


def _resolve_default_disk(fraction: float = DISK_FRACTION) -> str:
    """``fraction`` of the GPU node's LIVE allocatable ephemeral-storage, as a ``"<N>Gi"`` string.

    Whole-node-exclusive gangs have no co-tenants, so claim most of the node NVMe (the old fixed
    512GB default evicted long MoE steps once Ray's object store spilled to the metered /tmp).
    Queries kubectl for the MIN allocatable across 8-GPU nodes (never over-request a smaller node);
    falls back to FALLBACK_DISK_GIB if kubectl is unavailable (requires KUBECONFIG)."""
    import subprocess

    try:
        out = subprocess.run(
            ["kubectl", "get", "nodes", "-o",
             r'jsonpath={range .items[*]}{.status.capacity.nvidia\.com/gpu}{" "}'
             r'{.status.allocatable.ephemeral-storage}{"\n"}{end}'],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        allocs = [
            _parse_quantity_to_gib(p[1])
            for p in (line.split() for line in out.splitlines())
            if len(p) == 2 and p[0] == "8"
        ]
        if allocs:
            gib = int(min(allocs) * fraction)
            print(f"[rl-iris] --disk auto: {fraction:.0%} of node allocatable "
                  f"(min {min(allocs):.0f}GiB across {len(allocs)} GPU nodes) = {gib}Gi", flush=True)
            return f"{gib}Gi"
        print("[rl-iris] --disk auto: no 8-GPU nodes returned by kubectl; "
              f"using fallback {FALLBACK_DISK_GIB}Gi", flush=True)
    except Exception as exc:  # noqa: BLE001 - best-effort; fall back rather than block a launch
        print(f"[rl-iris] --disk auto: kubectl node query failed ({type(exc).__name__}: {exc}); "
              f"using fallback {FALLBACK_DISK_GIB}Gi", flush=True)
    return f"{FALLBACK_DISK_GIB}Gi"
# The gpu-rl image's RL venv (deps-only: torch 2.11 + vLLM fork + skyrl editable).
RL_PYTHON = "/opt/openthoughts/envs/rl/bin/python"
SKYRL_HOME = "/opt/skyrl"
# In-container source sync target. iris syncs the launcher's `workspace`
# (the OT-Agent repo) to /app and sets IRIS_WORKDIR=/app; putting /app first on
# PYTHONPATH makes the live synced OT-Agent code win over the image's baked
# /opt/openthoughts copy.
APP_DIR = "/app"


def _resolve_cluster_config_default() -> str:
    """Find the marin repo's cw-us-east-02a cluster YAML."""
    rel = f"lib/iris/config/{DEFAULT_CLUSTER}.yaml"
    candidates = [
        Path.home() / "Documents/marin" / rel,
        Path("/Users/benjaminfeuer/Documents/marin") / rel,
        Path(os.environ.get("MARIN_ROOT", "")) / rel,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return rel


def _default_secrets_env() -> Optional[str]:
    cand = os.environ.get("OT_AGENT_SECRETS_ENV") or os.path.expanduser("~/Documents/secrets.env")
    return cand if os.path.isfile(cand) else None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a MarinSkyRL RL training job on the Iris CoreWeave H100 cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- RL job args (mirror launch_rl_cloud.py) ---
    parser.add_argument(
        "--rl_config", required=True,
        help="Path to SkyRL/MarinSkyRL config YAML (repo-relative or absolute).",
    )
    parser.add_argument("--rl-config", dest="rl_config", help=argparse.SUPPRESS)

    parser.add_argument(
        "--model_path", required=True,
        help="Model path or HuggingFace ID (e.g., Qwen/Qwen3-8B).",
    )
    parser.add_argument("--model-path", dest="model_path", help=argparse.SUPPRESS)

    parser.add_argument(
        "--train_data", default="[]",
        help="Training data paths as a JSON list (e.g., '[\"org/dataset\"]').",
    )
    parser.add_argument("--train-data", dest="train_data", help=argparse.SUPPRESS)

    parser.add_argument(
        "--val_data", default="[]",
        help="Validation data paths as a JSON list.",
    )
    parser.add_argument("--val-data", dest="val_data", help=argparse.SUPPRESS)

    parser.add_argument(
        "--skyrl_override", action="append", default=[],
        help="SkyRL Hydra override (repeatable).",
    )
    parser.add_argument("--skyrl-override", dest="skyrl_override", action="append", help=argparse.SUPPRESS)

    parser.add_argument(
        "--experiments_dir", default="/app/experiments",
        help="In-container experiments output dir (on the synced /app workspace).",
    )
    parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

    # --- Resource / topology args (GPU multi-node) ---
    parser.add_argument(
        "--num-nodes", "--num_nodes", dest="num_nodes", type=int, default=1,
        help="Number of WHOLE H100 nodes to request EXCLUSIVELY, gang/co-scheduled "
             "(one iris task per node, all 8 GPUs each, coscheduled by leafgroup/IB).",
    )
    parser.add_argument(
        "--gpus-per-node", "--gpus_per_node", dest="gpus_per_node", type=int,
        default=DEFAULT_GPUS_PER_NODE,
        help="GPUs per node (CoreWeave nodes are 8x H100).",
    )
    parser.add_argument(
        "--gpu-variant", "--gpu_variant", dest="gpu_variant", default=DEFAULT_GPU_VARIANT,
        help="GPU variant (default H100).",
    )
    parser.add_argument(
        "--cpu", type=float, default=DEFAULT_CPU_PER_NODE,
        help="CPU cores per node.",
    )
    parser.add_argument(
        "--memory", default=DEFAULT_MEMORY_PER_NODE,
        help="Memory per node.",
    )
    parser.add_argument(
        "--disk", default=DEFAULT_DISK_PER_NODE,
        help=f"Ephemeral disk per node. Default 'auto' = {int(DISK_FRACTION * 100)}%% of the GPU "
             "node's live allocatable ephemeral-storage (whole-node-exclusive gangs have no "
             "co-tenants, so claim most of the node NVMe — keeps Ray object-spill / checkpoints "
             "clear of the ephemeral-storage eviction). Pass an explicit value (e.g. 4000GB) to override.",
    )
    parser.add_argument(
        "--ray-port", "--ray_port", dest="ray_port", type=int, default=6379,
        help="Port the cross-node Ray head binds.",
    )
    parser.add_argument(
        "--rendezvous-dir", "--rendezvous_dir", dest="rendezvous_dir", default=None,
        help="Shared object-store/path (gs://, s3://, or shared dir) for the multi-node "
             "Ray head/worker rendezvous. Required for --num-nodes>1. On cw-us-east-02a "
             "use an s3:// (R2) URI under the cluster's marin-na bucket, e.g. "
             "s3://marin-na/iris/rl-rdv/<job>; the cluster injects working R2 creds into "
             "every task pod (iris-task-env Secret), so no external creds are needed.",
    )
    parser.add_argument(
        "--trials-dir", "--trials_dir", dest="trials_dir", default="auto",
        help="Where Harbor writes per-trial agentic-RL rollout artifacts "
             "(terminal_bench_config.trials_dir). 'auto' (default) = "
             "s3://marin-na/iris/<job_name>/trace_jobs — a DURABLE R2 path the cw-us-east-02a "
             "pods reach via auto-injected creds, so rollouts survive pod GC and are inspectable "
             "post-hoc. 'local'/'off' = keep the config default (node-local "
             "/app/experiments/<run>/trace_jobs, EPHEMERAL — lost on GC, no shared FS/PVC). Or pass "
             "an explicit s3://, gs://, or path URI. NOTE: cw uses R2 (s3://marin-na), NOT gs://; "
             "ignored if you already set terminal_bench_config.trials_dir via --skyrl_override.",
    )

    # --- Iris submission args (mirror launch_eval_iris.py / IrisLauncher) ---
    parser.add_argument(
        "--cluster", default=DEFAULT_CLUSTER,
        help="Iris cluster name (default cw-us-east-02a).",
    )
    parser.add_argument(
        "--cluster-config", "--cluster_config", dest="cluster_config",
        default=_resolve_cluster_config_default(),
        help="Path to the iris cluster YAML (default: cw-us-east-02a in the marin repo).",
    )
    parser.add_argument(
        "--task-image", "--task_image", "--docker_image", "--docker-image",
        dest="task_image", default=DEFAULT_RL_DOCKER_IMAGE,
        help=f"Container image (default {DEFAULT_RL_DOCKER_IMAGE}).",
    )
    parser.add_argument(
        "--job-name", "--job_name", dest="job_name", default=None,
        help="Job name (auto-derived if not set).",
    )
    parser.add_argument(
        "--priority", default=DEFAULT_PRIORITY,
        choices=["production", "interactive", "batch"],
        help="Iris priority band.",
    )
    parser.add_argument(
        "--max-retries", "--max_retries", dest="max_retries", type=int, default=0,
        help="Max retries on failure (iris auto-retries preemptions separately).",
    )
    parser.add_argument(
        "--timeout", type=int, default=0,
        help="Job timeout in seconds (0 = no timeout).",
    )
    parser.add_argument(
        "--no-wait", dest="no_wait", action="store_true", default=False,
        help="Submit and detach instead of streaming logs.",
    )
    parser.add_argument(
        "--preemptible", dest="preemptible", action="store_true", default=None,
        help="Force scheduling on preemptible workers.",
    )
    parser.add_argument(
        "--no-preemptible", dest="preemptible", action="store_false",
        help="Force scheduling on non-preemptible workers.",
    )
    parser.add_argument(
        "--secrets-env", "--secrets_env", dest="secrets_env", default=_default_secrets_env(),
        help="KEY=VALUE env file injected into the task (HF_TOKEN, WANDB_API_KEY, etc.). "
             "Defaults to $OT_AGENT_SECRETS_ENV, else ~/Documents/secrets.env.",
    )
    parser.add_argument(
        "--skyrl-ref", "--skyrl_ref", dest="skyrl_ref", default=None,
        help="If set, `git fetch && git checkout <ref>` the baked MarinSkyRL clone at "
             "/opt/skyrl BEFORE running, so the live editable install picks up a newer "
             "(or pinned) commit than the one baked into the image. Use to apply a "
             "MarinSkyRL fix that landed AFTER the image was built without waiting for an "
             "image rebuild (deps are baked, but skyrl-train is an editable git clone, so "
             "a checkout is live). Default: unset = use whatever commit the image baked.",
    )
    parser.add_argument(
        "--dry-run", "--dry_run", dest="dry_run", action="store_true", default=False,
        help="Print the resolved config + in-container command without submitting.",
    )

    return parser


def load_config_extra_env(rl_config_path: str) -> dict[str, str]:
    """Read a top-level ``extra_env:`` mapping from the RL config YAML.

    On the SLURM/Apptainer path the runtime env lives under ``container.extra_env``
    and is emitted as shell ``export`` lines (hpc/rl_launch_utils.py). The Iris path
    has NO ``container:`` block (the gpu-rl Docker image is the runtime), so that
    plumbing never runs — without this, env declared in the YAML is silently
    dropped and only the launcher's hardcoded passthrough (HF/WANDB/DAYTONA) reaches
    the pod. This forwards a top-level ``extra_env:`` block (and, defensively,
    ``container.extra_env`` if a ported config still carries one) into the iris
    EnvironmentSpec so e.g. EPDIAG probe arms + R3/DCP guard env take effect.

    Values are coerced to str (YAML may parse "1"/true as int/bool). Returns {} if
    the file is unreadable or declares no extra_env (byte-identical behavior for the
    existing extra_env-less iris configs).
    """
    try:
        full = PROJECT_ROOT / rl_config_path
        path = full if full.exists() else Path(rl_config_path)
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[rl-iris] WARNING: could not read extra_env from {rl_config_path}: {exc}",
              file=sys.stderr)
        return {}
    extra = dict(raw.get("extra_env") or {})
    container_env = (raw.get("container") or {}).get("extra_env") or {}
    for k, v in container_env.items():
        extra.setdefault(k, v)
    out: dict[str, str] = {}
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        out[str(k)] = str(v)
    return out


def normalize(args: argparse.Namespace) -> None:
    """Validate + normalize. Keep rl_config repo-relative so it resolves on /app."""
    # Resolve rl_config to a repo-relative path (it must exist on the synced
    # /app workspace, NOT be an absolute host path).
    rl_cfg = Path(args.rl_config)
    if rl_cfg.is_absolute():
        try:
            args.rl_config = str(rl_cfg.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            raise SystemExit(
                f"--rl_config {args.rl_config!r} is absolute and not under the repo "
                f"({PROJECT_ROOT}); pass a repo-relative path so it resolves on /app."
            )
    # Verify it exists locally (so we fail fast before submitting).
    if not (PROJECT_ROOT / args.rl_config).exists():
        # Fall back to hpc/skyrl_yaml/<name>[.yaml] like launch_rl_cloud.py.
        yaml_dir = Path("hpc/skyrl_yaml")
        for cand in (yaml_dir / args.rl_config, yaml_dir / f"{args.rl_config}.yaml"):
            if (PROJECT_ROOT / cand).exists():
                args.rl_config = str(cand)
                break
        else:
            print(f"[rl-iris] WARNING: --rl_config {args.rl_config!r} not found under "
                  f"{PROJECT_ROOT}; the worker will error if it isn't on /app.",
                  file=sys.stderr)

    if args.num_nodes < 1:
        raise SystemExit("--num-nodes must be >= 1.")
    if args.num_nodes > 1 and not args.rendezvous_dir:
        raise SystemExit(
            "--num-nodes>1 requires --rendezvous-dir (a shared gs://, s3://, or path URI "
            "both head and worker nodes can reach) for the multi-node Ray rendezvous."
        )


def build_task_command(args: argparse.Namespace) -> List[str]:
    """Build the in-container command, multi-node-aware.

    The full pipeline that runs inside each task container:
      cd /app
      && export SKYRL_HOME + PYTHONPATH (live /app + skyrl-train win)
      && <RL_PYTHON> scripts/iris/start_rl_iris_controller.py
            --ray-port ... --rendezvous-dir ...
            -- <RL_PYTHON> -m rl.local.run_rl --rl_config ... --num_nodes N ...

    Rank 0 (IRIS_TASK_ID==0) starts the Ray head and runs run_rl.py (which, with
    RAY_ADDRESS set + --num_nodes>1, attaches to the cluster instead of starting a
    local one). Workers join Ray and park. We invoke the gpu-rl venv python by
    absolute path so it is used regardless of whichever venv iris's setup phase
    activates.
    """
    total_gpus = args.num_nodes * args.gpus_per_node

    # The MarinSkyRL training command rank 0 runs (run_rl.py owns config parse,
    # hydra-arg build, HF data resolution, and the SkyRL entrypoint launch).
    train_cmd: List[str] = [
        RL_PYTHON, "-m", "rl.local.run_rl",
        "--rl_config", args.rl_config,
        "--model_path", args.model_path,
        "--job_name", args.job_name,
        "--gpus", str(total_gpus),
        "--num_nodes", str(args.num_nodes),
        "--gpus_per_node", str(args.gpus_per_node),
        "--experiments_dir", args.experiments_dir,
        "--ray_port", str(args.ray_port),
    ]
    if args.train_data and args.train_data != "[]":
        train_cmd.extend(["--train_data", args.train_data])
    if args.val_data and args.val_data != "[]":
        train_cmd.extend(["--val_data", args.val_data])
    for override in (args.skyrl_override or []):
        train_cmd.extend(["--skyrl_override", override])

    # Durable Harbor rollout artifacts. The default (config trials_dir: null) resolves to a
    # node-local path on the rank-0 pod (/app/experiments/<run>/trace_jobs); cw-us-east-02a has
    # no shared FS/PVC and GCs pods on terminal, so those per-trial rollouts are lost when the
    # job ends. Point terminal_bench_config.trials_dir at a remote R2 URI (the cluster injects
    # working R2 creds, same store as the rendezvous) so rollouts persist + are inspectable
    # post-hoc. Skip if the user opted out (--trials-dir local) or already set it explicitly.
    trials_dir = (args.trials_dir or "auto").strip()
    user_set_trials = any("terminal_bench_config.trials_dir=" in o for o in (args.skyrl_override or []))
    if trials_dir.lower() not in ("local", "off", "none", "") and not user_set_trials:
        if trials_dir.lower() == "auto":
            trials_dir = f"s3://marin-na/iris/{args.job_name}/trace_jobs"
        train_cmd.extend(["--skyrl_override", f"++terminal_bench_config.trials_dir={trials_dir}"])

    # The controller wraps the training command for the multi-node Ray bootstrap.
    controller_cmd: List[str] = [
        RL_PYTHON, "scripts/iris/start_rl_iris_controller.py",
        "--ray-port", str(args.ray_port),
    ]
    if args.rendezvous_dir:
        controller_cmd.extend(["--rendezvous-dir", args.rendezvous_dir])
    # Per-NODE task-dataset staging. On a multi-node iris/CoreWeave slice each task
    # pod has its OWN node-local filesystem ($DCFT=/opt/openthoughts in the gpu-rl
    # image) — there is NO shared scratch like SLURM's GPFS. run_rl.py's
    # resolve_rl_train_data() extracts the HF task dataset to /opt/openthoughts/tasks/
    # but it runs ONLY on rank 0 (the head), so the Ray-scheduled rollout workers on
    # ranks 1..N-1 find an empty tasks dir and every rollout dies with
    # FileNotFoundError: .../task.toml -> reward always 0 (data-starved, doomed run).
    # Fix: forward --train-data to the controller so it can run the SAME extraction
    # on EVERY node before Ray starts, populating the identical node-local path on
    # all pods. Idempotent (on_exist=skip) — rank-0's later run_rl re-resolve is a
    # cheap no-op.
    if args.train_data and args.train_data != "[]":
        controller_cmd.extend(["--train-data", args.train_data])
    controller_cmd.append("--")
    controller_cmd.extend(train_cmd)

    # Wrap in a bash bootstrap: cd to the synced workspace and set PYTHONPATH so
    # live /app + skyrl-train win over the image's baked copies. Use the absolute
    # RL venv python (set above) — independent of iris's activated venv.
    pythonpath = f"{APP_DIR}:{SKYRL_HOME}/skyrl-train"
    # Optional: refresh the baked MarinSkyRL editable clone to a newer/pinned commit
    # before running (deps are baked, but skyrl-train is `pip install -e` over a git
    # clone, so a checkout is live without reinstall). Fetch is best-effort but the
    # checkout MUST succeed (the ref is the whole point), so it's under `set -e`.
    skyrl_refresh = ""
    if args.skyrl_ref:
        ref = shlex.quote(args.skyrl_ref)
        skyrl_refresh = (
            f"git -C {shlex.quote(SKYRL_HOME)} fetch --quiet --all || true; "
            f"git -C {shlex.quote(SKYRL_HOME)} checkout {ref}; "
            # Purge baked bytecode after the checkout. The gpu-rl image bakes
            # `.pyc` for the editable skyrl-train at its build-time commit; if those
            # were compiled with hash-based (UNCHECKED_HASH) invalidation, Python
            # does NOT recompile when `git checkout` swaps the `.py` underneath, so
            # a `--skyrl-ref` checkout SILENTLY runs the stale baked bytecode (proven
            # 2026-06-25: the norm_topk_prob fix at 518179d checked out, but the pod
            # raised at the pre-fix line numbers). Delete the cache so the live `.py`
            # is recompiled. Best-effort (|| true) — must not block on a read-only fs.
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name '*.pyc' -delete 2>/dev/null || true; "
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name __pycache__ -type d -prune -exec rm -rf {{}} + 2>/dev/null || true; "
            f"echo \"[rl-iris] MarinSkyRL now at $(git -C {shlex.quote(SKYRL_HOME)} rev-parse HEAD)\"; "
        )
    bash = (
        f"set -e; cd {APP_DIR}; "
        f"{skyrl_refresh}"
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export VLLM_USE_V1=1; "
        f"exec {shlex.join(controller_cmd)}"
    )
    return ["bash", "-c", bash]


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    normalize(args)

    if not args.job_name:
        args.job_name = f"rl-iris-{time.strftime('%Y%m%d-%H%M%S')}"

    # Load --secrets-env into os.environ on the launch host (so launch-host
    # hooks see it) AND collect them for injection into the task. Reuse the
    # IrisLauncher static helper (same semantics as launch_eval_iris.py).
    IrisLauncher.load_secrets_env_into_os_environ(args.secrets_env)

    command = build_task_command(args)

    # Per-task resources: a WHOLE node (8 H100 + IB), one task per node.
    gpu_spec = f"{args.gpu_variant}x{args.gpus_per_node}"

    # Resolve the "auto" disk default to ~80% of the node's live allocatable ephemeral-storage.
    # (Whole-node-exclusive gangs have no co-tenants → reserving disk is wasted; a too-low fixed
    # default evicted long MoE steps once Ray spilled to the metered /tmp. See _resolve_default_disk.)
    if str(args.disk).strip().lower() == "auto":
        args.disk = _resolve_default_disk()

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    print(f"[rl-iris] Job:        /{user}/{args.job_name}", flush=True)
    print(f"[rl-iris] Cluster:    {args.cluster}  ({args.cluster_config})", flush=True)
    print(f"[rl-iris] Image:      {args.task_image}", flush=True)
    print(f"[rl-iris] Topology:   {args.num_nodes} node(s) x {gpu_spec}  "
          f"(= {args.num_nodes * args.gpus_per_node} GPUs, exclusive, gang/leafgroup)", flush=True)
    print(f"[rl-iris] Per node:   cpu={args.cpu} memory={args.memory} disk={args.disk}", flush=True)
    print(f"[rl-iris] Priority:   {args.priority}", flush=True)
    print(f"[rl-iris] RL config:  {args.rl_config}  model={args.model_path}", flush=True)
    if args.num_nodes > 1:
        print(f"[rl-iris] Rendezvous: {args.rendezvous_dir}", flush=True)
    print(f"[rl-iris] Command:    {shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[rl-iris] --dry-run: not submitting", flush=True)
        return 0

    # Defer heavy iris imports so --dry-run / --help stay snappy.
    from iris.client import IrisClient
    from iris.cluster.config import IrisConfig
    from iris.cluster.types import EnvironmentSpec, Entrypoint
    from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults
    from iris.rpc import job_pb2

    # Per-task resources: whole node, all GPUs (no co-tenant → exclusive).
    resources = build_resources(
        None, gpu_spec, cpu=args.cpu, memory=args.memory, disk=args.disk
    )

    # Multi-node gang: replicas=num_nodes; for GPUs with replicas>1 this returns
    # CoschedulingConfig(group_by="leafgroup") — co-schedule all nodes on one IB
    # leaf fabric, atomically (Kueue gang admission on cw-us-east-02a).
    replicas, coscheduling = resolve_multinode_defaults(None, args.gpu_variant, args.num_nodes)

    resources_proto = resources.to_proto()
    constraints = build_job_constraints(
        resources_proto=resources_proto,
        tpu_variants=[],
        replicas=replicas,
        regions=None,
        zone=None,
        preemptible=args.preemptible,
    )

    priority_band = {
        "production": job_pb2.PRIORITY_BAND_PRODUCTION,
        "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
        "batch": job_pb2.PRIORITY_BAND_BATCH,
    }.get(args.priority, job_pb2.PRIORITY_BAND_UNSPECIFIED)

    # Env: secrets file values + the standard RL/iris-serve signals. iris injects
    # IRIS_TASK_ID / IRIS_NUM_TASKS / IRIS_ADVERTISE_HOST per task automatically.
    env_vars: dict[str, str] = {}
    # Forward the RL config YAML's top-level `extra_env:` block (the Iris analog of
    # the SLURM container.extra_env exports — see load_config_extra_env). Seeded
    # FIRST so the launcher's own signals (rendezvous/secrets, below) win on any
    # collision.
    config_extra_env = load_config_extra_env(args.rl_config)
    if config_extra_env:
        env_vars.update(config_extra_env)
        print(f"[rl-iris] Config extra_env: {', '.join(sorted(config_extra_env))}", flush=True)
    if args.rendezvous_dir:
        env_vars["OT_AGENT_IRIS_RENDEZVOUS_DIR"] = args.rendezvous_dir
    env_vars["OT_AGENT_IRIS_RAY_PORT"] = str(args.ray_port)
    # Forward the launch host's secrets (mirrors launch_eval_iris.py passthrough).
    #
    # IMPORTANT — do NOT forward AWS_*/R2_* here. The cw-us-east-02a cluster
    # projects an `iris-task-env` k8s Secret into EVERY task pod via `envFrom`
    # (because storage.remote_state_dir is an s3:// URI), and that secret already
    # carries the correct in-cluster R2 credentials + endpoint
    # (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL / AWS_REGION /
    # FSSPEC_S3). In K8s, explicit container `env` entries take precedence over
    # `envFrom`, so forwarding the launch host's AWS_* (which point at a
    # DIFFERENT account and lack AWS_ENDPOINT_URL) would CLOBBER the pod's R2
    # creds and make the s3://marin-na rendezvous (multi-node) silently target
    # real AWS S3 instead of R2. Let the cluster-injected R2 creds win; the
    # fsspec rendezvous in start_rl_iris_controller.py uses default credential
    # discovery and picks them up.
    #
    # Daytona credentials MUST be forwarded: agentic RL (terminal_bench / Harbor)
    # builds a Daytona sandbox per trial, and iris injects only HF/WANDB into the
    # task pod — nothing else. Without DAYTONA_API_KEY the worker's harbor client
    # raises DaytonaAuthenticationError on every env build, so no sandbox comes
    # up, the verifier never runs, and EVERY trajectory finalizes as
    # VerificationNotCompletedError with reward 0 (observed zeroing an entire
    # reverify rollout). Mirror the base IrisLauncher passthrough set
    # (hpc/iris_launch_utils.py) so the same creds reach the RL worker.
    #
    # WANDB routing default: the iris RL configs log to wandb (trainer.logger: wandb;
    # CoreWeave has egress). SkyRL's wandb.init passes project= but NOT entity=
    # (MarinSkyRL tracking.py), so without WANDB_ENTITY the run silently lands in the
    # API key's DEFAULT entity (e.g. nyu-dice-lab), not the team org. Default both to
    # the OT-Agent team here (matches hpc/dotenv/perlmutter.env) so every run lands in
    # dogml/OpenThoughts-Agent; an explicitly-set launch-host WANDB_ENTITY/PROJECT wins.
    os.environ.setdefault("WANDB_ENTITY", "dogml")
    os.environ.setdefault("WANDB_PROJECT", "OpenThoughts-Agent")
    for k in (
        "HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT",
        "DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN", "DAYTONA_ORGANIZATION_ID",
        "DAYTONA_API_URL",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "TOGETHER_API_KEY",
    ):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v

    iris_config = IrisConfig.load(args.cluster_config)
    bundle = iris_config.provider_bundle()
    controller_proto = iris_config.proto.controller
    if controller_proto.WhichOneof("controller") == "local":
        from iris.cluster.providers.local.cluster import LocalCluster
        controller_address = LocalCluster(iris_config.proto).start()
    else:
        controller_address = (
            iris_config.controller_address()
            or bundle.controller.discover_controller(controller_proto)
        )

    with bundle.controller.tunnel(controller_address) as controller_url:
        client = IrisClient.remote(controller_url, workspace=PROJECT_ROOT)
        entrypoint = Entrypoint.from_command(*command)
        job = client.submit(
            entrypoint=entrypoint,
            name=args.job_name,
            resources=resources,
            environment=EnvironmentSpec(env_vars=env_vars, extras=[]),
            constraints=constraints or None,
            coscheduling=coscheduling,
            replicas=replicas,
            max_retries_failure=args.max_retries,
            task_image=args.task_image,
            priority_band=priority_band,
            timeout=None if args.timeout == 0 else _seconds_to_duration(args.timeout),
        )
        full_job_id = str(job.job_id)
        print(f"[rl-iris] Submitted: {full_job_id}  (replicas={replicas}, "
              f"coscheduling={getattr(coscheduling, 'group_by', None)})", flush=True)

        if args.no_wait:
            return 0
        try:
            status = job.wait(stream_logs=True, timeout=float("inf"))
            exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
        except KeyboardInterrupt:
            print(f"[rl-iris] Terminating job {full_job_id}...", file=sys.stderr, flush=True)
            client.terminate_job(job.job_id)
            exit_code = 130
        print(f"[rl-iris] Job exit: {exit_code}", flush=True)
        return exit_code


def _seconds_to_duration(secs: int):
    from rigging.timing import Duration
    return Duration.from_seconds(secs)


if __name__ == "__main__":
    sys.exit(main())
