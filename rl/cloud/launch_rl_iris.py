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
# This digest == the immutable gitsha tag ``:gpu-rl-8bc5bdb4`` (OT-Agent commit
# 8bc5bdb4, "pin gpu-rl to MarinSkyRL 2d9feef + harbor 342729d5"): flash_attn
# 2.8.3 + flash_attn_2_cuda present, /opt/skyrl baked at MarinSkyRL 2d9feef
# (trials_dir raw-str fix), harbor BAKED at 342729d5 (reward-zeroing
# trial.paths.trial_dir fix) — both fixes now in the image, NO runtime sed
# patches needed. Validated by a no-runtime-patch RL smoke (reward>0 over >=2
# steps, num_failed_trajectories=0). Supersedes :gpu-rl-1b5a82d6 (sha256
# f571b88a) which had working flash_attn but LACKED these two fixes.
# When the gpu-rl image is rebuilt, bump this digest (use the immutable
# ``:gpu-rl-<gitsha>`` tag's digest, never the floating ``:gpu-rl``).
DEFAULT_RL_DOCKER_IMAGE = (
    "ghcr.io/open-thoughts/openthoughts-agent"
    "@sha256:df2cb77e49fc078c75e909f0f967af3205b2b7cf1c4656c5ab5cd907e37128d1"
)
DEFAULT_GPU_VARIANT = "H100"
DEFAULT_GPUS_PER_NODE = 8           # gd-8xh100ib-i128 = 8x H100-80GB + IB
DEFAULT_CPU_PER_NODE = 64.0
DEFAULT_MEMORY_PER_NODE = "512GB"
DEFAULT_DISK_PER_NODE = "512GB"
DEFAULT_PRIORITY = "interactive"
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
        help="Ephemeral disk per node.",
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

    # The controller wraps the training command for the multi-node Ray bootstrap.
    controller_cmd: List[str] = [
        RL_PYTHON, "scripts/iris/start_rl_iris_controller.py",
        "--ray-port", str(args.ray_port),
    ]
    if args.rendezvous_dir:
        controller_cmd.extend(["--rendezvous-dir", args.rendezvous_dir])
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
