#!/usr/bin/env python
"""Submit the EP=8 CROSS-NODE, NON-CIRCULAR weight-vs-disk diagnostic on CoreWeave.

The decisive measurement for the MoE token-salad (Class W). Brings up ONLY the FSDP
grouped+EP policy worker at the PROD MoE factorization EP=8 x FSDP=2 = 16 GPU, laid
out 4 nodes x 4 GPU/node so the 8 EP ranks of a group STRADDLE >=2 physical nodes
(verified in-job, not assumed). NO inference engine, NO rollout, NO Daytona. At gs0
(untrained base weights) it gathers layer-0's grouped experts.w1/w2/w3 via the REAL
on-GPU ``_gather_tensor`` (= ``gather_dtensor_strided_safe`` over the
``(_StridedShard(fsdp), Shard(ep))`` composite) and compares each expert row to the
BASE model's on-disk HF checkpoint (``safetensors.safe_open``) — a reference path that
NEVER touches the EP gather. NON-circular (fixes the prior EXP2 which compared two
traversals of the SAME gather).

Multi-node: reuses launch_rl_iris's exact submission machinery (image, leafgroup gang,
secrets, Ray rendezvous via start_rl_iris_controller.py). The MODIFIED fsdp_worker.py
(the new ``diag_ep8_*`` worker methods) + the diag driver are UNTRACKED local edits in
MarinSkyRL; we do NOT push (supervisor owns commits). So we base64-inject BOTH into
every pod's /opt/skyrl checkout before running, then rank 0 runs the diag as the Ray
driver attached to the cross-node cluster.

Usage:
    source /Users/benjaminfeuer/Documents/secrets.env
    export KUBECONFIG=~/.kube/coreweave-iris-gpu
    python scripts/iris/submit_ep8_disk_ref_diag.py [--dry-run] [--num-nodes 4] [--gpus-per-node 4]
"""
from __future__ import annotations

import argparse
import base64
import os
import shlex
import sys
import time
from pathlib import Path

from rl.cloud.launch_rl_iris import (  # noqa: E402
    APP_DIR,
    DEFAULT_CLUSTER,
    DEFAULT_GPU_VARIANT,
    DEFAULT_RL_DOCKER_IMAGE,
    PROJECT_ROOT,
    RL_PYTHON,
    SKYRL_HOME,
    _resolve_cluster_config_default,
    _default_secrets_env,
)
from hpc.iris_launch_utils import IrisLauncher  # noqa: E402

# Default geometry: EP=8 x FSDP=2 = 16 GPU as 4 nodes x 4 GPU/node => cross-node EP.
DEFAULT_NUM_NODES = 4
DEFAULT_GPUS_PER_NODE = 4
DEFAULT_EP = 8
DEFAULT_FSDP = 2
MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"

# Generous footprint (whole 30B base weight-load on each node).
CPU_PER_NODE = 48.0
MEMORY_PER_NODE = "1400GB"
DISK_PER_NODE = "512GB"

# The MODIFIED ground-truth files we must inject into every pod (untracked locally).
SKYRL_LOCAL = Path("/Users/benjaminfeuer/Documents/MarinSkyRL/skyrl-train")
_FSDP_WORKER = (SKYRL_LOCAL / "skyrl_train/workers/fsdp/fsdp_worker.py",
                "skyrl_train/workers/fsdp/fsdp_worker.py")
# fsdp_utils.py carries gather_dtensor_strided_safe (added ac44079, NOT in the baked
# 78d83a5 image); the injected fsdp_worker imports it. Inject so the gather resolves.
_FSDP_UTILS = (SKYRL_LOCAL / "skyrl_train/distributed/fsdp_utils.py",
               "skyrl_train/distributed/fsdp_utils.py")
_VLLM_ENGINE = (SKYRL_LOCAL / "skyrl_train/inference_engines/vllm/vllm_engine.py",
                "skyrl_train/inference_engines/vllm/vllm_engine.py")
# Client + ray-wrapper carry the begin/finish_weight_reload fan-out for the #1685 bracket.
_ENGINE_CLIENT = (SKYRL_LOCAL / "skyrl_train/inference_engines/inference_engine_client.py",
                  "skyrl_train/inference_engines/inference_engine_client.py")
_RAY_WRAP = (SKYRL_LOCAL / "skyrl_train/inference_engines/ray_wrapped_inference_engine.py",
             "skyrl_train/inference_engines/ray_wrapped_inference_engine.py")

# Diag registry: name -> {module, inject files, default geometry}. Select with --diag.
DIAGS = {
    # gather-only (CLEAN, run #1): EP=8 cross-node policy gather vs disk.
    "gather": {
        "module": "tests.gpu.diag_ep8_weight_disk_ref",
        "inject": [_FSDP_WORKER, _FSDP_UTILS,
                   (SKYRL_LOCAL / "tests/gpu/diag_ep8_weight_disk_ref.py",
                    "tests/gpu/diag_ep8_weight_disk_ref.py")],
        "num_nodes": 4, "gpus_per_node": 4,
    },
    # DISAGGREGATED engine-vs-disk (D1/D2): policy 2 nodes x 8 GPU + engine on a 3rd node.
    "disagg": {
        "module": "tests.gpu.diag_ep8_disagg_engine_disk",
        "inject": [_FSDP_WORKER, _FSDP_UTILS, _VLLM_ENGINE, _ENGINE_CLIENT, _RAY_WRAP,
                   (SKYRL_LOCAL / "tests/gpu/diag_ep8_disagg_engine_disk.py",
                    "tests/gpu/diag_ep8_disagg_engine_disk.py")],
        "num_nodes": 3, "gpus_per_node": 8,
    },
}


def stage_inject_files(inject_files) -> str:
    """Copy the modified ground-truth files into a staging dir UNDER PROJECT_ROOT so
    they ride along in the iris workspace upload (-> /app/<rel>). base64-inlining them
    into the single `bash -c` arg exceeds Linux MAX_ARG_STRLEN (128KB/arg) ->
    `exec /usr/bin/bash: argument list too long`. The workspace upload has no such limit.
    Returns the staging dir's POD path (under /app)."""
    import shutil

    stage_rel = "scripts/iris/_ep8_inject"
    stage_abs = Path(PROJECT_ROOT) / stage_rel
    if stage_abs.exists():
        shutil.rmtree(stage_abs)
    for local, rel in inject_files:
        dst = stage_abs / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dst)
    return f"{APP_DIR}/{stage_rel}"


def build_command(args) -> list[str]:
    skyrl_train_dir = f"{SKYRL_HOME}/skyrl-train"
    pythonpath = f"{APP_DIR}:{skyrl_train_dir}"

    # Files are staged into the workspace upload (see stage_inject_files); cp them into
    # the pod's /opt/skyrl checkout (a short command, no 128KB base64 arg).
    pod_stage = f"{APP_DIR}/scripts/iris/_ep8_inject"
    inject = ""
    for _local, rel in args.diag_cfg["inject"]:
        src = f"{pod_stage}/{rel}"
        dest = f"{skyrl_train_dir}/{rel}"
        inject += (
            f"mkdir -p $(dirname {shlex.quote(dest)}); "
            f"cp {shlex.quote(src)} {shlex.quote(dest)}; "
            f"echo '[ep8-diag] injected -> {dest}'; "
        )
    # Purge baked bytecode so the injected .py wins (hash-based pyc invalidation gotcha).
    inject += (
        f"find {shlex.quote(skyrl_train_dir)} -name '*.pyc' -delete 2>/dev/null || true; "
        f"find {shlex.quote(skyrl_train_dir)} -name __pycache__ -type d -prune "
        f"-exec rm -rf {{}} + 2>/dev/null || true; "
    )

    # Driver (runs only on rank 0 via the controller): the EP=8 diag. The controller
    # launches this with cwd=/app, where a sibling `tests/` dir SHADOWS skyrl-train's
    # `tests.gpu` package on `-m` resolution (ModuleNotFoundError: No module named
    # 'tests.gpu'). Wrap in bash that cd's into skyrl-train (and prepends it to
    # PYTHONPATH) so `-m tests.gpu...` resolves to the injected diag.
    driver_cmd = [
        "bash", "-lc",
        f"cd {shlex.quote(skyrl_train_dir)} && "
        f"export PYTHONPATH={shlex.quote(skyrl_train_dir)}:${{PYTHONPATH:-}} && "
        f"exec {RL_PYTHON} -m {args.diag_cfg['module']}",
    ]

    controller_cmd = [
        RL_PYTHON, "scripts/iris/start_rl_iris_controller.py",
        "--ray-port", str(args.ray_port),
    ]
    if args.rendezvous_dir:
        controller_cmd.extend(["--rendezvous-dir", args.rendezvous_dir])
    controller_cmd.append("--")
    controller_cmd.extend(driver_cmd)

    bash = (
        f"set -e; cd {APP_DIR}; "
        f"{inject}"
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export VLLM_USE_V1=1; "
        # diag geometry forwarded to the driver (read on rank 0). Superset for both diags.
        f"export DIAG_MODEL={shlex.quote(MODEL)}; "
        f"export DIAG_EP={args.ep}; export DIAG_FSDP={args.fsdp}; "
        f"export DIAG_NUM_GPUS={args.ep * args.fsdp}; "
        f"export DIAG_GPUS_PER_NODE={args.gpus_per_node}; "
        # disagg diag: policy fills whole nodes (8 GPU); engine TP/EP = prod salad geom.
        f"export DIAG_POLICY_GPUS_PER_NODE={args.gpus_per_node}; "
        f"export DIAG_ENGINE_TP={args.engine_tp}; export DIAG_ENGINE_EP={args.engine_ep}; "
        f"export DIAG_LAYER=0; "
        f"export DIAG_W13_STRICT={args.w13_strict}; export DIAG_LAYERS={shlex.quote(args.layers)}; "
        f"exec {shlex.join(controller_cmd)}"
    )
    return ["bash", "-c", bash]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diag", choices=sorted(DIAGS), default="disagg",
                    help="Which diagnostic to run (gather = run#1 policy-gather; "
                         "disagg = D1/D2 disaggregated engine-vs-disk).")
    ap.add_argument("--num-nodes", dest="num_nodes", type=int, default=None)
    ap.add_argument("--gpus-per-node", dest="gpus_per_node", type=int, default=None)
    ap.add_argument("--ep", type=int, default=DEFAULT_EP)
    ap.add_argument("--fsdp", type=int, default=DEFAULT_FSDP)
    ap.add_argument("--engine-tp", dest="engine_tp", type=int, default=2)
    ap.add_argument("--engine-ep", dest="engine_ep", type=int, default=2)
    ap.add_argument("--w13-strict", dest="w13_strict", type=int, default=1,
                    help="disagg: 1 = FIXED-order w13 compare (gate/up-swap probe); 0 = both-order tolerant.")
    ap.add_argument("--layers", default="0,24",
                    help="disagg: comma layers to sample (default '0,24' = layer0 + mid).")
    ap.add_argument("--ray-port", dest="ray_port", type=int, default=6379)
    ap.add_argument("--rendezvous-dir", dest="rendezvous_dir", default=None)
    ap.add_argument("--job-name", dest="job_name", default=None)
    ap.add_argument("--priority", default="interactive")
    ap.add_argument("--max-retries", dest="max_retries", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    args.diag_cfg = DIAGS[args.diag]
    if args.num_nodes is None:
        args.num_nodes = args.diag_cfg["num_nodes"]
    if args.gpus_per_node is None:
        args.gpus_per_node = args.diag_cfg["gpus_per_node"]

    total_gpus = args.num_nodes * args.gpus_per_node
    policy_gpus = args.ep * args.fsdp
    if args.diag == "gather":
        assert policy_gpus == total_gpus, (
            f"gather: EP*FSDP={policy_gpus} must == total GPUs {total_gpus}")
    else:
        # disagg: policy uses ep*fsdp GPUs (whole nodes); the remaining node(s) host the
        # engine. Require room: total >= policy_gpus + engine_tp*engine_ep, and policy
        # fills whole nodes (ep*fsdp divisible by gpus_per_node).
        eng_gpus = args.engine_tp * args.engine_ep
        assert policy_gpus % args.gpus_per_node == 0, (
            f"disagg: policy EP*FSDP={policy_gpus} must fill whole {args.gpus_per_node}-GPU nodes")
        assert total_gpus >= policy_gpus + eng_gpus, (
            f"disagg: total {total_gpus} < policy {policy_gpus} + engine {eng_gpus}")
    if args.num_nodes > 6:
        raise SystemExit(f"--num-nodes {args.num_nodes} exceeds the 6-node cap")

    if args.job_name is None:
        args.job_name = f"ep8-{args.diag}-diag-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.rendezvous_dir is None and args.num_nodes > 1:
        args.rendezvous_dir = f"s3://marin-na/iris/{args.job_name}/rdv"

    cluster_config = _resolve_cluster_config_default()
    pod_stage = stage_inject_files(args.diag_cfg["inject"])
    print(f"[ep8-diag] Staged inject files for workspace upload -> {pod_stage}", flush=True)
    command = build_command(args)
    gpu_spec = f"{DEFAULT_GPU_VARIANT}x{args.gpus_per_node}"

    print(f"[ep8-diag] Diag:       {args.diag} ({args.diag_cfg['module']})", flush=True)
    print(f"[ep8-diag] Job:        {args.job_name}", flush=True)
    print(f"[ep8-diag] Cluster:    {DEFAULT_CLUSTER}  ({cluster_config})", flush=True)
    print(f"[ep8-diag] Image:      {DEFAULT_RL_DOCKER_IMAGE}", flush=True)
    print(f"[ep8-diag] Topology:   {args.num_nodes} node(s) x {gpu_spec} = {total_gpus} GPU "
          f"(policy EP={args.ep}xFSDP={args.fsdp}={policy_gpus} GPU; "
          f"engine TP={args.engine_tp}xEP={args.engine_ep}), gang/leafgroup", flush=True)
    print(f"[ep8-diag] Model:      {MODEL}", flush=True)
    print(f"[ep8-diag] Rendezvous: {args.rendezvous_dir}", flush=True)
    print(f"[ep8-diag] Injecting:  {[r for _, r in args.diag_cfg['inject']]}", flush=True)
    print(f"[ep8-diag] Command:    {shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[ep8-diag] --dry-run: not submitting", flush=True)
        return 0

    secrets_env = _default_secrets_env()
    if secrets_env:
        IrisLauncher.load_secrets_env_into_os_environ(secrets_env)

    from iris.client import IrisClient
    from iris.cluster.config import IrisConfig
    from iris.cluster.types import EnvironmentSpec, Entrypoint
    from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults
    from iris.rpc import job_pb2

    resources = build_resources(None, gpu_spec, cpu=CPU_PER_NODE, memory=MEMORY_PER_NODE, disk=DISK_PER_NODE)
    replicas, coscheduling = resolve_multinode_defaults(None, DEFAULT_GPU_VARIANT, args.num_nodes)
    constraints = build_job_constraints(
        resources_proto=resources.to_proto(), tpu_variants=[], replicas=replicas,
        regions=None, zone=None, preemptible=False,
    )
    priority_band = {
        "production": job_pb2.PRIORITY_BAND_PRODUCTION,
        "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
        "batch": job_pb2.PRIORITY_BAND_BATCH,
    }.get(args.priority, job_pb2.PRIORITY_BAND_INTERACTIVE)

    env_vars: dict[str, str] = {}
    if args.rendezvous_dir:
        env_vars["OT_AGENT_IRIS_RENDEZVOUS_DIR"] = args.rendezvous_dir
    env_vars["OT_AGENT_IRIS_RAY_PORT"] = str(args.ray_port)
    for k in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v

    iris_config = IrisConfig.load(cluster_config)
    bundle = iris_config.provider_bundle()
    controller_proto = iris_config.proto.controller
    controller_address = (
        iris_config.controller_address() or bundle.controller.discover_controller(controller_proto)
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
            task_image=DEFAULT_RL_DOCKER_IMAGE,
            priority_band=priority_band,
            timeout=None,
        )
        full_job_id = str(job.job_id)
        print(f"[ep8-diag] Submitted: {full_job_id}  (replicas={replicas}, "
              f"coscheduling={getattr(coscheduling, 'group_by', None)})", flush=True)
        if args.no_wait:
            return 0
        try:
            status = job.wait(stream_logs=True, timeout=float("inf"))
            exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
        except KeyboardInterrupt:
            print(f"[ep8-diag] Terminating {full_job_id}...", file=sys.stderr, flush=True)
            client.terminate_job(job.job_id)
            exit_code = 130
        print(f"[ep8-diag] Job exit: {exit_code}", flush=True)
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
