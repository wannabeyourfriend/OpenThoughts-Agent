#!/usr/bin/env python
"""Submit the 1-node GPU verification smoke for the EP MoE weight-sync fix.

Runs ``tests/gpu/test_e2e_moe_rl_step.py`` (Qwen1.5-MoE-A2.7B-Chat, EP=2xFSDP=2
trainer colocated with an EP=2 vLLM engine, 4 GPUs / 1 NODE) on the prod gpu-rl
image (torch 2.11), with the MarinSkyRL clone at /opt/skyrl checked out to a
chosen ref (default: ac44079, the order-correct grouped-expert gather fix).

The test's DIRECT weight-equality gate (test_e2e_moe_rl_step.py:401-457) reads the
trainer's post-step HF expert weights via the SAME _gather_tensor path the fix
touches and compares byte-exact to the vLLM engine readback, for experts spanning
BOTH EP shards. PASS = ~fp32 epsilon; the bug (scrambled experts on 2.11) shows as
a large per-expert max-abs.

This reuses launch_rl_iris.py's IrisClient.submit machinery (image, secrets, gang)
but with a CUSTOM in-pod command (pytest + the skyrl-ref checkout + an explicit
in-pod torch.__version__ echo — the test only exercises the real
_StridedShard.is_shard()==False bug on torch 2.11).

1 node only. No >2-node job. Self-terminating (the test exits, the job ends).

Usage:
    python scripts/iris/submit_ep_sync_gpu_smoke.py --skyrl-ref ac44079 [--dry-run]
    python scripts/iris/submit_ep_sync_gpu_smoke.py --baked   # fail-before: NO checkout (baked 78d83a5)
"""
from __future__ import annotations

import argparse
import base64
import os
import shlex
import sys
from pathlib import Path

# Reuse the RL-iris launcher's resolved constants + helpers (single source of truth
# for the image digest, venv python, cluster config, secrets passthrough).
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

# The smoke runs on ONE whole node; the test uses 4 of the 8 GPUs.
GPUS_PER_NODE = 8
# Small footprint — this is a 4-GPU integration test, not the 131k EP8 weight-load.
CPU_PER_NODE = 48.0
MEMORY_PER_NODE = "1000GB"
DISK_PER_NODE = "512GB"
PRIORITY = "interactive"
# Default vehicle: the grouped+EP weight-sync coherence test. It loads
# Qwen1.5-MoE-A2.7B-Chat into a grouped-GEMM + EP=2 x FSDP=2 trainer (the prod
# `(_StridedShard, Shard)` composite), syncs into an EP=2 vLLM engine via the
# fixed `_gather_tensor` -> `broadcast_to_inference_engines`, and asserts post-sync
# responses match pre-sync (the synced-policy coherence gate). It does NOT run a
# training forward, so it sidesteps the HF-eager `qwen2_moe` forward crash that
# blocks test_e2e_moe_rl_step on this image — while exercising the SAME gather.
TEST_MODULE = "tests.gpu.test_expert_parallel_inference"
TEST_FUNC = "test_ep_weight_sync_grouped"
# Active vehicle: the sync-and-compare DIAGNOSTIC (no generation; value-compares the
# synced vLLM engine weights vs the FSDP source across ALL weight types -> corruption
# signature). 8 GPUs (8 ranks) on 1 node per the coordinator's scale guidance.
DIAG_MODULE = "tests.gpu.diag_ep_weight_sync_compare"
DIAG_NUM_GPUS = 8
# The diagnostic is a NEW untracked file in the LOCAL MarinSkyRL clone — it is NOT in
# the pod's git checkout (and we must NOT push). So we read it locally and inject it
# into the pod (base64, robust against shell quoting) at the path `-m DIAG_MODULE`
# expects, before running. It only imports test_expert_parallel_inference / utils,
# which ARE in the baked checkout.
LOCAL_DIAG_FILE = Path(
    "/Users/benjaminfeuer/Documents/MarinSkyRL/skyrl-train/tests/gpu/diag_ep_weight_sync_compare.py"
)
POD_DIAG_REL = "tests/gpu/diag_ep_weight_sync_compare.py"


def build_command(skyrl_ref: str | None) -> list[str]:
    """In-pod bash: optional skyrl-ref checkout (+ bytecode purge), assert torch
    2.11, run the pytest from the skyrl-train dir. Mirrors launch_rl_iris's
    skyrl_refresh block verbatim so the checkout semantics are identical."""
    skyrl_refresh = ""
    if skyrl_ref:
        ref = shlex.quote(skyrl_ref)
        skyrl_refresh = (
            f"git -C {shlex.quote(SKYRL_HOME)} fetch --quiet --all || true; "
            f"git -C {shlex.quote(SKYRL_HOME)} checkout {ref}; "
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name '*.pyc' -delete 2>/dev/null || true; "
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name __pycache__ -type d -prune "
            f"-exec rm -rf {{}} + 2>/dev/null || true; "
            f"echo \"[ep-smoke] MarinSkyRL now at $(git -C {shlex.quote(SKYRL_HOME)} rev-parse HEAD)\"; "
        )
    else:
        skyrl_refresh = (
            f"echo \"[ep-smoke] BAKED (no checkout) MarinSkyRL at "
            f"$(git -C {shlex.quote(SKYRL_HOME)} rev-parse HEAD)\"; "
        )

    pythonpath = f"{APP_DIR}:{SKYRL_HOME}/skyrl-train"
    torch_check = (
        f"{RL_PYTHON} -c "
        + shlex.quote(
            "import torch,sys;"
            "v=torch.__version__;"
            "print('[ep-smoke] in-pod torch.__version__ =',v, flush=True);"
            "from torch.distributed.tensor.placement_types import _StridedShard,Shard;"
            "s=_StridedShard(dim=0,split_factor=2);"
            "print('[ep-smoke] _StridedShard is Shard subclass =',issubclass(_StridedShard,Shard),"
            "'| is_shard() =',s.is_shard(), flush=True);"
            "sys.exit(0 if v.startswith('2.11') else 7)"
        )
        + "; "
    )
    skyrl_train_dir = f"{SKYRL_HOME}/skyrl-train"
    # Build the diag-injection snippet: base64-decode the local diag file into the
    # pod at the module path. (The diag is untracked locally; we don't push.)
    diag_b64 = base64.b64encode(LOCAL_DIAG_FILE.read_bytes()).decode("ascii")
    pod_diag_path = f"{skyrl_train_dir}/{POD_DIAG_REL}"
    diag_inject = (
        f"echo {shlex.quote(diag_b64)} | base64 -d > {shlex.quote(pod_diag_path)}; "
        f"echo '[ep-smoke] injected diagnostic -> {pod_diag_path}'; "
    )
    bash = (
        f"set -e; cd {APP_DIR}; "
        f"{skyrl_refresh}"
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export VLLM_USE_V1=1; "
        # Hard-gate: STOP if the pod is not torch 2.11 (the bug is 2.11-only; a 2.9
        # pod would pass even buggy and prove nothing).
        f"{torch_check}"
        f"cd {shlex.quote(skyrl_train_dir)}; "
        # The test module does `import pytest` at top (only uses pytest.skip, which
        # won't fire — we have 8 GPUs). The rl venv has NEITHER pytest NOR pip. Drop a
        # minimal stub `pytest` module on a PYTHONPATH dir so the import resolves; we
        # call the test function DIRECTLY (not via the pytest runner).
        f"mkdir -p /tmp/pystub; "
        f"printf '%s\\n' "
        + shlex.quote(
            "class _Skip(Exception):\n"
            "    pass\n"
            "def skip(msg='', *a, **k):\n"
            "    raise _Skip(msg)\n"
            "class _Mark:\n"
            "    def __getattr__(self, n):\n"
            "        def deco(*a, **k):\n"
            "            return (a[0] if a and callable(a[0]) else (lambda f: f))\n"
            "        return deco\n"
            "mark = _Mark()\n"
            "def fixture(*a, **k):\n"
            "    return (a[0] if a and callable(a[0]) else (lambda f: f))\n"
            "class raises:\n"
            "    def __init__(self, *a, **k):\n"
            "        pass\n"
            "    def __enter__(self):\n"
            "        return self\n"
            "    def __exit__(self, *a):\n"
            "        return True\n"
        )
        + " > /tmp/pystub/pytest.py; "
        f"export PYTHONPATH=/tmp/pystub:$PYTHONPATH; "
        # Inject the local (untracked) diagnostic into the pod's checkout (base64).
        f"{diag_inject}"
        # SYNC-AND-COMPARE diagnostic: NO generation. Bring up FSDP grouped+EP policy
        # + EP=2 vLLM engine, ONE weight-sync, then value-compare engine readback vs
        # FSDP source across ALL weight types -> emit the corruption signature.
        f"export DIAG_NUM_GPUS={DIAG_NUM_GPUS}; "
        f"echo '[ep-smoke] === running {DIAG_MODULE} (DIAG_NUM_GPUS={DIAG_NUM_GPUS}) ==='; "
        f"exec {RL_PYTHON} -m {DIAG_MODULE}"
    )
    return ["bash", "-c", bash]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skyrl-ref", dest="skyrl_ref", default="ac44079",
                    help="MarinSkyRL ref to checkout at /opt/skyrl (default ac44079 = the fix).")
    ap.add_argument("--nccl-disables", dest="nccl_disables", action="store_true",
                    help="NCCL A/B test arm: inject NCCL_P2P_DISABLE=1 + NCCL_NVLS_ENABLE=0 + "
                         "NCCL_COLLNET_ENABLE=0 (Jupiter's GH200 disables) into the pod env.")
    ap.add_argument("--baked", action="store_true",
                    help="Fail-before mode: do NOT checkout — use the image's baked 78d83a5.")
    ap.add_argument("--job-name", dest="job_name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    skyrl_ref = None if args.baked else args.skyrl_ref
    tag = "baked78d83a5" if args.baked else (args.skyrl_ref or "head")
    job_name = args.job_name or f"ep-sync-smoke-{tag}"
    cluster_config = _resolve_cluster_config_default()
    command = build_command(skyrl_ref)
    gpu_spec = f"{DEFAULT_GPU_VARIANT}x{GPUS_PER_NODE}"

    print(f"[ep-smoke] Job:       {job_name}", flush=True)
    print(f"[ep-smoke] Cluster:   {DEFAULT_CLUSTER} ({cluster_config})", flush=True)
    print(f"[ep-smoke] Image:     {DEFAULT_RL_DOCKER_IMAGE}", flush=True)
    print(f"[ep-smoke] Topology:  1 node x {gpu_spec} (test uses 4/8 GPUs)", flush=True)
    print(f"[ep-smoke] skyrl-ref: {skyrl_ref if skyrl_ref else 'BAKED (no checkout, 78d83a5)'}", flush=True)
    print(f"[ep-smoke] Command:   {shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[ep-smoke] --dry-run: not submitting", flush=True)
        return 0

    # Load secrets (HF_TOKEN etc.) into the launch-host env for passthrough.
    secrets_env = _default_secrets_env()
    if secrets_env:
        IrisLauncher.load_secrets_env_into_os_environ(secrets_env)

    from iris.client import IrisClient
    from iris.cluster.config import IrisConfig
    from iris.cluster.types import EnvironmentSpec, Entrypoint
    from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults
    from iris.rpc import job_pb2

    resources = build_resources(None, gpu_spec, cpu=CPU_PER_NODE, memory=MEMORY_PER_NODE, disk=DISK_PER_NODE)
    replicas, coscheduling = resolve_multinode_defaults(None, DEFAULT_GPU_VARIANT, 1)
    constraints = build_job_constraints(
        resources_proto=resources.to_proto(), tpu_variants=[], replicas=replicas,
        regions=None, zone=None, preemptible=False,
    )
    priority_band = job_pb2.PRIORITY_BAND_INTERACTIVE

    env_vars: dict[str, str] = {}
    for k in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v
    # NCCL A/B: re-add Jupiter's GH200 disables (dropped for H100). Injected into the
    # task pod env so BOTH the policy worker AND the vLLM engine Ray actors inherit
    # them before NCCL init. Tests whether NVLS/P2P on H100 corrupts the EP-expert
    # weight broadcast.
    if args.nccl_disables:
        env_vars["NCCL_P2P_DISABLE"] = "1"
        env_vars["NCCL_NVLS_ENABLE"] = "0"
        env_vars["NCCL_COLLNET_ENABLE"] = "0"
        print("[ep-smoke] NCCL A/B = TEST: NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_COLLNET_ENABLE=0", flush=True)
    else:
        print("[ep-smoke] NCCL A/B = CONTROL: NCCL defaults (P2P/NVLS ON)", flush=True)

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
            name=job_name,
            resources=resources,
            environment=EnvironmentSpec(env_vars=env_vars, extras=[]),
            constraints=constraints or None,
            coscheduling=coscheduling,
            replicas=replicas,
            max_retries_failure=0,
            task_image=DEFAULT_RL_DOCKER_IMAGE,
            priority_band=priority_band,
            timeout=None,
        )
        full_job_id = str(job.job_id)
        print(f"[ep-smoke] Submitted: {full_job_id}", flush=True)
        if args.no_wait:
            return 0
        try:
            status = job.wait(stream_logs=True, timeout=float("inf"))
            exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
        except KeyboardInterrupt:
            print(f"[ep-smoke] Terminating {full_job_id}...", file=sys.stderr, flush=True)
            client.terminate_job(job.job_id)
            exit_code = 130
        print(f"[ep-smoke] Job exit: {exit_code}  (state={status.state if 'status' in dir() else '?'})", flush=True)
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
