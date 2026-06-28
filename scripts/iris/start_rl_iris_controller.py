#!/usr/bin/env python3
"""Bootstrap a multi-node MarinSkyRL RL job on an iris GPU slice.

This is the RL analog of ``scripts/vllm/start_vllm_iris_controller.py`` (which
serves vLLM across a multi-host iris TPU slice). The crucial shared fact: iris
gang-schedules a multi-node job as N coscheduled tasks — one task per whole
node — and runs THIS SAME entrypoint on every node, injecting ``IRIS_TASK_ID``
/ ``IRIS_NUM_TASKS`` per task. SkyRL/MarinSkyRL (skyrl-train) is Ray-native: it
wants ONE cross-node Ray cluster and a single training driver that fans
policy/ref/inference actors across the Ray nodes.

So this script bootstraps that one cluster, then runs the driver on rank 0:

- **rank 0 (head):** ``ray start --head``; publish the head IP to a shared
  rendezvous file; wait until ``ray.nodes()`` shows all ``IRIS_NUM_TASKS``
  nodes joined; then ``exec`` the MarinSkyRL training command (``python -m
  <entrypoint> <hydra args>``) with ``RAY_ADDRESS`` pointing at the head, so
  skyrl-train's ``initialize_ray`` (which calls bare ``ray.init()``) attaches
  to the existing multi-node cluster instead of starting a fresh local one.
- **ranks 1..N-1 (workers):** read the head IP from the rendezvous, run
  ``ray start --address=<head_ip>:<port>``, verify they joined, then BLOCK
  until the head finishes (signalled via the rendezvous ``done`` marker) or
  SIGTERM. They contribute their 8 H100s to the Ray cluster; the driver on
  rank 0 schedules engine/policy workers onto them. They do NOT run the
  training driver.

Head-IP discovery
-----------------
iris injects ``IRIS_ADVERTISE_HOST`` (the task's routable IP under
``host_network: true`` — required for NCCL/IB on the CoreWeave slice) into
every task, so rank 0 uses it directly as the Ray head IP. iris does NOT inject
a peer/host list into the task env, so rank 0 publishes the head IP to a small
rendezvous file on a shared object store the launcher passes in
(``--rendezvous-dir`` / ``OT_AGENT_IRIS_RENDEZVOUS_DIR``). Ranks 1..N poll for
it. The rendezvous URI may be ``gs://``, ``s3://`` (CoreWeave R2), or a shared
local/NFS path — it is opened via ``fsspec`` so the storage backend is whatever
the URI scheme resolves to (CoreWeave uses R2/S3, not GCS).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

RENDEZVOUS_FILENAME = "ray_head.json"
DONE_FILENAME = "ray_head.done"


def _ray_bin() -> str:
    """Resolve the ``ray`` CLI from the SAME venv as the running interpreter.

    The launcher runs this controller via the RL venv python by absolute path
    (e.g. /opt/openthoughts/envs/rl/bin/python), but iris's uv-sync setup
    activates a DIFFERENT venv (/app/.venv) and leaves only that on $PATH — which
    has no `ray`. So a bare `ray` command resolves to nothing (FileNotFoundError).
    Use the `ray` binary that sits next to this interpreter; fall back to PATH.
    """
    import shutil

    candidate = os.path.join(os.path.dirname(sys.executable), "ray")
    if os.path.exists(candidate):
        return candidate
    found = shutil.which("ray")
    return found or "ray"

# Cold GPU nodes (image pull + setup) can take several minutes to reach the
# rendezvous, so these are generous.
DEFAULT_RENDEZVOUS_TIMEOUT = 1800
DEFAULT_CLUSTER_JOIN_TIMEOUT = 1800
POLL_INTERVAL = 5
# Tolerates clock skew between nodes and the time rank-0 needs to start Ray.
RENDEZVOUS_FRESHNESS_SLACK = 60


def _log(msg: str) -> None:
    print(f"[start_rl_iris_controller] {msg}", flush=True)


def stage_train_data(train_data_json: str) -> None:
    """Extract the HF task dataset(s) to this NODE's local task dir on EVERY node.

    WHY THIS EXISTS (the data-starvation bug, 2026-06-26): the agentic
    terminal_bench task dataset (e.g. ``DCAgent/exp_rpt_pymethods2test-large``) is
    an HF *parquet* repo that must be extracted into the on-disk
    ``$DCFT/tasks/<repo>/<instance>/{instruction.md,task.toml,...}`` layout the
    Harbor rollout reads. ``hpc.rl_launch_utils.resolve_rl_train_data`` does that
    extraction, but on the iris/CoreWeave path it runs only inside rank-0's
    ``run_rl.py`` driver, writing to ``$DCFT=/opt/openthoughts/tasks`` on the HEAD
    pod's NODE-LOCAL filesystem. CoreWeave task pods do NOT share a filesystem (no
    SLURM-style GPFS ``$SCRATCH``), so the Ray-scheduled rollout workers on ranks
    1..N-1 saw an empty tasks dir and every rollout died with
    ``FileNotFoundError: .../task.toml`` -> reward always 0 (doomed, data-starved).

    Fix: the controller runs on EVERY node, so stage here (before Ray bootstrap)
    using the SAME extraction routine. CoreWeave nodes have egress, so each pod
    fetches+extracts to the identical node-local path; the path strings rank-0's
    dataset object ships to the workers then resolve on every pod. Idempotent
    (``on_exist=skip`` + the stat short-circuit in ``_fix_task_permissions``), so a
    re-run / rank-0's later run_rl re-resolve is a cheap no-op.
    """
    import json as _json

    try:
        train_data = _json.loads(train_data_json)
    except (ValueError, TypeError):
        train_data = [train_data_json] if train_data_json else []
    if not train_data:
        return

    # Reuse the exact, SLURM-proven staging logic. PYTHONPATH already includes
    # /app (set by the launcher bootstrap), so hpc is importable.
    from hpc.rl_launch_utils import resolve_rl_train_data

    _log(f"Staging train_data on this node (rank {_rank()}/{_num_tasks()}): {train_data}")
    resolved = resolve_rl_train_data(train_data, on_exist="skip", verbose=True)
    _log(f"train_data staged to node-local paths: {resolved}")


def _rank() -> int:
    # IRIS_TASK_ID is the full task path (e.g. "/user/job/0"); on retried tasks
    # iris appends a ":N" retry suffix. The rank is the trailing path segment
    # with any retry suffix stripped.
    return int(os.environ.get("IRIS_TASK_ID", "0").rsplit("/", 1)[-1].split(":", 1)[0])


def _num_tasks() -> int:
    return int(os.environ.get("IRIS_NUM_TASKS", "1"))


def _own_ip() -> str:
    """Routable IP of this node.

    Prefers iris's ``IRIS_ADVERTISE_HOST`` (the routable IP iris computed for
    this task under ``host_network: true``); falls back to a UDP-socket probe.
    """
    advertised = os.environ.get("IRIS_ADVERTISE_HOST")
    if advertised and advertised != "127.0.0.1":
        return advertised
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Rendezvous — head publishes its IP, workers poll for it. Backend-agnostic via
# fsspec so the URI scheme (gs://, s3://, file://, plain path) selects storage.
# ---------------------------------------------------------------------------


def _fs_and_path(uri: str):
    """Return (fsspec filesystem, path) for ``uri``. Uses default credential
    discovery for the scheme (workload identity / instance creds / env keys)."""
    import fsspec

    fs, _, paths = fsspec.get_fs_token_paths(uri)
    return fs, paths[0]


def _rendezvous_uri(rendezvous_dir: str) -> str:
    return f"{rendezvous_dir.rstrip('/')}/{RENDEZVOUS_FILENAME}"


def _done_uri(rendezvous_dir: str) -> str:
    return f"{rendezvous_dir.rstrip('/')}/{DONE_FILENAME}"


def write_rendezvous(rendezvous_dir: str, head_ip: str, ray_port: int) -> None:
    uri = _rendezvous_uri(rendezvous_dir)
    payload = {
        "head_ip": head_ip,
        "port": ray_port,
        "num_tasks": _num_tasks(),
        "written_at": time.time(),
    }
    fs, path = _fs_and_path(uri)
    with fs.open(path, "w") as f:
        json.dump(payload, f)
    _log(f"Wrote rendezvous {uri}: head_ip={head_ip} port={ray_port}")


def poll_rendezvous(rendezvous_dir: str, timeout: int, min_written_at: float | None = None) -> dict:
    """Poll for the head's rendezvous file. Returns its parsed payload.

    Payloads with ``written_at`` older than ``min_written_at`` (minus slack) are
    treated as stale (from a prior iris task attempt) and ignored.
    """
    uri = _rendezvous_uri(rendezvous_dir)
    fs, path = _fs_and_path(uri)
    deadline = time.time() + timeout
    threshold = (min_written_at - RENDEZVOUS_FRESHNESS_SLACK) if min_written_at else None
    _log(f"Polling for rendezvous {uri} (timeout {timeout}s)...")
    while time.time() < deadline:
        try:
            if fs.exists(path):
                with fs.open(path, "r") as f:
                    payload = json.load(f)
                if payload.get("head_ip"):
                    written_at = payload.get("written_at", 0)
                    if threshold is not None and written_at < threshold:
                        _log(
                            f"Ignoring stale rendezvous (written_at={written_at:.0f} "
                            f"< threshold={threshold:.0f}); waiting for rank-0 rewrite."
                        )
                    else:
                        _log(f"Found rendezvous: {payload}")
                        return payload
        except Exception as exc:  # transient object-store hiccup
            _log(f"rendezvous poll error (will retry): {exc}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Worker rank {_rank()} timed out after {timeout}s waiting for "
        f"rank-0 rendezvous at {uri}. Did the head task fail to start?"
    )


def _set_marker(rendezvous_dir: str, name: str) -> None:
    uri = f"{rendezvous_dir.rstrip('/')}/{name}"
    try:
        fs, path = _fs_and_path(uri)
        with fs.open(path, "w") as f:
            f.write(str(time.time()))
    except Exception as exc:
        _log(f"Warning: could not write marker {uri}: {exc}")


def _marker_exists(rendezvous_dir: str, name: str, min_written_at: float | None = None) -> bool:
    uri = f"{rendezvous_dir.rstrip('/')}/{name}"
    try:
        fs, path = _fs_and_path(uri)
        if not fs.exists(path):
            return False
        if min_written_at is None:
            return True
        with fs.open(path, "r") as f:
            written_at = float(f.read().strip() or 0)
        return written_at >= (min_written_at - RENDEZVOUS_FRESHNESS_SLACK)
    except Exception:
        return False


def clear_rendezvous(rendezvous_dir: str) -> None:
    """Best-effort delete of the rendezvous + done markers (rank 0, on entry/exit)."""
    for name in (RENDEZVOUS_FILENAME, DONE_FILENAME):
        uri = f"{rendezvous_dir.rstrip('/')}/{name}"
        try:
            fs, path = _fs_and_path(uri)
            if fs.exists(path):
                fs.rm(path)
                _log(f"Removed {uri}")
        except Exception as exc:
            _log(f"Warning: could not remove {uri}: {exc}")


# ---------------------------------------------------------------------------
# Ray cluster bootstrap (mirrors start_vllm_iris_controller).
# ---------------------------------------------------------------------------


# Ray port allocation on cw-us-east-02a — PIN every named system port OUTSIDE the
# worker_ports range so Ray's own randomized agent ports can never collide with it.
#
# THE BUG: Ray assigns several system components (metrics_export, runtime_env_agent,
# dashboard_agent_grpc, dashboard_agent_listen, node/object_manager) by picking a
# RANDOM free port. By default it draws them from the SAME ephemeral zone as the
# default worker_ports range (10002–19999), and Ray's own pre-start validation then
# aborts the node when a random agent port lands inside worker_ports:
#   ValueError: Ray component worker_ports is trying to use a port number <N>
#   that is used by other components.
# Observed on THREE different components across launches:
#   - head:   metrics_export=19865        (grpmm-fix)
#   - worker: runtime_env_agent=15731     (grpmm-fix3, head-only metrics pin in place)
#   - worker: dashboard_agent_grpc=24543 + runtime_env_agent=28330  (grpmm-fix4)
# grpmm-fix4 proved that merely SHIFTING worker_ports (to 20000–29999) does NOT help:
# the random agent ports simply followed into the new range. The collision can be ANY
# of Ray's randomized agent ports, on EITHER head or worker.
#
# THE FIX (deterministic, complete): keep worker_ports at Ray's DEFAULT 10002–19999
# and PIN every agent port Ray would otherwise randomize to a fixed value in the low
# 8xxx band — OUTSIDE 10002–19999 and distinct from gcs(6379)/dashboard(8265)/
# client_server(10001). With no random port left to draw from the worker range, the
# validation can never trip. Applied on BOTH head and worker (run_worker is where the
# fix3/fix4 collisions hit). 8090 matches the repo precedent RAY_metrics_export_port=
# 8090 in scripts/torch/kimi-k2-tracegen-run-v2.sh; the rest are adjacent free 8xxx.
RAY_METRICS_EXPORT_PORT = 8090
RAY_RUNTIME_ENV_AGENT_PORT = 8092
RAY_DASHBOARD_AGENT_GRPC_PORT = 8093
RAY_DASHBOARD_AGENT_LISTEN_PORT = 8094
RAY_NODE_MANAGER_PORT = 8076
RAY_OBJECT_MANAGER_PORT = 8077


def _ray_port_flags() -> list[str]:
    """Ray port flags shared by head + worker: pin EVERY named system port that Ray
    would otherwise randomize to a fixed value OUTSIDE the default worker_ports range
    (10002–19999), so no random agent port can ever collide with worker_ports (see the
    collision note above). worker_ports is left at Ray's default."""
    return [
        f"--metrics-export-port={RAY_METRICS_EXPORT_PORT}",
        f"--runtime-env-agent-port={RAY_RUNTIME_ENV_AGENT_PORT}",
        f"--dashboard-agent-grpc-port={RAY_DASHBOARD_AGENT_GRPC_PORT}",
        f"--dashboard-agent-listen-port={RAY_DASHBOARD_AGENT_LISTEN_PORT}",
        f"--node-manager-port={RAY_NODE_MANAGER_PORT}",
        f"--object-manager-port={RAY_OBJECT_MANAGER_PORT}",
    ]


# --- R2 object-store spilling (added 2026-06-28) -----------------------------------
# WHY: Ray spills its object store to /tmp/ray/session*/ray_spilled_objects on LOCAL
# disk when plasma (~95GB) overflows. The fully-async RL generator over-produces
# rollouts during the slow (~55-min) first training step, so the spill grows ~370 G/h
# to >1.6 TB and the kubelet EVICTS rank-0 on its ephemeral-storage limit -> gang
# bounce. CoreWeave/iris task pods have R2 (Cloudflare S3-compatible) creds + endpoint
# in env (AWS_ENDPOINT_URL / AWS_*_KEY / AWS_REGION=auto), and boto3 honors
# AWS_ENDPOINT_URL natively, so we redirect Ray's spill to s3://marin-na/... instead of
# local disk. Validated 2026-06-28 in a running w13fix-r3 pod: with this exact config
# Ray spilled 25 objects to R2 and 0 to /tmp (see ray._private.external_storage
# ExternalStorageSmartOpenImpl, which does boto3.resource("s3") -> picks up the R2
# endpoint from env). NOTE: requires boto3 in the rl env (baked into the gpu-rl image).
# Set on BOTH head and worker: object spilling is per-raylet (node-local), the
# smart_open backend appends "_<node_id>" to the prefix so nodes never collide.
# Gate: OT_AGENT_RAY_SPILL_TO_R2 (default "1" = on); set to "0" to fall back to local
# /tmp spilling. Spill prefix is derived per-job from --rendezvous-dir so runs and
# task-retries within a run share one prefix without colliding across jobs.
RAY_SPILL_BUFFER_SIZE = 100 * 1024 * 1024  # 100MB multipart buffer (>=1MB recommended for remote)


def _ray_spill_uri(rendezvous_dir: str | None) -> str | None:
    """Per-job R2 spill prefix derived from the rendezvous dir, or None if R2 spilling
    is disabled / no rendezvous dir is available (single-node runs with no s3 dir fall
    back to local /tmp spilling)."""
    if os.environ.get("OT_AGENT_RAY_SPILL_TO_R2", "1") != "1":
        return None
    if not rendezvous_dir or not rendezvous_dir.startswith("s3://"):
        return None
    # SELF-GATE on boto3: Ray's smart_open spill backend imports boto3 directly, and it
    # is NOT in the gpu-rl image's rl env until the Dockerfile.gpu-rl boto3 add is baked.
    # On an image without boto3, return None -> clean fallback to local /tmp spill (no
    # `ray start` crash). R2 spilling AUTO-ACTIVATES once the rebuilt image ships boto3.
    try:
        import boto3  # noqa: F401
    except ImportError:
        _log("WARNING: boto3 missing -> Ray R2 object-spilling DISABLED (local /tmp fallback); "
             "rebuild gpu-rl image with boto3 (Dockerfile.gpu-rl) to enable. This run risks "
             "the ephemeral-storage eviction if its object store spills > the --disk limit.")
        return None
    return f"{rendezvous_dir.rstrip('/')}/ray_spill"


def _ray_spill_flags(spill_uri: str | None) -> list[str]:
    """Build the `--system-config` flag that redirects Ray object spilling to R2.

    The object_spilling_config VALUE is itself a JSON STRING (double-encoded), per Ray's
    system-config schema. min_spilling_size=0 forces every overflow to spill remotely
    rather than buffering small objects locally first."""
    if not spill_uri:
        return []
    spilling_config = json.dumps(
        {
            "type": "smart_open",
            "params": {"uri": spill_uri, "buffer_size": RAY_SPILL_BUFFER_SIZE},
        }
    )
    system_config = json.dumps(
        {"object_spilling_config": spilling_config, "min_spilling_size": 0}
    )
    return [f"--system-config={system_config}"]


def ray_start_head(head_ip: str, ray_port: int, spill_uri: str | None = None) -> None:
    cmd = [
        _ray_bin(), "start", "--head",
        f"--node-ip-address={head_ip}",
        f"--port={ray_port}",
        "--dashboard-host=0.0.0.0",
        *_ray_port_flags(),
        *_ray_spill_flags(spill_uri),
    ]
    if spill_uri:
        _log(f"Ray object spilling -> R2 prefix {spill_uri} (no local /tmp spill)")
    _log(f"Starting Ray HEAD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ray_start_worker(head_ip: str, ray_port: int, node_ip: str, spill_uri: str | None = None) -> None:
    cmd = [
        _ray_bin(), "start",
        f"--address={head_ip}:{ray_port}",
        f"--node-ip-address={node_ip}",
        *_ray_port_flags(),
        *_ray_spill_flags(spill_uri),
    ]
    if spill_uri:
        _log(f"Ray object spilling -> R2 prefix {spill_uri} (no local /tmp spill)")
    _log(f"Starting Ray WORKER: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ray_stop() -> None:
    try:
        subprocess.run([_ray_bin(), "stop", "--force"], check=False, timeout=60)
    except subprocess.TimeoutExpired:
        _log("Warning: 'ray stop' timed out")


def wait_for_nodes(ray_address: str, expected_nodes: int, timeout: int, rewrite_cb=None) -> None:
    """Block until the Ray cluster reports ``expected_nodes`` alive nodes.

    ``rewrite_cb`` (head only): a no-arg callable invoked on every poll to RE-PUBLISH
    the rendezvous so its ``written_at`` stays fresh. WHY: a worker pod on a cold node
    can start >RENDEZVOUS_FRESHNESS_SLACK (60s) after the head wrote the rendezvous;
    its ``poll_rendezvous(min_written_at=worker_start)`` then rejects the head's
    one-shot rendezvous as "stale" and waits forever for a rewrite, while the head
    waits forever for that 4th node — a mutual deadlock (observed: ep8 diag, task 3
    started 3 min late on a cold node). Rewriting each poll keeps the timestamp ahead
    of any late worker's freshness threshold without weakening the prior-ATTEMPT
    protection (a stale file from a dead PRIOR attempt is still never refreshed).
    """
    import ray

    deadline = time.time() + timeout
    _log(f"Waiting for {expected_nodes} Ray node(s) at {ray_address} (timeout {timeout}s)...")
    ray.init(address=ray_address, ignore_reinit_error=True)
    try:
        last_count = -1
        while time.time() < deadline:
            if rewrite_cb is not None:
                try:
                    rewrite_cb()
                except Exception as exc:
                    _log(f"Warning: rendezvous rewrite failed (will retry): {exc}")
            alive = [n for n in ray.nodes() if n.get("Alive")]
            count = len(alive)
            if count != last_count:
                _log(f"Ray nodes alive: {count}/{expected_nodes}")
                last_count = count
            if count >= expected_nodes:
                _log(f"All {expected_nodes} Ray node(s) joined. Resources: {ray.cluster_resources()}")
                return
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"Only {last_count}/{expected_nodes} Ray nodes joined within {timeout}s.")
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# Roles.
# ---------------------------------------------------------------------------


def capture_termination_artifacts(rendezvous_dir: str | None, reason: str) -> None:
    """On teardown, snapshot a FAST diagnostic summary to the rendezvous store
    BEFORE the pod is reaped. Best-effort; never raises; bounded to finish inside
    the k8s grace period.

    WHY: an iris/k8s-level termination (ephemeral-storage EVICTION, cgroup OOM,
    VRAM OOM) sends the controller a plain SIGTERM and leaves NOTHING in the iris
    finelog — and the per-node Ray logs are deleted with the pod, so the real
    cause is unrecoverable post-mortem (seen 2026-06-28, rl-q36-35b-w13fix: rank-0
    EVICTED for ephemeral-storage>512Gi mid-training-step; no traceback anywhere,
    k8s event TTL'd in ~1h). This persists disk-hogs + GPU mem + df + dmesg-OOM,
    keyed by task id, to ``<rendezvous_dir>/term_artifacts/`` so the next probe
    reads the true cause (disk vs VRAM-OOM vs RAM-OOM)."""
    if not rendezvous_dir:
        return
    import subprocess as _sp

    task_id = os.environ.get("IRIS_TASK_ID", "unknown").replace("/", "_")
    ts = int(time.time())

    def _run(cmd: str, timeout: int = 7) -> str:
        try:
            return _sp.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout).stdout
        except Exception as exc:  # noqa: BLE001 - best-effort
            return f"<{cmd!r} failed: {exc}>"

    summary = "\n".join([
        f"=== TERMINATION ARTIFACT task={task_id} ts={ts} reason={reason} ===",
        "--- df -h /tmp /dev/shm ---", _run("df -h /tmp /dev/shm 2>&1"),
        "--- top /tmp disk hogs (ephemeral-storage eviction cause) ---",
        _run("du -sh /tmp/* /tmp/ray/session*/logs /tmp/ray/session*/*spill* 2>/dev/null | sort -rh | head -25", 9),
        "--- nvidia-smi (VRAM OOM cause) ---",
        _run("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>&1"),
        "--- top RSS procs (host-RAM OOM cause) ---",
        _run("ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -12"),
        "--- dmesg OOM/kill tail ---",
        _run("dmesg 2>/dev/null | grep -iE 'oom|killed process|out of memory|Xid' | tail -10"),
    ])
    try:
        uri = f"{rendezvous_dir.rstrip('/')}/term_artifacts/{task_id}_{ts}.txt"
        fs, path = _fs_and_path(uri)
        with fs.open(path, "w") as f:
            f.write(summary)
        _log(f"[term-capture] wrote termination artifact -> {uri}")
    except Exception as exc:  # noqa: BLE001 - still emit to finelog as fallback
        _log(f"[term-capture] upload FAILED ({exc}); emitting inline:\n{summary}")


def run_head(args: argparse.Namespace, train_argv: list[str]) -> int:
    num_tasks = _num_tasks()
    head_ip = _own_ip()
    ray_port = args.ray_port
    ray_address = f"{head_ip}:{ray_port}"
    _log(f"ROLE=head rank=0/{num_tasks} head_ip={head_ip} ray_port={ray_port}")

    # On iris task retry, a rendezvous file from a previous attempt still points
    # at a now-dead head. Purge before starting the new head.
    if num_tasks > 1 and args.rendezvous_dir:
        clear_rendezvous(args.rendezvous_dir)

    ray_start_head(head_ip, ray_port, spill_uri=_ray_spill_uri(args.rendezvous_dir))

    if num_tasks > 1:
        if not args.rendezvous_dir:
            raise ValueError(
                "Multi-node iris slice (IRIS_NUM_TASKS>1) requires --rendezvous-dir "
                "(or OT_AGENT_IRIS_RENDEZVOUS_DIR) so worker ranks can find the head IP."
            )
        write_rendezvous(args.rendezvous_dir, head_ip, ray_port)
        # Re-publish the rendezvous each poll so a late cold-node worker never sees it
        # as "stale" (see wait_for_nodes docstring — prevents the freshness deadlock).
        wait_for_nodes(
            ray_address, num_tasks, args.cluster_join_timeout,
            rewrite_cb=lambda: write_rendezvous(args.rendezvous_dir, head_ip, ray_port),
        )
    else:
        _log("Single-node slice: skipping rendezvous and multi-node wait.")

    env = os.environ.copy()
    env["RAY_ADDRESS"] = ray_address  # skyrl-train's bare ray.init() attaches here
    env["PYTHONUNBUFFERED"] = "1"

    _log("Launching MarinSkyRL training driver:")
    _log("  " + " ".join(train_argv))
    sys.stdout.flush()
    sys.stderr.flush()

    process = subprocess.Popen(train_argv, env=env, start_new_session=True)

    def _shutdown(signum, _frame) -> None:
        _log(f"Received signal {signum}; terminating training driver and stopping Ray...")
        # Capture FIRST (before teardown mutates disk/GPU state) — a SIGTERM here is
        # often a k8s eviction / OOM whose cause survives nowhere else.
        capture_termination_artifacts(args.rendezvous_dir, f"signal {signum} (head rank 0)")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=60)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if args.rendezvous_dir and num_tasks > 1:
            _set_marker(args.rendezvous_dir, DONE_FILENAME)
        ray_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    exit_code = process.wait()
    if exit_code != 0:
        capture_termination_artifacts(args.rendezvous_dir, f"driver exit_code={exit_code} (head rank 0)")
    # Signal workers to unpark, then tear down.
    if args.rendezvous_dir and num_tasks > 1:
        _set_marker(args.rendezvous_dir, DONE_FILENAME)
    ray_stop()
    if args.rendezvous_dir and num_tasks > 1:
        clear_rendezvous(args.rendezvous_dir)
    return exit_code


def run_worker(args: argparse.Namespace) -> int:
    worker_start = time.time()
    rank = _rank()
    num_tasks = _num_tasks()
    node_ip = _own_ip()
    _log(f"ROLE=worker rank={rank}/{num_tasks} node_ip={node_ip}")

    if not args.rendezvous_dir:
        raise ValueError(
            "Worker rank requires --rendezvous-dir (or OT_AGENT_IRIS_RENDEZVOUS_DIR) "
            "to discover the head IP."
        )

    payload = poll_rendezvous(args.rendezvous_dir, args.rendezvous_timeout, min_written_at=worker_start)
    head_ip = payload["head_ip"]
    ray_port = int(payload.get("port", args.ray_port))
    ray_address = f"{head_ip}:{ray_port}"

    ray_start_worker(head_ip, ray_port, node_ip, spill_uri=_ray_spill_uri(args.rendezvous_dir))
    wait_for_nodes(ray_address, num_tasks, args.cluster_join_timeout)
    _log(f"Worker rank {rank} joined Ray cluster at {ray_address}; parking until the head finishes.")

    stop = threading.Event()

    def _shutdown(signum, _frame) -> None:
        _log(f"Worker rank {rank} received signal {signum}; stopping Ray.")
        # A SIGTERM on a worker node is often a k8s eviction/OOM of that node (it
        # hosts the training actors' GPUs); capture its disk/GPU state before reap.
        capture_termination_artifacts(args.rendezvous_dir, f"signal {signum} (worker rank {rank})")
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block until the head publishes the done marker (training finished) or we
    # are signalled. The training driver on rank 0 schedules actors onto this
    # node's GPUs; this process just keeps the Ray node alive.
    while not stop.is_set():
        if _marker_exists(args.rendezvous_dir, DONE_FILENAME, min_written_at=worker_start):
            _log(f"Worker rank {rank} saw head done-marker; shutting down.")
            break
        time.sleep(POLL_INTERVAL)
    ray_stop()
    return 0


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Bootstrap one cross-node Ray cluster on an iris GPU slice and run "
        "the MarinSkyRL training driver on rank 0. Everything after `--` is the "
        "training command (e.g. `python -m skyrl_train.entrypoints.main_base <hydra args>`).",
    )
    parser.add_argument(
        "--ray-port",
        type=int,
        default=int(os.environ.get("OT_AGENT_IRIS_RAY_PORT", "6379")),
        help="Port the Ray head binds (default 6379).",
    )
    parser.add_argument(
        "--rendezvous-dir",
        default=os.environ.get("OT_AGENT_IRIS_RENDEZVOUS_DIR"),
        help="Shared object-store/dir for the head/worker rendezvous (gs://, s3://, "
        "or a shared path). Defaults to $OT_AGENT_IRIS_RENDEZVOUS_DIR.",
    )
    parser.add_argument(
        "--rendezvous-timeout",
        type=int,
        default=DEFAULT_RENDEZVOUS_TIMEOUT,
        help=f"Seconds workers poll for the head rendezvous (default {DEFAULT_RENDEZVOUS_TIMEOUT}).",
    )
    parser.add_argument(
        "--cluster-join-timeout",
        type=int,
        default=DEFAULT_CLUSTER_JOIN_TIMEOUT,
        help=f"Seconds to wait for all nodes to join the Ray cluster (default {DEFAULT_CLUSTER_JOIN_TIMEOUT}).",
    )
    parser.add_argument(
        "--train-data",
        default=os.environ.get("OT_AGENT_IRIS_TRAIN_DATA", ""),
        help="JSON list of train_data HF dataset(s) to stage (extract to the node-local "
        "task dir) on EVERY node before Ray starts. Required for agentic terminal_bench "
        "rollouts on a multi-node slice with no shared filesystem.",
    )
    args, train_argv = parser.parse_known_args()
    # argparse leaves the `--` separator out of train_argv; strip a leading one
    # if the shell passed it through.
    if train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]
    if not train_argv:
        parser.error("No training command given. Pass it after `--`.")
    return args, train_argv


def _print_env_snapshot() -> None:
    _log("environment snapshot:")
    for key in (
        "IRIS_TASK_ID", "IRIS_NUM_TASKS", "IRIS_ADVERTISE_HOST",
        "RAY_ADDRESS", "SKYRL_HOME", "PYTHONPATH", "HF_HOME",
        "NUM_INFERENCE_ENGINES", "POLICY_NUM_NODES", "TENSOR_PARALLEL_SIZE",
    ):
        print(f"  {key}={os.environ.get(key, '<unset>')}", flush=True)


def main() -> None:
    args, train_argv = parse_args()
    _print_env_snapshot()
    # Stage the task dataset on THIS node before Ray bootstrap (head + every worker).
    # Without this, only rank-0 has the extracted tasks and the rollout workers die
    # with FileNotFoundError on task.toml (see stage_train_data docstring).
    if args.train_data:
        stage_train_data(args.train_data)
    rank = _rank()
    if rank == 0:
        exit_code = run_head(args, train_argv)
    else:
        exit_code = run_worker(args)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
