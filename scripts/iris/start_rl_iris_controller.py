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

# Cold GPU nodes (image pull + setup) can take several minutes to reach the
# rendezvous, so these are generous.
DEFAULT_RENDEZVOUS_TIMEOUT = 1800
DEFAULT_CLUSTER_JOIN_TIMEOUT = 1800
POLL_INTERVAL = 5
# Tolerates clock skew between nodes and the time rank-0 needs to start Ray.
RENDEZVOUS_FRESHNESS_SLACK = 60


def _log(msg: str) -> None:
    print(f"[start_rl_iris_controller] {msg}", flush=True)


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


def ray_start_head(head_ip: str, ray_port: int) -> None:
    cmd = [
        "ray", "start", "--head",
        f"--node-ip-address={head_ip}",
        f"--port={ray_port}",
        "--dashboard-host=0.0.0.0",
    ]
    _log(f"Starting Ray HEAD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ray_start_worker(head_ip: str, ray_port: int, node_ip: str) -> None:
    cmd = [
        "ray", "start",
        f"--address={head_ip}:{ray_port}",
        f"--node-ip-address={node_ip}",
    ]
    _log(f"Starting Ray WORKER: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ray_stop() -> None:
    try:
        subprocess.run(["ray", "stop", "--force"], check=False, timeout=60)
    except subprocess.TimeoutExpired:
        _log("Warning: 'ray stop' timed out")


def wait_for_nodes(ray_address: str, expected_nodes: int, timeout: int) -> None:
    """Block until the Ray cluster reports ``expected_nodes`` alive nodes."""
    import ray

    deadline = time.time() + timeout
    _log(f"Waiting for {expected_nodes} Ray node(s) at {ray_address} (timeout {timeout}s)...")
    ray.init(address=ray_address, ignore_reinit_error=True)
    try:
        last_count = -1
        while time.time() < deadline:
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

    ray_start_head(head_ip, ray_port)

    if num_tasks > 1:
        if not args.rendezvous_dir:
            raise ValueError(
                "Multi-node iris slice (IRIS_NUM_TASKS>1) requires --rendezvous-dir "
                "(or OT_AGENT_IRIS_RENDEZVOUS_DIR) so worker ranks can find the head IP."
            )
        write_rendezvous(args.rendezvous_dir, head_ip, ray_port)
        wait_for_nodes(ray_address, num_tasks, args.cluster_join_timeout)
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

    ray_start_worker(head_ip, ray_port, node_ip)
    wait_for_nodes(ray_address, num_tasks, args.cluster_join_timeout)
    _log(f"Worker rank {rank} joined Ray cluster at {ray_address}; parking until the head finishes.")

    stop = threading.Event()

    def _shutdown(signum, _frame) -> None:
        _log(f"Worker rank {rank} received signal {signum}; stopping Ray.")
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
    rank = _rank()
    if rank == 0:
        exit_code = run_head(args, train_argv)
    else:
        exit_code = run_worker(args)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
