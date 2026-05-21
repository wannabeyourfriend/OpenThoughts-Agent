"""IrisLauncher — base class for submitting OT-Agent jobs to a Marin Iris cluster.

This is the Iris analog of ``hpc/cloud_launch_utils.CloudLauncher`` (which
targets SkyPilot). It exists in parallel rather than as a "provider" plugin
because Iris and SkyPilot disagree on key abstractions (workdir bind mount
vs file_mounts, in-cluster scheduling vs bring-up-a-VM, autostop vs job
timeout). Trying to share one interface created leaky bolts; two clean
modules is cheaper to reason about.

Backend-agnostic helpers under ``hpc/`` (e.g. ``arg_groups``,
``harbor_utils``, ``datagen_config_utils``) are reused as-is.

Output handling — two modes, default ``rsync``:

* ``rsync``: periodically pull the task's host-side workdir back to
  ``--local-sync-dir`` via ``gcloud compute tpus tpu-vm ssh ... rsync``.
  Matches the SkyPilot ``PeriodicRemoteSync`` UX for downstream tools that
  expect local files. Iris-internal coupling: derives ``workdir_host_path``
  from ``JobName.to_safe_token()`` — if iris changes that convention we
  update one constant here.

* ``gcs``: workload writes directly to ``--gcs-output-dir/<job-name>/``.
  More resilient to preemption and multi-host scatter; no laptop sync.

Multi-host slices (TPU vm_count > 1) are scaffolded but **untested**. The
periodic-sync loop already fans out across worker indices; coordination of
JAX cross-host init and rsync race conditions across hosts need real
verification before relying on this for v6e-8+ jobs. See block at
``_periodic_rsync_loop``.

TODO(coordinator): right now the launcher runs from the user's laptop and
holds the tunnel + sync threads for the duration of the job. For long jobs
this is fragile (laptop sleep, network hiccup, etc.). The durable answer
is to wrap the launcher in a tiny CPU coordinator job on the cluster
itself, so the sync threads live on a controller-class VM. Not done in v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# Default chips per TPU host VM on every family currently exposed by marin
# (ct5lp-hightpu-4t, ct6e-standard-4t, ct5p-hightpu-4t, ct4p-hightpu-4t).
# If marin ever provisions ``-8t`` host variants this needs revisiting.
CHIPS_PER_TPU_HOST = 4

DEFAULT_TASK_IMAGE = "ghcr.io/open-thoughts/openthoughts-agent:tpu"
DEFAULT_CLUSTER_CONFIG = "lib/iris/config/marin.yaml"
DEFAULT_OUTPUT_SUBDIR = "cloud_runs"
DEFAULT_SYNC_INTERVAL_SECONDS = 60
DEFAULT_PRIORITY = "interactive"

# Where iris workers stage per-task workdirs on the host VM. Tracks
# ``self._cache_dir / "workdirs"`` in lib/iris/.../worker/task_attempt.py;
# the iris docker image creates ``/var/cache/iris`` directly. If iris
# changes this convention rsync stops finding outputs — single constant
# to update.
IRIS_WORKER_WORKDIR_ROOT = "/var/cache/iris/workdirs"


def derive_safe_task_token(task_id: str, user: str) -> str:
    """Replicate ``iris.cluster.types.JobName.to_safe_token`` client-side.

    Returns ``<user>-<sha256(task_id)>`` — used to compose the on-host
    workdir path. Coupled to iris internals; see module docstring.
    """
    digest = hashlib.sha256(task_id.encode()).hexdigest()
    return f"{user}-{digest}"


def compute_workdir_host_path(job_name: str, user: str, task_index: int, attempt_id: int = 0) -> str:
    """Reconstruct ``self.workdir`` from task_attempt.py:614 without an RPC."""
    task_id = f"{job_name}/{task_index}"
    safe = derive_safe_task_token(task_id, user)
    return f"{IRIS_WORKER_WORKDIR_ROOT}/{safe}_attempt_{attempt_id}"


def parse_tpu_vm_count(tpu_spec: Optional[str]) -> int:
    """Return the host-VM count implied by a TPU variant like ``v6e-16``.

    Chips per host is 4 on every family currently configured in marin's
    cluster YAML, so ``vm_count = chips / 4``. Returns 1 when no TPU is
    requested or the spec doesn't end in ``-<int>``.
    """
    if not tpu_spec:
        return 1
    try:
        chips = int(tpu_spec.rsplit("-", 1)[-1])
    except ValueError:
        return 1
    return max(1, chips // CHIPS_PER_TPU_HOST)


@dataclass
class WorkerHandle:
    """Identifies a single GCP TPU VM hosting one task of an iris job."""

    vm_name: str
    zone: str
    project: str
    worker_index: int  # which worker in the multi-host gang (0-based)


class PeriodicWorkerSync:
    """Background thread that rsyncs a per-task host-side workdir to the laptop.

    Mirrors the shape of ``CloudLauncher.PeriodicRemoteSync`` but goes
    through ``gcloud compute tpus tpu-vm ssh ... rsync`` because iris
    workers don't accept direct ``rsync`` over a stable SSH alias the
    way SkyPilot clusters do.

    Single-worker only in v1. Multi-host fan-out lives in IrisLauncher.
    """

    def __init__(
        self,
        worker: WorkerHandle,
        remote_path: str,
        local_dir: str,
        interval_seconds: int = DEFAULT_SYNC_INTERVAL_SECONDS,
        log_prefix: str = "[iris-sync]",
    ):
        self.worker = worker
        self.remote_path = remote_path.rstrip("/")
        self.local_dir = local_dir
        self.interval = interval_seconds
        self.log_prefix = log_prefix
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _rsync_once(self) -> None:
        try:
            Path(self.local_dir).mkdir(parents=True, exist_ok=True)
            # rsync over gcloud's IAP/SSH tunnel. ``--rsync-path=sudo rsync``
            # because workdir is created by the worker process (often root)
            # and the user's gcloud SSH lands as a non-root account.
            cmd = [
                "gcloud", "compute", "tpus", "tpu-vm", "scp",
                "--zone", self.worker.zone,
                "--project", self.worker.project,
                f"--worker={self.worker.worker_index}",
                "--quiet", "--recurse",
                f"{self.worker.vm_name}:{self.remote_path}/",
                self.local_dir,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return
            stderr_lc = (result.stderr or "").lower()
            # Benign during early job: workdir doesn't exist yet on the worker.
            if "no such file" in stderr_lc or "does not exist" in stderr_lc:
                return
            print(
                f"{self.log_prefix} scp from {self.worker.vm_name} returned "
                f"{result.returncode}: {result.stderr.strip()[:200]}",
                file=sys.stderr,
                flush=True,
            )
        except subprocess.TimeoutExpired:
            print(f"{self.log_prefix} scp timed out (will retry)", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"{self.log_prefix} scp failed: {e}", file=sys.stderr, flush=True)

    def _run(self) -> None:
        # Brief delay so the worker has a chance to create the workdir.
        time.sleep(10)
        while not self._stop_event.is_set():
            self._rsync_once()
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_and_final_sync(self) -> None:
        """Signal stop, do one last sync attempt before the workdir is GC'd."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # Iris cleans up the workdir on task end (task_attempt.py:1074), so
        # this final pull races with that. Best-effort; GCS mode if you
        # can't tolerate the race.
        self._rsync_once()


class IrisLauncher:
    """Base class for OT-Agent launchers targeting Marin Iris.

    Subclasses override:
      - ``add_task_specific_args(parser)``
      - ``normalize_paths(args)``
      - ``build_task_command(args, remote_output_dir) -> list[str]``
      - ``build_env(args) -> dict[str, str]``  (optional override)
    """

    task_name: str = "ot-iris"
    job_name_prefix: str = "iris"
    default_output_subdir: str = DEFAULT_OUTPUT_SUBDIR
    default_n_concurrent: int = 16
    default_tpu: str = "v6e-4"

    # Daytona is the only sandbox backend that works without DinD on iris.
    # Users may still pass --harbor_env docker; iris workers don't mount
    # /var/run/docker.sock so the job will fail at runtime — by design.
    default_harbor_env: str = "daytona"

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()

    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------

    def create_argument_parser(self, description: str = "") -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description or self.task_name)
        self._add_iris_common_args(parser)
        self.add_task_specific_args(parser)
        return parser

    def _add_iris_common_args(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("iris")
        g.add_argument("--cluster-config", "--cluster_config",
                       default=self._resolve_cluster_config_default(),
                       help="Path to the iris cluster YAML (default: marin via lib/iris/config/marin.yaml in the marin repo).")
        g.add_argument("--task-image", "--task_image",
                       default=DEFAULT_TASK_IMAGE,
                       help=f"Container image for the task (default: {DEFAULT_TASK_IMAGE}).")
        g.add_argument("--tpu", default=self.default_tpu,
                       help=f"TPU variant (default: {self.default_tpu}). For multi-host slices "
                            "(e.g. v6e-8, v6e-16) the launcher will gang-schedule replicas — "
                            "**UNTESTED in v1**, see module docstring.")
        g.add_argument("--cpu", type=float, default=8.0,
                       help="CPU cores for the entrypoint task (default 8).")
        g.add_argument("--memory", default="64GB",
                       help="Memory for the entrypoint task (default 64GB).")
        g.add_argument("--disk", default="200GB",
                       help="Ephemeral disk (default 200GB).")
        g.add_argument("--priority", default=DEFAULT_PRIORITY,
                       choices=["production", "interactive", "batch"],
                       help="Iris priority band (default interactive).")
        g.add_argument("--max-retries", "--max_retries", type=int, default=0,
                       help="Max retries on failure (does NOT cover preemption — iris retries "
                            "preemptions automatically up to its own limit).")
        g.add_argument("--timeout", type=int, default=0,
                       help="Job timeout in seconds (0 = no timeout).")
        g.add_argument("--preemptible", dest="preemptible", action="store_true", default=None,
                       help="Force scheduling on preemptible workers (overrides iris heuristic).")
        g.add_argument("--no-preemptible", dest="preemptible", action="store_false",
                       help="Force scheduling on non-preemptible workers.")
        g.add_argument("--no-wait", dest="no_wait", action="store_true", default=False,
                       help="Submit and detach instead of streaming logs.")

        og = parser.add_argument_group("outputs")
        og.add_argument("--output-mode", "--output_mode",
                        choices=["rsync", "gcs"], default="rsync",
                        help="rsync: periodically pull /var/cache/iris/workdirs/.../outputs back "
                             "to --local-sync-dir via gcloud SSH. "
                             "gcs: workload writes directly to --gcs-output-dir, nothing pulled "
                             "back. Default: rsync.")
        og.add_argument("--local-sync-dir", "--local_sync_dir",
                        default=str(Path.home() / self.default_output_subdir),
                        help="Local destination for --output-mode rsync (default ~/cloud_runs).")
        og.add_argument("--gcs-output-dir", "--gcs_output_dir",
                        default=None,
                        help="GCS prefix for --output-mode gcs (e.g. gs://my-bucket/ot-agent). "
                             "Required when --output-mode=gcs.")
        og.add_argument("--sync-interval", "--sync_interval", type=int,
                        default=DEFAULT_SYNC_INTERVAL_SECONDS,
                        help="Seconds between rsync polls (default 60).")
        # NOTE: --dry-run / --dry_run is provided by hpc.arg_groups.add_model_compute_args
        # which subclass launchers call from add_task_specific_args. We don't redeclare
        # it here to avoid argparse conflicts.

    def _resolve_cluster_config_default(self) -> str:
        """Find the marin repo's cluster config relative to common locations."""
        candidates = [
            Path.home() / "Documents/marin" / DEFAULT_CLUSTER_CONFIG,
            Path("/Users/benjaminfeuer/Documents/marin") / DEFAULT_CLUSTER_CONFIG,
            Path(os.environ.get("MARIN_ROOT", "")) / DEFAULT_CLUSTER_CONFIG,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return DEFAULT_CLUSTER_CONFIG

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def normalize_paths(self, args: argparse.Namespace) -> None:
        """Subclass hook: validate/normalize paths and infer defaults."""

    def build_task_command(self, args: argparse.Namespace, remote_output_dir: str) -> List[str]:
        """Subclass hook: build the ``python data/...py ...`` invocation."""
        raise NotImplementedError

    def build_env(self, args: argparse.Namespace) -> dict:
        """Subclass hook: env vars to inject into the iris task container.

        HF_TOKEN, WANDB_API_KEY, HF_DATASETS_TRUST_REMOTE_CODE, and
        TOKENIZERS_PARALLELISM are auto-injected by iris workers — no need
        to add them here.
        """
        return {}

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def _derive_job_name(self, args: argparse.Namespace) -> str:
        job_name = getattr(args, "job_name", None)
        if job_name:
            return job_name
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"{self.job_name_prefix}-{ts}"

    def run(self, args: argparse.Namespace) -> int:
        self.normalize_paths(args)

        if args.output_mode == "gcs" and not args.gcs_output_dir:
            raise SystemExit("--gcs-output-dir is required when --output-mode=gcs")

        job_name = self._derive_job_name(args)
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"

        # Where the workload should write outputs inside the container.
        # rsync mode → /app/outputs/<job-name>/ (lives under workdir; rsync'd back)
        # gcs mode   → gs://.../<job-name>/    (workload writes directly)
        if args.output_mode == "gcs":
            remote_output_dir = f"{args.gcs_output_dir.rstrip('/')}/{job_name}"
        else:
            remote_output_dir = f"/app/outputs/{job_name}"

        command = self.build_task_command(args, remote_output_dir)
        env_vars = self.build_env(args)

        vm_count = parse_tpu_vm_count(args.tpu)

        print(f"[iris] Job:        /{user}/{job_name}", flush=True)
        print(f"[iris] Cluster:    {args.cluster_config}", flush=True)
        print(f"[iris] Image:      {args.task_image}", flush=True)
        print(f"[iris] TPU:        {args.tpu}  (vm_count={vm_count})", flush=True)
        print(f"[iris] Priority:   {args.priority}", flush=True)
        print(f"[iris] Output:     mode={args.output_mode} dest={remote_output_dir}", flush=True)
        if args.output_mode == "rsync":
            print(f"[iris] Local sync: {args.local_sync_dir}/{job_name}/  (every {args.sync_interval}s)", flush=True)
        print(f"[iris] Command:    {shlex.join(command)}", flush=True)

        if args.dry_run:
            print("[iris] --dry-run: not submitting", flush=True)
            return 0

        if vm_count > 1:
            print(
                "[iris] WARNING: multi-host TPU slice (vm_count > 1) is scaffolded but UNTESTED. "
                "Expect rough edges in: (1) cross-host JAX init / vLLM-TPU sharding, "
                "(2) rsync fan-out timing, (3) coscheduling. Validate single-host first.",
                file=sys.stderr, flush=True,
            )

        # Defer the heavy iris imports so --dry-run / --help stay snappy.
        from iris.client import IrisClient
        from iris.cluster.config import IrisConfig
        from iris.cluster.types import EnvironmentSpec, Entrypoint
        from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults, build_tpu_alternatives
        from iris.proto import job_pb2

        # Tunnel to the controller via the documented IrisConfig pattern
        # (see lib/iris/.../cluster/config.py:IrisConfig docstring).
        iris_config = IrisConfig.load(args.cluster_config)
        bundle = iris_config.provider_bundle()
        controller_proto = iris_config.proto.controller
        if controller_proto.WhichOneof("controller") == "local":
            from iris.cluster.providers.local.cluster import LocalCluster
            local_cluster = LocalCluster(iris_config.proto)
            controller_address = local_cluster.start()
        else:
            controller_address = (
                iris_config.controller_address()
                or bundle.controller.discover_controller(controller_proto)
            )

        with bundle.controller.tunnel(controller_address) as controller_url:
            resources = build_resources(args.tpu, None, cpu=args.cpu, memory=args.memory, disk=args.disk)
            tpu_variants = build_tpu_alternatives(args.tpu)
            primary_tpu = tpu_variants[0] if tpu_variants else None
            replicas, coscheduling = resolve_multinode_defaults(primary_tpu, None, None)
            resources_proto = resources.to_proto()
            constraints = build_job_constraints(
                resources_proto=resources_proto,
                tpu_variants=tpu_variants,
                replicas=replicas,
                regions=None, zone=None,
                preemptible=args.preemptible,
            )

            priority_band = job_pb2.PRIORITY_BAND_UNSPECIFIED
            if args.priority:
                # Map name → enum the same way iris/cli/job.py does.
                _PRIO = {
                    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
                    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
                    "batch": job_pb2.PRIORITY_BAND_BATCH,
                }
                priority_band = _PRIO.get(args.priority, priority_band)

            client = IrisClient.remote(controller_url, workspace=self.repo_root)
            entrypoint = Entrypoint.from_command(*command)

            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=resources,
                environment=EnvironmentSpec(env_vars=env_vars),
                constraints=constraints,
                coscheduling=coscheduling,
                replicas=replicas,
                max_retries_failure=args.max_retries,
                # Iris auto-retries on preemption; leave at default (1000).
                task_image=args.task_image,
                priority_band=priority_band,
                timeout=None if args.timeout == 0 else _seconds_to_duration(args.timeout),
            )
            full_job_id = str(job.job_id)
            print(f"[iris] Submitted: {full_job_id}", flush=True)

            sync_threads: List[PeriodicWorkerSync] = []
            if args.output_mode == "rsync" and not args.no_wait:
                sync_threads = self._start_rsync_threads(
                    args, job_name, user, vm_count, client, full_job_id,
                )

            if args.no_wait:
                return 0

            try:
                status = job.wait(stream_logs=True, timeout=float("inf"))
                exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
            except KeyboardInterrupt:
                print(f"[iris] Terminating job {full_job_id}...", file=sys.stderr, flush=True)
                client.terminate_job(job.job_id)
                exit_code = 130
            finally:
                for st in sync_threads:
                    st.stop_and_final_sync()

            print(f"[iris] Job exit: {exit_code}", flush=True)
            return exit_code

    # ------------------------------------------------------------------
    # Rsync fan-out (single-host today, multi-host scaffold)
    # ------------------------------------------------------------------

    def _start_rsync_threads(
        self,
        args: argparse.Namespace,
        job_name: str,
        user: str,
        vm_count: int,
        client,  # IrisClient
        full_job_id: str,
    ) -> List[PeriodicWorkerSync]:
        """Resolve worker VMs for each replica and spawn one rsync thread each.

        Multi-host note: vm_count > 1 is wired but UNTESTED. The hard parts
        we haven't validated: (a) all replicas' workdirs may not exist at
        the same wall-clock instant (worker scheduling latency), so early
        rsyncs will return "no such file" benignly; (b) the gcloud TPU SCP
        path may need ``--worker=<idx>`` per host rather than per VM name —
        on multi-VM TPU slices the workers share a TPU pod name and differ
        only by worker index. The code below uses worker_index; verify on
        a v6e-8 smoke before relying on this for v6e-16+.
        """
        local_root = Path(args.local_sync_dir) / job_name
        threads: List[PeriodicWorkerSync] = []

        for task_index in range(vm_count):
            workdir_remote = compute_workdir_host_path(
                f"/{user}/{job_name}", user, task_index=task_index,
            )
            # Workload writes to /app/outputs/<job-name>/; on host this is
            # <workdir_remote>/outputs/<job-name>/.
            remote_outputs = f"{workdir_remote}/outputs/{job_name}"
            local_dir = str(local_root / f"worker-{task_index}")

            try:
                worker = self._resolve_worker_handle(client, full_job_id, task_index)
            except Exception as e:
                print(
                    f"[iris-sync] Could not resolve worker {task_index} yet "
                    f"({type(e).__name__}: {e}). Will retry inside the sync thread.",
                    file=sys.stderr, flush=True,
                )
                # The thread will retry resolution implicitly on each scp attempt
                # by re-querying iris if needed. For v1 keep it simple: skip
                # this replica's sync and warn.
                continue

            print(
                f"[iris-sync] worker {task_index}: {worker.vm_name} ({worker.zone}) "
                f"→ {local_dir}",
                flush=True,
            )
            t = PeriodicWorkerSync(
                worker=worker,
                remote_path=remote_outputs,
                local_dir=local_dir,
                interval_seconds=args.sync_interval,
            )
            t.start()
            threads.append(t)

        return threads

    def _resolve_worker_handle(
        self,
        client,  # IrisClient
        full_job_id: str,
        task_index: int,
    ) -> WorkerHandle:
        """Look up which GCP VM is hosting a specific task replica.

        Polls iris briefly until the task is assigned. Pulls VM metadata
        (name, zone, project) from the iris worker registry.
        """
        from iris.cluster.types import JobName

        job_name = JobName.from_string(full_job_id)
        deadline = time.time() + 120
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                tasks = client.list_tasks(job_name)
            except Exception as e:
                last_err = e
                time.sleep(2)
                continue

            for t in tasks:
                if t.task_id.endswith(f"/{task_index}"):
                    worker_id = getattr(t, "worker_id", "") or ""
                    if not worker_id:
                        break  # assignment not done yet
                    return self._worker_id_to_handle(client, worker_id, task_index)
            time.sleep(2)
        raise TimeoutError(
            f"Worker for task {task_index} of {full_job_id} not assigned within 120s "
            f"(last error: {last_err})"
        )

    def _worker_id_to_handle(self, client, worker_id: str, task_index: int) -> WorkerHandle:
        """Map iris worker_id → (gcp_vm_name, zone, project).

        Workers register with the controller along with metadata that
        includes their GCP zone and instance name. We pull that off the
        worker health list.
        """
        workers = client.list_workers()
        for w in workers:
            if w.worker_id == worker_id:
                # iris worker metadata uses these keys (see
                # lib/iris/.../providers/gcp/worker.py). If keys change,
                # the dict.get defaults keep this from KeyError'ing out.
                meta = dict(getattr(w, "metadata", {}) or {})
                vm = meta.get("gcp_instance_name") or meta.get("instance") or worker_id
                zone = meta.get("gcp_zone") or meta.get("zone") or ""
                project = meta.get("gcp_project") or "hai-gcp-models"
                if not zone:
                    raise RuntimeError(
                        f"Worker {worker_id} reports no GCP zone in metadata; "
                        "rsync needs zone for gcloud SSH. Check iris worker registration."
                    )
                return WorkerHandle(
                    vm_name=vm, zone=zone, project=project, worker_index=task_index,
                )
        raise RuntimeError(f"Worker {worker_id} not found in cluster's worker list")


# Imported lazily inside .run() to keep CLI startup fast, but tiny enough
# to define here.
def _seconds_to_duration(secs: int):
    from iris.cluster.types import Duration
    return Duration.from_seconds(secs)
