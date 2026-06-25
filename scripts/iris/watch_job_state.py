#!/usr/bin/env python3
"""Authoritative iris job-state watcher.

Poll the *authoritative job lifecycle state* of an iris job on an interval, emit
a line on every state transition, and exit with a clear terminal verdict the
moment the job leaves RUNNING (succeeded / failed / killed / worker_failed /
unschedulable) **OR disappears from the cluster entirely**.

Why this exists
---------------
The prior monitor watched log-line *content* — it grepped rank-0 logs for the
strings ``EPDIAG_FWD1`` / ``DEADLOCK`` / ``TERMINAL``. A clean termination,
eviction, preemption, or an early crash that never prints one of those strings
makes the content-watch see no matching line, so the supervising agent goes idle
believing the job is "still running" while the job + its pods have already left
the cluster. (Confirmed against ``/benjaminfeuer/rl-131k-cpdcp2r3``: it ended in
state ``killed`` / "Terminated by user" with 0 pods on the cluster — a clean
termination the content-watch could never have caught.)

The fix is to poll the **authoritative iris job state**, not log content. The
iris controller retains terminal job records after the pods are reaped, so even
a job that has fully vanished from k8s still reports its terminal state via
``iris job summary``. That is the signal this tool watches.

Authoritative state source
---------------------------
``iris --cluster=<C> job summary <job_id> --json`` (the richest single-job call:
``state`` + ``error`` + ``exit_code`` + per-task states + ``finished_at``). It is
backed by the controller's ``GetJobStatus`` / ``ListTasks`` RPCs and works for
running *and* completed/terminal jobs. If that call fails transiently we fall
back to the lighter ``iris query "SELECT state FROM jobs WHERE job_id=..."``
which returns the numeric state. As a final cross-check (and to catch the
"disappeared from cluster / 0 pods" case explicitly) we can count live pods via
``kubectl``.

iris JobState enum (lib/iris/src/iris/rpc/job.proto):
    0 UNSPECIFIED  1 PENDING  2 BUILDING  3 RUNNING
    4 SUCCEEDED    5 FAILED   6 KILLED    7 WORKER_FAILED   8 UNSCHEDULABLE

Usage
-----
    # one-shot: print the current authoritative state and exit
    python scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once

    # watch on a 60s interval until the job leaves RUNNING (then exit)
    python scripts/iris/watch_job_state.py /benjaminfeuer/<job> --interval 60

Exit codes: 0 succeeded · 1 failed/killed/worker_failed/unschedulable · 2 the
job is absent from the controller AND has 0 pods (disappeared) · 3 watch error.

Importable: ``get_job_state(job_id, cluster)`` returns a ``JobStateSnapshot``;
``watch(job_id, ...)`` runs the poll loop and returns the terminal snapshot, so a
supervising agent can use this as the watch primitive instead of grepping logs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

# --- iris invocation ------------------------------------------------------
# The CoreWeave GPU (k8s) backend needs a `kubernetes` install that the bare
# marin `.venv` lacks; the otagent env has it AND ships the iris CLI, so use
# that binary by default. Override with $IRIS_BIN if it ever moves.
IRIS_BIN = os.environ.get(
    "IRIS_BIN", "/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris"
)
DEFAULT_CLUSTER = "cw-us-east-02a"  # the GPU RL cluster; use "marin" for TPU jobs

# JobState int -> friendly name (lib/iris/src/iris/rpc/job.proto).
STATE_NAMES = {
    0: "unspecified",
    1: "pending",
    2: "building",
    3: "running",
    4: "succeeded",
    5: "failed",
    6: "killed",
    7: "worker_failed",
    8: "unschedulable",
}
NAME_TO_INT = {v: k for k, v in STATE_NAMES.items()}

RUNNING_STATES = {"pending", "building", "running", "unspecified"}
TERMINAL_STATES = {"succeeded", "failed", "killed", "worker_failed", "unschedulable"}

# Retry/backoff for the iris CLI call (matches analyze_job_history.py's posture:
# a transient tunnel/RPC blip should not be read as a terminal transition).
IRIS_ATTEMPTS = 3
IRIS_BACKOFFS = (2, 5)


@dataclass
class JobStateSnapshot:
    """One observation of a job's authoritative lifecycle state."""

    job_id: str
    state: str  # friendly name, or "absent" when the controller has no record
    state_int: int | None
    error: str = ""
    exit_code: int | None = None
    failure_count: int | None = None
    preemption_count: int | None = None
    task_count: int | None = None
    completed_count: int | None = None
    task_state_counts: dict[str, int] = field(default_factory=dict)
    finished_at_ms: int | None = None
    source: str = ""  # "summary" | "query" | "absent"
    pods_alive: int | None = None  # kubectl cross-check, if run

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES or self.state == "absent"

    @property
    def is_running(self) -> bool:
        return self.state in RUNNING_STATES

    def verdict_line(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        parts = [f"[{ts}] {self.job_id} state={self.state}"]
        if self.state_int is not None:
            parts.append(f"({self.state_int})")
        if self.source:
            parts.append(f"src={self.source}")
        if self.pods_alive is not None:
            parts.append(f"pods={self.pods_alive}")
        if self.completed_count is not None and self.task_count is not None:
            parts.append(f"tasks={self.completed_count}/{self.task_count}")
        if self.exit_code is not None:
            parts.append(f"exit={self.exit_code}")
        if self.preemption_count:
            parts.append(f"preempts={self.preemption_count}")
        if self.error:
            parts.append(f"error={self.error!r}")
        return " ".join(parts)


# ---------- iris CLI helpers (authoritative state) ----------


def _run_iris(args: list[str], cluster: str, timeout: int = 180) -> subprocess.CompletedProcess:
    cmd = [IRIS_BIN, f"--cluster={cluster}", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def get_job_summary(job_id: str, cluster: str) -> JobStateSnapshot | None:
    """Authoritative primary: ``iris job summary <job_id> --json``.

    Returns a snapshot, or None on a (retryable) failure. A job the controller
    has never heard of yields an error we surface as None so the caller can fall
    back to the SQL/pod path and decide "absent".
    """
    last_err = ""
    for attempt in range(IRIS_ATTEMPTS):
        try:
            proc = _run_iris(["job", "summary", job_id, "--json"], cluster)
        except subprocess.TimeoutExpired:
            last_err = "timeout"
            if attempt < len(IRIS_BACKOFFS):
                time.sleep(IRIS_BACKOFFS[attempt])
            continue
        if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
            data = json.loads(proc.stdout)
            state = (data.get("state") or "unspecified").lower()
            return JobStateSnapshot(
                job_id=job_id,
                state=state,
                state_int=NAME_TO_INT.get(state),
                error=data.get("error", "") or "",
                exit_code=data.get("exit_code"),
                failure_count=data.get("failure_count"),
                preemption_count=data.get("preemption_count"),
                task_count=data.get("task_count"),
                completed_count=data.get("completed_count"),
                task_state_counts=data.get("task_state_counts", {}) or {},
                source="summary",
            )
        last_err = (proc.stderr or proc.stdout)[-400:]
        if attempt < len(IRIS_BACKOFFS):
            time.sleep(IRIS_BACKOFFS[attempt])
    print(f"  [summary] failed after {IRIS_ATTEMPTS} attempts: {last_err}", file=sys.stderr)
    return None


def get_job_state_via_query(job_id: str, cluster: str) -> JobStateSnapshot | None:
    """Fallback: ``iris query "SELECT state FROM jobs WHERE job_id=..."``.

    Lighter than ``summary`` (numeric state only). Returns None on failure or an
    empty result (no such job → caller treats as candidate for "absent").
    """
    sql = f"SELECT job_id, state FROM jobs WHERE job_id='{job_id}'"
    try:
        proc = _run_iris(["query", sql, "-f", "csv"], cluster)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    rows = [ln for ln in proc.stdout.strip().splitlines() if ln and "," in ln]
    # first line is the header "job_id,state"
    data_rows = [r for r in rows if not r.startswith("job_id,")]
    if not data_rows:
        return None
    try:
        _jid, state_s = data_rows[0].rsplit(",", 1)
        state_int = int(state_s)
    except (ValueError, IndexError):
        return None
    return JobStateSnapshot(
        job_id=job_id,
        state=STATE_NAMES.get(state_int, f"state_{state_int}"),
        state_int=state_int,
        source="query",
    )


# ---------- kubectl cross-check (the disappeared / 0-pods case) ----------


def count_live_pods(job_id: str) -> int | None:
    """Count cluster pods whose name carries the job's short name.

    Requires KUBECONFIG to point at the cluster (e.g. ~/.kube/coreweave-iris-gpu).
    Returns the pod count, or None if kubectl is unavailable / errors. iris job
    pods are named after the job's short id (last path segment); a vanished job
    has 0. This is the explicit cross-check for the case the content-watch
    missed: pods gone but no terminal log string ever printed.
    """
    short = job_id.rstrip("/").rsplit("/", 1)[-1]
    try:
        proc = subprocess.run(
            ["kubectl", "get", "pods", "-A", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    return sum(1 for ln in proc.stdout.splitlines() if short in ln)


# ---------- the authoritative single observation ----------


def get_job_state(job_id: str, cluster: str = DEFAULT_CLUSTER, check_pods: bool = False) -> JobStateSnapshot:
    """Return the current authoritative state of ``job_id``.

    Order: ``job summary --json`` (primary) → ``query`` (fallback) → if both say
    "no such job" AND (when ``check_pods``) the cluster has 0 matching pods, the
    job has disappeared → state="absent" (a TERMINAL signal). If the controller
    is simply unreachable we raise, so a transient outage is not misread as a
    terminal transition.
    """
    snap = get_job_summary(job_id, cluster)
    if snap is None:
        snap = get_job_state_via_query(job_id, cluster)

    pods = count_live_pods(job_id) if check_pods else None

    if snap is None:
        # Controller has no record (or both calls failed). Distinguish
        # "disappeared" from "controller unreachable" via the pod count.
        if pods == 0:
            return JobStateSnapshot(
                job_id=job_id, state="absent", state_int=None,
                source="absent", pods_alive=0,
                error="no controller record AND 0 pods on cluster (disappeared)",
            )
        raise RuntimeError(
            f"could not read state for {job_id}: iris summary+query both failed "
            f"and pod count is {pods} (controller unreachable, or job not yet placed)"
        )

    snap.pods_alive = pods
    return snap


# ---------- the watch loop ----------


def watch(
    job_id: str,
    cluster: str = DEFAULT_CLUSTER,
    interval: int = 60,
    check_pods: bool = True,
    max_polls: int | None = None,
) -> JobStateSnapshot:
    """Poll authoritative state until the job leaves RUNNING; return the terminal
    snapshot. Emits a line on EVERY state transition (and the first observation).
    """
    print(
        f"[watch] {job_id} on cluster={cluster} every {interval}s "
        f"(authoritative iris job-state poll; check_pods={check_pods})",
        file=sys.stderr,
    )
    prev_state: str | None = None
    polls = 0
    while True:
        polls += 1
        try:
            snap = get_job_state(job_id, cluster, check_pods=check_pods)
        except RuntimeError as e:
            # Transient: report, keep watching (do NOT treat as terminal).
            print(f"  [watch] transient read error: {e}", file=sys.stderr)
            snap = None
        if snap is not None:
            if snap.state != prev_state:
                print(snap.verdict_line(), flush=True)
                prev_state = snap.state
            if snap.is_terminal:
                print(f"[watch] TERMINAL: {snap.state}", file=sys.stderr)
                return snap
        if max_polls is not None and polls >= max_polls:
            print(f"[watch] reached max_polls={max_polls}; stopping", file=sys.stderr)
            return snap if snap is not None else JobStateSnapshot(
                job_id=job_id, state="unspecified", state_int=0, source="timeout"
            )
        time.sleep(interval)


_EXIT_FOR_STATE = {
    "succeeded": 0,
    "failed": 1,
    "killed": 1,
    "worker_failed": 1,
    "unschedulable": 1,
    "absent": 2,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_id", help="iris job id, e.g. /benjaminfeuer/rl-131k-cpdcp2r3")
    ap.add_argument("--cluster", default=DEFAULT_CLUSTER, help=f"iris cluster (default: {DEFAULT_CLUSTER}; use 'marin' for TPU)")
    ap.add_argument("--interval", type=int, default=60, help="poll interval seconds (default 60)")
    ap.add_argument("--once", action="store_true", help="print current state once and exit")
    ap.add_argument("--no-pods", action="store_true", help="skip the kubectl pod cross-check")
    ap.add_argument("--max-polls", type=int, default=None, help="stop after N polls even if still running")
    ap.add_argument("--json", action="store_true", help="emit the final snapshot as JSON")
    args = ap.parse_args()

    check_pods = not args.no_pods
    try:
        if args.once:
            snap = get_job_state(args.job_id, args.cluster, check_pods=check_pods)
            print(snap.verdict_line(), flush=True)
        else:
            snap = watch(args.job_id, args.cluster, interval=args.interval, check_pods=check_pods, max_polls=args.max_polls)
    except RuntimeError as e:
        print(f"[watch] ERROR: {e}", file=sys.stderr)
        return 3

    if args.json:
        from dataclasses import asdict
        print(json.dumps(asdict(snap), indent=2, default=str))

    return _EXIT_FOR_STATE.get(snap.state, 3 if not snap.is_terminal else 0)


if __name__ == "__main__":
    sys.exit(main())
