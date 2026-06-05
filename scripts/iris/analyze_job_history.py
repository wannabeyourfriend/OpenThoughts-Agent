#!/usr/bin/env python3
"""Pull the entire log history for an iris job and produce summary statistics.

The naive ``iris job logs <job> --tail --max-lines N`` truncates by line count,
not by time, and verbose Ray state-dumps + fd-monitor frames crowd out the
throughput emissions. This script paginates the log via fixed time windows
(``--since-ms`` + ``--no-tail``) and filters at the python level to only retain
lines we actually need (cycle boundaries, vLLM throughput emissions). The
filtered stream is cached to ``/tmp/iris_history_<job>.filtered.log`` so re-runs
are fast.

Three sections of stats are computed:
  §1 preemption count + time-to-preempt distribution
  §2 trace progress per cycle (from harbor GCS output)
  §3 serving throughput stats (full + warmup-excluded)

A markdown report is written to ``--output``; a JSON sidecar is written
alongside.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

IRIS_BIN = "/Users/benjaminfeuer/Documents/marin/.venv/bin/iris"
GCS_ROOT = "gs://marin-models-us/ot-agent"

# Iris log line: "[HH:MM:SS] task=<task> | <content>"
LINE_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\] task=(\S+) \| (.*)$")

# Cycle boundary on a given task: the tpu-inference-patch fires exactly once per
# attempt boot. We use task=/.../0 boundaries as the cycle marker.
PATCH_APPLIED_RE = re.compile(r"\[tpu-inference-patch\] APPLIED:")

# Rendezvous polling has an embedded epoch-seconds timestamp:
# "[start_vllm_iris_controller] Polling for rendezvous ... min written_at <epoch>)..."
RENDEZVOUS_RE = re.compile(r"min written_at (\d+)\)")

# Throughput line from vLLM's logger has an embedded MM-DD HH:MM:SS:
# "(APIServer pid=N) INFO 05-30 07:07:36 [loggers.py:271] Engine ...: Avg prompt
# throughput: X tokens/s, Avg generation throughput: Y tokens/s, Running: N reqs,
# Waiting: W reqs, ..."
THROUGHPUT_RE = re.compile(
    r"INFO (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) .*?"
    r"Avg prompt throughput:\s+([0-9.]+) tokens/s,\s+"
    r"Avg generation throughput:\s+([0-9.]+) tokens/s,\s+"
    r"Running:\s+(\d+) reqs,\s+Waiting:\s+(\d+) reqs"
)

# Preempt-edge markers (any of these on any task signal a cycle ending soon).
# We separately count cycles from APPLIED markers on task=0; these are
# diagnostic only, useful for cross-checking.
PREEMPT_EDGE_RE = re.compile(
    r"(raylet\) Raylet is terminated|EngineCore died|worker_lost_spec|Preempted by /[^/]+/)"
)

# Filter regex used when pulling logs from iris (keep only lines we care about).
# We INCLUDE the iris control-plane lines (without a [HH:MM:SS] prefix) so we can
# detect the "tunnel ready" boundary; we filter those out at line-prefix-parse
# time later.
KEEP_RE = re.compile(
    r"(\[tpu-inference-patch\] APPLIED|"
    r"start_vllm_iris_controller\] Polling for rendezvous|"
    r"raylet\) Raylet is terminated|"
    r"EngineCore died|"
    r"worker_lost_spec|"
    r"Preempted by /[^/]+/|"
    r"Avg generation throughput)"
)

# Spinner / progress lines we definitely want to drop fast (these dominate the
# raw log volume). The KEEP_RE above already excludes them implicitly, so this
# is belt-and-suspenders / not used at the moment.
DROP_SUFFIX = ("running agent...", "[fd-monitor]")


@dataclass
class Cycle:
    idx: int
    cycle_start: datetime
    cycle_end: datetime
    did_serve: bool
    time_to_first_serve_s: float | None
    duration_s: float
    serving_samples_in_cycle: int = 0
    non_empty_trials_in_cycle: int = 0


@dataclass
class ServingSample:
    ts: datetime
    prompt_tps: float
    gen_tps: float
    running: int
    waiting: int
    cycle_idx: int = -1
    elapsed_in_cycle_s: float = 0.0


@dataclass
class TrialStatus:
    trial_name: str
    has_trajectory: bool
    trajectory_mtime: datetime | None
    trajectory_size: int


@dataclass
class JobAnalysis:
    job_id: str
    job_name: str
    submitted_at: datetime
    started_at: datetime
    current_time: datetime
    total_runtime_s: float
    iris_preemption_count: int | None
    state: int

    # §1
    cycles: list[Cycle] = field(default_factory=list)
    preempt_count_from_log: int = 0

    # §2
    total_trial_dirs: int = 0
    non_empty_trials: int = 0
    harbor_n_completed: int = 0
    harbor_n_errored: int = 0
    harbor_n_running: int = 0
    harbor_n_pending: int = 0
    harbor_exception_stats: dict[str, int] = field(default_factory=dict)
    harbor_started_at: str | None = None
    harbor_updated_at: str | None = None
    harbor_n_total_trials: int = 0

    # §3
    serving_samples: list[ServingSample] = field(default_factory=list)


# ---------- Iris CLI / subprocess helpers ----------


def run_iris_query(sql: str) -> list[dict[str, str]]:
    """Run ``iris query`` and parse CSV. iris prints I-level logs to stderr."""
    cmd = [IRIS_BIN, "--cluster", "marin", "query", sql, "-f", "csv"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [line for line in proc.stdout.strip().splitlines() if line]
    if not lines:
        return []
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","), strict=False)) for line in lines[1:]]


def get_job_metadata(job_id: str) -> dict[str, int | str]:
    sql = (
        "SELECT job_id, submitted_at_ms, started_at_ms, state "
        f"FROM jobs WHERE job_id='{job_id}'"
    )
    rows = run_iris_query(sql)
    if not rows:
        raise RuntimeError(f"no job row found for {job_id}")
    row = rows[0]
    return {
        "job_id": row["job_id"],
        "submitted_at_ms": int(row["submitted_at_ms"]),
        "started_at_ms": int(row["started_at_ms"]),
        "state": int(row["state"]),
    }


def get_job_summary_preemptions(job_id: str) -> int | None:
    cmd = [IRIS_BIN, "--cluster", "marin", "job", "summary", job_id]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # "State: running  exit=0  failures=0  preemptions=37"
    m = re.search(r"preemptions=(\d+)", proc.stdout)
    if not m:
        return None
    return int(m.group(1))


# ---------- Iris log paging ----------


def fetch_filtered_log(
    job_id: str, since_ms: int, max_lines: int = 200_000
) -> tuple[list[str], str | None, int, int]:
    """Fetch a page of logs from iris, returning only lines matching KEEP_RE.

    Returns (kept_lines, last_line_seen, total_lines_read, used_max). ``used_max``
    is the page-size that actually succeeded (== max_lines unless we had to
    halve to dodge a server timeout). ``last_line_seen`` is the very last raw
    line emitted by iris (regardless of filter); we use its ``[HH:MM:SS]``
    prefix to advance the page cursor.

    Halves ``max_lines`` and retries on RPC timeout / non-zero exit. Iris's log
    server returns ``StatsError: Request timed out`` when the server-side
    aggregation for a big page exceeds its deadline; a smaller page is
    materially cheaper to assemble and will normally succeed.
    """
    tries = 0
    cur_max = max_lines
    last_err: str = ""
    while tries < 4:
        tries += 1
        cmd = [
            IRIS_BIN,
            "--cluster",
            "marin",
            "job",
            "logs",
            job_id,
            "--since-ms",
            str(since_ms),
            "--max-lines",
            str(cur_max),
            "--no-tail",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        kept: list[str] = []
        last_line: str | None = None
        total = 0
        assert proc.stdout is not None
        for line in proc.stdout:
            total += 1
            if LINE_RE.match(line):
                last_line = line.rstrip("\n")
            if KEEP_RE.search(line):
                kept.append(line.rstrip("\n"))
        _, err = proc.communicate()
        if proc.returncode == 0:
            return kept, last_line, total, cur_max
        last_err = err
        if "timed out" in err.lower() or "deadline" in err.lower():
            print(
                f"  [retry] iris timed out at max_lines={cur_max}; halving",
                file=sys.stderr,
            )
            cur_max = max(cur_max // 2, 20_000)
            continue
        # other error — propagate
        break
    raise RuntimeError(
        f"iris job logs returned exit {proc.returncode} after {tries} tries; last err:\n{last_err[-500:]}"
    )


def paginate_full_log(
    job_id: str,
    submitted_at_ms: int,
    end_ms: int,
    cache_path: Path,
    refresh: bool,
    page_max_lines: int = 200_000,
) -> list[str]:
    """Paginate the entire job log, caching the filtered stream to disk.

    We start at ``submitted_at_ms`` and ask iris for up to ``page_max_lines``
    lines from that point onward. We then advance the cursor to the wall-clock
    timestamp of the last line in the page (minus a 1s safety overlap, to make
    sure we don't miss anything that shares the second). We stop when:
      (a) cursor reaches ``end_ms``, OR
      (b) the cursor failed to advance from the previous page (we caught up to
          head and the next ``--since-ms`` will just return the same trailing
          lines).

    The iris log timestamp has only HH:MM:SS, so we maintain a UTC day counter
    based on monotonic comparison: when HH:MM:SS goes backwards by > 12 hours
    relative to the previous page-end, increment the day.
    """
    if cache_path.exists() and not refresh:
        print(f"[cache] reusing {cache_path}", file=sys.stderr)
        return cache_path.read_text().splitlines()

    print(f"[fetch] paginating logs from {submitted_at_ms} to {end_ms}", file=sys.stderr)
    seen: set[str] = set()
    out: list[str] = []
    cursor_ms = submitted_at_ms
    page_idx = 0
    # Sticky page size: once a page had to halve to succeed, keep using that
    # smaller size for subsequent pages. Saves wasted 30s timeouts.
    sticky_max = page_max_lines
    while cursor_ms < end_ms:
        page_idx += 1
        page_lines, last_line, total, used_max = fetch_filtered_log(
            job_id, cursor_ms, max_lines=sticky_max
        )
        if used_max < sticky_max:
            print(
                f"  [sticky] reducing page_max_lines from {sticky_max} to {used_max}",
                file=sys.stderr,
            )
            sticky_max = used_max
        new = 0
        for line in page_lines:
            if line not in seen:
                seen.add(line)
                out.append(line)
                new += 1
        print(
            f"  page {page_idx}: cursor={cursor_ms}, raw={total}, kept={len(page_lines)}, new={new}",
            file=sys.stderr,
        )
        if last_line is None:
            print(f"  page {page_idx}: empty page, stopping", file=sys.stderr)
            break
        m = LINE_RE.match(last_line)
        if not m:
            print(f"  page {page_idx}: unparseable last line, stopping", file=sys.stderr)
            break
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Reconstruct date by anchoring to cursor's own date and walking forward
        # day-by-day while the proposed timestamp falls before cursor_ms+1s.
        cursor_dt = datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc)
        candidate = datetime(
            cursor_dt.year, cursor_dt.month, cursor_dt.day, hh, mm, ss,
            tzinfo=timezone.utc,
        )
        # advance by whole days until candidate >= cursor_dt (so wall-clock
        # rollovers within a page work)
        while candidate < cursor_dt:
            candidate = candidate + timedelta(days=1)
        new_cursor_dt = candidate - timedelta(seconds=1)
        new_cursor_ms = int(new_cursor_dt.timestamp() * 1000)
        if new_cursor_ms <= cursor_ms:
            # No progress => we've caught up to head
            print(
                f"  page {page_idx}: cursor didn't advance ({new_cursor_ms} <= {cursor_ms}); stopping",
                file=sys.stderr,
            )
            break
        # Cap so we don't go past `now`
        if new_cursor_ms > end_ms:
            cursor_ms = end_ms
        else:
            cursor_ms = new_cursor_ms
    print(f"[fetch] total kept lines: {len(out)}", file=sys.stderr)
    cache_path.write_text("\n".join(out) + "\n")
    return out


# ---------- Log line parsing ----------


def parse_log_lines(
    filtered_lines: list[str], submitted_at: datetime
) -> tuple[list[tuple[datetime, str, str]], list[ServingSample]]:
    """Parse filtered lines into (timestamp, task, content) tuples + serving samples.

    The iris ``[HH:MM:SS]`` outer prefix has no date, so we walk lines in order
    and bump a day-counter whenever HH:MM:SS goes backwards. The starting day is
    ``submitted_at``'s UTC date.

    For lines containing the vLLM throughput emission, we use the embedded
    ``MM-DD HH:MM:SS`` as the authoritative timestamp (still year-less but
    dated).

    For lines containing the rendezvous-polling ``min written_at <epoch_s>``, we
    use that epoch.
    """
    parsed: list[tuple[datetime, str, str]] = []
    samples: list[ServingSample] = []
    cur_date = submitted_at.astimezone(timezone.utc).date()
    last_hms: tuple[int, int, int] | None = None

    for line in filtered_lines:
        m = LINE_RE.match(line)
        if not m:
            continue
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        task = m.group(4)
        content = m.group(5)

        # day-rollover via monotonic HH:MM:SS
        hms = (hh, mm, ss)
        if last_hms is not None and hms < last_hms:
            # going backwards by > 12 hours => assume day-rollover. The iris log
            # stream isn't perfectly ordered task-to-task; small backward jumps
            # within a minute are normal. So only roll the day if we went
            # backward by a lot.
            secs_now = hh * 3600 + mm * 60 + ss
            secs_prev = last_hms[0] * 3600 + last_hms[1] * 60 + last_hms[2]
            if secs_prev - secs_now > 12 * 3600:
                cur_date = cur_date + timedelta(days=1)
        last_hms = hms

        outer_ts = datetime(
            cur_date.year,
            cur_date.month,
            cur_date.day,
            hh,
            mm,
            ss,
            tzinfo=timezone.utc,
        )

        # Prefer embedded epoch (rendezvous lines)
        ts = outer_ts
        rm = RENDEZVOUS_RE.search(content)
        if rm:
            ts = datetime.fromtimestamp(int(rm.group(1)), tz=timezone.utc)

        tm = THROUGHPUT_RE.search(content)
        if tm:
            # Use embedded MM-DD HH:MM:SS (no year). Anchor year to outer_ts.year
            # to disambiguate at year boundaries; that's the right year for both
            # of these jobs.
            mo, dy = int(tm.group(1)), int(tm.group(2))
            th, tmin, tsec = int(tm.group(3)), int(tm.group(4)), int(tm.group(5))
            try:
                ts = datetime(outer_ts.year, mo, dy, th, tmin, tsec, tzinfo=timezone.utc)
            except ValueError:
                ts = outer_ts
            samples.append(
                ServingSample(
                    ts=ts,
                    prompt_tps=float(tm.group(6)),
                    gen_tps=float(tm.group(7)),
                    running=int(tm.group(8)),
                    waiting=int(tm.group(9)),
                )
            )

        parsed.append((ts, task, content))

    parsed.sort(key=lambda x: x[0])
    samples.sort(key=lambda s: s.ts)
    return parsed, samples


def build_cycles(
    job_id: str,
    parsed: list[tuple[datetime, str, str]],
    samples: list[ServingSample],
    end_time: datetime,
) -> list[Cycle]:
    """Build cycle list from APPLIED markers on task=0.

    Each task=0 'tpu-inference-patch] APPLIED:' line marks the boot of a fresh
    attempt on rank 0. There are several APPLIED lines per boot (one per patch
    applied); we take the FIRST per boot via clustering by proximity.
    """
    task0 = f"{job_id}/0"
    cycle_starts: list[datetime] = []
    last_boundary: datetime | None = None
    for ts, task, content in parsed:
        if task != task0:
            continue
        if not PATCH_APPLIED_RE.search(content):
            continue
        # cluster: only count the first APPLIED line within a 60s window
        if last_boundary is None or (ts - last_boundary).total_seconds() > 60:
            cycle_starts.append(ts)
        last_boundary = ts

    cycles: list[Cycle] = []
    for i, start in enumerate(cycle_starts):
        end = cycle_starts[i + 1] if i + 1 < len(cycle_starts) else end_time
        duration_s = (end - start).total_seconds()
        # serving samples in this cycle
        in_cycle = [s for s in samples if start <= s.ts < end]
        did_serve = len(in_cycle) > 0
        first_serve = None
        if did_serve:
            first_serve = (in_cycle[0].ts - start).total_seconds()
        cycles.append(
            Cycle(
                idx=i,
                cycle_start=start,
                cycle_end=end,
                did_serve=did_serve,
                time_to_first_serve_s=first_serve,
                duration_s=duration_s,
                serving_samples_in_cycle=len(in_cycle),
            )
        )

    # tag samples with cycle_idx + elapsed_in_cycle
    for sample in samples:
        for cycle in cycles:
            if cycle.cycle_start <= sample.ts < cycle.cycle_end:
                sample.cycle_idx = cycle.idx
                sample.elapsed_in_cycle_s = (sample.ts - cycle.cycle_start).total_seconds()
                break

    return cycles


# ---------- GCS / harbor trial parsing ----------


def list_trial_trajectories(job_name: str) -> list[TrialStatus]:
    """Return TrialStatus for every trial dir (with or without trajectory.json).

    We do this in two passes:
      (a) list trial dirs (cheap, one listing)
      (b) list trajectory.json files with mtime+size (one recursive ls)
    Then we join.
    """
    root = f"{GCS_ROOT}/{job_name}/{job_name}"

    # (a) all trial dirs. Tolerate the dir not existing — that happens when
    # the job has never reached SERVING + harbor never wrote a single trial
    # (e.g., preempt-storm before first compile finishes). Treat as zero
    # trials.
    cmd = ["gsutil", "ls", f"{root}/"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if "matched no objects" in (proc.stderr or "").lower():
            return []
        # Any other gsutil failure (auth, network) — surface it.
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )
    trial_dirs: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.endswith("/"):
            continue
        if line.endswith(f"/{job_name}/"):
            continue  # root self
        trial_name = line.rstrip("/").rsplit("/", 1)[-1]
        if trial_name.startswith("_"):
            continue
        trial_dirs.append(trial_name)

    # (b) trajectory.json files
    cmd = ["gsutil", "ls", "-l", f"{root}/**/agent/trajectory.json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    traj_info: dict[str, tuple[datetime, int]] = {}
    # gsutil ls -l format: "<size> <YYYY-MM-DDTHH:MM:SSZ> <gs://...>"
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[-1].startswith("gs://"):
            continue
        size_s, mtime_s, path = parts[0], parts[1], parts[-1]
        try:
            size = int(size_s)
        except ValueError:
            continue
        try:
            mtime = datetime.strptime(mtime_s, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        # path like ../<trial_name>/agent/trajectory.json
        parts2 = path.split("/")
        if len(parts2) < 3:
            continue
        # find 'agent' index
        try:
            agent_idx = parts2.index("agent")
        except ValueError:
            continue
        trial_name = parts2[agent_idx - 1]
        traj_info[trial_name] = (mtime, size)

    out: list[TrialStatus] = []
    for name in trial_dirs:
        if name in traj_info:
            mtime, size = traj_info[name]
            out.append(TrialStatus(name, True, mtime, size))
        else:
            out.append(TrialStatus(name, False, None, 0))
    return out


def fetch_harbor_result(job_name: str) -> dict | None:
    """Cat <root>/<job_name>/<job_name>/result.json."""
    path = f"{GCS_ROOT}/{job_name}/{job_name}/result.json"
    proc = subprocess.run(
        ["gsutil", "cat", path], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


# ---------- Stats helpers ----------


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def stats_dict(values: list[float]) -> dict[str, float]:
    if not values:
        return {k: float("nan") for k in ("n", "mean", "median", "p10", "p25", "p50", "p75", "p90", "p99", "min", "max")}
    return {
        "n": float(len(values)),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": pct(values, 0.10),
        "p25": pct(values, 0.25),
        "p50": pct(values, 0.50),
        "p75": pct(values, 0.75),
        "p90": pct(values, 0.90),
        "p99": pct(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


# ---------- Report rendering ----------


def fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "n/a"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{100*x:.1f}%"


def render_markdown(a: JobAnalysis, warmup_seconds: float) -> str:
    lines: list[str] = []
    lines.append(f"# Iris job history analysis: `{a.job_id}`")
    lines.append("")
    lines.append(f"- **submitted_at**: {a.submitted_at.isoformat()}")
    lines.append(f"- **started_at**: {a.started_at.isoformat()}")
    lines.append(f"- **current_time** (analysis): {a.current_time.isoformat()}")
    lines.append(f"- **total_runtime**: {fmt_duration(a.total_runtime_s)}")
    lines.append(f"- **state**: {a.state} (3=RUNNING)")
    lines.append("")

    # §1
    lines.append("## §1 Preemption analysis")
    lines.append("")
    n_cycles = len(a.cycles)
    n_serving = sum(1 for c in a.cycles if c.did_serve)
    n_dead_in_compile = n_cycles - n_serving
    lines.append(
        f"- **cycles detected (log-derived)**: {n_cycles} "
        f"(=> preempts from log = {a.preempt_count_from_log})"
    )
    lines.append(f"- **iris job summary preemptions=**: {a.iris_preemption_count}")
    if a.iris_preemption_count is not None and a.iris_preemption_count != a.preempt_count_from_log:
        lines.append(
            f"  - **discrepancy**: log shows {a.preempt_count_from_log}, "
            f"iris shows {a.iris_preemption_count} "
            f"(diff={a.preempt_count_from_log - a.iris_preemption_count})"
        )
    lines.append(f"- **cycles that reached SERVING**: {n_serving}")
    lines.append(f"- **cycles that died in compile (no throughput emission)**: {n_dead_in_compile}")
    lines.append("")
    serving_durations = [c.duration_s for c in a.cycles if c.did_serve]
    first_serve = [c.time_to_first_serve_s for c in a.cycles if c.did_serve and c.time_to_first_serve_s is not None]
    if serving_durations:
        sd_stats = stats_dict(serving_durations)
        lines.append("### Serving-cycle survival time (cycles that did_serve)")
        lines.append("")
        lines.append(f"- mean: {fmt_duration(sd_stats['mean'])}")
        lines.append(f"- median: {fmt_duration(sd_stats['median'])}")
        lines.append(f"- p25 / p75: {fmt_duration(sd_stats['p25'])} / {fmt_duration(sd_stats['p75'])}")
        lines.append(f"- max: {fmt_duration(sd_stats['max'])}")
        lines.append("")
    if first_serve:
        fs_stats = stats_dict(first_serve)
        lines.append("### Time-to-first-serve (cold compile cost)")
        lines.append("")
        lines.append(f"- mean: {fmt_duration(fs_stats['mean'])}")
        lines.append(f"- median: {fmt_duration(fs_stats['median'])}")
        lines.append("")

    # cycle table (top 30 + bottom 5 if huge)
    lines.append("### Per-cycle table")
    lines.append("")
    lines.append("| idx | cycle_start (UTC) | duration | did_serve | t_first_serve | samples | trials_finalized | cumulative_trials |")
    lines.append("|-----|--------------------|----------|-----------|---------------|---------|------------------|-------------------|")
    cum_trials = 0
    for c in a.cycles:
        cum_trials += c.non_empty_trials_in_cycle
        tfs = fmt_duration(c.time_to_first_serve_s) if c.time_to_first_serve_s is not None else "—"
        lines.append(
            f"| {c.idx} | {c.cycle_start.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{fmt_duration(c.duration_s)} | {c.did_serve} | {tfs} | "
            f"{c.serving_samples_in_cycle} | {c.non_empty_trials_in_cycle} | {cum_trials} |"
        )
    lines.append("")

    # §2
    lines.append("## §2 Trace progress")
    lines.append("")
    lines.append(f"- **harbor n_total_trials**: {a.harbor_n_total_trials}")
    lines.append(f"- **harbor n_completed_trials**: {a.harbor_n_completed} (includes errored)")
    lines.append(f"- **harbor n_errored_trials**: {a.harbor_n_errored}")
    lines.append(f"- **harbor n_running_trials**: {a.harbor_n_running}")
    lines.append(f"- **harbor n_pending_trials**: {a.harbor_n_pending}")
    lines.append(f"- **harbor started_at**: {a.harbor_started_at}")
    lines.append(f"- **harbor updated_at**: {a.harbor_updated_at}")
    lines.append("")
    lines.append(f"- **trial dirs on GCS**: {a.total_trial_dirs}")
    lines.append(f"- **non-empty trials (trajectory.json exists)**: {a.non_empty_trials}")
    empty = a.total_trial_dirs - a.non_empty_trials
    empty_rate = (empty / a.total_trial_dirs) if a.total_trial_dirs else float("nan")
    lines.append(f"- **empty trials (no trajectory.json)**: {empty}")
    lines.append(f"- **empty-rate**: {fmt_pct(empty_rate)}")
    lines.append("")
    if a.harbor_exception_stats:
        lines.append("### Harbor exception breakdown")
        lines.append("")
        for k, v in sorted(a.harbor_exception_stats.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {k}: {v}")
        lines.append("")

    # §3
    lines.append("## §3 Serving stats")
    lines.append("")
    all_samples = a.serving_samples
    lines.append(f"- **total serving samples emitted**: {len(all_samples)}")
    lines.append("")
    if all_samples:
        gen_full = [s.gen_tps for s in all_samples]
        prompt_full = [s.prompt_tps for s in all_samples]
        running_full = [float(s.running) for s in all_samples]
        waiting_full = [float(s.waiting) for s in all_samples]
        sat_full = sum(1 for s in all_samples if s.waiting > 0) / len(all_samples)

        gen_warm = [s.gen_tps for s in all_samples if s.elapsed_in_cycle_s >= warmup_seconds]
        prompt_warm = [s.prompt_tps for s in all_samples if s.elapsed_in_cycle_s >= warmup_seconds]
        running_warm = [float(s.running) for s in all_samples if s.elapsed_in_cycle_s >= warmup_seconds]
        waiting_warm = [float(s.waiting) for s in all_samples if s.elapsed_in_cycle_s >= warmup_seconds]
        sat_warm = (
            sum(1 for s in all_samples if s.elapsed_in_cycle_s >= warmup_seconds and s.waiting > 0)
            / max(len(gen_warm), 1)
        )

        def render(label: str, gen: list[float], prompt: list[float], running: list[float], waiting: list[float], sat: float) -> None:
            lines.append(f"### {label}")
            lines.append("")
            g = stats_dict(gen)
            p = stats_dict(prompt)
            r = stats_dict(running)
            w = stats_dict(waiting)
            lines.append(f"- samples: {int(g['n'])}")
            lines.append(f"- gen_tps: mean={g['mean']:.1f}, median={g['median']:.1f}, "
                         f"p10={g['p10']:.1f}, p50={g['p50']:.1f}, p90={g['p90']:.1f}, "
                         f"p99={g['p99']:.1f}, peak={g['max']:.1f}, min={g['min']:.1f}")
            lines.append(f"- prompt_tps: mean={p['mean']:.1f}, peak={p['max']:.1f}")
            lines.append(f"- running: mean={r['mean']:.2f}, peak={int(r['max'])}")
            lines.append(f"- waiting: mean={w['mean']:.2f}, peak={int(w['max'])}")
            lines.append(f"- saturation rate (waiting>0): {fmt_pct(sat)}")
            lines.append("")

        render("Full (all samples)", gen_full, prompt_full, running_full, waiting_full, sat_full)
        render(
            f"Warmup-excluded (elapsed_in_cycle >= {warmup_seconds:.0f}s)",
            gen_warm, prompt_warm, running_warm, waiting_warm, sat_warm,
        )

        if gen_full and gen_warm:
            full_mean = statistics.fmean(gen_full)
            warm_mean = statistics.fmean(gen_warm)
            diff_pct = abs(full_mean - warm_mean) / full_mean if full_mean else 0
            if diff_pct < 0.05:
                lines.append(
                    f"_Warmup exclusion changed gen_tps mean by {fmt_pct(diff_pct)} — "
                    f"warmup did not meaningfully bias the all-samples stats._"
                )
            else:
                lines.append(
                    f"_Warmup exclusion changed gen_tps mean by {fmt_pct(diff_pct)} "
                    f"(full={full_mean:.1f}, warm={warm_mean:.1f}) — the first "
                    f"{int(warmup_seconds)}s of each cycle pulls the mean down._"
                )
            lines.append("")

    # bottom-line
    lines.append("## Bottom line")
    lines.append("")
    productive = a.non_empty_trials
    pre = a.preempt_count_from_log
    runtime_h = a.total_runtime_s / 3600.0
    lines.append(
        f"Over **{runtime_h:.1f} hours** of runtime, the job survived **{pre} preempts** "
        f"({n_serving}/{n_cycles} cycles reached the serving phase) and produced "
        f"**{productive} non-empty trial trajectories** out of "
        f"{a.total_trial_dirs} trial dirs "
        f"(empty-rate {fmt_pct(empty_rate)})."
    )
    if a.iris_preemption_count is not None and a.iris_preemption_count != pre:
        lines.append(
            f"Note: iris reports preemptions={a.iris_preemption_count}, log shows "
            f"{pre}. Pick the more trustworthy depending on the discrepancy "
            f"magnitude (log can miss a boundary if the cycle aborted before "
            f"tpu-inference-patch fired)."
        )
    lines.append("")
    return "\n".join(lines)


def analysis_to_json(a: JobAnalysis) -> dict:
    """Serialise JobAnalysis to a JSON-safe dict (datetimes as ISO strings)."""

    def _conv(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return {k: _conv(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_conv(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _conv(v) for k, v in obj.items()}
        return obj

    return {
        "job_id": a.job_id,
        "job_name": a.job_name,
        "submitted_at": a.submitted_at.isoformat(),
        "started_at": a.started_at.isoformat(),
        "current_time": a.current_time.isoformat(),
        "total_runtime_s": a.total_runtime_s,
        "iris_preemption_count": a.iris_preemption_count,
        "preempt_count_from_log": a.preempt_count_from_log,
        "state": a.state,
        "cycles": [
            {
                "idx": c.idx,
                "cycle_start": c.cycle_start.isoformat(),
                "cycle_end": c.cycle_end.isoformat(),
                "duration_s": c.duration_s,
                "did_serve": c.did_serve,
                "time_to_first_serve_s": c.time_to_first_serve_s,
                "serving_samples_in_cycle": c.serving_samples_in_cycle,
                "non_empty_trials_in_cycle": c.non_empty_trials_in_cycle,
            }
            for c in a.cycles
        ],
        "total_trial_dirs": a.total_trial_dirs,
        "non_empty_trials": a.non_empty_trials,
        "harbor_n_completed": a.harbor_n_completed,
        "harbor_n_errored": a.harbor_n_errored,
        "harbor_n_running": a.harbor_n_running,
        "harbor_n_pending": a.harbor_n_pending,
        "harbor_n_total_trials": a.harbor_n_total_trials,
        "harbor_exception_stats": a.harbor_exception_stats,
        "harbor_started_at": a.harbor_started_at,
        "harbor_updated_at": a.harbor_updated_at,
        "serving_samples_count": len(a.serving_samples),
        # Don't dump the full sample stream; keep summary
        "serving_summary": _summarise_samples(a.serving_samples),
    }


def _summarise_samples(samples: list[ServingSample]) -> dict:
    if not samples:
        return {}
    return {
        "gen_tps": stats_dict([s.gen_tps for s in samples]),
        "prompt_tps": stats_dict([s.prompt_tps for s in samples]),
        "running": stats_dict([float(s.running) for s in samples]),
        "waiting": stats_dict([float(s.waiting) for s in samples]),
        "saturation_rate": sum(1 for s in samples if s.waiting > 0) / len(samples),
    }


# ---------- Main ----------


def analyze(job_id: str, output: Path, refresh: bool, warmup_seconds: float) -> JobAnalysis:
    job_name = job_id.rsplit("/", 1)[-1]
    meta = get_job_metadata(job_id)
    submitted_at = datetime.fromtimestamp(meta["submitted_at_ms"] / 1000, tz=timezone.utc)
    started_at = datetime.fromtimestamp(meta["started_at_ms"] / 1000, tz=timezone.utc)
    state = meta["state"]
    now = datetime.now(timezone.utc)
    total_runtime = (now - submitted_at).total_seconds()
    print(
        f"[{job_name}] submitted={submitted_at.isoformat()}, "
        f"runtime={total_runtime/3600:.1f}h, state={state}",
        file=sys.stderr,
    )

    iris_preempts = get_job_summary_preemptions(job_id)
    print(f"[{job_name}] iris preemptions={iris_preempts}", file=sys.stderr)

    cache_path = Path(f"/tmp/iris_history_{job_name}.filtered.log")
    end_ms = int(now.timestamp() * 1000)
    filtered = paginate_full_log(
        job_id,
        meta["submitted_at_ms"],
        end_ms,
        cache_path=cache_path,
        refresh=refresh,
    )
    parsed, samples = parse_log_lines(filtered, submitted_at)
    print(
        f"[{job_name}] parsed lines={len(parsed)}, samples={len(samples)}",
        file=sys.stderr,
    )

    cycles = build_cycles(job_id, parsed, samples, end_time=now)
    print(f"[{job_name}] cycles detected: {len(cycles)}", file=sys.stderr)

    # GCS trial inspection
    trials = list_trial_trajectories(job_name)
    non_empty = [t for t in trials if t.has_trajectory]
    print(
        f"[{job_name}] trial dirs={len(trials)}, non_empty={len(non_empty)}",
        file=sys.stderr,
    )

    # Assign non-empty trials to cycles by trajectory mtime
    for c in cycles:
        c.non_empty_trials_in_cycle = sum(
            1
            for t in non_empty
            if t.trajectory_mtime is not None
            and c.cycle_start <= t.trajectory_mtime < c.cycle_end
        )

    # Harbor result.json
    harbor = fetch_harbor_result(job_name)
    a = JobAnalysis(
        job_id=job_id,
        job_name=job_name,
        submitted_at=submitted_at,
        started_at=started_at,
        current_time=now,
        total_runtime_s=total_runtime,
        iris_preemption_count=iris_preempts,
        state=state,
        cycles=cycles,
        preempt_count_from_log=max(len(cycles) - 1, 0),
        serving_samples=samples,
        total_trial_dirs=len(trials),
        non_empty_trials=len(non_empty),
    )
    if harbor:
        stats = harbor.get("stats", {})
        a.harbor_n_completed = stats.get("n_completed_trials", 0)
        a.harbor_n_errored = stats.get("n_errored_trials", 0)
        a.harbor_n_running = stats.get("n_running_trials", 0)
        a.harbor_n_pending = stats.get("n_pending_trials", 0)
        a.harbor_n_total_trials = harbor.get("n_total_trials", 0)
        a.harbor_started_at = harbor.get("started_at")
        a.harbor_updated_at = harbor.get("updated_at")
        evals = stats.get("evals", {})
        for _ev_name, ev_data in evals.items():
            es = ev_data.get("exception_stats", {})
            for k, vs in es.items():
                a.harbor_exception_stats[k] = a.harbor_exception_stats.get(k, 0) + len(vs)

    return a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_id", help="iris job id, e.g. /benjaminfeuer/qwen122b-q12-v1-...")
    ap.add_argument("--output", required=True, help="markdown report output path")
    ap.add_argument("--refresh", action="store_true", help="ignore cached log")
    ap.add_argument("--warmup-seconds", type=float, default=180.0)
    ap.add_argument(
        "--gsutil-sample",
        type=int,
        default=0,
        help="(unused, accepted for compat) cap GCS trial inspection",
    )
    args = ap.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    a = analyze(args.job_id, output, args.refresh, args.warmup_seconds)

    md = render_markdown(a, warmup_seconds=args.warmup_seconds)
    output.write_text(md)

    json_path = output.with_suffix(output.suffix + ".json")
    json_path.write_text(json.dumps(analysis_to_json(a), indent=2))

    print(f"\nReport written to: {output}", file=sys.stderr)
    print(f"JSON sidecar:     {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
