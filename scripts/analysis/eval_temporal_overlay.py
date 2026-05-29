#!/usr/bin/env python3
"""Overlay eval-time reward markers on the RL-time temporal axis.

Extends ``temporal_trace_analysis.py``: that script bins RL training
traces by timestamp; this one additionally takes one or more
post-training eval trace datasets (one per RL checkpoint) and plots
their summary rewards as points on the same time axis, anchored at the
checkpoint's wall-clock time.

Answers research question (3) at the time-series level: **do behavioral
changes observed during RL persist when the policy is run on held-out
eval tasks?** If the eval markers track the RL reward curve, the gains
generalize. If they plateau below or diverge, the gains were either
overfitting or non-stationary.

Usage:
    python -m scripts.analysis.eval_temporal_overlay \\
        --rl-traces      penfever/rl-training-traces \\
        --eval-traces    penfever/eval@step-500:2026-05-25T12:00 \\
        --eval-traces    penfever/eval@step-1000:2026-05-26T18:30 \\
        --bin-hours      4 \\
        --output         /path/temporal_overlay.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.utils import (  # noqa: E402
    Trace,
    load_traces,
    mean_reward_per_trial,
)


def _floor_to_bin(dt: datetime, bin_size: timedelta, origin: datetime) -> datetime:
    """Floor *dt* into a bin of size *bin_size* relative to *origin*."""
    delta = dt - origin
    bin_index = int(delta.total_seconds() // bin_size.total_seconds())
    return origin + bin_size * bin_index


def _bin_rl_reward(
    traces: List[Trace], bin_hours: float
) -> Tuple[List[datetime], List[float]]:
    """Bin RL traces by date, return parallel (centers, mean_reward) arrays."""
    dated = [t for t in traces if t.date is not None]
    if not dated:
        return [], []
    origin = min(t.date for t in dated)
    bin_size = timedelta(hours=bin_hours)
    buckets: Dict[datetime, List[float]] = defaultdict(list)
    for t in dated:
        if t.reward is None:
            continue
        b = _floor_to_bin(t.date, bin_size, origin)
        buckets[b].append(t.reward)
    centers = sorted(buckets)
    means = [sum(buckets[c]) / len(buckets[c]) for c in centers]
    # Shift to mid-bin for plotting.
    centers = [c + bin_size / 2 for c in centers]
    return centers, means


def _parse_eval_spec(spec: str) -> Tuple[str, Optional[datetime], Optional[str]]:
    """Parse ``<source>[@<label>][:<iso-timestamp>]``.

    Examples:
      ``penfever/foo``                  → (source, None, None)
      ``penfever/foo@step-500``         → (source, None, "step-500")
      ``penfever/foo:2026-05-25T12:00`` → (source, datetime, None)
      ``penfever/foo@step-500:2026-05-25T12:00`` → all three
    """
    label: Optional[str] = None
    ts: Optional[datetime] = None
    # Split off timestamp last (search for last ":" that looks like an ISO date)
    # ISO timestamps have ":" embedded, so just split on the LAST ":" before any 'T'.
    # We use a simpler convention: timestamps must be tagged with "@label:ts" or just
    # ":ts" at the end. ts is everything after the last colon if it parses as datetime.
    if ":" in spec:
        head, _, tail = spec.rpartition(":")
        # Tail could be HH:MM (still part of ISO) — try a few splits.
        for candidate_tail, candidate_head in [
            (tail, head),
            (f"{head.rpartition(':')[2]}:{tail}", head.rpartition(":")[0]) if ":" in head else (tail, head),
        ]:
            try:
                ts = datetime.fromisoformat(candidate_tail)
                head = candidate_head
                break
            except (TypeError, ValueError):
                ts = None
        else:
            head = spec
    else:
        head = spec
    if "@" in head:
        source, _, label = head.partition("@")
    else:
        source = head
    return source, ts, label


def _eval_marker(spec: str, max_rows: Optional[int]) -> Dict[str, Any]:
    source, ts, label = _parse_eval_spec(spec)
    traces = load_traces(source, max_rows=max_rows)
    mean_r = mean_reward_per_trial([t.raw for t in traces])
    return {
        "spec": spec,
        "source": source,
        "label": label or source,
        "timestamp": ts,
        "n": len(traces),
        "mean_reward": mean_r,
    }


def _plot(rl_centers, rl_means, eval_markers, output_path: Path, bin_hours: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval-temporal-overlay] matplotlib unavailable; skipping plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    if rl_centers:
        ax.plot(rl_centers, rl_means, "-o", label=f"RL-time reward (bin={bin_hours}h)", color="#36c", markersize=4)
    for m in eval_markers:
        if m["timestamp"] is None or m["mean_reward"] is None:
            continue
        ax.scatter([m["timestamp"]], [m["mean_reward"]], color="#a30", s=80, marker="D", zorder=5)
        ax.annotate(
            m["label"],
            (m["timestamp"], m["mean_reward"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("mean reward per trial")
    ax.set_title("RL-time reward (line) with post-RL eval-checkpoint rewards (markers)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rl-traces", required=True, help="RL-time training trace source (HF/JSONL/dir)")
    parser.add_argument(
        "--eval-traces",
        action="append",
        default=[],
        help=(
            "Eval-time trace source with optional label/timestamp: "
            "'<source>[@<label>][:<iso-timestamp>]'. Repeat for multiple "
            "checkpoints. Timestamp is required to place the marker on the "
            "time axis."
        ),
    )
    parser.add_argument("--bin-hours", type=float, default=4.0, help="Hours per RL-time bin")
    parser.add_argument("--output", type=Path, required=True, help="PNG output path (JSON sidecar also written)")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows per source (smoke testing)")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    rl_traces = load_traces(args.rl_traces, max_rows=args.max_rows)
    rl_centers, rl_means = _bin_rl_reward(rl_traces, args.bin_hours)
    eval_markers = [_eval_marker(s, args.max_rows) for s in args.eval_traces]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot(rl_centers, rl_means, eval_markers, args.output, args.bin_hours)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "rl": {
                    "bin_hours": args.bin_hours,
                    "points": [
                        {"t": c.isoformat(), "mean_reward": m}
                        for c, m in zip(rl_centers, rl_means)
                    ],
                },
                "eval": [
                    {
                        "label": m["label"],
                        "source": m["source"],
                        "timestamp": m["timestamp"].isoformat() if m["timestamp"] else None,
                        "n": m["n"],
                        "mean_reward": m["mean_reward"],
                    }
                    for m in eval_markers
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[eval-temporal-overlay] wrote {args.output} ({len(rl_centers)} RL bins, {len(eval_markers)} eval markers)")
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
