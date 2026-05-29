#!/usr/bin/env python3
"""Diff behavioral metrics + failure-mode distributions between two trace datasets.

Answers research question (1): **What model behaviors are changing as a
result of RL?** Run this with the pre-RL trace dataset (or the SFT-only
eval traces) as ``--before`` and the post-RL trace dataset as ``--after``.

The script:
- Loads both trace datasets via the shared :func:`scripts.analysis.utils.load_traces`
  loader (HF id, JSONL, or directory of ``result.json`` all supported).
- Aggregates macro-level metrics: reward, turn count, conversation token
  count, error-type distribution, failure-mode distribution (from the
  ``failure_mode_analysis`` column if ``update_hf_failure_modes`` was
  run first).
- Diffs each metric. Emits a markdown report ranking the failure modes
  that rose / fell most, plus the tasks whose pass/fail status flipped.

Usage:
    python -m scripts.analysis.behavioral_delta \\
        --before  penfever/pre-rl-eval-traces \\
        --after   penfever/post-rl-eval-traces \\
        --output  /path/behavioral_delta.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.utils import (  # noqa: E402
    Trace,
    count_tokens,
    get_tiktoken_encoder,
    group_by_task,
    load_traces,
)


@dataclass
class Summary:
    """Aggregated behavioral metrics for one trace dataset."""
    label: str
    n: int
    mean_reward: Optional[float]
    mean_turns: Optional[float]
    mean_tokens: Optional[float]
    error_dist: Dict[str, int]
    failure_mode_dist: Dict[str, int]
    pass_set: set  # tasks with at least one reward > 0
    fail_set: set  # tasks with no reward > 0 (and at least one trial)
    fm_coverage: float  # fraction of rows with a failure_mode label


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(label: str, traces: Sequence[Trace]) -> Summary:
    encoder = get_tiktoken_encoder()
    rewards = [t.reward for t in traces if t.reward is not None]
    turns = [t.turns for t in traces if t.turns > 0]
    tokens = [count_tokens(t.conversation, encoder) for t in traces if t.conversation]
    err = Counter(t.error_type for t in traces if t.error_type)
    fm = Counter(t.failure_mode for t in traces if t.failure_mode)
    fm_coverage = sum(1 for t in traces if t.failure_mode) / len(traces) if traces else 0.0
    pass_set: set = set()
    fail_set: set = set()
    for task, rows in group_by_task(traces).items():
        if any((r.reward or 0.0) > 0 for r in rows):
            pass_set.add(task)
        elif rows:
            fail_set.add(task)
    return Summary(
        label=label,
        n=len(traces),
        mean_reward=_mean(rewards),
        mean_turns=_mean(turns),
        mean_tokens=_mean(tokens),
        error_dist=dict(err),
        failure_mode_dist=dict(fm),
        pass_set=pass_set,
        fail_set=fail_set,
        fm_coverage=fm_coverage,
    )


def _delta_str(before: Optional[float], after: Optional[float], pct: bool = False) -> str:
    if before is None or after is None:
        return "—"
    d = after - before
    if pct:
        return f"{d*100:+.1f}pp"
    return f"{d:+.2f}"


def _rank_dist_delta(before: Dict[str, int], after: Dict[str, int]) -> List[tuple]:
    """Return (key, before_share, after_share, delta_share) sorted by |delta|.

    Shares are normalized to fractions of the total in each dataset, so
    different sample sizes don't bias the diff.
    """
    bt = sum(before.values()) or 1
    at_ = sum(after.values()) or 1
    keys = set(before) | set(after)
    rows = []
    for k in keys:
        b = before.get(k, 0) / bt
        a = after.get(k, 0) / at_
        rows.append((k, b, a, a - b))
    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    return rows


def write_markdown_report(
    before: Summary, after: Summary, output_path: Path, top_n: int = 15
) -> None:
    lines: List[str] = [
        "# Behavioral Delta Report",
        "",
        f"- **Before**: `{before.label}` ({before.n} rows, failure-mode coverage {before.fm_coverage:.0%})",
        f"- **After**:  `{after.label}` ({after.n} rows, failure-mode coverage {after.fm_coverage:.0%})",
        "",
        "## Macro metrics",
        "",
        "| metric | before | after | delta |",
        "|---|---|---|---|",
        f"| mean reward | {before.mean_reward if before.mean_reward is None else f'{before.mean_reward:.3f}'} | "
        f"{after.mean_reward if after.mean_reward is None else f'{after.mean_reward:.3f}'} | "
        f"{_delta_str(before.mean_reward, after.mean_reward)} |",
        f"| mean turns | {before.mean_turns if before.mean_turns is None else f'{before.mean_turns:.1f}'} | "
        f"{after.mean_turns if after.mean_turns is None else f'{after.mean_turns:.1f}'} | "
        f"{_delta_str(before.mean_turns, after.mean_turns)} |",
        f"| mean tokens (conv) | {before.mean_tokens if before.mean_tokens is None else f'{before.mean_tokens:.0f}'} | "
        f"{after.mean_tokens if after.mean_tokens is None else f'{after.mean_tokens:.0f}'} | "
        f"{_delta_str(before.mean_tokens, after.mean_tokens)} |",
        "",
    ]

    fm_delta = _rank_dist_delta(before.failure_mode_dist, after.failure_mode_dist)
    if fm_delta:
        lines += [
            "## Failure-mode distribution shifts",
            "",
            "Each row is a failure-mode label produced by the GPT judge "
            "(`update_hf_failure_modes.py`); shares are normalized within "
            "each dataset.",
            "",
            "| mode | before share | after share | delta |",
            "|---|---|---|---|",
        ]
        for key, b, a, d in fm_delta[:top_n]:
            lines.append(f"| `{key}` | {b:.1%} | {a:.1%} | {d*100:+.1f}pp |")
        lines.append("")
    else:
        lines += [
            "## Failure-mode distribution shifts",
            "",
            "_No failure-mode annotations found on either side. Run "
            "`scripts/analysis/update_hf_failure_modes.py` on both repos first "
            "to populate the `failure_mode_analysis` column._",
            "",
        ]

    err_delta = _rank_dist_delta(before.error_dist, after.error_dist)
    if err_delta:
        lines += [
            "## Error-type distribution shifts",
            "",
            "| error | before share | after share | delta |",
            "|---|---|---|---|",
        ]
        for key, b, a, d in err_delta[:top_n]:
            lines.append(f"| `{key}` | {b:.1%} | {a:.1%} | {d*100:+.1f}pp |")
        lines.append("")

    # Task-level flips
    common_tasks = (before.pass_set | before.fail_set) & (after.pass_set | after.fail_set)
    newly_passing = sorted(t for t in common_tasks if t in before.fail_set and t in after.pass_set)
    newly_failing = sorted(t for t in common_tasks if t in before.pass_set and t in after.fail_set)
    lines += [
        "## Task-level pass/fail flips",
        "",
        f"- Common tasks across both datasets: **{len(common_tasks)}**",
        f"- Newly passing after RL (was-fail → now-pass): **{len(newly_passing)}**",
        f"- Newly failing after RL (was-pass → now-fail): **{len(newly_failing)}**",
        "",
    ]
    if newly_passing:
        lines.append("### Newly passing tasks")
        lines.append("")
        for t in newly_passing[:top_n]:
            lines.append(f"- `{t}`")
        if len(newly_passing) > top_n:
            lines.append(f"- _...and {len(newly_passing) - top_n} more_")
        lines.append("")
    if newly_failing:
        lines.append("### Newly failing tasks (regressions)")
        lines.append("")
        for t in newly_failing[:top_n]:
            lines.append(f"- `{t}`")
        if len(newly_failing) > top_n:
            lines.append(f"- _...and {len(newly_failing) - top_n} more_")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Also drop a JSON sidecar with the raw counts so downstream tooling
    # (e.g., the orchestrator) doesn't need to re-parse the markdown.
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "before": {
                    "label": before.label,
                    "n": before.n,
                    "mean_reward": before.mean_reward,
                    "mean_turns": before.mean_turns,
                    "mean_tokens": before.mean_tokens,
                    "error_dist": before.error_dist,
                    "failure_mode_dist": before.failure_mode_dist,
                    "fm_coverage": before.fm_coverage,
                },
                "after": {
                    "label": after.label,
                    "n": after.n,
                    "mean_reward": after.mean_reward,
                    "mean_turns": after.mean_turns,
                    "mean_tokens": after.mean_tokens,
                    "error_dist": after.error_dist,
                    "failure_mode_dist": after.failure_mode_dist,
                    "fm_coverage": after.fm_coverage,
                },
                "task_flips": {
                    "newly_passing": newly_passing,
                    "newly_failing": newly_failing,
                    "common_count": len(common_tasks),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", required=True, help="Pre-RL trace source (HF id, JSONL, or dir)")
    parser.add_argument("--after", required=True, help="Post-RL trace source (HF id, JSONL, or dir)")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report path (sidecar .json also written)")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--top-n", type=int, default=15, help="Top N rows per ranked section")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    before = load_traces(args.before, max_rows=args.max_rows)
    after = load_traces(args.after, max_rows=args.max_rows)
    if not before or not after:
        print(f"[behavioral-delta] empty trace set: before={len(before)} after={len(after)}", file=sys.stderr)
        return 2
    s_before = summarize(args.before, before)
    s_after = summarize(args.after, after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(s_before, s_after, args.output, top_n=args.top_n)
    print(f"[behavioral-delta] wrote {args.output} (+ {args.output.with_suffix('.json').name})")
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
