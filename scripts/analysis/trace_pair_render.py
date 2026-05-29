#!/usr/bin/env python3
"""Render same-task trace pairs side-by-side for human inspection.

Answers research question (3): **Do behavioral changes persist in post-RL
eval traces?** The macro deltas from ``behavioral_delta.py`` say *what*
shifted; this script surfaces *what those shifts look like in practice*.

For each task present in both datasets, picks one representative trial
from each side (highest reward, then most recent) and renders both
conversations as columns in a single HTML page. Differences in turn
count, response length, and final reward are flagged inline.

Usage:
    python -m scripts.analysis.trace_pair_render \\
        --before  penfever/pre-rl-eval-traces \\
        --after   penfever/post-rl-eval-traces \\
        --output  /path/trace_pairs.html \\
        --top-n   25
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.utils import (  # noqa: E402
    Trace,
    group_by_task,
    load_traces,
)


def _pick_representative(rows: Sequence[Trace]) -> Trace:
    """Pick the highest-reward trial, then the most-recent within ties."""
    return max(
        rows,
        key=lambda t: (
            t.reward if t.reward is not None else -1.0,
            t.date.timestamp() if t.date else 0.0,
        ),
    )


def _messages_of(raw: dict) -> List[dict]:
    msgs = raw.get("messages") or raw.get("conversations") or []
    if not isinstance(msgs, list):
        return []
    return msgs


def _format_message_html(msg: dict, max_chars: int = 4000) -> str:
    role = html.escape(str(msg.get("role", "?")))
    content = msg.get("content")
    if isinstance(content, list):
        # Sometimes content is a list of {type, text} dicts.
        content = "\n".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    elif not isinstance(content, str):
        content = str(content)
    truncated = ""
    if len(content) > max_chars:
        truncated = f"<div class=trunc>... [{len(content) - max_chars} more chars truncated]</div>"
        content = content[:max_chars]
    return (
        f'<div class="msg role-{role}">'
        f'<div class="role">{role}</div>'
        f'<pre class="content">{html.escape(content)}</pre>'
        f"{truncated}"
        f"</div>"
    )


def _trial_panel_html(t: Trace) -> str:
    msgs = _messages_of(t.raw)
    reward = "—" if t.reward is None else f"{t.reward:.3f}"
    err = f" · err: <code>{html.escape(t.error_type)}</code>" if t.error_type else ""
    fm = f" · failure-mode: <code>{html.escape(t.failure_mode)}</code>" if t.failure_mode else ""
    header = (
        f'<div class=trial-header>'
        f'reward: <b>{reward}</b> · turns: <b>{len(msgs)}</b>'
        f'{err}{fm}'
        f'<div class=trial-source>{html.escape(t.source or "")}</div>'
        f'</div>'
    )
    body = "\n".join(_format_message_html(m) for m in msgs)
    return header + body


CSS = """
* { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
body { margin: 0; padding: 1rem; background: #fafafa; color: #111; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1rem; margin-top: 2rem; padding: .5rem; background: #eef; border-radius: 4px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem; }
.col { background: white; padding: .5rem; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }
.col h3 { margin: 0 0 .5rem 0; padding: .25rem .5rem; font-size: .85rem; }
.col.before h3 { background: #ffe6e6; }
.col.after h3 { background: #e6ffe6; }
.trial-header { font-size: .85rem; color: #444; padding: .25rem; border-bottom: 1px solid #eee; }
.trial-source { font-family: monospace; font-size: .75rem; color: #888; }
.msg { margin: .5rem 0; border-left: 3px solid #ccc; padding-left: .5rem; }
.msg.role-user { border-left-color: #36c; }
.msg.role-assistant { border-left-color: #3a8; }
.msg.role-system { border-left-color: #888; }
.msg.role-tool { border-left-color: #a83; }
.role { font-size: .7rem; text-transform: uppercase; color: #666; letter-spacing: .04em; }
.content { white-space: pre-wrap; font-size: .8rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: .25rem 0; line-height: 1.4; }
.trunc { font-size: .7rem; color: #999; font-style: italic; }
.delta-tag { display: inline-block; padding: .1rem .4rem; border-radius: 3px; font-size: .7rem; margin-left: .25rem; }
.delta-tag.up { background: #cfc; color: #060; }
.delta-tag.dn { background: #fcc; color: #600; }
.delta-tag.flat { background: #eee; color: #555; }
"""


def _delta_tag(before: float, after: float, kind: str) -> str:
    if abs(after - before) < 1e-9:
        return f'<span class="delta-tag flat">{kind} unchanged</span>'
    cls = "up" if after > before else "dn"
    return f'<span class="delta-tag {cls}">{kind} {after - before:+.2f}</span>'


def render_html(
    pairs: List[tuple], before_label: str, after_label: str, output_path: Path
) -> None:
    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Trace pair render</title>",
        f"<style>{CSS}</style>",
        "</head><body>",
        f"<h1>Trace pair render: <code>{html.escape(before_label)}</code> vs "
        f"<code>{html.escape(after_label)}</code></h1>",
        f"<p>{len(pairs)} task pairs rendered (highest-reward trial from each side).</p>",
    ]
    for task, before, after in pairs:
        rb = before.reward if before.reward is not None else 0.0
        ra = after.reward if after.reward is not None else 0.0
        tb = len(_messages_of(before.raw))
        ta = len(_messages_of(after.raw))
        parts.append(
            f"<h2>Task: <code>{html.escape(task)}</code> "
            f"{_delta_tag(rb, ra, 'reward')} "
            f"{_delta_tag(tb, ta, 'turns')}</h2>"
        )
        parts.append('<div class="pair">')
        parts.append(f'<div class="col before"><h3>Before — {html.escape(before_label)}</h3>')
        parts.append(_trial_panel_html(before))
        parts.append("</div>")
        parts.append(f'<div class="col after"><h3>After — {html.escape(after_label)}</h3>')
        parts.append(_trial_panel_html(after))
        parts.append("</div></div>")
    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _select_pairs(
    before: Dict[str, List[Trace]],
    after: Dict[str, List[Trace]],
    top_n: int,
    prefer: str,
) -> List[tuple]:
    """Pick representative pairs, ordered by what's most interesting.

    ``prefer`` choices:
      * ``flips`` — tasks where pass/fail status flipped (default; most
        interesting for "did behavior change matter?")
      * ``reward-delta`` — tasks with the largest absolute reward swing
      * ``any`` — first ``top_n`` common tasks sorted alphabetically
    """
    common = sorted(set(before) & set(after))
    pairs = []
    for task in common:
        b = _pick_representative(before[task])
        a = _pick_representative(after[task])
        pairs.append((task, b, a))
    if prefer == "flips":
        # A flip means reward sign-of-success changed.
        def _flipped(tup):
            _, b, a = tup
            bp = (b.reward or 0.0) > 0
            ap = (a.reward or 0.0) > 0
            return 1 if bp != ap else 0
        pairs.sort(key=lambda t: (-_flipped(t), abs((t[2].reward or 0.0) - (t[1].reward or 0.0))), reverse=False)
        # Resort with proper key: flips first (1 > 0), then by abs delta.
        pairs.sort(key=lambda t: (_flipped(t), abs((t[2].reward or 0.0) - (t[1].reward or 0.0))), reverse=True)
    elif prefer == "reward-delta":
        pairs.sort(key=lambda t: abs((t[2].reward or 0.0) - (t[1].reward or 0.0)), reverse=True)
    return pairs[:top_n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", required=True, help="Pre-RL trace source")
    parser.add_argument("--after", required=True, help="Post-RL trace source")
    parser.add_argument("--output", type=Path, required=True, help="HTML output path")
    parser.add_argument("--top-n", type=int, default=20, help="Number of task pairs to render")
    parser.add_argument(
        "--prefer",
        choices=("flips", "reward-delta", "any"),
        default="flips",
        help="How to rank which task pairs to render first",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows loaded per side")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    before = load_traces(args.before, max_rows=args.max_rows)
    after = load_traces(args.after, max_rows=args.max_rows)
    bb = group_by_task(before)
    aa = group_by_task(after)
    pairs = _select_pairs(bb, aa, args.top_n, args.prefer)
    if not pairs:
        print("[trace-pair-render] no common tasks between before/after", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_html(pairs, args.before, args.after, args.output)
    print(f"[trace-pair-render] wrote {args.output} ({len(pairs)} pairs)")
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
