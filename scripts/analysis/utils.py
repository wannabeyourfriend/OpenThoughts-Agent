"""Shared utilities for analysis scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore[assignment]

_EPISODE_PATTERN = re.compile(r"(\d+)")


# ---------------------------------------------------------------------------
# JSONL iteration
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[Dict]:
    """Yield parsed dicts from a JSONL file, raising on malformed lines."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number}: {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# Conversation text extraction
# ---------------------------------------------------------------------------

def _extract_from_message_content(content) -> str:
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            parts.append(text)
    elif isinstance(content, list):
        for item in content:
            parts.append(_extract_from_message_content(item))
    return "\n".join(p for p in parts if p)


def extract_conversation_text(record) -> str:
    """Extract the full concatenated text from a conversation record.

    Handles ``messages``, ``conversations``, and various fallback fields.
    """
    if isinstance(record, dict):
        messages = record.get("messages") or record.get("conversations")
        if isinstance(messages, list):
            collected: list[str] = []
            for message in messages:
                if isinstance(message, dict):
                    if "content" in message:
                        collected.append(_extract_from_message_content(message["content"]))
                    for key in ("value", "text"):
                        value = message.get(key)
                        if isinstance(value, str):
                            collected.append(value)
                elif isinstance(message, str):
                    collected.append(message)
            combined = "\n".join(chunk for chunk in collected if chunk)
            if combined:
                return combined
        for field in ("conversation", "text", "prompt", "content"):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                return value
    return json.dumps(record, ensure_ascii=False)


def count_turns(record) -> int:
    """Estimate the number of turns (messages) in a record."""
    if isinstance(record, dict):
        messages = record.get("messages") or record.get("conversations")
        if isinstance(messages, list):
            return len(messages)
        turn_count = record.get("turn_count")
        if isinstance(turn_count, int):
            return turn_count
    return 0


# ---------------------------------------------------------------------------
# Episode number extraction
# ---------------------------------------------------------------------------

def extract_episode_numbers(values: Iterable) -> List[int]:
    """Parse integer episode indices from various formats (str, int, dict)."""
    episodes: List[int] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            episodes.append(int(value))
            continue
        if isinstance(value, str):
            cleaned = value.replace("-", " ").replace("_", " ")
            match = _EPISODE_PATTERN.search(cleaned)
            if match:
                episodes.append(int(match.group(1)))
            continue
        if isinstance(value, dict):
            inner = value.get("episode")
            if isinstance(inner, (int, float)):
                episodes.append(int(inner))
            elif isinstance(inner, str):
                cleaned_inner = inner.replace("-", " ").replace("_", " ")
                match = _EPISODE_PATTERN.search(cleaned_inner)
                if match:
                    episodes.append(int(match.group(1)))
    return episodes


# ---------------------------------------------------------------------------
# Token counting (tiktoken)
# ---------------------------------------------------------------------------

def get_tiktoken_encoder():
    """Return a tiktoken encoder, or ``None`` if tiktoken is unavailable."""
    if tiktoken is None:
        return None
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder) -> int:
    """Count tokens in *text* using *encoder*, falling back to whitespace split."""
    if not text:
        return 0
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))
    return len(text.split())


# ---------------------------------------------------------------------------
# HuggingFace dataset loading
# ---------------------------------------------------------------------------

def load_hf_trace_dataset(repo_id: str, split: str = "train"):
    """Load a HuggingFace dataset with a helpful error message on failure."""
    from datasets import load_dataset
    try:
        return load_dataset(repo_id, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{repo_id}' (split={split}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Reward / result extraction
# ---------------------------------------------------------------------------

def extract_reward(record) -> Optional[float]:
    """Extract a numeric reward from a record's ``result`` field.

    Returns ``None`` if the value is missing, null-like, or non-numeric.
    """
    if isinstance(record, dict):
        candidate = record.get("result")
    else:
        candidate = record
    if candidate is None:
        return None
    if isinstance(candidate, (int, float)):
        return float(candidate)
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if not stripped or stripped.lower() in ("none", "null", ""):
            return None
        try:
            return float(stripped)
        except (TypeError, ValueError):
            return None
    return None


def mean_reward_per_trial(rows: list) -> Optional[float]:
    """Compute the flat mean reward across all trials (Harbor-style 'accuracy').

    This matches Harbor's Mean metric: every trial contributes equally,
    with errors/missing results counted as 0. No per-task grouping.
    """
    values = []
    for row in rows:
        reward = extract_reward(row)
        values.append(reward if reward is not None else 0.0)
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Infrastructure-error filtering (matches Harbor drop_ei logic)
# ---------------------------------------------------------------------------

DEFAULT_DROP_EXCEPTIONS: frozenset[str] = frozenset(
    [
        "AgentEnvironmentTimeoutError",
        "DaytonaError",
        "DaytonaRateLimitError",
        "DaytonaNotFoundError",
        "EnvironmentStartTimeoutError",
        "SandboxBuildFailedError",
        "PodmanHPCTimeoutError",
        "PodmanHPCCommandError",
        "ApptainerTimeoutError",
        "ApptainerCommandError",
    ]
)


def filter_ei(
    rows: list[dict],
    drop_exceptions: frozenset[str] = DEFAULT_DROP_EXCEPTIONS,
) -> list[dict]:
    """Drop rows whose result is an infrastructure error type."""
    filtered = []
    for row in rows:
        error = extract_error_type(row)
        if error is not None and error in drop_exceptions:
            continue
        filtered.append(row)
    return filtered


def tasks_with_n_attempts(
    rows: list[dict],
    n_attempts: int,
) -> set[str]:
    """Return tasks that have at least *n_attempts* rows after filtering."""
    from collections import Counter
    counts = Counter(row["task"] for row in rows)
    return {task for task, count in counts.items() if count >= n_attempts}


def mean_reward_per_trial_ei(
    rows: list[dict],
    drop_exceptions: frozenset[str] = DEFAULT_DROP_EXCEPTIONS,
    n_attempts: int = 1,
) -> Optional[float]:
    """Mean reward after dropping infra-errored trials and incomplete tasks.

    Mirrors Harbor's MeanDropEI metric.
    """
    clean = filter_ei(rows, drop_exceptions)
    complete_tasks = tasks_with_n_attempts(clean, n_attempts)
    values = []
    for row in clean:
        if row["task"] not in complete_tasks:
            continue
        reward = extract_reward(row)
        values.append(reward if reward is not None else 0.0)
    if not values:
        return None
    return sum(values) / len(values)


def ei_common_tasks(
    all_datasets: dict[str, list[dict]],
    drop_exceptions: frozenset[str] = DEFAULT_DROP_EXCEPTIONS,
    n_attempts: int = 1,
) -> set[str]:
    """Return tasks present and complete (post-EI-filter) in ALL datasets."""
    per_model: list[set[str]] = []
    for rows in all_datasets.values():
        clean = filter_ei(rows, drop_exceptions)
        per_model.append(tasks_with_n_attempts(clean, n_attempts))
    if not per_model:
        return set()
    return set.intersection(*per_model)


def extract_error_type(record) -> Optional[str]:
    """Extract an error type name from a record's ``result`` field.

    Returns the string value when it is not parseable as a number (i.e. it's
    an exception class name like ``"AgentTimeoutError"``).  Returns ``None``
    for numeric results or missing values.
    """
    if isinstance(record, dict):
        candidate = record.get("result")
    else:
        candidate = record
    if candidate is None:
        return None
    if isinstance(candidate, (int, float)):
        return None
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if not stripped or stripped.lower() in ("none", "null"):
            return None
        try:
            float(stripped)
            return None  # it's a number, not an error
        except (TypeError, ValueError):
            return stripped
    return None


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

def extract_date(record) -> Optional[datetime]:
    """Parse the ``date`` field of a record into a :class:`datetime`.

    Accepts ISO 8601 strings.  Returns ``None`` on failure.
    """
    if isinstance(record, dict):
        candidate = record.get("date")
    else:
        candidate = record
    if candidate is None:
        return None
    if isinstance(candidate, datetime):
        return candidate
    if isinstance(candidate, str):
        try:
            return datetime.fromisoformat(candidate)
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Task identity
# ---------------------------------------------------------------------------

def task_id_of(record) -> Optional[str]:
    """Best-effort canonical task identifier for a trace row.

    Trace datasets disagree on which field holds the task identity — some
    use ``task``, others ``task_name``, others bury it inside a nested
    dict. This helper handles the common cases.
    """
    if not isinstance(record, dict):
        return None
    for key in ("task", "task_name", "task_id", "trial_name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# ---------------------------------------------------------------------------
# Trace dataclass + unified loader
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    """Normalized view over a single trace row.

    Field-extraction (``extract_reward``, ``extract_error_type``, etc.) is
    eager and cached here so downstream analyses don't repeat the work.
    The ``raw`` dict is kept for one-off field access.
    """
    raw: Dict[str, Any]
    task: Optional[str] = None
    reward: Optional[float] = None
    error_type: Optional[str] = None
    date: Optional[datetime] = None
    turns: int = 0
    conversation: str = ""
    failure_mode: Optional[str] = None
    # Optional cached token count (filled on demand to avoid tokenizer cost).
    tokens: Optional[int] = None
    # Origin tag (e.g. "hf:penfever/dataset", "jsonl:/path/foo.jsonl",
    # "dir:/path/results"). Useful when cross-referencing traces from
    # multiple sources in the same analysis pass.
    source: Optional[str] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any], source: Optional[str] = None) -> "Trace":
        # update_hf_failure_modes writes to "failure_mode_analysis" by default;
        # accept either the new short name or the legacy long name.
        fm = row.get("failure_mode") or row.get("failure_mode_analysis")
        if isinstance(fm, dict):
            # GPT-5 judge returns a dict; collapse to its 'mode' field if present.
            fm = fm.get("mode") or fm.get("category") or fm.get("summary")
        return cls(
            raw=row,
            task=task_id_of(row),
            reward=extract_reward(row),
            error_type=extract_error_type(row),
            date=extract_date(row),
            turns=count_turns(row),
            conversation=extract_conversation_text(row),
            failure_mode=fm if isinstance(fm, str) else None,
            source=source,
        )


def _iter_results_in_dir(root: Path) -> Iterator[Dict[str, Any]]:
    """Walk a directory of trial folders, yielding their result.json contents.

    Mirrors the eval-trace layout (``<root>/<trial-name>/result.json`` plus
    optional ``agent/``, ``verifier/``). Each yielded row carries the
    trial_name + the conversation pulled from ``agent/conversation.json``
    when present, so downstream code can treat it like an HF row.
    """
    for trial_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            payload = {"result": payload}
        payload.setdefault("trial_name", trial_dir.name)
        # Pull the conversation if it sits at the conventional path.
        conv_path = trial_dir / "agent" / "conversation.json"
        if conv_path.exists():
            try:
                conv = json.loads(conv_path.read_text(encoding="utf-8"))
                if isinstance(conv, list):
                    payload.setdefault("messages", conv)
            except (json.JSONDecodeError, OSError):
                pass
        yield payload


def load_traces(
    source: Union[str, Path],
    *,
    split: str = "train",
    max_rows: Optional[int] = None,
) -> List[Trace]:
    """Load traces from any of the supported source types.

    Supported ``source`` formats:
      - HuggingFace dataset id (``"penfever/foo-bar"``)
      - JSONL path (``.jsonl`` or ``.json`` suffix)
      - Local directory of trial folders (each holding ``result.json``)

    Returns a list of :class:`Trace` instances. ``max_rows`` caps the
    number of rows loaded (useful for smoke tests).
    """
    if isinstance(source, str) and not Path(source).expanduser().exists():
        # Treat as HF repo id.
        ds = load_hf_trace_dataset(source, split=split)
        traces: List[Trace] = []
        for i, row in enumerate(ds):
            if max_rows is not None and i >= max_rows:
                break
            traces.append(Trace.from_row(row, source=f"hf:{source}"))
        return traces

    path = Path(source).expanduser().resolve()
    if path.is_file() and path.suffix in (".jsonl", ".json"):
        rows = list(iter_jsonl(path))
        if max_rows is not None:
            rows = rows[:max_rows]
        return [Trace.from_row(r, source=f"jsonl:{path}") for r in rows]

    if path.is_dir():
        rows: List[Dict[str, Any]] = []
        for row in _iter_results_in_dir(path):
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
        return [Trace.from_row(r, source=f"dir:{path}") for r in rows]

    raise ValueError(
        f"Cannot resolve traces source {source!r}: not an existing JSONL, "
        "result.json directory, or HF dataset id."
    )


def group_by_task(traces: Sequence[Trace]) -> Dict[str, List[Trace]]:
    """Bucket traces by their canonical task id; drops rows with no task."""
    out: Dict[str, List[Trace]] = {}
    for t in traces:
        if t.task is None:
            continue
        out.setdefault(t.task, []).append(t)
    return out
