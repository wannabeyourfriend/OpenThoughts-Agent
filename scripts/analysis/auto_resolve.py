"""Resolve the orchestrator's args from a (rl_traces, model_repo) pair.

Given just a training-trace dataset and the trained-model repo, the
orchestrator can fill the remaining inputs by querying the Supabase
metadata DB and inspecting the model repo on the HuggingFace Hub:

  --post-rl-eval     ← sandbox_jobs(model_id = <model>).hf_traces_link
  --post-rl-eval-ts  ← models.training_end (post-RL checkpoint mtime)
  --baseline-eval    ← sandbox_jobs(model_id = base_model_id).hf_traces_link
  --baseline-eval-ts ← models.training_start  (snapshot the RL forked from)
  --training-log-dir ← <model_repo>/training_logs/* if present
                      (downloaded to a local snapshot dir on demand)

The Supabase calls reuse ``database.unified_db`` so we inherit its auth
+ retry handling. HF lookups use the standard ``huggingface_hub`` client.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Resolved:
    """Result of an autofill resolve. Any field may be ``None`` if not found."""
    post_rl_eval: Optional[str] = None
    post_rl_eval_ts: Optional[str] = None  # ISO 8601
    baseline_eval: Optional[str] = None
    baseline_eval_ts: Optional[str] = None  # ISO 8601
    training_log_dir: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "post_rl_eval": self.post_rl_eval,
            "post_rl_eval_ts": self.post_rl_eval_ts,
            "baseline_eval": self.baseline_eval,
            "baseline_eval_ts": self.baseline_eval_ts,
            "training_log_dir": self.training_log_dir,
        }


def parse_hf_url(value: str) -> str:
    """Normalize a HuggingFace URL or bare repo id to ``owner/name``.

    Accepts:
      - ``https://huggingface.co/owner/name``
      - ``https://huggingface.co/datasets/owner/name``
      - ``https://huggingface.co/owner/name/tree/main/...``
      - ``owner/name``
    """
    value = (value or "").strip()
    if not value:
        raise ValueError("empty HF reference")
    if not value.startswith("http"):
        # Assume bare "owner/name" — strip leading "datasets/" if present.
        if value.startswith("datasets/"):
            value = value[len("datasets/"):]
        return value
    parsed = urlparse(value)
    parts = [p for p in parsed.path.split("/") if p]
    # Drop a leading "datasets" segment.
    if parts and parts[0] == "datasets":
        parts = parts[1:]
    # Keep just the owner/name pair (drop /tree/main/... etc.).
    if len(parts) < 2:
        raise ValueError(f"could not parse owner/name from HF URL {value!r}")
    return f"{parts[0]}/{parts[1]}"


# ---------------------------------------------------------------------------
# Supabase queries
# ---------------------------------------------------------------------------


def _supabase_client(use_admin: bool = False):
    """Return a Supabase client via the existing unified_db helper."""
    from database.unified_db.utils import get_supabase_client
    return get_supabase_client(use_admin=use_admin)


def _model_by_name(client, name: str) -> Optional[Dict[str, Any]]:
    """Look up a model row by ``name`` (the HF repo id). Returns None if missing."""
    resp = client.table("models").select("*").eq("name", name).limit(1).execute()
    return resp.data[0] if resp.data else None


def _latest_eval_job_for_model(client, model_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent sandbox_job for the given model_id with an hf_traces_link.

    "Most recent" = highest ``ended_at`` (fall back to ``started_at``,
    then ``created_at``). Picks any benchmark — caller can pre-filter if
    they want a specific one.
    """
    resp = (
        client.table("sandbox_jobs")
        .select("id,hf_traces_link,benchmark_id,started_at,ended_at,created_at,job_status,metrics")
        .eq("model_id", model_id)
        .order("ended_at", desc=True)
        .order("started_at", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    if not resp.data:
        return None
    for row in resp.data:
        # Need a non-null hf_traces_link for the resolver to be useful.
        if row.get("hf_traces_link"):
            return row
    return None


# ---------------------------------------------------------------------------
# HuggingFace training-logs lookup
# ---------------------------------------------------------------------------


def _download_training_logs(model_repo: str, dest_dir: Path) -> Optional[Path]:
    """Try to snapshot ``<model_repo>/training_logs/*`` into ``dest_dir``.

    Returns the local dir path if the model repo has a ``training_logs/``
    directory and we successfully pulled it; ``None`` otherwise. Cheap
    listing first via ``list_repo_files`` so we don't trigger a download
    for repos that don't have logs.
    """
    try:
        from huggingface_hub import list_repo_files, snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not installed; can't fetch training_logs/")
        return None
    try:
        files = list_repo_files(repo_id=model_repo, repo_type="model")
    except Exception as exc:
        logger.warning("list_repo_files(%s) failed: %s", model_repo, exc)
        return None
    if not any(f.startswith("training_logs/") for f in files):
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        local = snapshot_download(
            repo_id=model_repo,
            repo_type="model",
            allow_patterns=["training_logs/**"],
            local_dir=str(dest_dir),
        )
    except Exception as exc:
        logger.warning("snapshot_download(%s, training_logs/**) failed: %s", model_repo, exc)
        return None
    log_dir = Path(local) / "training_logs"
    return log_dir if log_dir.exists() else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(
    rl_traces: str,
    model_repo: str,
    *,
    training_log_cache: Optional[Path] = None,
    fetch_training_logs: bool = True,
) -> Resolved:
    """Autofill the post-RL/baseline eval + ts + training_log_dir from supabase + HF.

    Args:
        rl_traces: Training-trace dataset (HF id or URL). Used only to
            populate ``notes`` cross-reference; the resolver doesn't need
            to query it.
        model_repo: Trained-model HF id or URL.
        training_log_cache: Where to drop the snapshotted training_logs/.
            Defaults to ``~/.ot-agent/training_logs/<safe_repo>/``.
        fetch_training_logs: If False, skip the HF snapshot step entirely
            (just emit the URL convention as a note).

    Returns:
        A :class:`Resolved` with whatever could be filled. Fields that
        couldn't be resolved stay ``None``; ``notes`` carries the reason.
    """
    out = Resolved()
    try:
        model_name = parse_hf_url(model_repo)
    except Exception as exc:
        out.notes.append(f"could not parse model_repo {model_repo!r}: {exc}")
        return out

    try:
        rl_name = parse_hf_url(rl_traces)
        out.notes.append(f"rl_traces normalized to {rl_name}")
    except Exception:
        pass  # rl_traces is informational only here

    # ---- Supabase lookup ----
    try:
        client = _supabase_client()
    except Exception as exc:
        out.notes.append(f"supabase client unavailable: {exc}")
        client = None

    model_row: Optional[Dict[str, Any]] = None
    if client is not None:
        try:
            model_row = _model_by_name(client, model_name)
        except Exception as exc:
            out.notes.append(f"models lookup failed: {exc}")

    if model_row is None:
        out.notes.append(f"no row in models table with name={model_name!r}")
    else:
        out.post_rl_eval_ts = model_row.get("training_end") or out.post_rl_eval_ts
        out.baseline_eval_ts = model_row.get("training_start") or out.baseline_eval_ts
        if not out.post_rl_eval_ts:
            out.notes.append("models.training_end is null; post_rl_eval_ts not set")

        # Post-RL eval comes from the model's own sandbox_jobs.
        if client is not None:
            try:
                eval_job = _latest_eval_job_for_model(client, model_row["id"])
                if eval_job and eval_job.get("hf_traces_link"):
                    out.post_rl_eval = eval_job["hf_traces_link"]
                    out.notes.append(
                        f"post_rl_eval ← sandbox_jobs[{eval_job['id']}] "
                        f"(benchmark={eval_job.get('benchmark_id')}, status={eval_job.get('job_status')})"
                    )
                else:
                    out.notes.append(f"no sandbox_jobs with hf_traces_link for model_id={model_row['id']}")
            except Exception as exc:
                out.notes.append(f"sandbox_jobs lookup (post-RL) failed: {exc}")

            # Baseline eval = the model's base_model_id's sandbox_jobs.
            base_id = model_row.get("base_model_id")
            if base_id:
                try:
                    base_eval = _latest_eval_job_for_model(client, base_id)
                    if base_eval and base_eval.get("hf_traces_link"):
                        out.baseline_eval = base_eval["hf_traces_link"]
                        out.notes.append(
                            f"baseline_eval ← sandbox_jobs[{base_eval['id']}] for base_model_id={base_id}"
                        )
                    else:
                        out.notes.append(f"no sandbox_jobs with hf_traces_link for base_model_id={base_id}")
                except Exception as exc:
                    out.notes.append(f"sandbox_jobs lookup (baseline) failed: {exc}")
            else:
                out.notes.append("models.base_model_id is null; baseline_eval can't be auto-resolved")

    # ---- HF training_logs/ snapshot ----
    if fetch_training_logs:
        if training_log_cache is None:
            safe = model_name.replace("/", "__")
            training_log_cache = Path.home() / ".ot-agent" / "training_logs" / safe
        log_dir = _download_training_logs(model_name, training_log_cache)
        if log_dir is not None:
            out.training_log_dir = str(log_dir)
            out.notes.append(f"training_log_dir ← snapshotted {model_name}/training_logs/ to {log_dir}")
        else:
            out.notes.append(
                f"no training_logs/ in HF repo {model_name} "
                f"(expected at https://huggingface.co/{model_name}/tree/main/training_logs)"
            )
    else:
        out.notes.append("fetch_training_logs=False; training_log_dir not populated")

    return out
