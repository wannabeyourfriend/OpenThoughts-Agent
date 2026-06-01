"""IrisLauncher — base class for submitting OT-Agent jobs to a Marin Iris cluster.

This is the Iris analog of ``hpc/cloud_launch_utils.CloudLauncher`` (which
targets SkyPilot). It exists in parallel rather than as a "provider" plugin
because Iris and SkyPilot disagree on key abstractions (workdir bind mount
vs file_mounts, in-cluster scheduling vs bring-up-a-VM, autostop vs job
timeout). Trying to share one interface created leaky bolts; two clean
modules is cheaper to reason about.

Backend-agnostic helpers under ``hpc/`` (e.g. ``arg_groups``,
``harbor_utils``, ``datagen_config_utils``) are reused as-is.

Output handling — GCS only. The workload writes directly to
``--gcs-output-dir/<job-name>/`` (default
``gs://marin-eu-west4/ot-agent/``; override with ``$OT_AGENT_GCS_OUTPUT_ROOT``
or the flag). A local fetch daemon (``hpc.iris_fetch_daemon``, planned)
polls the iris controller and pulls completed jobs into
``~/.ot-agent/runs/<job-name>/``. The previous "rsync from worker
workdir" mode was removed on 2026-05-22: the worker workdir is on
ephemeral tmpfs and iris GCs it at task end, so any laptop-side rsync
loop is fragile by construction; see
``notes/marin/flows/iris-outputs-redesign.md`` for the post-mortem.

Multi-host slices (TPU vm_count > 1) are scaffolded but only validated
on v6e-8. Confirm cross-host JAX init + coscheduling before relying on
larger slices.
"""

from __future__ import annotations

import argparse
import json as _json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from hpc.local_paths import PATHS as LOCAL_PATHS, ensure as ensure_local_paths
from hpc.iris_job_registry import register_submission, get_latest_by_job_name

DEFAULT_TASK_IMAGE = "ghcr.io/open-thoughts/openthoughts-agent:tpu"
DEFAULT_CLUSTER_CONFIG = "lib/iris/config/marin.yaml"
DEFAULT_PRIORITY = "interactive"

# Default GCS prefix for workload outputs. EU-region matches where most
# of our v6e-preemptible TPU slices land; us-region jobs incur small
# cross-region writes (eval outputs are ~MB-scale, so this is fine).
# Override with $OT_AGENT_GCS_OUTPUT_ROOT or the --gcs-output-dir flag.
DEFAULT_GCS_OUTPUT_ROOT = "gs://marin-eu-west4/ot-agent"


# Maps the multi-region prefix in a single-region zone name
# ("us-east5" → "us") to the GCS multi-region bucket that's free/cheap
# from any single-region in that prefix. We use multi-region buckets
# (gs://marin-models-us, gs://marin-models-eu) rather than per-zone
# buckets so a job that gets re-scheduled into a sibling zone within
# the same continent (e.g. us-east5 → us-central1) still reads/writes
# locally.
_REGION_PREFIX_TO_BUCKET = {
    "us": "gs://marin-models-us",
    "europe": "gs://marin-models-eu",
}


def _region_prefix(region: str) -> Optional[str]:
    """Return the multi-region prefix ('us', 'europe') for a single-region.

    >>> _region_prefix("us-east5")
    'us'
    >>> _region_prefix("europe-west4")
    'europe'
    >>> _region_prefix("asia-northeast1")  # unmapped
    None
    """
    for prefix in _REGION_PREFIX_TO_BUCKET:
        if region.startswith(prefix + "-"):
            return prefix
    return None


def _gcs_bucket_for_region(region: str) -> Optional[str]:
    """Return the cheapest GCS bucket for workers in ``region``.

    Returns the multi-region bucket (``gs://marin-models-us`` /
    ``gs://marin-models-eu``) matching the region's continent. Returns
    None for unmapped regions — callers should fall back rather than
    silently emit a wrong bucket.
    """
    prefix = _region_prefix(region)
    return _REGION_PREFIX_TO_BUCKET.get(prefix) if prefix else None


_GCS_URI_RE = re.compile(r"gs://(marin-models-(?:us|eu))(?:/|$)")


def _scan_yaml_for_gcs_paths(yaml_path: Path) -> List[tuple]:
    """Return ``[(field_dotted_path, gcs_uri, bucket), ...]`` for every gs://marin-models-{us,eu}/...
    string in the YAML at ``yaml_path``.

    Walks the parsed YAML recursively. ``field_dotted_path`` is the
    dotted location ('vllm_server.model_path', 'engine.model', etc.) for
    error messages. ``bucket`` is the matched bucket name (the part used
    for region-fit checks). Returns [] if the file isn't readable or
    isn't valid YAML.
    """
    try:
        import yaml  # PyYAML is already a project dep
    except ImportError:
        return []
    try:
        text = yaml_path.read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
    except (OSError, yaml.YAMLError):
        return []

    matches: List[tuple] = []

    def walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{prefix}[{i}]")
        elif isinstance(node, str):
            m = _GCS_URI_RE.search(node)
            if m:
                matches.append((prefix, node, m.group(1)))

    walk(doc, "")
    return matches


def assert_yaml_regions_match_pin(yaml_paths: List[Path], pinned_region: str) -> None:
    """Fail-fast when any YAML hardcodes a GCS bucket in the wrong region.

    Iris pins the job to ``pinned_region`` at submit time. If a YAML
    points at ``gs://marin-models-us/...`` and the worker will live in
    ``europe-west4``, every model-weight read crosses continents at GCS
    egress prices. We refuse to submit in that case and tell the user
    which path is wrong + which bucket to use.

    The launcher already pins the *output* bucket automatically; weight
    paths inside config YAMLs are advisory (we don't auto-rewrite them
    on purpose — silent rewrites mask which mirror is canonical and have
    failed before when the EU mirror was incomplete). The contract is:
    keep your YAML's bucket consistent with the variant's natural
    region, or run with ``--gcs-output-dir`` explicitly to opt out of
    the region pin entirely.
    """
    expected_bucket = _gcs_bucket_for_region(pinned_region)
    if not expected_bucket:
        return  # Unmapped region (e.g. asia-*); can't validate.
    expected_bucket_name = expected_bucket.removeprefix("gs://")  # "marin-models-us"
    violations: List[str] = []
    for path in yaml_paths:
        if not path or not Path(path).is_file():
            continue
        for field, uri, bucket in _scan_yaml_for_gcs_paths(Path(path)):
            if bucket != expected_bucket_name:
                violations.append(
                    f"  {path}: {field} = {uri!r}\n"
                    f"    bucket {bucket!r} but iris pinned region {pinned_region} "
                    f"expects {expected_bucket_name!r}"
                )
    if violations:
        raise SystemExit(
            "[iris] YAML model paths point at the wrong region for the iris-pinned "
            f"worker (region={pinned_region}, expected bucket={expected_bucket_name}). "
            "Cross-region GCS reads are expensive; refusing to submit.\n\n"
            + "\n".join(violations)
            + f"\n\nFix: swap the bucket in the YAML to {expected_bucket!r}, or "
            "pass --gcs-output-dir explicitly to opt out of the region pin."
        )


def _iris_query(cluster_config: str, sql: str, timeout: int = 30) -> List[dict]:
    """Run a SQL query against the iris controller, return parsed rows.

    Shells out to the ``iris query -f json`` CLI rather than wrangling
    the connectrpc client directly: it handles tunnel setup, auth, and
    the protobuf round-trip for us. ``shutil.which`` picks up the iris
    binary from whichever venv the launcher is running in (e.g. the
    otagent conda env on the launch host).
    """
    iris_bin = shutil.which("iris")
    if iris_bin is None:
        # Fall back to the marin venv if iris isn't on PATH. This is
        # the developer's expected layout per [[iris-python-env]].
        iris_bin = "/Users/benjaminfeuer/Documents/marin/.venv/bin/iris"
    if not Path(iris_bin).exists():
        raise RuntimeError(
            "iris CLI not found on PATH or at the documented marin venv "
            "location; can't run dynamic region discovery."
        )
    cmd = [iris_bin, "--config", str(cluster_config), "query", "-f", "json", sql]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"iris query failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    out = result.stdout.strip()
    if not out:
        return []
    return _json.loads(out)


def discover_region_for_tpu(
    cluster_config: str, tpu_spec: str
) -> tuple[Optional[str], List[dict]]:
    """Pick the region with the most v5p/v6e capacity for ``tpu_spec``.

    Queries iris's worker_attributes table to find every region that has
    workers of the requested device variant, ranks them by unassigned-
    worker count (warm/idle workers first), and returns the top region
    plus the full ranked breakdown.

    Returns ``(region, rows)`` where ``region`` may be ``None`` if no
    workers are visible (e.g., scale-group exists but has been scaled to
    zero). Caller falls back to the static default in that case.

    The pin happens at submit time and iris's scheduler honors region
    constraints across preempt-retries, so a job picked into us-east5
    stays in us-east5 (or another us-* zone iris adds later if the
    constraint is multi-region) for its whole lifetime — preserving
    harbor's resume invariant on a stable ``jobs_dir`` location.
    """
    # slice_id is NULL/empty for unassigned (warm) workers. Rank regions
    # by unassigned count (more warm = better odds of avoiding a queue)
    # then by total (tie-breaker for hot regions still likely to recycle
    # soonest).
    sql = (
        "SELECT wa.str_value AS region, "
        "COUNT(*) AS total, "
        "SUM(CASE WHEN w.slice_id IS NULL OR w.slice_id = '' THEN 1 ELSE 0 END) AS unassigned "
        f"FROM workers w JOIN worker_attributes wa ON w.worker_id = wa.worker_id "
        f"WHERE w.device_variant = '{tpu_spec}' AND wa.key = 'region' "
        "GROUP BY wa.str_value "
        "ORDER BY unassigned DESC, total DESC, region"
    )
    rows = _iris_query(cluster_config, sql)
    # Drop rows for regions we can't route to a known bucket — picking
    # one would just re-create the cross-continent problem.
    candidates = [r for r in rows if r.get("region") and _gcs_bucket_for_region(r["region"])]
    if not candidates:
        return None, rows
    return candidates[0]["region"], rows


def parse_tpu_vm_count(tpu_spec: Optional[str]) -> int:
    """Return the host-VM count for a TPU variant via iris's topology table.

    Delegates to iris ``get_tpu_topology`` — the same source iris uses to
    auto-set replicas — so the count matches reality across families. The
    old ``chips / 4`` arithmetic was wrong on two axes: chips-per-host
    varies (v6e-8 packs 8 chips on a single host, v5p hosts have 4), and
    v5p suffixes count cores, not chips (``v5p-32`` = 32 cores = 16 chips =
    4 hosts, not ``32 / 4 = 8``). Returns 1 when no TPU is requested; an
    unrecognized variant raises ``ValueError`` from ``get_tpu_topology``
    (fail fast rather than guess).
    """
    if not tpu_spec:
        return 1
    from iris.cli.job import get_tpu_topology

    return get_tpu_topology(tpu_spec).vm_count


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
                       help=f"TPU variant (default: {self.default_tpu}).")
        g.add_argument("--replicas", type=int, default=1,
                       help="Replica count passed to iris submit (default 1). "
                            "For a multi-host TPU slice iris REQUIRES one task "
                            "per VM and auto-scales replicas=1 -> vm_count "
                            "(iris adjust_tpu_replicas); those tasks form ONE "
                            "JAX device mesh (jax.distributed.initialize) that a "
                            "single engine spans — this is required, not "
                            "duplication. Use N*vm_count to request N slices. "
                            "NOTE: run_tracegen currently runs harbor on EVERY "
                            "task; the per-host duplication is the harbor layer, "
                            "not the replica count (fix = gate harbor to the "
                            "driver rank, see #69).")
        g.add_argument("--cpu", type=float, default=8.0,
                       help="CPU cores for the entrypoint task (default 8).")
        g.add_argument("--memory", default="256GB",
                       help="Memory for the entrypoint task (default 256GB). "
                            "v6e workers have 720GB total, so 256GB covers "
                            "HF weight loading for models up to ~120B bf16 "
                            "or ~400B AWQ-4-bit with comfortable headroom. "
                            "Bump for larger models; drop to 64GB for small "
                            "smokes if you want to be polite to the queue.")
        g.add_argument("--disk", default="100GB",
                       help="Ephemeral disk (default 100GB). marin's v6e/v5p "
                            "workers cap per-VM disk at 100GB; requests above "
                            "that queue forever waiting on the autoscaler which "
                            "can't provision a larger-disk worker. For models "
                            "whose weights exceed 100GB, use --load-format "
                            "runai_streamer + gs://-hosted weights instead of "
                            "bumping disk.")
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
        g.add_argument("--extras", action="append", default=None,
                       help="OpenThoughts-Agent extras to install in the iris worker's "
                            "/app/.venv via `uv sync --extra <name>`. Repeatable. "
                            "Default: ['datagen-tpu'] (matches the :tpu task image's "
                            "intended dep set). Pass --extras '' to install no extras.")

        og = parser.add_argument_group("outputs")
        og.add_argument("--gcs-output-dir", "--gcs_output_dir",
                        default=os.environ.get("OT_AGENT_GCS_OUTPUT_ROOT", DEFAULT_GCS_OUTPUT_ROOT),
                        help=f"GCS prefix for workload outputs; workload writes to "
                             f"<this>/<job-name>/. Defaults to $OT_AGENT_GCS_OUTPUT_ROOT or "
                             f"{DEFAULT_GCS_OUTPUT_ROOT}. The fetch daemon "
                             f"(hpc.iris_fetch_daemon) pulls completed jobs from here into "
                             f"{LOCAL_PATHS.runs}/<job-name>/.")

        rg = parser.add_argument_group("resume")
        rg.add_argument("--resume-from", "--resume_from", dest="resume_from", default=None,
                        help="Resume harbor state from a previously-submitted iris job "
                             "(by job_name; looked up in the local registry "
                             f"at {LOCAL_PATHS.state}/iris_jobs.db). The new iris job gets a "
                             "fresh timestamped name (for iris-level uniqueness), but the "
                             "harbor --job_name and --jobs-dir are routed at the old job's "
                             "GCS path so harbor's _maybe_init_existing_job picks up the "
                             "existing trial results and only runs the unmatched remaining "
                             "trials. No config gating: per the user's direction (2026-05-24), "
                             "OT-Agent and harbor already validate compatibility on resume.")

        sg = parser.add_argument_group("secrets")
        # Default to $OT_AGENT_SECRETS_ENV, then ~/Documents/secrets.env if it
        # exists — the canonical location for the user's credentials file.
        # File values override the os.environ passthrough below, so this is
        # the safer-by-default path: without it, a stale shell-cached
        # DAYTONA_API_KEY can ride into the iris worker and cause harbor's
        # auto_snapshot path to fail with "Sandbox not found" even when the
        # snapshot is ACTIVE on the right org.
        _default_secrets = os.environ.get("OT_AGENT_SECRETS_ENV") or os.path.expanduser(
            "~/Documents/secrets.env"
        )
        if not os.path.isfile(_default_secrets):
            _default_secrets = None
        sg.add_argument("--secrets-env", "--secrets_env", default=_default_secrets,
                        help="Path to a KEY=VALUE env file (~/Documents/secrets.env style). "
                             "Every entry is loaded into the iris task's env_vars at submit "
                             "time. Pairs with the hardcoded launcher passthrough list "
                             "(DAYTONA_API_KEY, OPENAI_API_KEY, etc.) — file values win on "
                             "conflict, explicit `-e` iris-CLI flags can't override since we "
                             "use IrisClient.submit() directly. Lines starting with '#' and "
                             "blank lines are ignored; leading 'export ' is stripped. "
                             "Defaults to $OT_AGENT_SECRETS_ENV, else ~/Documents/secrets.env "
                             "if it exists.")
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

    @staticmethod
    def load_secrets_env_into_os_environ(secrets_env: Optional[str]) -> int:
        """Read ``secrets_env`` (KEY=VALUE) into ``os.environ`` on the launch host.

        Called early in ``normalize_paths`` so launch-host hooks that read
        ``os.environ`` directly (e.g., ``get_daytona_api_key_override`` in the
        snapshot pre-build) see the file values, not the shell's possibly-stale
        cache. The iris worker still gets the same values via the existing
        ``--secrets-env`` parser in ``run()``.

        **File values override existing os.environ entries.** An explicit
        secrets file is more intentional than a shell-cached value, and the
        common failure mode is a stale ``DAYTONA_API_KEY`` lingering in a
        zsh shell that gets propagated to the iris worker — producing
        "Sandbox not found" on every harbor trial because the snapshot
        lives on a different Daytona org. This matches the existing
        worker-side semantics at ``run()``'s ``--secrets-env`` parser
        (``env_vars[k] = v  # file values override passthrough``).

        Returns the number of keys loaded. Returns 0 silently when
        ``secrets_env`` is None or the file is missing.
        """
        if not secrets_env:
            return 0
        path = Path(secrets_env).expanduser().resolve()
        if not path.is_file():
            return 0
        loaded = 0
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                v = v[1:-1]
            if not k:
                continue
            os.environ[k] = v  # file overrides shell — see docstring
            loaded += 1
        return loaded

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

        if not args.gcs_output_dir:
            raise SystemExit(
                "--gcs-output-dir is required (set OT_AGENT_GCS_OUTPUT_ROOT or pass the flag)."
            )

        # Dynamic region discovery + pin. When the user did NOT explicitly
        # set --gcs-output-dir or $OT_AGENT_GCS_OUTPUT_ROOT (i.e., the value
        # is still the cross-continent default), query iris for which
        # region has v5p/v6e capacity for this TPU spec, override the
        # output bucket to the matching multi-region bucket, and pin the
        # iris job to that region so preempt-retry can't cross-continent
        # failover. Skipped on --resume-from (the existing job's bucket
        # is authoritative) and when no TPU is requested.
        args._pinned_region = None
        user_set_output = bool(
            os.environ.get("OT_AGENT_GCS_OUTPUT_ROOT")
            or args.gcs_output_dir != DEFAULT_GCS_OUTPUT_ROOT
        )
        if (
            not user_set_output
            and not getattr(args, "resume_from", None)
            and getattr(args, "tpu", None)
        ):
            try:
                region, rows = discover_region_for_tpu(args.cluster_config, args.tpu)
            except Exception as exc:
                print(
                    f"[iris] Region discovery failed ({exc}); falling back to "
                    f"static default {DEFAULT_GCS_OUTPUT_ROOT}. Cross-region "
                    "egress possible if iris schedules outside that bucket's "
                    "region.",
                    file=sys.stderr, flush=True,
                )
            else:
                if region is None:
                    print(
                        f"[iris] No workers visible for --tpu={args.tpu}; falling "
                        f"back to static default {DEFAULT_GCS_OUTPUT_ROOT}.",
                        file=sys.stderr, flush=True,
                    )
                else:
                    bucket = _gcs_bucket_for_region(region)
                    args.gcs_output_dir = f"{bucket}/ot-agent"
                    args._pinned_region = region
                    summary = ", ".join(
                        f"{r['region']}: {r.get('unassigned', 0)} warm / "
                        f"{r.get('total', 0)} total"
                        for r in rows if r.get('region')
                    )
                    print(
                        f"[iris] Region pin: --tpu={args.tpu} → {region} "
                        f"(bucket {bucket}). Capacity: {summary or 'none reported'}.",
                        flush=True,
                    )

        # Fail-fast on cross-region YAML model paths. Iris just pinned the
        # job to a region; if a YAML hardcodes a GCS bucket in the wrong
        # continent (e.g. model_path=gs://marin-models-us/... but pinned
        # region=europe-west4), every weight read would cross continents.
        # We don't auto-rewrite — that masks broken mirrors. We refuse to
        # submit and tell the user exactly which field is wrong.
        if args._pinned_region:
            yaml_attrs = ("datagen_config", "harbor_config", "eval_config",
                          "config", "harbor_yaml")
            yaml_paths = [
                Path(getattr(args, attr))
                for attr in yaml_attrs
                if getattr(args, attr, None)
            ]
            assert_yaml_regions_match_pin(yaml_paths, args._pinned_region)

        # --resume-from: look up a previously-submitted job and route the
        # new task's harbor command at the old GCS path. The iris-level
        # job_name still gets a fresh timestamp (iris rejects duplicates);
        # the harbor-level identity (jobs_dir + job_name) is preserved so
        # harbor's _maybe_init_existing_job (job.py:203) finds existing
        # trial results and skips them. We stash the old harbor identity
        # on the args namespace so subclass build_task_command can read it.
        resume_target = getattr(args, "resume_from", None)
        if resume_target:
            prev = get_latest_by_job_name(resume_target)
            if prev is None:
                raise SystemExit(
                    f"--resume-from {resume_target!r}: no record found in "
                    f"{LOCAL_PATHS.state}/iris_jobs.db. Available recent jobs:\n"
                    "  python -c 'from hpc.iris_job_registry import list_all; "
                    "[print(r.job_name) for r in list_all(limit=20)]'"
                )
            ts = time.strftime("%Y%m%d-%H%M%S")
            args.job_name = getattr(args, "job_name", None) or f"{prev.job_name}-resume-{ts}"
            args._harbor_job_name_override = prev.job_name
            args._resume_gcs_output_dir = prev.gcs_output_dir
            print(
                f"[iris] Resume mode: harbor job_name={prev.job_name}  "
                f"gcs={prev.gcs_output_dir}",
                flush=True,
            )

        job_name = self._derive_job_name(args)
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"

        # The workload writes outputs directly to GCS; the fetch daemon
        # pulls them back to LOCAL_PATHS.runs/<job-name>/ on completion.
        if resume_target:
            # Resume: point at the OLD job's full GCS path so harbor finds
            # its existing config.json / trial dirs. Do NOT re-join job_name.
            remote_output_dir = args._resume_gcs_output_dir.rstrip("/")
        else:
            remote_output_dir = f"{args.gcs_output_dir.rstrip('/')}/{job_name}"

        # Make sure the local managed tree exists so the daemon (and any
        # downstream consumers) find LOCAL_PATHS.runs/ on first run.
        ensure_local_paths(
            LOCAL_PATHS.home, LOCAL_PATHS.state, LOCAL_PATHS.runs, LOCAL_PATHS.logs,
        )

        command = self.build_task_command(args, remote_output_dir)
        env_vars = self.build_env(args)

        # iris-serve gating. iris runs the entrypoint on EVERY VM of the slice
        # (one task/VM; adjust_tpu_replicas scales replicas=1 -> vm_count), so
        # the worker's LocalHarborRunner.run() must (a) bring up ONE cross-host
        # Ray cluster via scripts/vllm/start_vllm_iris_controller.py instead of
        # the SLURM/single-host start_vllm_ray_controller.py, and (b) gate
        # harbor to the driver rank (IRIS_TASK_ID==0). This env var is the
        # signal — it is only ever set here, on the iris entrypoint path. The
        # rendezvous dir is a shared gs:// location (under the job's GCS output
        # prefix) where rank 0 publishes the Ray head IP for the worker ranks.
        env_vars.setdefault("OT_AGENT_IRIS_SERVE", "1")
        env_vars.setdefault(
            "OT_AGENT_IRIS_RENDEZVOUS_DIR",
            f"{remote_output_dir.rstrip('/')}/_ray_rendezvous",
        )

        # Wire the iris controller's XLA persistent cache to the same
        # region-matched bucket we picked above. The worker appends
        # /<cpu_tag>/<model_tag>/ to namespace per-microarch (the only
        # axis the JAX cache key doesn't already discriminate; see
        # [[xla-persistent-cache-cross-host-poison]]). One shared base
        # across all jobs is fine — JAX hashes the HLO into per-config
        # subdirs of its own beneath the dir we hand it. Disable with
        # OT_AGENT_XLA_CACHE_BASE=disabled in the environment.
        if args.gcs_output_dir:
            cache_root = args.gcs_output_dir.rstrip("/").rsplit("/ot-agent", 1)[0]
            env_vars.setdefault(
                "OT_AGENT_XLA_CACHE_BASE",
                f"{cache_root}/ot-agent/xla_cache",
            )

        # Default extras = ["datagen-tpu"]; allow override via repeated --extras
        # or --extras '' (single empty) to install nothing extra.
        if args.extras is None:
            extras = ["datagen-tpu"]
        else:
            extras = [e for e in args.extras if e]

        # OT-Agent's build_support.py syncs the sft/llamafactory git submodule
        # at every setuptools.build_meta call (i.e. every editable install),
        # even when no sft-* extra is being installed. Inside the iris worker
        # container there's no git remote configured for that submodule, so
        # the sync errors out with exit 128. The build_support helper already
        # supports an escape hatch — opt in when no sft-* extra is requested.
        if not any(e.startswith("sft-") for e in extras):
            env_vars.setdefault("OT_AGENT_SKIP_SFT_SYNC", "1")

        # OT-Agent uses setuptools with [tool.setuptools.packages.find]
        # listing several top-level dirs (hpc, eval, data, ...). When iris's
        # entrypoint runs `python eval/local/run_eval.py`, Python sets
        # sys.path[0] to /app/eval/local, not /app — so `from hpc.* import
        # ...` raises ModuleNotFoundError. Setting PYTHONPATH=/app at boot
        # exposes the top-level dirs but in iris workers ALSO triggers
        # "unknown location" namespace-package resolution for some real
        # wheels (e.g. pydantic), so we can't use that. Instead we rewrite
        # the user command into a tiny python -c bootstrap that appends
        # /app to sys.path AFTER the venv has been activated and the
        # interpreter has built its initial path — namespace package
        # machinery has already cached real packages, so appending /app at
        # the end is safe.

        # The :tpu image sets ENV VIRTUAL_ENV=/opt/openthoughts/.venv so its
        # own preinstalled wheels are visible at container start. iris's
        # entrypoint runs `uv sync ...` from /app, then `source .venv/bin/
        # activate`, expecting `.venv` to live under /app. uv honors the
        # existing VIRTUAL_ENV unless told otherwise, so without this it
        # installs deps into /opt/openthoughts/.venv and then activates an
        # empty /app/.venv at run time — every `import pydantic` fails.
        # Force uv to use /app/.venv via UV_PROJECT_ENVIRONMENT, which has
        # higher precedence than VIRTUAL_ENV.
        env_vars.setdefault("UV_PROJECT_ENVIRONMENT", "/app/.venv")
        # Also clear VIRTUAL_ENV so `uv pip install` (used by iris for
        # cloudpickle/py-spy/memray) lands in /app/.venv, not the image's
        # preinstalled venv at /opt/openthoughts/.venv.
        env_vars.setdefault("VIRTUAL_ENV", "/app/.venv")
        # Force uv to materialize wheel contents into the venv instead of
        # symlinking them from /root/.cache/uv/archive-v0/... . On iris
        # workers, the uv cache lives in a tmpfs / different mount than the
        # venv: `uv sync` builds the symlinks during sync, but when the user
        # command runs the cache target is unreadable, so Python sees e.g.
        # /app/.venv/.../pydantic/__init__.py as a broken symlink and falls
        # back to namespace-package resolution — `from pydantic import
        # BaseModel` then raises "cannot import name BaseModel from
        # 'pydantic' (unknown location)". Copy mode avoids the symlink path
        # entirely. Confirmed via _iris_diag.py: pydantic/__init__.py was a
        # symlink to /root/.cache/uv/archive-v0/BYLjs1LAJOgakDOL/... which
        # didn't exist at runtime.
        env_vars.setdefault("UV_LINK_MODE", "copy")
        # Forward Ray/vLLM subprocess stdout/stderr to the parent process so
        # they appear in ``iris job logs``. Without this, vLLM controller
        # crashes during init (e.g. before ``write_endpoint_json`` runs) leave
        # no diagnostic trail in iris — the workload exits with the generic
        # "vLLM controller exited before writing the endpoint JSON" symptom
        # and the actual stacktrace is only in the per-task workdir
        # ``logs/vllm_controller.log``, which rsync hasn't picked up yet.
        env_vars.setdefault("OT_AGENT_INHERIT_SUBPROC_LOGS", "1")
        # Skip the vLLM --help flag-discovery probe in start_vllm_ray_controller.
        # On vllm-tpu (0.20.0) the import path of vllm.entrypoints.openai
        # cold-bootstraps libtpu inside a subprocess.run, which can hang for
        # multi-minute stretches and deadlock the parent controller with no
        # diagnostic output. The launcher emits a stable known-good set of
        # flags so skipping discovery is safe.
        env_vars.setdefault("VLLM_SKIP_FLAG_DISCOVERY", "1")
        # Skip the pre-Popen Ray probe in start_vllm_ray_controller. The probe
        # calls ray.init/cluster_resources/shutdown to print diagnostics, but
        # on the v6e-4 TPU runtime this sequence has been observed to hang
        # silently right after "Connected to Ray cluster". The probe isn't
        # load-bearing — vLLM does its own ray.init internally.
        env_vars.setdefault("VLLM_SKIP_RAY_PROBE", "1")

        # tpu_inference resolves MODEL_IMPL_TYPE=auto → flax_nnx for many
        # architectures (Gemma4, Qwen3.5Moe, etc), but flax_nnx doesn't
        # support AWQ weights (`NotImplementedError: awq quantization
        # method not supported. Supported methods are dict_keys([None,
        # 'fp8'])`). For our AWQ workloads (QuantTrio Qwen3.5-397B-AWQ,
        # QuantTrio MiniMax-M2.7-AWQ, etc) we need the PyTorch-XLA
        # ('vllm') path, which routes through tpu_inference's
        # VllmAWQConfig override. Marin's own native vllm_server.py
        # sets the same default — see lib/marin/src/marin/inference/
        # vllm_server.py:276.
        env_vars.setdefault("MODEL_IMPL_TYPE", "vllm")

        # NO cross-run persistent XLA compilation cache.
        #
        # A shared GCS JAX compilation cache (gs://marin-models-us/ot-agent/
        # xla-cache) was tried for Qwen122B-FP8 to skip the ~100-min cold
        # compile, but it is UNSAFE across iris's heterogeneous host pool.
        # JAX's persistent cache key for an XLA:CPU AOT executable does not
        # pin the exact host CPU feature set. The FP8→bf16 weight dequant
        # runs as an XLA:CPU kernel; a cache entry compiled on a host with
        # AVX `+prefer-no-gather/+prefer-no-scatter` and loaded on a host
        # lacking them logs `cpu_aot_loader.cc:220 ... could lead to ...
        # SIGILL` and produces DIVERGENT host-side weight arrays per
        # process. On a multi-host slice (v5p-32 DP=2) the replicated-weight
        # `device_put` then fails with `AssertionError: ArrayImpl passed to
        # device_put is not the same on each process`, killing all engine
        # cores (qwen122b v8d/v8e, 2026-05-27).
        #
        # Within a single slice the host VMs are homogeneous, so a fresh
        # per-run compile is self-consistent. Eating the cold compile each
        # run is correct; re-enabling a persistent cache requires either a
        # CPU-feature-keyed cache dir or pinning a single host machine type.

        # Run:AI Model Streamer config so `--load-format runai_streamer`
        # can pull safetensors from S3-compatible storage on workers that
        # can't disk-cache the full model (>50 GB total weights vs the
        # 100 GB v6e per-VM disk cap).
        #
        # AWS_ENDPOINT_URL is NOT setdefault'd here — it must come from
        # the user's ~/Documents/secrets.env via --secrets-env so the
        # launcher works against any S3-compatible target (real AWS,
        # MinIO at Jülich, GCS-S3-interop, etc.). The original Plan A
        # was GCS-S3-interop via HMAC keys (set AWS_ENDPOINT_URL=
        # https://storage.googleapis.com), but the user lacked
        # storage.hmacKeys.create on hai-gcp-models — we pivoted to
        # MinIO@Jülich, accessed via LAION_ENDPOINT.
        env_vars.setdefault("RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING", "False")
        env_vars.setdefault("AWS_EC2_METADATA_DISABLED", "true")

        # Forward sandbox-backend / external-API credentials from the
        # launcher's shell env into the iris worker. Iris auto-injects
        # HF_TOKEN, WANDB_API_KEY, HF_DATASETS_TRUST_REMOTE_CODE, and
        # TOKENIZERS_PARALLELISM but nothing else, so harbor's Daytona
        # client and other API-key-driven integrations need explicit
        # passthrough. The user typically loads these via
        # `source ~/Documents/secrets.env` before invoking the launcher.
        # Missing-from-env entries are skipped silently — harbor will
        # surface its own "DAYTONA_API_KEY not set" error if it actually
        # needs one. setdefault keeps any explicit -e overrides above.
        _LAUNCHER_ENV_PASSTHROUGH = (
            "DAYTONA_API_KEY",
            "DAYTONA_JWT_TOKEN",
            "DAYTONA_ORGANIZATION_ID",
            "DAYTONA_API_URL",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "TOGETHER_API_KEY",
            "FIREWORKS_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        )
        for _k in _LAUNCHER_ENV_PASSTHROUGH:
            _v = os.environ.get(_k)
            if _v:
                env_vars.setdefault(_k, _v)

        # --secrets-env loader. SkyPilot mounted this file into the container
        # and sourced it remotely; iris has no file_mounts so we parse it
        # client-side and copy KEY=VALUE pairs into env_vars. File entries
        # override the os.environ passthrough above (an explicit file is more
        # intentional than an inherited shell env).
        if getattr(args, "secrets_env", None):
            secrets_path = Path(args.secrets_env).expanduser().resolve()
            if not secrets_path.exists():
                raise FileNotFoundError(f"--secrets-env file not found: {secrets_path}")
            loaded: list[str] = []
            for line_no, raw_line in enumerate(secrets_path.read_text().splitlines(), 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue  # malformed; skip
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip()
                # Strip matching surrounding quotes if present.
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                if not k:
                    continue
                env_vars[k] = v  # file values override passthrough
                loaded.append(k)
            print(
                f"[iris] Secrets:    loaded {len(loaded)} entries from "
                f"{secrets_path}: {', '.join(sorted(loaded))}",
                flush=True,
            )

        # Alias S3-compat credentials → AWS_* env vars for runai_streamer.
        # ~/Documents/secrets.env may carry several S3-compat credential
        # pairs:
        #   - LAION_ACCESS_KEY + LAION_SECRET_KEY + LAION_ENDPOINT
        #     (MinIO@Jülich)
        #   - MARIN_HMAC_ACCESS_ID + MARIN_HMAC_SECRET
        #     (GCS S3-interop via hai-gcp-models HMAC keys; endpoint is
        #     always https://storage.googleapis.com, no separate env var)
        # plus the real-AWS pair (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
        # which `_load_secrets` already populated in env_vars by name.
        #
        # The C SDK can only carry ONE credential pair, so we need to
        # pick the right one based on which endpoint the YAML targets.
        # Priority: MARIN_HMAC_* (GCS S3-interop) > LAION_* (MinIO Jülich).
        # If the YAML pre-sets AWS_ENDPOINT_URL we honor that and pick
        # the credential pair matching it; otherwise default to LAION
        # (the historical pre-MARIN-HMAC behavior).
        endpoint_in_yaml = env_vars.get("AWS_ENDPOINT_URL")
        is_marin_endpoint = (endpoint_in_yaml is not None and
                              "storage.googleapis.com" in endpoint_in_yaml)
        has_marin = (
            "MARIN_HMAC_ACCESS_ID" in env_vars
            and "MARIN_HMAC_SECRET" in env_vars
        )
        has_laion = "LAION_ENDPOINT" in env_vars
        aliased: list[str] = []

        if has_marin and (is_marin_endpoint or not has_laion):
            env_vars.setdefault("AWS_ENDPOINT_URL", "https://storage.googleapis.com")
            env_vars["AWS_ACCESS_KEY_ID"] = env_vars["MARIN_HMAC_ACCESS_ID"]
            env_vars["AWS_SECRET_ACCESS_KEY"] = env_vars["MARIN_HMAC_SECRET"]
            aliased = [
                "AWS_ENDPOINT_URL ← https://storage.googleapis.com",
                "AWS_ACCESS_KEY_ID ← MARIN_HMAC_ACCESS_ID",
                "AWS_SECRET_ACCESS_KEY ← MARIN_HMAC_SECRET",
            ]
        elif has_laion:
            # AWS_ENDPOINT_URL: only auto-fill from LAION_ENDPOINT if YAML
            # didn't pre-set it.
            if "AWS_ENDPOINT_URL" not in env_vars:
                env_vars["AWS_ENDPOINT_URL"] = env_vars["LAION_ENDPOINT"]
                aliased.append("AWS_ENDPOINT_URL ← LAION_ENDPOINT")
            if "LAION_ACCESS_KEY" in env_vars:
                env_vars["AWS_ACCESS_KEY_ID"] = env_vars["LAION_ACCESS_KEY"]
                aliased.append("AWS_ACCESS_KEY_ID ← LAION_ACCESS_KEY")
            if "LAION_SECRET_KEY" in env_vars:
                env_vars["AWS_SECRET_ACCESS_KEY"] = env_vars["LAION_SECRET_KEY"]
                aliased.append("AWS_SECRET_ACCESS_KEY ← LAION_SECRET_KEY")

        if aliased:
            print(
                f"[iris] Aliased for runai_streamer S3 against "
                f"{env_vars.get('AWS_ENDPOINT_URL', '<unset>')}: "
                f"{', '.join(aliased)}",
                flush=True,
            )

        vm_count = parse_tpu_vm_count(args.tpu)

        local_dest = LOCAL_PATHS.runs / job_name

        print(f"[iris] Job:        /{user}/{job_name}", flush=True)
        print(f"[iris] Cluster:    {args.cluster_config}", flush=True)
        print(f"[iris] Image:      {args.task_image}", flush=True)
        print(f"[iris] TPU:        {args.tpu}  (vm_count={vm_count})", flush=True)
        print(f"[iris] Priority:   {args.priority}", flush=True)
        print(f"[iris] Extras:     {extras or '(none)'}", flush=True)
        print(f"[iris] Output:     {remote_output_dir}", flush=True)
        print(f"[iris] Fetch dest: {local_dest}/  (via hpc.iris_fetch_daemon)", flush=True)
        print(f"[iris] Command:    {shlex.join(command)}", flush=True)

        if args.dry_run:
            print("[iris] --dry-run: not submitting", flush=True)
            return 0

        if vm_count > 1:
            print(
                "[iris] NOTE: multi-host TPU slice (vm_count > 1). Validated on v6e-8 "
                "(2026-05-22 smoke #10); larger slices need their own validation pass.",
                file=sys.stderr, flush=True,
            )

        # Defer the heavy iris imports so --dry-run / --help stay snappy.
        from iris.client import IrisClient
        from iris.cluster.config import IrisConfig
        from iris.cluster.types import EnvironmentSpec, Entrypoint
        from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults, build_tpu_alternatives
        from iris.rpc import job_pb2

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
            # --replicas defaults to 1; for a multi-host TPU iris's
            # adjust_tpu_replicas (in client.submit) auto-scales 1 -> vm_count
            # because every VM in the slice must run a task to join the one
            # JAX device mesh. So this does NOT reduce the task count on a
            # multi-host slice (that is required for the mesh) — pass N*vm_count
            # to request N slices. The per-host *harbor* duplication is a
            # separate run_tracegen issue (run harbor on the driver rank only).
            replicas, coscheduling = resolve_multinode_defaults(
                primary_tpu, None, args.replicas
            )
            resources_proto = resources.to_proto()
            # Pin the job to the region we discovered at submit time, so
            # preempt-retries land back in the same continent and our
            # output bucket stays local.
            pinned_region = getattr(args, "_pinned_region", None)
            constraints = build_job_constraints(
                resources_proto=resources_proto,
                tpu_variants=tpu_variants,
                replicas=replicas,
                regions=(pinned_region,) if pinned_region else None,
                zone=None,
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

            # Wrap the user command in a bash bootstrap that:
            #   (1) re-syncs deps with --link-mode=copy to materialize wheel
            #       contents into /app/.venv, replacing the broken symlinks
            #       iris's build phase left behind (iris hardcodes
            #       --link-mode symlink at lib/iris/.../runtime/entrypoint.py
            #       and its DockerRuntime runs setup in a build container, so
            #       the symlinked /root/.cache/uv/archive-v0/... targets do
            #       not exist in the run container — every `import pydantic`
            #       resolves to a namespace package and `from pydantic import
            #       BaseModel` raises "unknown location"). Confirmed via
            #       eval/local/_iris_diag.py.
            #   (2) runs the original user command via a python -c
            #       bootstrap that appends /app to sys.path. See block
            #       above for why we can't just set PYTHONPATH=/app.
            # The first command arg is the entrypoint script (e.g.
            # eval/local/run_eval.py); the rest are passed through as
            # argv[1:]. `python -c '<bootstrap>' <script> args...` makes
            # sys.argv = ['-c', <script>, *args], so the bootstrap rewrites
            # sys.argv to drop the '-c' and run the script via
            # runpy.run_path with __name__ == '__main__'.
            if command and command[0] == "python" and len(command) >= 2:
                script_path = command[1]
                script_argv = command[2:]
                py_bootstrap = (
                    "import sys; "
                    "sys.path.append('/app'); "
                    "sys.argv = sys.argv[1:]; "
                    "import runpy; "
                    "runpy.run_path(sys.argv[0], run_name='__main__')"
                )
                # Build the uv sync flags to mirror what iris runs, but with
                # --link-mode=copy. --all-packages + --extra entries are the
                # only project-shape flags that matter here; everything else
                # (python version, frozen) iris's build already validated.
                extras_flags = " ".join(
                    f"--extra {shlex.quote(e.split(':', 1)[-1])}" for e in extras
                )
                # Use --reinstall to force uv to rewrite every package into
                # the venv as copies, replacing the broken symlinks iris's
                # build phase produced. Without --reinstall uv sees the
                # existing .dist-info entries, declares "already installed",
                # and skips — the broken symlinks stay broken.
                # IRIS_DEBUG_UV_SYNC=1 turns this on; defaults to quiet so
                # the run-phase resync logs don't drown the user output.
                quiet = "" if os.environ.get("IRIS_DEBUG_UV_RESYNC") else "--quiet"
                resync_cmd = (
                    "cd /app && "
                    f"uv sync {quiet} --frozen --reinstall --link-mode=copy "
                    f"--all-packages --no-group dev {extras_flags}".rstrip()
                )
                # Runtime patch step: apply ot-agent-side workarounds to
                # third-party packages that ship in the wheel (currently
                # the tpu-inference hbm_usage_bytes multi-host bug). Runs
                # after `uv sync` so we're patching the freshly-installed
                # copies, before the workload exec. The script is
                # idempotent and prints a one-line status per patch.
                patch_cmd = "python scripts/iris/patch_tpu_inference.py"
                # Quote the python -c body and script argv for the bash -c
                # invocation. We use a single shlex.join for the python
                # invocation so spaces/quotes in argv survive.
                py_invoke = shlex.join(
                    ["python", "-c", py_bootstrap, script_path, *script_argv]
                )
                bash_cmd = f"set -e; {resync_cmd}; {patch_cmd}; exec {py_invoke}"
                wrapped = ["bash", "-c", bash_cmd]
                entrypoint = Entrypoint.from_command(*wrapped)
            else:
                entrypoint = Entrypoint.from_command(*command)

            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=resources,
                environment=EnvironmentSpec(env_vars=env_vars, extras=extras),
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

            # Record the job in the local registry so the fetch daemon
            # knows where to pull outputs from on completion. Failures
            # here are non-fatal — the job is already submitted, and the
            # user can re-register later via `python -m hpc.iris_fetch_daemon
            # fetch <job-id>` once that module lands.
            try:
                register_submission(
                    job_id=full_job_id,
                    job_name=job_name,
                    submitted_at_iso=datetime.now(timezone.utc).isoformat(),
                    gcs_output_dir=remote_output_dir,
                    local_dest=local_dest,
                    cluster_config=str(args.cluster_config),
                )
            except Exception as e:
                print(f"[iris] WARN: could not register job locally: {e}", file=sys.stderr, flush=True)

            if args.no_wait:
                return 0

            try:
                status = job.wait(stream_logs=True, timeout=float("inf"))
                exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
            except KeyboardInterrupt:
                print(f"[iris] Terminating job {full_job_id}...", file=sys.stderr, flush=True)
                client.terminate_job(job.job_id)
                exit_code = 130

            print(f"[iris] Job exit: {exit_code}", flush=True)
            return exit_code

# Imported lazily inside .run() to keep CLI startup fast, but tiny enough
# to define here.
def _seconds_to_duration(secs: int):
    # Duration moved from iris.cluster.types to rigging.timing on a
    # marin/iris refactor; iris.client imports from rigging.timing now.
    from rigging.timing import Duration
    return Duration.from_seconds(secs)
