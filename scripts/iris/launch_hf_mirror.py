#!/usr/bin/env python3
"""Submit ``mirror_hf_to_gcs.py`` as an iris job for one or more HF repos.

Marin has no CPU-only worker pools as of 2026-05-22 — every iris worker
is TPU-typed — so the mirror runs on the smallest TPU slice we can ask
for (v6e-4). The TPU sits idle; we're paying for the worker's CPU /
RAM / network. Acceptable for a one-shot ~30-60 min mirror, but don't
schedule this against a busy queue.

Usage::

    python -m scripts.iris.launch_hf_mirror \\
        --repo cyankiwi/MiniMax-M2.7-AWQ-4bit \\
        --repo QuantTrio/GLM-5.1-AWQ \\
        --gcs-prefix gs://marin-eu-west4/ot-agent/models/
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse the proper IrisLauncher infrastructure so we get the same
# bash-bootstrap (uv sync --link-mode=copy + sys.path append) + secrets
# forwarding + GCS-default plumbing as the eval/tracegen launchers.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from hpc.iris_launch_utils import IrisLauncher
from hpc.launch_utils import PROJECT_ROOT


class HfMirrorIrisLauncher(IrisLauncher):
    """Iris launcher for the HF→GCS mirror script."""

    task_name = "ot-hf-mirror"
    job_name_prefix = "hf-mirror"
    default_tpu = "v6e-4"

    # Default fan-out: one multi-region bucket in US + one in EU. Together
    # they cover all 6 iris worker regions (us-central1, us-central2,
    # us-east1, us-east5, us-west4, europe-west4) with zero cross-region
    # egress at read time. Override with --gcs-prefix to mirror to a
    # different set of locations.
    DEFAULT_GCS_PREFIXES = (
        "gs://marin-models-us/ot-agent/models",
        "gs://marin-models-eu/ot-agent/models",
    )

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--repo", action="append", required=True,
                            help="HF model repo id (repeatable).")
        parser.add_argument(
            "--gcs-prefix", "--gcs_prefix", action="append", default=None,
            help="GCS prefix; each repo lands under <prefix>/<repo>/. "
                 "Repeatable — every prefix gets a full mirror, so a single "
                 "HF download fans out to every region. Defaults to the "
                 "marin-models-us + marin-models-eu multi-region buckets, "
                 "which together cover all iris worker regions with zero "
                 "cross-region egress.",
        )
        parser.add_argument("--job_name",
                            help="Override the auto-generated iris job name.")
        parser.add_argument("--dry_run", action="store_true",
                            help="Print the command without submitting.")

    def normalize_paths(self, args: argparse.Namespace) -> None:
        if not args.gcs_prefix:
            args.gcs_prefix = list(self.DEFAULT_GCS_PREFIXES)
        bad = [p for p in args.gcs_prefix if not p.startswith("gs://")]
        if bad:
            raise SystemExit(
                f"--gcs-prefix entries must start with gs:// (got {bad!r})"
            )

    def build_task_command(self, args: argparse.Namespace, remote_output_dir: str) -> list[str]:
        # remote_output_dir is the GCS prefix the daemon will fetch from
        # on completion. The mirror script doesn't write outputs there —
        # it streams the actual HF shards to args.gcs_prefix. Pass it
        # through so the registry record + daemon fetch still work; the
        # daemon will just find an empty (or nearly-empty) directory.
        cmd: list[str] = ["python", "scripts/iris/mirror_hf_to_gcs.py"]
        for prefix in args.gcs_prefix:
            cmd.extend(["--gcs-prefix", prefix])
        for repo in args.repo:
            cmd.extend(["--repo", repo])
        return cmd


def main() -> None:
    launcher = HfMirrorIrisLauncher(PROJECT_ROOT)
    parser = launcher.create_argument_parser(
        description="Mirror HF model repos to GCS via an iris worker.",
    )
    args = parser.parse_args()
    sys.exit(launcher.run(args))


if __name__ == "__main__":
    main()
