---
name: datagen-reclaim-stale-snapshots
description: >-
  Reclaim idle Daytona SNAPSHOTS org-wide to free space under the 60-snapshot cap,
  using scripts/daytona/daytona_snapshot_manager.py. Deletes only harbor__ per-
  environment snapshots idle past a threshold (default 120 min) — NEVER the shared
  base/template images (daytonaio/sandbox:*, daytona-*, windows-*), which do not
  rebuild-on-demand. Use when a datagen/eval launch hits SnapshotCapExceeded, when a
  monitor sweep finds the cap full, or as routine cap hygiene. This is the org-wide
  RECLAIM tool; to shrink a SINGLE dataset's unique-environment count instead, use
  datagen-reduce-dataset-snapshots. Snapshots are a DIFFERENT resource from sandboxes
  (for sandbox cleanup use utils-cleanup-stale-sandboxes).
---

# datagen-reclaim-stale-snapshots

Free Daytona **snapshot** quota by deleting `harbor__<hash>__snapshot` per-
environment snapshots that have been **idle past a threshold**, so a blocked
datagen/eval launch can pre-build its environments under the org's hard cap of
**60**.

## Why this is safe (and what it must never touch)

Harbor's Daytona backend builds one snapshot per unique task-environment hash
(`harbor__<hash>__snapshot`). These are **rebuilt on demand** by harbor's
`auto_snapshot` path, so deleting an idle one only costs a rebuild the next time
that exact environment is needed — never data loss.

The **base/template images are different** — `daytonaio/sandbox:*`,
`daytona-gpu`, `daytona-medium/large/small`, `windows-*`. They are shared
org-wide and do **not** rebuild-on-demand; deleting one breaks sandbox creation
for everyone. The tool's `--name-prefix harbor__` default (below) guarantees
only `harbor__` snapshots are ever flagged deletable — base images are always
kept, regardless of idle time.

For a Daytona *snapshot*, `state=active` is just "built and available" — it does
**not** mean a sandbox is currently running off it. So an idle `active` harbor
snapshot is precisely the reclaim target; the tool keys staleness on
`last_used_at` (falling back to `created_at` only when a snapshot was never
used). An environment that a live job is actively pulling shows `idle < threshold`
and is spared.

## Which org

The datagen/eval snapshot pre-build runs on the **`cli` org**, whose key is
`DAYTONA_API_KEY` (this is where `SnapshotCapExceeded` fires for a TPU datagen
launch). Target it with `--api-key-env DAYTONA_API_KEY`. (The RL rollouts use a
separate org, `DAYTONA_RL_API_KEY`; datagen never fills that one.)

## How to run

Always **audit (read-only) first**, then delete.

```bash
cd /Users/benjaminfeuer/Documents/OpenThoughts-Agent
source /Users/benjaminfeuer/miniconda3/etc/profile.d/conda.sh && conda activate otagent
source /Users/benjaminfeuer/Documents/secrets.env
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
SCRIPT=scripts/daytona/daytona_snapshot_manager.py

# 1. AUDIT — read-only, see what WOULD be reclaimed (120 min = 0.0833 days idle)
$PY $SCRIPT --api-key-env DAYTONA_API_KEY --stale-days 0.0833

# 2. RECLAIM — actually delete the stale harbor__ snapshots
$PY $SCRIPT --api-key-env DAYTONA_API_KEY --stale-days 0.0833 --delete-stale --yes
```

`--stale-days` takes a float, so sub-day thresholds are minutes/1440 (120 min =
`0.0833`, 1 h = `0.0417`, 6 h = `0.25`). **120 min is the standing default** for
unblocking a launch — long enough that a live job's actively-rotating
environments (idle < 2 h) are spared, short enough to reclaim the leftovers of
completed jobs. Raise it if you want to be more conservative.

## Flags (this tool)

| Flag | Default | Purpose |
|---|---|---|
| `--api-key-env` | `DAYTONA_DATA_API_KEY` | Org key env var. **Use `DAYTONA_API_KEY`** for the cli/datagen org. |
| `--stale-days` | 14 | Idle-days threshold (float). Use `0.0833` for 120 min. |
| `--name-prefix` | `harbor__` | Only names starting with this are deletable. **Do not widen it** — the default guards base images. |
| `--delete-stale` | off (dry-run) | Actually delete. |
| `--yes` | off | Skip the confirm prompt (for scripted/cron use). |
| `--json` | off | Machine-readable output. |

## Gotchas

- **Source `secrets.env` + otagent env** — the `daytona` SDK lives in the otagent
  conda env; use the full interpreter path.
- **Never pass `--name-prefix ''`** with `--delete-stale` — that would make base
  images deletable. The tool is safe only with the `harbor__` default.
- The old cap workaround was "delete only MISSING `harbor__` snapshots" (memory
  `daytona_snapshot_cap`); that stalls when there are 0 MISSING (all ACTIVE).
  This idle-based reclaim supersedes it: idle ACTIVE `harbor__` snapshots are the
  real reclaim pool.
- Reclaiming forces a rebuild the next time a live job hits a deleted
  environment — harmless (auto_snapshot rebuilds) but adds a one-time
  per-environment build latency. 120 min avoids this for actively-rotating envs.

## In the monitor sweep

The every-3-hours Iris sweep runs the **audit** each tick and **reclaims** when a
datagen refill is blocked (or the cap is ≥ ~58/60). See
**monitor-cron-sweep-iris** §5.
