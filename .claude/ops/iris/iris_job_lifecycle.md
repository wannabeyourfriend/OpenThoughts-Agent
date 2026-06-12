# Iris jobs on Google TPU cloud — launch, monitor, tear down

How to run OpenThoughts-Agent jobs on Marin's Iris-managed Google TPU cloud
without burning money or hitting the recurring footguns. Hardware specs (per-chip
HBM/FLOPs, slice totals) live in `iris_google_tpu_cloud_hardware.md`; this doc is
the operational lifecycle + pitfalls. Canonical upstream ops: `marin:lib/iris/OPS.md`.

Conventions used below: `IRIS = /Users/benjaminfeuer/Documents/marin/.venv/bin/iris`
(or `conda activate marin && uv run iris`); launch from the **otagent py3.12
conda env**; `source /Users/benjaminfeuer/Documents/secrets.env` first.

---

## 1. Launch

Two entrypoints (both submit `--no-wait`; the launchd fetch daemon mirrors
outputs back — see §2):
- **datagen / tracegen** → `data/cloud/launch_tracegen_iris.py`
- **eval** → `eval/cloud/launch_eval_iris.py`

Both forward `--harbor_config`, `--model` (or infer from `--datagen_config`),
`--tpu`, `--n_concurrent`, `--secrets-env`, `--upload_hf_repo`, `--gcs-output-dir`,
`--no-wait`, and auto-inject `--harbor_extra_arg=--jobs-dir=<gcs_output_dir>/<job>`
so harbor writes through fsspec/UPath straight to GCS. (See the `run-datagen-iris`
/ `run-eval-iris` skills for full launch templates.)

**Before you submit, get three things right — region, disk, node shape:**

### Region — the #1 cost footgun (cross-region egress)
The model weight buckets are **regional**: `gs://marin-models-us/...` and
`gs://marin-models-eu/...`. Reading a US-bucket model from a EU-region worker
(or vice-versa) is **cross-continent egress** — a major cost driver, and project
policy forbids it (`AGENTS.md`: never read/write large data across GCS regions;
never use Storage Transfer Service across regions). Rules:
- Keep **model bucket, `jobs_dir`/output bucket, and worker region in the same
  multi-region** (all US, or all EU).
- The launcher auto-pins the job to the region with the most capacity for the
  TPU type (`hpc/iris_launch_utils.py:discover_region_for_tpu`) and routes output
  to the matching multi-region bucket. **The static default
  (`DEFAULT_GCS_OUTPUT_ROOT`) is `gs://marin-eu-west4/...`** — so if you neither
  set `--gcs-output-dir` nor let the pin run, a US placement reads EU = egress.
- Passing `--gcs-output-dir gs://marin-models-us/ot-agent` **opts out of the
  region pin** (places on the first free worker in any US region — the fix for a
  collapsed single-region pool, see §3 preemption/stuck-PENDING) while keeping
  output in the US multi-region. Only do this with a US model bucket.

### Local disk — the ~100 GB/node ceiling
Each TPU **worker node has only ~100 GB of local disk.** Consequences:
- **Stream the model from GCS** (`--load-format runai_streamer`, gs:// URIs) —
  do NOT download a full large checkpoint to local disk (122B-FP8 alone is
  ~122 GB and won't fit).
- **Write `jobs_dir` to `gs://`, never local.** Per-trial outputs (trajectories,
  verifier dirs) go straight to the bucket; harbor must not buffer the whole job
  on the node.
- On **memory-heavy/repo-based datasets**, also keep harbor's RSS bounded with
  `release_trial_payloads_in_memory: true` (in `ctx32k_verified.yaml`) — otherwise
  the orchestrator accumulates completed-trial payloads and OOMs the container
  (host RAM, ~256 GB, distinct from the 100 GB disk).
- The `--disk` flag defaults to 5 GB ephemeral; raising it does not change the
  node's physical ceiling.

### Node shape — get chip/host counts from the codebase, not arithmetic
**Do not assume "chips ÷ 4 = hosts."** That heuristic is wrong (v5p variants
count *cores not chips*, and v6e single-host packs up to 8 chips). Authoritative
sources, in order:
- **Host/process count:** `iris.cli.job.get_tpu_topology("<variant>").vm_count`.
  Known good values: `v5p-8 → 1`, `v5p-16 → 2`, `v5p-32 → 4`, `v6e-8 → 1`,
  `v6e-16 → 4`. The launcher uses this to auto-set `--replicas`.
- **v5p naming is CORES, not chips:** `v5p-N` = N cores = **N/2 chips**. So
  `v5p-8` = 4 chips (1 host), `v5p-32` = 16 chips (4 hosts). **Tensor-parallel
  degree must be ≤ chip count, not core count** — TP=8 will not fit a v5p-8 (only
  4 chips).
- **Live capacity + real chip counts:** query the cluster's `workers` table —
  ```bash
  $IRIS --cluster=marin query "SELECT device_variant, count(*) workers, sum(total_tpu_count) chips FROM workers WHERE device_type='tpu' GROUP BY device_variant ORDER BY device_variant" -f csv
  ```
- **Pools / variants / zones available:** `marin:lib/iris/config/marin.yaml`
  (v5e/v6e/v5p/v4 families, sizes, zones). Per-chip HBM and slice totals are in
  `iris_google_tpu_cloud_hardware.md`.
- For 122B-FP8 specifically: it fits **v5p** (95 GB HBM/chip) but **not v6e-8**
  (32 GB/chip, 256 GB/slice) — weights + MoE footprint + compile peak exceed it.

**Cold-compile budget:** 122B-FP8 first-serve compile can take ~60 min. Pass
`--health_max_attempts 600`; the default (~50 min) kills the job before it serves.

---

## 2. Monitor

```bash
# Iris-side state (1=PENDING 2=starting 3=RUNNING 4=SUCCEEDED 5=FAILED 6=KILLED)
$IRIS --cluster=marin query "SELECT job_id, state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%' ORDER BY job_id DESC" -f csv

# Full-history analyzer (paginates the whole log; don't eyeball --tail)
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python scripts/iris/analyze_job_history.py /benjaminfeuer/<job> --refresh
#   sidecar JSON: total_runtime_s, iris_preemption_count, cycles[], serving_summary.{gen_tps,running}, non_empty_trials/total_trial_dirs, harbor_exception_stats

# Fetch daemon (mirrors GCS outputs → ~/.ot-agent/runs/<job>/ and captures .iris-job.log)
python -m hpc.iris_fetch_daemon status      # heartbeat should be ALIVE
$IRIS --cluster=marin job logs -f /benjaminfeuer/<job>    # live workload logs (controller tunnel)
```
Outputs land in `~/.ot-agent/runs/<job>/` (daemon rsync) + `.iris-job.log`. Use
the productive-trial rate (`non_empty/total`) and `harbor_exception_stats` as the
health signal; gen tok/s varies by dataset (short-task sets run lower by nature).
Jobs on `:tpu` image `ae085bc8`+ **auto-upload** their HF repo on a state-4
success — verify the repo exists before any manual rescue.

---

## 3. Pitfalls & recovery

### Preemption (these are `--preemptible` workers)
- Preemption is normal and frequent; a single slice can take 10+ preempts in a
  few hours. Each preempt → a fresh worker → **cold XLA recompile** (~60 min on
  v5p-8 cold; ~13-20 min warm).
- **XLA persistent cache** makes warm restarts fast. It is namespaced per
  CPU-microarch and per model under `OT_AGENT_XLA_CACHE_BASE` (region-matched
  bucket, auto-set on iris) — do not point two different host CPUs at the same
  cache subdir (cross-host poison → wrong execution).
- **harbor resumes from the gs:// `jobs_dir`** — completed trials persist across
  preempts, so progress is not lost, only the recompile time.
- `IRIS_TASK_ID` gains a `:N` suffix on retried/preempted attempts
  (e.g. `/user/job/0:2`) — any rank-parsing must strip it
  (`.rsplit('/',1)[-1].split(':',1)[0]`) or it crashes the moment a rank is retried.
- **Stuck PENDING** = no capacity for that TPU type in the pinned region (a
  preemptible pool can scale to zero). A finished job does NOT free its snapshot
  or instantly free capacity. Fix: relaunch **unpinned** with
  `--gcs-output-dir gs://marin-models-us/ot-agent` so iris places on any free US
  worker. Kill the stuck submission first only with user permission.

### Daytona snapshot cap
Launches build a per-env Daytona snapshot; the shared `cli` org caps at 60. On
`SnapshotCapExceeded`, delete **only `MISSING`-state `harbor__*` snapshots**
(broken builds, safe). NEVER broad-prune (`cleanup_unused_snapshots`) on the
shared org — it removes ACTIVE snapshots other jobs (yours or teammates') depend
on. Snippet in the `run-datagen-iris` skill.

### Local-storage growth on the launch host
The daemon mirror under `~/.ot-agent/runs/` (and `.iris-job.log`, which can be
10s of MB) accumulates across jobs. `python -m hpc.local_paths inventory` lists
sizes; `... clean --older-than 30d --apply` purges old runs. (This is the launch
*host*; distinct from the 100 GB worker-node ceiling in §1.)

### Empty GCS prefix after a "successful" job
Means the workload didn't route through UPath. Confirm
`--harbor_extra_arg=--jobs-dir=<gcs>` is in the submitted command
(`iris job bug-report <id>`), and that the harbor pin is the UPath-aware build.

---

## 4. Teardown

```bash
# Kill a job (ONLY with explicit user permission for a RUNNING/placed job)
$IRIS --cluster=marin job kill /benjaminfeuer/<job>
```
- **Rescue banked traces** before/after a kill if the repo didn't auto-create:
  `gsutil -m rsync -r gs://marin-models-us/ot-agent/<job>/<job>/ /tmp/<job>/` then
  `scripts/harbor/make_and_upload_trace_dataset.py --job_dir /tmp/<job> --repo_id penfever/<slug>-... --episodes last --filter none --skip_register`.
- **NEVER** `iris cluster restart` / stop / bounce the cluster without explicit
  user approval — it kills every running job. `job kill` is job-scoped and safe
  (with permission); cluster ops are not.
- Releasing capacity: killing the job frees its workers; there is no separate
  teardown step for the TPU slice (iris reclaims preemptible workers).

---

## Authoritative references
- `marin:lib/iris/OPS.md` — cluster lifecycle, controller, SQL, GCP ops (read first).
- `marin:lib/iris/config/marin.yaml` — pools, variants, zones.
- `iris.cli.job.get_tpu_topology(variant)` — vm_count / chip topology (don't guess).
- `iris_google_tpu_cloud_hardware.md` (this dir) — per-chip + slice hardware specs.
- `notes/marin/tech.md` — the OT-Agent↔iris fetch-daemon architecture + flag cheat sheet.
