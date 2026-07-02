# Finelog R2 archive creds — running `analyze_job_history.py` against CoreWeave from the Mac

> **TL;DR.** `analyze_job_history.py` (and anything that reads the finelog **archive** half) needs **R2
> credentials** to list/read `s3://marin-na/finelog/cw-us-east-02a`. The Mac does **not** have them; they live
> only in the cluster's `iris`-namespace secret **`finelog-cw-use02a-env`** (`AWS_*` keys incl.
> `AWS_ENDPOINT_URL`). Source that secret into the env — **values never printed** — before running the
> analyzer against `--cluster cw-us-east-02a`. Without it the run crashes
> `FileNotFoundError: The specified bucket does not exist` (s3fs silently falls back to **real AWS S3**, where
> `marin-na` does not exist).

## Why this bites (the failure mode)

The analyzer fetches each job's complete log as **live ∪ GCS-archive**, deduped on `seq` (see the
`analyze-job-history-iris` skill). The split differs by cluster, and the **archive half** is where the cred
gap is:

| cluster | finelog `client_url` | LIVE half | ARCHIVE half (`remote_log_dir`) | archive creds |
|---|---|---|---|---|
| `marin` (TPU) | set | IAP proxy (`marin-login login marin`) | `gs://…` (GCS) | the IAP/ADC session covers it |
| `cw-us-east-02a` (CoreWeave) | **None** | **k8s tunnel** (no IAP needed) | **`s3://marin-na/finelog/cw-us-east-02a`** (**R2**) | **R2 creds — NOT on the Mac** |

So on CoreWeave the live half is *easier* than the skill implies (a tunnel, no IAP login), but the **archive
half needs R2 creds** the Mac lacks. `fsspec.url_to_fs("s3://marin-na/…")` with no R2 endpoint/creds resolves
to **AWS S3**, and the bucket isn't there → the run aborts in `_list_namespace_segments → fs.ls` **before**
duckdb even reads, so it is *not* caught by the script's compaction-race retry (that only catches
404/NoSuchKey mid-read). The crash is a hard `FileNotFoundError`.

This is the **same Mac-lacks-marin-na-R2-creds** fact noted for `trials_dir` in `iris_tools.md`
(`analyze_job_history.py` entry: "the launch-host Mac lacks marin-na R2 creds, so all R2 ops run INSIDE the
pod"). For finelog we work around it from the Mac by borrowing the pod's creds out of the k8s secret.

## Where the creds are (and the var-name trap)

- **Secret:** `finelog-cw-use02a-env` in the **`iris`** namespace (the finelog deployment's env). The
  sibling `iris-task-env` carries the same `AWS_*` set (it's what every task pod gets) plus a `FSSPEC_S3`
  key — either secret works; prefer `finelog-cw-use02a-env` for this purpose.
- **Keys (NAMES only):** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`,
  `AWS_REGION`, `AWS_DEFAULT_REGION`. The `AWS_ENDPOINT_URL` value is the CoreWeave R2 endpoint
  `https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com` (also non-secretly present as
  `object_storage_endpoint` in `~/Documents/marin/lib/iris/config/cw-us-east-02a.yaml`).
- **⚠ Var-name trap.** The cluster-config header *comment* documents the creds as
  `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`, but **botocore/s3fs read `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT_URL`** — which is exactly how the secret stores them. Use the
  `AWS_*` names; modern botocore picks up `AWS_ENDPOINT_URL` automatically, so **no** explicit
  `endpoint_url`/`FSSPEC_S3_ENDPOINT_URL`/`client_kwargs` is needed for the listing to work.
- **Do NOT** confuse these with the LAION `AWS_*` in `~/Documents/secrets.env` (a *different* S3 store,
  `LAION_ENDPOINT`). Source the R2 creds in a shell where you have **not** also sourced `secrets.env`, or the
  LAION values clobber R2 and you're back to "bucket does not exist."

## The plumbing (validated 2026-06-29 — values never printed)

Source the secret into the env via a base64-decode loop that pipes straight into `export`, so the credential
values stay out of stdout / your scrollback / any log:

```bash
export KUBECONFIG=~/.kube/coreweave-iris-gpu              # HARD prereq (cw kubeconfig)
# Borrow the pod's R2 creds out of the iris-ns secret — values are decoded inline, never echoed:
while IFS=$'\t' read -r k v; do export "$k=$(printf %s "$v" | base64 -d)"; done \
  < <(kubectl -n iris get secret finelog-cw-use02a-env \
        -o go-template='{{range $k,$v := .data}}{{$k}}{{"\t"}}{{$v}}{{"\n"}}{{end}}')
# sanity (prints only yes/no, never the value):
echo "R2 endpoint set? $([ -n "$AWS_ENDPOINT_URL" ] && echo yes || echo no)"
```

Then run the analyzer under the **marin venv** (it must import `finelog`/`rigging`/`duckdb`) with the
otagent-env iris binary (the marin `.venv` iris can't drive CoreWeave):

```bash
IRIS_BIN=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris \
/Users/benjaminfeuer/Documents/marin/.venv/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_job_history.py \
  /benjaminfeuer/<job> --cluster cw-us-east-02a \
  --output /tmp/<job>_history.md --refresh
```

Quick isolated check that the creds are wired (lists the archive root in seconds):

```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/python -c "
import fsspec; from finelog.deploy.config import load_finelog_config
cfg=load_finelog_config('cw-us-east-02a'); fs,_=fsspec.url_to_fs(cfg.remote_log_dir)
print('LS OK:', len(fs.ls(cfg.remote_log_dir, detail=False)), 'entries')"
# expect: LS OK: 4 entries  (iris.task_status, log, zephyr.stage, ...)
```

## Caveats / gotchas

- **Terminal vs running jobs.** For a **terminal** (old) job the live tunnel often has nothing left — the
  archive (R2) is the real source, so the R2 creds are mandatory. For a **running** job you need *both* (live
  tunnel for the recent L0 tail + R2 for the compacted history); a live-half failure still surfaces as a
  loud coverage gap, not a silent fragment.
- **GPU-RL jobs have no harbor trial sidecars**, so `analyze_job_history.py` §2 is empty and most of its
  value is gone — for GPU-RL diagnosis use **rl-job-health-deep-dive** instead. But the *log-acquisition*
  machinery here (live ∪ R2-archive) is generic and is the right way to pull a terminal RL run's full finelog
  history (e.g. the gs-1 wedge-diff baseline): the default `FINELOG_CONTAINS_PATTERNS` filter is TPU/datagen-
  tuned, so to capture RL signals (`WORKER_FORWARD_ENTER`, `global_step`, `[weight-sync]`, mesh_fsdp watchdog)
  swap those `contains(data, …)` patterns when reusing `fetch_live`/`fetch_gcs`.
- **Secret hygiene.** These are shared marin-infra R2 creds. Source them by the loop above (never paste a
  value into a prompt, file, or chat); a subagent that needs them gets *this procedure*, not the values
  (per the supervisor secrets rule). They do not belong in `secrets.env` either — borrow from the live
  secret each time so a rotation can't leave a stale copy on disk.

## Cross-references
- **Skill:** `analyze-job-history-iris` (the analyzer how-to + sidecar parsing; this doc supplies its missing
  CoreWeave-archive cred step).
- **Tools catalog:** `iris_tools.md` (`analyze_job_history.py` entry + the "Mac lacks marin-na R2 creds /
  R2 ops run in-pod" note for `trials_dir`).
- **Cluster config:** `~/Documents/marin/lib/iris/config/cw-us-east-02a.yaml` (`object_storage_endpoint`,
  `remote_state_dir`, the `R2_*` header comment).
- **GPU-RL diagnosis:** `rl-job-health-deep-dive`.
