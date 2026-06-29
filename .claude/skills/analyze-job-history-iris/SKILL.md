---
name: analyze-job-history-iris
description: Run the Iris harbor job-history analyzer (scripts/iris/analyze_job_history.py) on a datagen/eval job and read its JSON sidecar for trustworthy throughput / preemption / productive-trial stats. Use whenever a status check needs REAL metrics (gen tok/s, cycles, non_empty rate, harbor exceptions) instead of an eyeballed log tail. The analyzer paginates the ENTIRE job history and is slow (minutes), which is why naive runs (and subagents) get it wrong — this skill is the reliable recipe.
---

# analyze-job-history-iris

> **📍 Iris orientation — read first.** Before acting on anything in this skill, read the Iris **tools
> catalog** (`.claude/ops/iris/iris_tools.md`) and the Iris **ops directory** (`.claude/ops/iris/` — the
> CoreWeave GPU particulars in `coreweave_gpu_ops.md`, the TPU `marin` particulars in `iris_job_lifecycle.md`).
> They carry the binding access/preamble/gotchas and the helper-script inventory the steps below rely on.

`scripts/iris/analyze_job_history.py` paginates an Iris job's **full** log via
fixed time windows (not `--tail`, which truncates by line count and lets Ray
state-dumps + `[fd-monitor]` frames crowd out the throughput emissions), then
computes: §1 preemption count + time-to-preempt, §2 per-cycle trace progress
(from harbor GCS output), §3 serving throughput. It writes a markdown report to
`--output` and a **JSON sidecar** to `<output>.json`. **Always read the sidecar
with python — never eyeball the markdown.**

## The one pitfall: it is SLOW — run it foreground and WAIT

A single `--refresh` run re-pages the whole history and takes **minutes**
(scales with runtime × cycles; a multi-day, 40k-task job can take 5–15+ min).
The failure mode — what keeps going wrong — is treating it like a quick command:
backgrounding it, or letting a per-command timeout fire, then yielding/“pausing
to wait” before it finishes. It is NOT a background job; nothing re-wakes you.

**Run it in the FOREGROUND with a long explicit timeout and let it return.**
- Use a Bash `timeout` of **≥ 900000 ms (15 min)** per call.
- Do NOT `run_in_background`. Do NOT abort and report "still running."
- If one 15-min call doesn't finish, run it **once more** (the filtered log is
  cached, so the second run resumes fast) before giving up.

## Command

```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_job_history.py \
  <job_id> --output /tmp/$(basename <job_id>)_history.md --refresh
```

- `--refresh` ignores the cached filtered log (`/tmp/iris_history_<job>.filtered.log`).
  **Omit `--refresh`** to re-parse the existing cache instantly — use that when you
  only need to re-read stats for a job you already paged this session.
- **Cluster.** Default targets **marin/TPU**. For a CoreWeave job pass
  `--cluster cw-us-east-02a` AND have `KUBECONFIG=~/.kube/coreweave-iris-gpu` in
  the env AND use the otagent-env python above (the marin `.venv` iris cannot
  drive CoreWeave). **But:** GPU-RL jobs have **no harbor trial sidecars**, so §2
  is empty and most of the value is gone — for GPU-RL use **rl-job-health-deep-dive**
  instead. This analyzer is for **harbor-shaped jobs (datagen / agentic eval)**.

## Parse the sidecar (exact keys)

```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python - /tmp/$(basename <job_id>)_history.md.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
g = d.get("serving_summary", {}).get("gen_tps", {}) or {}
r = d.get("serving_summary", {}).get("running", {}) or {}
cyc = d.get("cycles", []) or []
served = [c for c in cyc if c.get("did_serve")]
tfs = served[0]["time_to_first_serve_s"] if served else None
ne, tot = d.get("non_empty_trials"), d.get("total_trial_dirs")
rate = (ne / tot) if (ne is not None and tot) else None
exc = sorted((d.get("harbor_exception_stats") or {}).items(), key=lambda kv: -kv[1])[:5]
print(f"runtime_h        : {round(d.get('total_runtime_s',0)/3600, 2)}")
print(f"preemptions      : {d.get('iris_preemption_count')}  (from_log={d.get('preempt_count_from_log')})")
print(f"state            : {d.get('state')}")
print(f"cycles           : total={len(cyc)} served={len(served)}")
print(f"t_first_serve_s  : {tfs}")
print(f"gen_tps          : n={g.get('n')} mean={round(g.get('mean',0),1)} peak={round(g.get('max',0),1)} median={round(g.get('median',0),1)}")
print(f"running          : mean={round(r.get('mean',0),1)} peak={round(r.get('max',0),1)}")
print(f"saturation_rate  : {d.get('serving_summary',{}).get('saturation_rate')}")
print(f"productive trials: {ne}/{tot} = {round(rate*100,1) if rate is not None else None}%")
print(f"harbor counts    : completed={d.get('harbor_n_completed')} errored={d.get('harbor_n_errored')} running={d.get('harbor_n_running')} pending={d.get('harbor_n_pending')} total={d.get('harbor_n_total_trials')}")
print(f"harbor_updated_at: {d.get('harbor_updated_at')}  (started {d.get('harbor_started_at')})")
print(f"top exceptions   : {exc}")
PY
```

Note the sidecar stores **`total_runtime_s`** (seconds — divide by 3600), not
`runtime_h`; `gen_tps`/`running` expose `max` (use as "peak"), not `peak`;
`harbor_exception_stats` is a `{name: count}` dict. S1 datagen baseline ≈ **gen
mean 400 / peak 1115** tok/s; short-task datasets run lower — judge health by the
**productive trial rate** (`non_empty/total`), not tok/s alone.

## ALSO report: mean reward + completed/total tasks (NOT in the sidecar)

Two fields the user always wants are **not** in the JSON sidecar — they live only
on harbor's live TUI **progress line** in the job logs, in the form
`<completed>/<total> Mean: <reward>`:

```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin job logs <job_id> --max-lines 8000 2>/dev/null \
  | grep -aoE '[0-9]+/[0-9]+ Mean: [-0-9.]+' | tail -1
# e.g. "11129/15713 Mean: 0.429"  ->  completed/total tasks = 11129/15713 (71% of dataset),  mean reward = 0.429
```

- **completed / total tasks** = the `N/M` — progress against the **whole dataset**
  (M is the dataset's task count). This is DIFFERENT from the sidecar's
  `non_empty_trials/total_trial_dirs`, which is the productive rate among *attempted*
  trials. Report both: dataset progress (N/M) and productive rate (sidecar).
- **mean reward** = the `Mean: X` — the running mean verifier reward across completed
  trials. (`grep -aoE '… Mean: …'` strips the box-drawing/ANSI junk that trails the
  bar — don't print the raw line.)
- For a CoreWeave job use `--cluster cw-us-east-02a` (+ `KUBECONFIG` + otagent iris).
- Every analyzer report MUST include these two alongside the sidecar stats.

## Running many jobs / offloading to a subagent

The analyzer is the slow part of every monitor tick. To keep the main loop free,
dispatch it to a subagent — but the prompt MUST pin the foreground-and-wait
discipline or the subagent yields early (the recurring bug). Use this template
verbatim per job (or list several; tell it to run them one at a time):

> Run `analyze_job_history.py` on `<job_id>` (cluster `marin`). It paginates the
> full history and is SLOW (many minutes) — this is expected. Run it FOREGROUND
> with a 900000 ms (15 min) timeout; do NOT background it, do NOT abort and say
> "still running." If it doesn't finish in one call, run it once more (the cache
> makes the retry fast). Then parse the JSON sidecar at
> `/tmp/<basename>_history.md.json` with python (keys: `total_runtime_s`,
> `iris_preemption_count`, `cycles[].did_serve`/`time_to_first_serve_s`,
> `serving_summary.gen_tps.{n,mean,max}`, `serving_summary.running.{mean,max}`,
> `non_empty_trials`, `total_trial_dirs`, `harbor_exception_stats`,
> `harbor_updated_at`). ALSO run
> `iris --cluster=marin job logs <job_id> --max-lines 8000 | grep -aoE '[0-9]+/[0-9]+ Mean: [-0-9.]+' | tail -1`
> and report **mean reward** and **completed/total tasks** from it (these are NOT
> in the sidecar). Return a compact key:value report + a one-line health read.
> Do not paste raw markdown/logs.

If a dispatched subagent stops early with results unparsed, **resume it** (don't
relaunch): tell it to report any sidecar that already exists, then finish the
rest foreground. A completed `<output>.json` on disk means that job is done even
if the agent yielded — re-parse it directly.

## Related skills
- **monitor-cron-sweep** / **monitor-job-tables** — the status sweeps that call this.
- **rl-job-health-deep-dive** — the GPU-RL equivalent (no harbor sidecars; uses the finelog).
