---
name: rl-standard-job-cleanup
description: >-
  Preserve + publish a finished STANDARD (non-agentic GRPO) SkyRL RL checkpoint — the Delphi/rlvr/dapo
  math-and-reasoning cells launched via rl-standard-launch-leonardo (raw sbatch of hpc/skyrl_yaml/leonardo/*,
  logger=console, NO Harbor/Daytona/trace_jobs). Covers: cancel pending retries, pick the BEST checkpoint by
  the trailing-5 EMA of reward via `parse_skyrl_metrics.py --format standard` (chain-aware, capped at the
  latest saved step), flatten weights to repo root, secret-scan, `hf upload` (Leonardo sbatch-tunnel) to
  laion/<run>-<step>-<size>B with the size suffix DERIVED FROM THE EXPORTED WEIGHTS (never the base-model
  name), DB register (--training-type RL + cross-user FK pre-check) ONLY for DB-registerable cells, clean up,
  and fire the Delphi downstream eval suite on the post-RL ckpt (defers to eval-standard-launch §5b). The ONLY
  artifacts are the model + the metric CSVs/report + the tracker scores — there is NO trace dataset. Use when
  a standard/non-agentic GRPO RL run finishes and needs uploading + (maybe) registering. DISTINCT from
  rl-agentic-job-cleanup, which is the AGENTIC (Harbor/Daytona) path with a companion trace dataset.
---

# rl-standard-job-cleanup

Cleanup for a finished **STANDARD (non-agentic) GRPO** SkyRL run on Leonardo — the math/reasoning cells
(`delphi-…`, `rlvr_math`, `dapo`, gsm8k/aime) launched per **`rl-standard-launch-leonardo`** (raw `sbatch`
of `hpc/skyrl_yaml/leonardo/*`, `logger=console`, **no Harbor / no Daytona / no `trace_jobs/`**). The final
HF artifact is **`laion/<run_name>-<step>-<size>B`** (weights at repo root) + the metric CSVs/report; a
Supabase `models` row (`training_type=RL`) **only if the cell's series is DB-registerable** (§7).

> **This is NOT the agentic cleanup.** `rl-agentic-job-cleanup` publishes an agentic Harbor/Daytona model PLUS a
> companion trace dataset `penfever/<job>` and reads `trace_jobs/`. Standard GRPO has **none of that** — no
> `trace_jobs/`, no `trainer_log.jsonl` (logger=console → never written), no trial/batch-error outputs. If
> the run came from `rl-standard-launch-leonardo`, use THIS skill.

**Cross-cutting rules (from CLAUDE.md / memory — they bite):**
- **`hf upload`, NEVER `hf upload-large-folder`** (deprecated stub + deadlocks on HF LFS 429s).
- **`--private` is a no-value flag** — never `--private false`. Default policy is PUBLIC; just omit it.
- Run `parse_skyrl_metrics.py` from the **`otagent` conda env** (matplotlib/pandas; the RL uv venv lacks them).
- **Leonardo login-node killer (~100 s):** a login-node `hf upload` is SIGKILLed at ~100 s regardless of
  detach. **Use the sbatch+tunnel upload pattern** (`.claude/ops/leonardo/ops.md` → "Leonardo HF Upload — Use
  sbatch, NOT the Login Node"). The tunnel depends on the `~/.ssh/leonardo_daytona` step-ca cert (~12 h) —
  a fast publickey failure means refresh the cert.
- **GPFS hygiene:** never `find`/`du` on `$WORK`; locate the `.out` via `scontrol show job <id> -o`
  (`StdOut=`/`%Z`) or a depth-1 `ls`; `rm -rf` cleanup runs detached (no sizing first).
- **Run on a login node** (login02/03/04 — login01 false-drains).

## 0. Cancel pending retries
Cancel queued retry/chain jobs for the same run so they don't relaunch mid-upload (and a fresh-cell relaunch
would `rm -rf` the very `CKPT_DIR` you're about to upload — see the `RESUME_MODE` wipe gotcha in
`rl-standard-launch-leonardo`):
```bash
squeue -u $USER --format='%.18i %.90j %.8T' | grep <RUN_NAME>
scancel <retry_job_ids>
```

## 1. Locate the run dir + the FULL `.out` chain
Standard RL writes to `$WORK/rl_ckpts/<RUN_NAME>` (`$WORK=/leonardo_work/AIFAC_5C0_290/bfeuer00`):
```bash
WORK=/leonardo_work/AIFAC_5C0_290/bfeuer00
RUN_DIR=$WORK/rl_ckpts/<RUN_NAME>
ls -d $RUN_DIR/exports/global_step_*/        # the HF-exportable checkpoints (every hf_save_interval steps)
cat $RUN_DIR/latest_ckpt_global_step.txt     # the last saved step — the selection CAP
```
The `.out` files are `%x_%j.out` in the **sbatch CWD** — `hpc/skyrl_yaml/leonardo/<JobName>_<jobid>.out`
(NOT an `experiments/<job>/logs/` dir; standard RL has no such dir). A chain restart = one `.out` per link:
```bash
cd $WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
ls -lt *<RUN_NAME>*_*.out      # collect ALL links of the chain (sorted by jobid/time)
```

## 2. Pick the BEST checkpoint + emit the metrics — `parse_skyrl_metrics.py --format standard`
The tool computes the **trailing-5 EMA of `reward/avg_raw_reward`** (α=1/3), intersects it with the
on-disk `exports/global_step_<N>/` set, and caps the choice at `latest_ckpt_global_step.txt`. It ALSO emits
the full metric surface (`metrics.csv`, `vllm_metrics.csv`, `report.md`, `reward_plot.png` — reward +
entropy/grad_norm collapse overlay). Run it from the **otagent** env.

**Tool-presence pre-check (the cluster clone may predate `--format standard`).** This tool only landed
recently; a cluster clone can be older, and a full `git pull` here risks CONFLICTING with live-applied local
edits (e.g. a running eval's `eval/leonardo/eval_harbor.sbatch`). Confirm the flag exists, and if it
does NOT, sync **only this one file** — never a full `git pull`:
```bash
cd $WORK/code/OpenThoughts-Agent
scripts/analysis/parse_skyrl_metrics.py --help 2>/dev/null | grep -q 'standard' || \
  grep -q "format" scripts/analysis/parse_skyrl_metrics.py || echo "MISSING --format standard"
# If missing, surgically fetch JUST the tool (does NOT touch any other live-applied edit):
git fetch origin penfever/working
git checkout origin/penfever/working -- scripts/analysis/parse_skyrl_metrics.py
```

**The CLI takes a single `log_folder` + a single `output_folder` positional** (it globs `log_folder` by
`--pattern`, default `*.out`) — it does **NOT** accept individual `.out` files as multiple positionals. So
**stage the run's real `.out` chain links into a clean folder first**, then pass that folder. Staging into a
dedicated `outlogs/` also conveniently excludes the tiny per-worker `*_rayw*.out` / `*_rayhead*.out` logs
(which would otherwise be globbed and pollute the parse):
```bash
OUTLOGS=$WORK/rl_cleanup/<RUN_NAME>/outlogs
OUT=$WORK/rl_cleanup/<RUN_NAME>/metrics
mkdir -p $OUTLOGS
# Copy ALL real chain-link TRAINING .out logs (one per restart link) — NOT the _rayw*/_rayhead worker logs:
cd $WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
for f in *<RUN_NAME>*_*.out; do
  case "$f" in *_rayw*|*_rayhead*) continue;; esac   # skip per-worker logs
  cp "$f" $OUTLOGS/
done
ls $OUTLOGS/      # sanity: only the real <JobName>_<jobid>.out training links

/leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/bin/python \
  scripts/analysis/parse_skyrl_metrics.py \
  --format standard \
  --run_dir $RUN_DIR \
  --save_every 20 \
  $OUTLOGS \
  $OUT
```
- **`--save_every`** MUST equal the run's `trainer.hf_save_interval` (Delphi default **20** — the first save
  is `global_step_20`; selection excludes the first save and starts at `2*save_every`). Verify it in the
  `.out` launch line (`trainer.hf_save_interval=…`) before trusting the cap.
- **CHAIN-AWARE — stage ALL real `.out` links of the chain into `$OUTLOGS`**, not one. A chain's first
  link(s) often have **0 train lines** (engine bring-up only); a single mid-chain `.out` under-covers the EMA
  window and can pick the wrong step. The selector takes `first-seen-wins` per step, so overlapping links are
  safe to include. Stage the training links ONLY — exclude the `*_rayw*` / `*_rayhead*` per-worker logs.
- The tool prints the EMA table + `CHOSEN STEP: <best>` and writes the same into `report.md`. Record
  `BEST=<best>`; the export to publish is `$RUN_DIR/exports/global_step_<BEST>/policy/`.
- If it reports **NO STEP CHOSEN** (e.g. only `global_step_20` exists, or no reward lines parsed), inspect
  `report.md`'s EMA table: a scancelled-early run's only eligible ckpt may be the largest saved multiple of
  `save_every` ≤ cap — fall back to that and note it. Never publish the first save (`global_step_20`).

## 3. Flatten model files to the upload-dir ROOT
HF model files MUST sit at the base of the uploaded dir — not under `policy/`:
```bash
UPLOAD_DIR=$WORK/rl_cleanup/<RUN_NAME>/upload-<BEST>
mkdir -p $UPLOAD_DIR
cp $RUN_DIR/exports/global_step_<BEST>/policy/* $UPLOAD_DIR/
ls $UPLOAD_DIR/      # *.safetensors, config.json, tokenizer files all at root
```

## 4. Derive the SIZE SUFFIX from the EXPORTED WEIGHTS (not the run/base name) — CRITICAL
The HF repo is `laion/<run_name>-<BEST>-<size>B`. **Compute `<size>B` from the exported model itself** —
its `config.json` + the safetensors param count. **Do NOT parse the size from the run/base name** (e.g. the
`delphi-…-32p07b-…` token): that token is **misleading** — a model with `hidden_size=3840,
num_hidden_layers=37` is **~9.7B**, NOT 32B. Trust the weights, not the string.
```bash
/leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/bin/python - "$UPLOAD_DIR" <<'PY'
import json, sys, glob, os
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
H = cfg.get("hidden_size"); L = cfg.get("num_hidden_layers"); V = cfg.get("vocab_size")
# 1) Exact: sum param counts from the safetensors index (preferred).
idx = glob.glob(os.path.join(d, "*.safetensors.index.json"))
total = None
if idx:
    wm = json.load(open(idx[0])).get("metadata", {})
    total = wm.get("total_parameters") or wm.get("total_size")  # total_size = BYTES, not params
    if total and "total_parameters" not in wm:
        total = None  # only had byte size; fall through to header sum
if total is None:
    # Sum tensor numels from each shard header (no torch load needed).
    import struct
    total = 0
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__": continue
            shp = meta.get("shape", [])
            cnt = 1
            for s in shp: cnt *= s
            total += cnt
# 2) Cross-check with the transformer estimate 12*L*H^2 + embeddings(2*V*H).
est = 12 * L * H * H + 2 * (V or 0) * H if (H and L) else None
print(f"config: hidden_size={H} layers={L} vocab={V}")
print(f"summed params = {total/1e9:.2f}B" if total else "summed params = (unknown)")
if est: print(f"estimate 12*L*H^2 + 2*V*H = {est/1e9:.2f}B (sanity cross-check)")
b = total or est
print(f"\nSIZE SUFFIX => {round(b/1e9)}B  (repo: laion/<run_name>-<BEST>-{round(b/1e9)}B)")
PY
```
Use the **summed-param** value (the estimate is only a cross-check; if they disagree by >~15 %, prefer the
summed count and note the discrepancy). Round to the nearest whole B (e.g. 9.7B → `10B`; state the rounding
you used). The size suffix is **required**.

## 5. Copy the launch config + scan for secrets
```bash
cp $WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo/run_delphi_math_rl.sh   $UPLOAD_DIR/rl_run_script.sh 2>/dev/null
cp $WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo/sbatch_delphi_math_rl.sh $UPLOAD_DIR/rl_sbatch.sh    2>/dev/null
trufflehog filesystem $UPLOAD_DIR --no-update    # if installed; else the grep fallback:
grep -rIE '(sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|hf_[a-zA-Z0-9]{34}|eyJ[a-zA-Z0-9._-]+)' $UPLOAD_DIR
```
Remove/redact anything found before upload (HF runs TruffleHog post-upload; catch it first).

## 6. Stage the metrics alongside the model, then upload — `laion/<run_name>-<BEST>-<size>B`
Fold the metric outputs into the upload dir so they ride along with the weights (this is Leonardo's only
W&B-equivalent — runs are `WANDB_MODE=offline`):
```bash
mkdir -p $UPLOAD_DIR/training_logs
cp $OUT/metrics.csv $OUT/vllm_metrics.csv $OUT/report.md $OUT/reward_plot.png $UPLOAD_DIR/training_logs/ 2>/dev/null
cp $WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo/<JobName>_*.out      $UPLOAD_DIR/training_logs/    2>/dev/null
```
Upload via the **Leonardo sbatch-tunnel** (NOT a login-node `hf upload` — it dies at ~100 s). Use the sbatch
template in `.claude/ops/leonardo/ops.md` ("sbatch template for HF upload"); point `cd` at `$UPLOAD_DIR` and
the command at:
```bash
$CMD_PREFIX /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/bin/hf upload \
    laion/<run_name>-<BEST>-<size>B . . --repo-type=model
```
`hf upload` is **additive** (safe to re-run; resumes from `.cache/huggingface/`). Never use
`huggingface_hub.upload_folder()` without `delete_patterns=[]` — it clobbers files absent locally.

## 7. Register in the DB (`--training-type RL`) — ONLY if the series is DB-registerable
**Confirm the cell's series is DB-registerable BEFORE registering.** Several standard-RL series are
**HF-upload-ONLY** — the Delphi scaling-laws SFT grid is HF-only by policy, and the *RL* cells of that grid
may be too. **If it is ambiguous whether THIS run's series is no-DB, STOP and flag it — do NOT auto-register
a no-DB series** (a scaling-laws cell in the model registry pollutes it; the artifacts are consumed by an
eval grid, not the registry). Honor any `enable_db_registration: false` and documented no-DB series
(`crud-otagent-supabase` → "Per-series exceptions"). If the series IS registerable:

**Base model (`--base-model`)** = the checkpoint the RL trained FROM — the sbatch/run-script `MODEL_PATH`
(read it from the `.out` launch line `trainer.policy.model.path=…` or the sbatch `MODEL_PATH=` token), e.g.
`laion/delphi-1e21-…-sft`. **`--base-model` is resolved to the `base_model_id` self-FK**, so it must name an
already-registered `models` row — if the base isn't registered, register/point it first (per
`crud-otagent-supabase`), or the FK lands wrong.

Standard GRPO does **not** auto-push a canonical `laion/<run>` duplicate the way agentic SkyRL does, so there
is usually no auto-row to delete. **Still run the cross-user FK pre-check** before ANY delete/mutate if you
do find a stray row, and restrict all writes to rows you own:
```python
other_users_fk = (c.table("sandbox_jobs").select("id,username,model_id")
    .eq("model_id", stray_row_id).neq("username", os.environ.get("USER","bfeuer00")).execute())
if other_users_fk.data:
    print(f"SKIPPING delete — {len(other_users_fk.data)} other-user rows FK'd; surface + leave it.")
```
Then register the `-<BEST>-<size>B` repo (`--training-type RL` is REQUIRED — the script defaults to SFT):
```bash
python scripts/database/manual_db_push.py \
  --hf-model-id laion/<run_name>-<BEST>-<size>B \
  --base-model laion/delphi-…-sft \
  --dataset-name <rlvr_math|gsm8k|aime|…> \
  --training-type RL
```

## 8. Clean up the run dir
Only after the upload (and DB register, if applicable) is **confirmed on HF**, free disk. Detached `rm -rf`
of the GPFS run dir — **never `du`/`find` to size it first**; verify inode reclaim afterward
(`jutil project dataquota -p <project>` / `df -i`):
```bash
nohup rm -rf $WORK/rl_ckpts/<RUN_NAME> >/dev/null 2>&1 &   # detached; keep $UPLOAD_DIR until HF verified
```
Keep `$WORK/rl_cleanup/<RUN_NAME>/` (staging + metrics) until you've confirmed the HF repo lists the
weights + `training_logs/`, then remove it too.

## 9. Fire the Delphi eval suite on the post-RL checkpoint
Once the ckpt is HF-uploaded (§6) and verified, score it on the lab's fixed downstream suite. **This step
DEFERS to `eval-standard-launch` → "§5b. Evaluate a (post-RL) checkpoint on the Delphi eval suite"** — the
home for the method; do not re-document it here. That section runs the canonical `delphi_eval.sbatch`
(MATH500 / AIME24 10-seed / gsm8k, pass@1, temp 0.7, delphi_v0 template, STAGE=`rl`) on Leonardo against the
uploaded `laion/` repo. One-line invocation (pre-cache the repo on the login node first, per that skill §3):
```bash
RUN=<run_name>-<BEST>-<size>B   # the repo §6 just published
sbatch --job-name="delphi-eval-$RUN" \
  /leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/delphi_eval.sbatch laion/$RUN $RUN rl
```
Add the submitted row to **`main_rl_evals/SCORES.md`** (`🚀 eval submitted` + job id); harvest the scores via
`eval-standard-cleanup`. **HF-upload-only, NEVER DB** — same series rule as §7; the eval records scalar scores
in the tracker, it does NOT create or require a models DB row.

---

## What this cleanup drops vs the agentic one (don't go looking for them)
- **No training-trace dataset** (`make_and_upload_trace_dataset` / `penfever/<job>`) — standard GRPO has no
  Harbor rollouts and no `trace_jobs/`. Omit the agentic "Training Traces" README section entirely.
- **No `trial_stats.csv` / `batch_errors`** — those come from per-trial `result.json` (agentic only); the
  parser's trace/batch emitters cleanly no-op under `--format standard`.
- **No `cp trainer_log.jsonl`** — `logger=console`, so the file is never written. The `.out` chain is the
  log of record; `metrics.csv`/`vllm_metrics.csv`/`report.md`/`reward_plot.png` are the published metric
  artifacts. **`vllm_metrics.csv` IS produced + uploaded** under `--format standard` (the vLLM stat-logger
  extraction is format-agnostic and rides along into the upload's `training_logs/`).
