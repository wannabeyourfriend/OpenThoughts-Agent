---
name: rl-agentic-launch-iris
description: >-
  Launch / relaunch agentic MarinSkyRL (SkyRL GRPO) RL on Marin's Iris / CoreWeave GPU cluster
  (cw-us-east-02a, 8x H100-80GB + InfiniBand per node) via `python -m rl.cloud.launch_rl_iris` + the
  gpu-rl Docker image (NO Apptainer SIF). Covers the dense 8B FSDP2 arms (seqnorm + TIS) and the
  MoE 30B-A3B arms (CP + DCP=2 + R3 @ 131k) — the exact launcher flag set (`--rl_config`, `--model_path`,
  `--train_data`, `--num-nodes`, `--rendezvous-dir`, `--job-name`, `--priority`, `--cpu`, `--max-retries`),
  the gang/leafgroup/Kueue multi-node Ray rendezvous, the iris config-authoring rules (NO container block,
  load-bearing top-level `extra_env:` forwarding, disaggregated placement + explicit `num_inference_engines`,
  SIF→Docker env translation), and the bring-up gotchas learned this week (`--cpu 48`, `--max-retries ≥1`).
  Use when asked to launch / relaunch an agentic SkyRL RL run on Iris / CoreWeave. Cluster access/hardware
  particulars live in .claude/ops/iris/coreweave_gpu_ops.md (this skill defers to it). Reference:
  rl/cloud/launch_rl_iris.py, scripts/iris/start_rl_iris_controller.py, .claude/ops/iris/coreweave_gpu_ops.md.
---

# rl-agentic-launch-iris

> **⚠ Local clone = ground truth (CLAUDE.md §Always).** ALL code/config edits (OpenThoughts-Agent +
> MarinSkyRL + the vLLM fork) go in the local Mac checkouts → commit → (push). **The Iris launcher uploads
> the LOCAL workspace to `/app`**, so a local commit takes effect on the next launch *immediately* — you do
> NOT push-then-pull-on-a-cluster here (there is no Iris clone to pull). Still **never** hand-edit on a
> remote, leave divergent state, or patch-by-rsync. Bake this into every subagent you dispatch.

Agentic RL on Iris runs through **`python -m rl.cloud.launch_rl_iris`** (MarinSkyRL / SkyRL, GRPO, FSDP2 or
MoE-EP). Each rollout is a real **Harbor** agent episode against a **Daytona** sandbox (the `terminal_bench`
generator). The target is Marin's **CoreWeave `cw-us-east-02a`** cluster — **8x H100-80GB + InfiniBand per
node**, whole-node exclusive, gang/leafgroup-coscheduled (NOT SLURM, NOT TPU). The **gpu-rl Docker image IS
the runtime** — there is no Apptainer SIF and no `hpc.launch`.

This skill is the GPU/CoreWeave analog of `rl-agentic-launch-jupiter`, and like the Jupiter/Leonardo launch
skills it **defers cluster-access/hardware particulars to its ops doc** —
**`.claude/ops/iris/coreweave_gpu_ops.md`** (kubeconfig/access, the H100 node shape + NCCL rationale,
gang/Kueue/rendezvous mechanics, and the binding `--cpu 48` / `--max-retries` gotchas). This skill keeps the
launch HOW-TO (flag set, config-authoring rules, bring-up checklist). The TPU-centric Iris job lifecycle
(datagen/eval monitor / teardown / preemption / Daytona-cap) is a DIFFERENT cluster — see
`.claude/ops/iris/iris_job_lifecycle.md`.

## 1. Prereqs (the pre-launch preamble)

> **Cluster access, hardware, and scheduling particulars → `.claude/ops/iris/coreweave_gpu_ops.md`**
> (kubeconfig `~/.kube/coreweave-iris-gpu`, the `cw-us-east-02a` cluster + access-verify commands, the H100
> node shape + NCCL-defaults rationale, gang/Kueue/`s3://`-rendezvous mechanics, the gpu-rl image's
> deps-only/source-synced model, and the binding `--cpu 48` / `--max-retries` gotchas). Read it once; this
> section keeps only what you type to launch.

Launch from the local Mac, **otagent py3.12 conda env**:
```bash
source /Users/benjaminfeuer/Documents/secrets.env     # HF_TOKEN, WANDB_*, DAYTONA_* (forwarded into the pod)
export KUBECONFIG=~/.kube/coreweave-iris-gpu           # the CoreWeave GPU cluster kubeconfig (see ops doc)
# otagent python = /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python (symlinks fail in the sandbox)
```
- **Confirm access** before submitting (synchronous `iris`/`kubectl` only — never background them; commands
  + the ~36-node H100 headroom check are in the ops doc's "Verify access").
- **Cluster config** auto-resolves to `~/Documents/marin/lib/iris/config/cw-us-east-02a.yaml`; override with
  `--cluster-config` only if it moved.
- **gpu-rl image:** deps-only (RL venv `/opt/openthoughts/envs/rl` + vLLM fork + MarinSkyRL editable
  `/opt/skyrl` + harbor), **pinned by immutable `@sha256:` digest** in
  `rl/cloud/launch_rl_iris.py:DEFAULT_RL_DOCKER_IMAGE` (NOT the floating `:gpu-rl` tag — it stale-caches).
  Source is synced at runtime so first-party edits live without a rebuild; **bump the digest** on an image
  rebuild (full rationale in the ops doc).

## 2. The canonical launch

Lifted from the validated config headers (each iris config's header carries its own ready-to-run command):
```bash
source /Users/benjaminfeuer/Documents/secrets.env
python -m rl.cloud.launch_rl_iris \
  --rl_config hpc/skyrl_yaml/iris/<cfg>.yaml \
  --model_path <hf-id> \
  --train_data '["<HF-repo-or-harbor-task-set>"]' \
  --num-nodes N \
  --gpus_per_node 8 \
  --cpu 48 \
  --max-retries 1 \
  --rendezvous-dir s3://marin-na/iris/rl-<slug>/<run> \
  --job-name <name> \
  --priority interactive \
  --no-wait
```
**What VARIES per arm:** `--rl_config` (the recipe), `--model_path`, `--train_data`, `--num-nodes` (must match
the config's GPU budget — see §3). **What is essentially fixed on Iris:** `--gpus_per_node 8` (CoreWeave nodes
are 8x H100 — also FORCES policy/ref gpus-per-node, see §4), `--cpu 48` (NOT the 64 default — see §6),
`--max-retries 1` (the transient HF weight-flake — §6), `--priority interactive` (band; `production`/`batch`
exist), `--no-wait` (submit + detach; without it the launcher streams logs and a `KeyboardInterrupt`
terminates the job). Flag glossary:

- **`--rl_config`** — repo-relative path under `hpc/skyrl_yaml/iris/`. It must exist on the synced `/app`
  workspace; the launcher resolves an absolute path back to repo-relative and fails fast if it's outside the
  repo. (`--rl-config` hyphenated alias also accepted; same for `--model-path`/`--train-data` etc.)
- **`--model_path`** — HF id (e.g. `Qwen/Qwen3-8B`, `Qwen/Qwen3-Coder-30B-A3B-Instruct`). CoreWeave nodes
  have egress → the model is pulled from HF **online** (NOT offline; do NOT set `HF_HUB_OFFLINE`).
- **`--train_data`** — a JSON-list string `'["..."]'` (HF repo `DCAgent/…` / `laion/…`, or a harbor task set).
- **`--num-nodes N`** — number of **WHOLE H100 nodes** requested EXCLUSIVELY, gang/leafgroup-coscheduled (one
  iris task per node, all 8 GPUs each). `--num_nodes` underscore alias also works. (See §3 + §5.)
- **`--rendezvous-dir`** — **REQUIRED for `--num-nodes>1`** (the launcher hard-errors otherwise). The shared
  store the multi-node Ray head/workers rendezvous through. On cw-us-east-02a use an **`s3://` (R2) URI under
  the cluster's `marin-na` bucket** (e.g. `s3://marin-na/iris/rl-<slug>/<run>`); the cluster injects working
  R2 creds into every task pod (the `iris-task-env` Secret), so **no external creds** are needed and you must
  **NOT forward `AWS_*`/`R2_*`** (it would clobber the pod's R2 creds and silently target real AWS S3). Use a
  fresh sub-path per run so a stale head file from a prior attempt isn't picked up.
- **`--job-name`** — controls the iris job id `/benjaminfeuer/<name>`; set it explicitly so monitoring +
  teardown land on a predictable name. (Auto-derived `rl-iris-<ts>` if unset.)
- **`--priority`** — `production` / `interactive` / `batch` band.
- **`--cpu` / `--memory` / `--disk`** — per-node resources (defaults 64 / 512GB / 512GB; **set `--cpu 48`**).
- **`--max-retries K`** — iris re-brings-up the gang on a FAILURE up to K times (preemptions retry
  separately). **Use ≥1** (§6).
- **`--skyrl-ref <git-ref>`** — `git fetch && checkout` the baked `/opt/skyrl` MarinSkyRL clone to a
  newer/pinned commit BEFORE running (deps are baked, but skyrl-train is editable → the checkout is live).
  Use to pick up a MarinSkyRL fix that landed AFTER the image build without rebuilding the image.
- **`--skyrl_override '++a.b.c=val'`** — repeatable Hydra override (last-wins over the yaml).
- **`--dry-run`** — print the resolved config + in-container command without submitting (always dry-run a new
  config first: confirm the hydra args show the placement / `num_inference_engines` / TP / extra_env you
  intend — see the resume log's VALIDATED block for the pattern).

## 3. Config map + node count (`--num-nodes` MUST match the config)

`--num-nodes = total_GPUs_in_config / 8`. Derive the GPU budget from the yaml
(`policy_num_nodes`, `ref_num_nodes`, `num_inference_engines × inference_engine_tensor_parallel_size`):

| Config (`hpc/skyrl_yaml/iris/…`) | Model | Layout | GPUs → `--num-nodes` |
|---|---|---|---|
| `smoke_seqnorm_tis.yaml` | Qwen3-8B (smoke) | colocated, all-null (derives) | 8 → **1** (or 16 → **2**) |
| `56GPU_seqnorm_tis.yaml` | dense 8B (seqnorm + TIS) | disaggregated: 1 node policy/ref + 48×TP1 engines | 56 → **7** |
| `8node_qwen3_30b_a3b_131k_cp_dcp2_r3.yaml` | **Qwen3-Coder-30B-A3B (MoE)** | disaggregated: 4 nodes policy (EP8×FSDP2×CP2=32) + 4×TP8/DCP2 engines | 64 → **8** |

- **Smoke first.** `smoke_seqnorm_tis.yaml` is the launcher-validation smoke (same seqnorm+TIS code path,
  toy scale, ≥2 steps in minutes); it runs unchanged at `--num-nodes 1` (no rendezvous needed) OR `2` (needs
  a rendezvous-dir). Use it to validate the launcher / a new image digest end-to-end before a real arm.
- The dense-8B model is typically `Qwen/Qwen3-8B` (or a `laion/…` 8B); the MoE arm is
  `Qwen/Qwen3-Coder-30B-A3B-Instruct`. Common train sets: `DCAgent/exp_rpt_pymethods2test-large`, etc.
- These iris configs are **ports of the Jupiter prod configs** (same experiment) — the header of each iris
  config documents exactly what was carried verbatim vs. changed (geometry + env translation). Read it.

## 4. Config-authoring rules for `hpc/skyrl_yaml/iris/`

Porting a Jupiter (Apptainer SIF) config to Iris (Docker) — the load-bearing rules:

- **NO `container:` / SIF / apptainer / conda / binds / pydeps block.** The gpu-rl image IS the container
  (RL venv `/opt/openthoughts/envs/rl`, MarinSkyRL `/opt/skyrl`, workspace synced to `/app`). The launcher +
  `start_rl_iris_controller.py` wire all of that; none of it belongs in the cluster-agnostic SkyRL/Hydra yaml.
- **Top-level `extra_env:` is FORWARDED and LOAD-BEARING.** On the SLURM path runtime env lives under
  `container.extra_env` and is emitted as shell `export`s; the Iris path has no `container:` block, so that
  plumbing never runs. **`launch_rl_iris.py:load_config_extra_env()`** reads a **top-level `extra_env:`**
  mapping (and, defensively, a leftover `container.extra_env`) and merges it into the iris `EnvironmentSpec`.
  **Without it the YAML's env is SILENTLY DROPPED** and only the launcher's hardcoded HF/WANDB/DAYTONA
  passthrough reaches the pod — e.g. the EPDIAG probe arm + the R3/DCP guard env never take effect (this is
  the fix that unblocked the sel_rows capture). The launcher seeds `extra_env` FIRST so its own signals
  (rendezvous/secrets) win on a collision. Values are coerced to str (k8s env are strings).
- **SIF→Docker env translation (what to DROP / KEEP / hardcode in `extra_env:`):**
  - **DROP** all `APPTAINERENV_*` duplicates, `TRITON_LIBCUDA_PATH`, `LIBRARY_PATH=/.singularity.d/libs`
    (SIF-only), and `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` (CoreWeave has egress).
  - **DROP** the GH200/SIF NCCL disables **`NCCL_P2P_DISABLE` / `NCCL_NVLS_ENABLE` / `NCCL_COLLNET_ENABLE`**
    — on H100+IB they would CRIPPLE the intra-node NVLink all-reduce that a TP=8 engine (DCP) depends on. Use
    **NCCL defaults** (NVLink intra-node + IB inter-node); keep `NCCL_DEBUG=INFO` + the observability/raised
    timeouts (`SKYRL_WORKER_NCCL_TIMEOUT_IN_S`, `TORCH_NCCL_*`). `disable_custom_all_reduce: true` STAYS.
  - **HARDCODE** `LD_LIBRARY_PATH: /opt/openthoughts/envs/rl/lib` (the RL conda prefix) — **NOT
    `$CONDA_PREFIX/lib`**: the launcher injects env as literal k8s values and **k8s does NOT shell-expand
    `$VAR`**, so a literal `$CONDA_PREFIX/lib` is a broken path.
  - **KEEP** the vLLM serve flags (`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`),
    `PYTORCH_CUDA_ALLOC_CONF`, and (for the MoE R3/DCP arm) the guard env `VLLM_ALLOW_ROUTED_EXPERTS_DCP=1`
    (+ `VLLM_MQ_MAX_CHUNK_BYTES_MB`, `VLLM_ROUTED_EXPERTS_SIDE_TIMEOUT_SECONDS`,
    `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`) and the EPDIAG probe arm — all as **plain top-level `extra_env:`**.
- **Disaggregated (`colocate_all: false`) vs colocated (`true`):** the smoke is colocated (policy/ref/engines
  share GPUs via vLLM sleep-mode, all placement counts null → derived). The real arms are **disaggregated**
  (policy/ref FSDP on a disjoint set of nodes from the vLLM generation engines).
- **Set `num_inference_engines` EXPLICITLY when disaggregated.** `build_skyrl_hydra_args` only DERIVES it
  (`= total_gpus / TP`) when the YAML leaves it `null`. On a disaggregated split that derivation is WRONG —
  it counts the policy-only nodes too. E.g. the 30B config: `(8×8)//8 = 8` would be derived, but only **4**
  engines exist (4 nodes are policy-only). **4 is authoritative; pin it.** (The smoke can leave it null
  *because* it's colocated — every node is both policy and engine.)
- **`--gpus_per_node 8` FORCES `policy_num_gpus_per_node` / `ref_num_gpus_per_node` to 8**
  (`build_skyrl_hydra_args` overrides the YAML values when the flag is set). `policy_num_nodes` /
  `ref_num_nodes` ARE honored from the YAML. So in a port: divide the Jupiter node counts by 2 (GH200 4-GPU
  → H100 8-GPU nodes) for `policy_num_nodes`/`ref_num_nodes`; list `*_num_gpus_per_node: 8` for documentation
  (the flag overrides it anyway).
- **MoE / DCP placement constraints** (hard — don't hand-edit around them): a **TP=8 engine must place
  intra-node on ONE 8-GPU node** (NVLink decode, no cross-node TP — this is exactly what CoreWeave's 8-GPU
  nodes unblock vs Jupiter's 4-GPU nodes). `inference_engine_mp_backend: false` (RAY executor, R3+DCP
  mandatory) + `async_scheduling: false` (R3 guard). DCP ceiling `dcp ≤ tp/num_kv_heads`; MoE dim-0 guard
  `(num_experts // ep) % fsdp == 0`; `policy_strict_spread_pg: true` reserves the policy nodes up front so
  the engines land on the disjoint gen nodes. All carried verbatim from the Jupiter prod config.

## 5. Gang scheduling + multi-node Ray rendezvous

- **`--num-nodes N` → `replicas=N`** whole H100x8 tasks. For GPUs with replicas>1,
  `resolve_multinode_defaults` returns **`CoschedulingConfig(group_by="leafgroup")`** — all N nodes
  co-scheduled on one InfiniBand leaf fabric, all-or-nothing. cw-us-east-02a enables **Kueue gang admission**
  (`host_network: true` for NCCL/IB) → the N-task gang is admitted **atomically** (all N whole nodes granted,
  or the job queues). At submit you'll see `replicas=N, coscheduling=leafgroup`, then the pods sit
  **SchedulingGated** (normal Kueue gang pre-admit) until admitted.
- **One controller runs on every node** (`scripts/iris/start_rl_iris_controller.py`); iris injects
  `IRIS_TASK_ID` / `IRIS_NUM_TASKS` / `IRIS_ADVERTISE_HOST` per task. **Rank 0** starts the Ray head, writes
  the head IP to the rendezvous file (`ray_head.json` under `--rendezvous-dir`), waits for all N nodes to
  join, then runs the MarinSkyRL driver (`run_rl.py`) with `RAY_ADDRESS` set so SkyRL's bare `ray.init()`
  **attaches** to the cluster. **Ranks 1..N-1** poll the rendezvous for the head IP, `ray start
  --address=…`, then park (contributing their 8 GPUs) until rank 0 publishes the `ray_head.done` marker. On a
  retry, rank 0 purges the stale rendezvous and rewrites (workers ignore stale files via `written_at`).

## 6. Bring-up gotchas LEARNED (the pre-flight checklist)

These were paid for this week (`agent_logs/2026-06-25_coreweave_131k_cpdcp2r3_resume.md`) — apply them up front:

- **`--cpu 48`, NOT 64.** CoreWeave nodes are ~128 cores but carry ~64–68 cores of persistent
  system/daemonset overhead, so at the **64 default only ~2/32 nodes have ≥64 free cores** → an N-node
  single-IB-leaf gang (leafgroup) can't be satisfied and sits **SchedulingGated** forever with a Kueue
  `topology 'infiniband' allows to fit only 2 out of N pod(s)` message. **48 cores fits all nodes** and the
  gang admits immediately (QuotaReserved=True). Memory 512GB is fine.
- **`--max-retries ≥1`** for the transient HF weight-resolution flake. At scale (e.g. 32 FSDP ranks each
  resolving sharded safetensors online) a single rank can hit a transient HF Hub HTTP/EOF failure →
  transformers reports the generic `… does not appear to have a file named model.safetensors`; with
  `max_retries=0` that one rank kills the whole gang (Ray SIGKILLs it). `--max-retries 1` re-brings-up the
  gang on that failure (time-only cost). A first-party retry-wrapper around the weight resolution has landed
  in MarinSkyRL (commit `0b2b05b`); keep `--max-retries ≥1` as belt-and-suspenders. *(DURABLE alternative:
  pre-stage the model into the image's HF cache / a shared snapshot before the FSDP workers start, or raise
  `HF_HUB_DOWNLOAD_TIMEOUT`. Particulars: `.claude/ops/iris/coreweave_gpu_ops.md §Binding gotchas`.)*
- **TP=8 must place intra-node on an 8-GPU node** (NVLink decode) — guaranteed by `--gpus_per_node 8` + the
  per-engine STRICT_PACK PG; do not split a TP=8 engine across nodes.
- **DCP / CP / R3 / EPDIAG env must reach the pod** — that's the `extra_env:` forwarding (§4). After a launch,
  confirm from the rank-0 log that the guard ENGAGED (not rejected): SkyRL `_validate_dcp_cfg
  VLLM_ALLOW_ROUTED_EXPERTS_DCP=1: allowing …` AND vLLM-fork `vllm.py … allowing --enable-return-routed-experts
  with decode_context_parallel_size=2`. The separate `envs.py "Unknown vLLM environment variable detected:
  VLLM_ALLOW_ROUTED_EXPERTS_DCP"` line is ONLY the env-registry whitelist not listing the fork-added var — it
  is **NOT** a no-op when the fork's own code reads the var (it does).
- **Transient self-healing on bring-up is normal:** a `ghcr.io` blob EOF → `ImagePullBackOff` self-heals
  (kubelet retries); `shm_broadcast: No available shared memory broadcast block found in 60s` is **benign**
  (engines idle-wait while the policy mesh loads weights). Don't salvage these.

## 7. Standing constraints (do NOT violate)

- **Daytona RL concurrency ≤ 6 RUNNING per cluster.** PENDING/SchedulingGated gangs that haven't admitted
  don't count once they fail out, but a RUNNING gang does — don't launch a 7th concurrent RL job.
- **`enable_db_registration: false`** in every iris RL yaml (it is). DB registration is a **manual cleanup
  step**, never a launch flag. Do NOT flip it on.
- **The a3 series is CONCLUDED** — do NOT launch / refill / auto-advance a3 rows. The active iris arms are
  the seqnorm/TIS dense-8B port and the CP+DCP2+R3 MoE port.
- **Never alter config/hparams mid-series** — propose a separate experiment; don't mutate an in-flight arm.
- **Daytona snapshot caps are HARD** — clean STALE snapshots, never raise the cap
  (`.claude/projects/daytona/daytona.md`; the iris/shared `cli` org delete-only-`MISSING` rule is in
  `iris_job_lifecycle.md` §3).
- **Never kill a RUNNING job without explicit permission**; **never** `iris cluster restart`/stop/bounce
  (kills every running job). `iris job kill /benjaminfeuer/<job>` is job-scoped (with permission).

## 8. Monitoring + completion

> **⚠ LIVENESS / TERMINAL DETECTION = AUTHORITATIVE STATE-POLL, NEVER A LOG-STRING WATCH.**
> The canonical way to know whether a run is still alive — and to catch the moment it leaves RUNNING — is to
> **poll the authoritative iris job lifecycle state** (`iris job summary --json` → `state`), NOT to grep rank-0
> logs for a content string (`EPDIAG_FWD1` / `DEADLOCK` / `TERMINAL`, etc.). A clean kill, eviction,
> preemption, or an early crash that never prints one of those strings makes a content-watch see no matching
> line, so the watching agent goes idle believing the run is "still running" while the job + its pods have
> already left the cluster. This is exactly how the watch on `rl-131k-cpdcp2r3` missed a terminal transition:
> the run ended `killed` / "Terminated by user" with **0 pods**, but no terminal string was ever emitted, so
> the log-grep never fired. **Log-content greps are ONLY for the sel_rows / EPDIAG / throughput science (via
> `analyze_job_history.py`) — never for liveness or terminal detection.**

- **Watch a run (the primitive — use this, do not grep logs for liveness):**
  ```bash
  PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python   # otagent env: ships iris + a WORKING kubernetes
  # one-shot authoritative state (state + error + per-task + pod cross-check):
  $PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json
  # watch until the run leaves RUNNING, emit a line on every transition, exit terminal:
  $PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --interval 60
  #   exit 0 succeeded · 1 failed/killed/worker_failed/unschedulable · 2 absent (disappeared/0-pods) · 3 error
  ```
  It polls `iris job summary --json` (authoritative; falls back to the SQL `query`), and — crucially — treats
  **"no controller record AND 0 pods" as a TERMINAL `absent` verdict**, the case the old content-watch missed.
  Importable: `from scripts.iris.watch_job_state import get_job_state, watch`. (A supervising agent should use
  `watch(...)` / `get_job_state(...)` as the watch primitive.)
- **Manual state / logs** (synchronous calls only — no background `iris`/`kubectl`; use the **otagent** iris
  binary — the bare marin `.venv/bin/iris` lacks a working `kubernetes` for the cw k8s controller backend):
  ```bash
  # iris-side state (0=UNSPECIFIED 1=PENDING 2=BUILDING 3=RUNNING 4=SUCCEEDED 5=FAILED 6=KILLED
  #                  7=WORKER_FAILED 8=UNSCHEDULABLE) — the watcher already wraps this:
  /Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster=cw-us-east-02a query \
    "SELECT job_id,state FROM jobs WHERE job_id LIKE '/benjaminfeuer/<job>%'" -f csv
  # richest single-job authoritative call (state + error + exit + per-task states + finished_at):
  /Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster=cw-us-east-02a job summary /benjaminfeuer/<job> --json
  # via the python SDK (matches the launcher's client):
  #   IrisClient.status(JobName.from_wire("/benjaminfeuer/<job>"))
  #   IrisClient.fetch_task_logs(JobName.from_wire("/benjaminfeuer/<job>/0"))   # rank 0 = the driver
  ```
  Rank-0 (`/…/<job>/0`) is the training driver; ranks 1..N-1 are Ray workers that clean-exit on the head's
  done-marker (so on a healthy run the head failing shows `1 task failed, N-1 succeeded`).
- **What healthy bring-up looks like** (read it from the rank-0 log, in order): gang **admitted**
  (QuotaReserved/Admitted=True, pods leave SchedulingGated) → all N pods **Running** → Ray **rendezvous +
  all N nodes joined** (`Ray nodes alive: N/N`) → SkyRL driver up (`total training steps …`, dataloader
  built) → **engines up** (`InferenceEngineClient initialized with K engines`, each loading its shards 100%;
  for DCP the worker name is `Worker_TP0_DCP0`) → **policy mesh loaded** (the FSDP ranks load weights 5→100%)
  → first training step. A 30B/131k arm can take ~1h to the first rollout and ~2.5–3h to the gs1 forward.
- **Mandatory progress columns** (per `monitor-job-tables`): entropy / log_ratio / grad_norm + reward — not
  just step. Use `monitor-cron-sweep` for the cross-cluster cadence. For full-history **science** (sel_rows /
  EPDIAG / throughput stats) use the windowed pagination (`scripts/iris/analyze_job_history.py`), **NOT**
  `iris job logs --tail` (under-samples 10–100×) — but that is for the science only; **liveness/terminal
  detection is the state-poll watcher above**, never a log-string grep.
- **On completion → `rl-job-cleanup`** (best-ckpt selection, HF upload, the **manual** Supabase DB
  registration, trace export). `enable_db_registration` stays false at launch.
- **Teardown:** `iris job kill /benjaminfeuer/<job>` (with permission). Rescue banked traces from the
  gs:///s3:// jobs/rendezvous path if the trace upload didn't auto-run (`iris_job_lifecycle.md` §4).

---

## Operating notes
- **The Iris launcher uploads the LOCAL workspace to `/app`** (PYTHONPATH puts `/app` + `/opt/skyrl/skyrl-train`
  first), so a LOCAL commit takes effect on the next launch immediately — no cluster pull. To pick up a
  MarinSkyRL fix that landed after the image build, use `--skyrl-ref <commit>` (live editable checkout) rather
  than rebuilding the image. The vLLM fork, being compiled, is fixed only by an image rebuild (then bump the
  digest).
- **Always `--dry-run` a new/edited config** and eyeball the resolved hydra args: `colocate_all`,
  `policy_num_nodes`, the EXPLICIT `num_inference_engines`, TP/DCP/CP/EP/FSDP, and that the intended
  `extra_env:` keys forward with **no `NCCL_P2P_DISABLE` leak**. (See the resume log's VALIDATED block.)
- **This is the agentic terminal_bench/Harbor/Daytona path.** `DAYTONA_*` MUST be forwarded (the launcher
  passes it) — without it every trajectory finalizes `VerificationNotCompletedError` reward 0. Do NOT forward
  `AWS_*`/`R2_*` (clobbers the pod's injected R2 creds → the s3:// rendezvous silently hits real AWS).
