# Iris CoreWeave GPU cluster (`cw-us-east-02a`) — access & particulars

Cluster particulars for the **GPU RL path** on Marin's Iris: the CoreWeave
`cw-us-east-02a` H100 cluster (8× H100-80GB + InfiniBand per node), driven via
`python -m rl.cloud.launch_rl_iris` + the `gpu-rl` Docker image. This is the
ACCESS/HARDWARE/SCHEDULING reference; the launch HOW-TO (flag set, config-authoring
rules, bring-up checklist) lives in the **`rl-agentic-launch-iris`** skill — ops =
particulars, skill = procedure.

> **Scope — this is the GPU cloud, not the TPU cloud.** The rest of `.claude/ops/iris/`
> (`iris_job_lifecycle.md`, `iris_google_tpu_cloud_hardware.md`,
> `iris_eval_fixed_snapshot_template_scoping.md`) is the Iris **Google TPU** cloud
> (the `marin` cluster: datagen/eval via `data/cloud/launch_tracegen_iris.py` /
> `eval/cloud/launch_eval_iris.py`, regional `gs://` buckets, preemptible v5p/v6e
> slices). THIS doc is a DIFFERENT physical cluster — CoreWeave H100 GPUs, the
> `cw-us-east-02a` cluster, R2/`s3://` rendezvous, gang/Kueue admission — reached
> through the same `iris` SDK but with none of the TPU regional-egress / XLA-cache /
> 100 GB-node-disk mechanics. Don't cross-apply the TPU doc's region/disk/preemption
> rules here. (Named `coreweave_gpu_ops.md` rather than `ops.md` precisely so it does
> not read as "THE iris ops doc" over the TPU docs that share this dir.)

---

## Access

Launch from the **local Mac**, **otagent py3.12 conda env**
(`/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` — symlinks fail in the
sandbox; use the full interpreter path). There is **no cluster login / no SSH** — you
talk to the cluster through the `iris` SDK over a controller tunnel, and the launcher
uploads your **local** workspace to `/app` (so a local commit takes effect on the next
launch immediately — no push-then-pull, there is no Iris clone to pull).

**Pre-launch preamble:**
```bash
source /Users/benjaminfeuer/Documents/secrets.env   # HF_TOKEN, WANDB_*, DAYTONA_* — forwarded into the pod
export KUBECONFIG=~/.kube/coreweave-iris-gpu         # REQUIRED — the CoreWeave GPU cluster kubeconfig
```
- **`export KUBECONFIG=~/.kube/coreweave-iris-gpu` is a HARD PREREQUISITE for every
  CoreWeave job/query — set it in the same shell before any `iris`/`kubectl`/watcher
  call.** This Mac's **default `KUBECONFIG` (`~/.kube/config`) points at a DIFFERENT
  context** (TPU/`marin`/other), so without the export, `kubectl` inspects the wrong
  cluster and `iris` cw commands open the tunnel against the wrong backend — you can get
  misleading "0 pods / not found" / auth errors that look like a dead job but are really
  the wrong kubeconfig. Exporting it explicitly is also a good general safeguard even when
  the default happens to be benign. (It's `export`ed into the shell env, so it persists for
  the whole session; re-export in any fresh shell or background call.)
- **`~/.kube/coreweave-iris-gpu`** is the CoreWeave GPU cluster's kubeconfig (distinct
  from any TPU/`marin` context). `kubectl` reads it via `KUBECONFIG`; the `iris` SDK
  reaches the cluster through the controller tunnel it opens from the cluster YAML (the
  kubeconfig is what backs direct `kubectl` inspection of pods/nodes/Kueue workloads).
- **`source ~/Documents/secrets.env` is load-bearing**, not optional: the launcher
  forwards `HF_TOKEN` / `WANDB_*` / `DAYTONA_*` from the launch host's env into the task
  pod. Without `DAYTONA_*` every agentic trajectory finalizes `VerificationNotCompletedError`
  reward 0 (no sandbox comes up); without `HF_TOKEN` weight/data resolution fails.
- **Cluster config** auto-resolves to the marin repo's
  `~/Documents/marin/lib/iris/config/cw-us-east-02a.yaml`
  (`launch_rl_iris.py:_resolve_cluster_config_default`); override with `--cluster-config`
  only if it moved.
- **`iris` CLI binary** = `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris`. **Use the otagent-env
  binary, NOT the bare `marin/.venv/bin/iris`** — the cw cluster is a k8s controller backend, and reaching it
  (even just to open the tunnel) instantiates `CloudK8sService`, which imports `kubernetes`. The marin `.venv`
  ships only a broken/partial `kubernetes` (dist-info present, module not importable) → every cw `iris`
  command (`job summary`, `query`, `job list`, …) dies with `ImportError: Install iris[controller]`. The
  **otagent** env has a working `kubernetes` 35.0.0 + the editable iris package, so its `iris` binary drives cw
  cleanly. (`conda activate marin && uv run iris` also works IF that venv has the controller extra; the
  otagent binary is the reliable default.) All `iris`/`kubectl` calls must be **SYNCHRONOUS** — never
  background them.

**Verify access** (cheap, before submitting):
```bash
# iris-side: my live jobs (JobState: 0=UNSPECIFIED 1=PENDING 2=BUILDING 3=RUNNING 4=SUCCEEDED
#                          5=FAILED 6=KILLED 7=WORKER_FAILED 8=UNSCHEDULABLE)
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster=cw-us-east-02a query \
  "SELECT job_id,state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%'" -f csv

# k8s-side: H100 node headroom (an N-node gang needs N WHOLE free 8-GPU nodes)
kubectl get nodes        # with KUBECONFIG=~/.kube/coreweave-iris-gpu  (Ready count ONLY — see trap below)
```

**Are nodes actually free? Use Kueue + a per-node free-GPU count — NOT `kubectl get nodes` or a pod request-sum.**
`kubectl get nodes` shows *Ready*, not *free*; and a naive "sum `requests.nvidia.com/gpu` over running pods vs
36×8" is wrong twice over (verified 2026-06-26): (a) allocatable GPUs is **~256, not 288** — only **~32 of the
~36 Ready nodes carry 8 GPUs** (the rest are util/control nodes with 0 GPU), and (b) the request-sum **undercounts**
because some pods declare GPUs via `limits`, not `requests`. Both errors make a busy cluster look free → you
relaunch into contention and get **preempted by higher-priority `/power` jobs** (interactive < production). Use the
two authoritative signals instead:

```bash
# (1) Kueue ClusterQueue = the SCHEDULER'S OWN accounting — this is literally what decides gang admission.
kubectl get clusterqueue                 # PENDING WORKLOADS column: 0 = no admission backlog (good sign)
kubectl get clusterqueue iris-cq -o json | python3 -c 'import json,sys; d=json.load(sys.stdin)["status"]; \
print("admitted:",d.get("admittedWorkloads"),"pending:",d.get("pendingWorkloads")); \
print([{r["name"]:r.get("total") for r in f["resources"]} for f in d.get("flavorsUsage",[])])'
#   (nominalQuota lives under .spec.resourceGroups[].flavors[].resources[].nominalQuota; note quotas mix units
#    like "1G" for memory, so parse GPU separately — don't int() the whole map.)

# (2) CORRECT free whole-nodes = per-node (allocatable_gpu - sum of bound-pod GPU req/limit), count nodes with >=8 free.
kubectl get nodes -o json | python3 -c '
import json,sys,subprocess
nodes=json.load(sys.stdin)["items"]
alloc={n["metadata"]["name"]:int(n["status"]["allocatable"].get("nvidia.com/gpu",0)) for n in nodes}
pods=json.loads(subprocess.run(["kubectl","get","pods","-A","-o","json"],capture_output=True,text=True).stdout)["items"]
used={}
for p in pods:
    if p.get("status",{}).get("phase") in ("Succeeded","Failed"): continue
    nn=p.get("spec",{}).get("nodeName")
    if not nn: continue
    g=0
    for c in p["spec"].get("containers",[]):
        r=c.get("resources",{}); req=r.get("requests",{}) or {}; lim=r.get("limits",{}) or {}
        g+=int(req.get("nvidia.com/gpu", lim.get("nvidia.com/gpu",0)))
    used[nn]=used.get(nn,0)+g
gpu_nodes=sum(1 for a in alloc.values() if a>0)
free=sum(1 for n,a in alloc.items() if a>0 and a-used.get(n,0)>=8)
print(f"GPU nodes:{gpu_nodes}  fully-free 8-GPU nodes:{free}  total free GPUs:{sum(max(0,a-used.get(n,0)) for n,a in alloc.items())}")'
```
Decision rule for relaunching an idle gang: only submit when **`pendingWorkloads == 0`** AND **fully-free 8-GPU
nodes ≥ N** (the gang size; e.g. ≥16 for a 30B+35B pair). If `/power` is bursting (free nodes oscillating), either
wait for it to drain or escalate to `--priority production` — do NOT churn-relaunch at interactive into contention.
The in-container invocation the launcher ultimately drives is
`uv run iris --cluster=cw-us-east-02a job run …` (the SDK `IrisClient.submit` path);
you do not type that by hand — `python -m rl.cloud.launch_rl_iris` builds it.

**Monitor liveness — state-poll, NOT a log-string watch.** To know whether a run is still alive (and to catch
the moment it leaves RUNNING) poll the **authoritative iris job lifecycle state**, never grep rank-0 logs for a
content string. A clean kill / eviction / preemption / early crash often emits **no** terminal log line, and
the pods are reaped, so a content-watch sits idle while the job is gone (this is how the `rl-131k-cpdcp2r3`
watch missed the run ending `killed`/"Terminated by user" with 0 pods). The watch primitive:
```bash
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
$PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json    # authoritative state now
$PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --interval 60     # watch until terminal
#   wraps `iris job summary --json` (auth) + SQL `query` fallback + kubectl pod cross-check;
#   "no record AND 0 pods" => terminal `absent`. Importable: get_job_state() / watch().
```
Log-content greps (`scripts/iris/analyze_job_history.py`) are for the sel_rows / EPDIAG / throughput **science
only** — never for liveness/terminal detection. (The launch HOW-TO's §8 carries the full monitoring rule.)

**Finelog retains the FULL job log — it is NOT capped at ~1600 init lines** (an earlier note claimed a storage
cap; that was wrong). The whole log, init → crash, is
retrievable by **time-window pagination** with the **cw-capable iris binary**
(`/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris` — the marin `.venv` iris has a broken `kubernetes`
import and CANNOT drive cw). The only real truncation is `--tail`'s line cap; `--since-ms <submitted_at_ms>
--no-tail` returns everything:
```bash
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris   # KUBECONFIG=~/.kube/coreweave-iris-gpu
$IRIS --cluster cw-us-east-02a query \
  "SELECT job_id,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code FROM jobs WHERE job_id LIKE '%<job>%'"
$IRIS --cluster cw-us-east-02a job logs /benjaminfeuer/<job> --since-ms <submitted_at_ms> --max-lines 500000 --no-tail
```
Proven 2026-06-25 on dead `rl-131k-cpdcp2r3-v2`: one `--no-tail --since-ms <submit>` returned all **2275** lines
spanning the full **10:09:08 → 10:16:21** lifetime, recovering the rank-0 fc cause
(`ModuleNotFoundError: No module named 'torchtitan'` in `fsdp_utils.py:667 apply_ep` — the MoE EP path needs
torchtitan, not in the gpu-rl image). The analyzer now takes `--cluster` + `--iris-bin` (auto-resolves the
cw-capable iris) so it works for cw out of the box.

---

## Hardware (node shape)

CoreWeave `cw-us-east-02a` node = **8× H100-80GB + InfiniBand**, requested **whole-node
exclusive** (`H100x8`, one iris task per node, no co-tenants). ~**36 H100 nodes** total
in the cluster.

- **~128 CPU cores per node, BUT ~64–68 cores are persistent system/daemonset overhead**
  → only ~48–60 cores are actually free per node. This is THE reason `--cpu 48` admits a
  multi-node gang and `--cpu 64` does not (see Scheduling below).
- **Whole-node-exclusive ⇒ REQUEST ALL the node's allocatable resources** (no co-tenants, so
  under-requesting is wasted capacity AND a footgun). Node allocatable ≈ **128 CPU / ~2014 GiB
  mem / 8 GPU**. The launcher defaults (`launch_rl_iris.py`) now request the full node: **`--cpu 48`**
  (the max-admittable — >~60 fails the IB gang), **`--memory 1800GB`** (≈ the full ~2 TB, leaving
  daemonset headroom), **`--gpus_per_node 8`**, `--disk 512GB` (rendezvous/ckpts go to R2, not
  node-local). ⚠ The old **`--memory 512GB`** default was a CGROUP-OOM footgun: the FSDP weight-load
  on an EP=8 + `cpu_offload` policy rank peaks above 512 GB while the NODE sits <200 GB used → the
  *container* OOM-killer fires though the node has ~1.8 TB free. Always request ~the full node.
- **NVLink intra-node + InfiniBand inter-node.** This is the headline difference from
  Jupiter's GH200 4-GPU nodes: a TP=8 vLLM engine places **intra-node on ONE 8-GPU node**
  over NVLink (decode), with no cross-node TP — exactly the placement Jupiter's 4-GPU
  nodes could never satisfy for the MoE DCP=2 arm.
- **NCCL DEFAULTS — use them (validated; the MoE-salad doubt was FALSIFIED 2026-06-27).**
  On H100+IB, do NOT set the GH200/SIF disables (`NCCL_P2P_DISABLE` / `NCCL_NVLS_ENABLE=0` /
  `NCCL_COLLNET_ENABLE=0`): they would cripple the intra-node NVLink all-reduce a TP=8 (DCP)
  engine depends on. NCCL defaults give NVLink intra-node + IB inter-node. Keep the
  observability/raised-timeout env (`NCCL_DEBUG=INFO`, `SKYRL_WORKER_NCCL_TIMEOUT_IN_S`,
  `TORCH_NCCL_*`).
  - *(2026-06-27 A/B:* run `rl-131k-cpdcp2r3-think2507-r9` re-added all three disables on a
    reproducing 30B MoE (TP=2+EP=2) — env verified in-pod, nothing else changed — and the
    served policy was **still CJK token-salad**. NCCL P2P/NVLS is therefore **NOT** the
    MoE-salad cause. **RESOLVED:** the salad was the FusedMoE `w13` gate/up swap not re-applied
    on the disaggregated RL weight update (FlashInfer-CUTLASS on H100 swaps `[gate;up]→[up;gate]`
    on disk-load but the per-chunk RL update skipped it) — fixed by `SKYRL_W13_RELOAD_BRACKET`
    (MarinSkyRL `2bb70a88`; default on; see the marinskyrl project doc). Record:
    `agent_logs/2026-06-27_coreweave_nccl_defaults_doubt.md`.)*
- **Egress: CoreWeave nodes have internet.** Models/data are pulled from HF **online** —
  do NOT set `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` (contrast Leonardo/Jupiter compute
  nodes, which have none). The cost is the transient HF-weight-resolution flake below.
- **Storage/scratch:** ephemeral per-node disk via the `--disk` request (default `512GB`);
  multi-node Ray rendezvous + any banked traces go through the R2/`s3://` rendezvous
  bucket (below), not node-local disk. There is no shared persistent POSIX scratch like
  Leonardo's `$WORK` — checkpoints/exports go to HF / the object store.

---

## Scheduling & multi-node particulars

- **Gang scheduling.** `--num-nodes N` → `replicas=N` whole `H100x8` tasks. For GPUs with
  replicas>1, `resolve_multinode_defaults` returns
  **`CoschedulingConfig(group_by="leafgroup")`** — all N nodes co-scheduled on **one
  InfiniBand leaf fabric**, all-or-nothing. `cw-us-east-02a` enables **Kueue gang
  admission** (`kueue.cluster_queue: iris-cq`, `host_network: true` for NCCL/IB), so the
  N-task gang admits **atomically** (all N whole nodes granted, or it queues). At submit
  you see `replicas=N, coscheduling=leafgroup`; pods then sit **SchedulingGated** (normal
  Kueue gang pre-admit) until admitted.
- **The single-IB-leaf gang constraint is what `--cpu 48` is about.** The gang must fit on
  ONE IB leaf; with `--cpu 64` only ~2/32 nodes have ≥64 free cores (the daemonset
  overhead above), so an N-node single-leaf gang can never be satisfied and sits
  SchedulingGated forever with a Kueue `topology 'infiniband' allows to fit only 2 out of
  N pod(s)` message. `--cpu 48` fits all nodes → admits immediately (QuotaReserved=True).
- **Multi-node Ray rendezvous via an `s3://` (R2) bucket.** `--num-nodes>1` REQUIRES
  `--rendezvous-dir` (the launcher hard-errors otherwise). On `cw-us-east-02a` use an
  **`s3://` URI under the cluster's `marin-na` bucket** (R2), e.g.
  `s3://marin-na/iris/rl-<slug>/<run>`. The cluster injects working R2 creds into every
  task pod via the **`iris-task-env` k8s Secret** (`envFrom`, because
  `storage.remote_state_dir` is an `s3://` URI), so **no external creds are needed** — and
  you must **NOT forward `AWS_*`/`R2_*`**: explicit container `env` overrides `envFrom`, so
  forwarding the launch host's `AWS_*` (different account, no `AWS_ENDPOINT_URL`) clobbers
  the pod's R2 creds and silently targets real AWS S3. Use a **fresh sub-path per run** so
  a stale head file from a prior attempt isn't picked up. Mechanism: one
  `start_rl_iris_controller.py` per node; rank 0 writes `ray_head.json` to the rendezvous,
  workers poll for it and join; rank 0 publishes `ray_head.done` on completion.
- **The `gpu-rl` image is deps-only; source is synced at runtime.** The image
  (`ghcr.io/open-thoughts/openthoughts-agent`, pinned by **immutable `@sha256:` digest**
  in `launch_rl_iris.py:DEFAULT_RL_DOCKER_IMAGE` — NOT the floating `:gpu-rl` tag, which
  stale-caches under `imagePullPolicy: IfNotPresent`) bakes the RL conda venv
  (`/opt/openthoughts/envs/rl`: torch 2.11 + the **vLLM fork built from source** +
  flash-attn 2.8.3), **MarinSkyRL editable** at `/opt/skyrl`, and harbor. The launcher
  syncs the **local OT-Agent workspace to `/app`** (first on PYTHONPATH) → first-party
  edits live on the next launch **without an image rebuild**. A MarinSkyRL fix that landed
  after the image build can be picked up live via `--skyrl-ref <commit>` (editable
  checkout); only the compiled vLLM fork requires an image rebuild (then **bump the
  digest**, using the immutable `:gpu-rl-<gitsha>` tag's digest).

---

## Binding gotchas

> **⚠ `--cpu 48`, NOT the 64 default** (paid for 2026-06-24,
> `agent_logs/2026-06-25_coreweave_131k_cpdcp2r3_resume.md`). At `--cpu 64` the N-node
> single-IB-leaf gang can't be satisfied (only ~2/32 nodes have ≥64 free cores after the
> ~64-core daemonset overhead) and sits **SchedulingGated forever** with the Kueue
> `topology 'infiniband' allows to fit only 2 out of N pod(s)` message. `--cpu 48` fits
> all nodes and admits immediately. Memory `512GB` is fine.

> **⚠ `--max-retries ≥1` for the transient HF weight-resolution flake.** At scale (e.g. 32
> FSDP ranks each resolving sharded safetensors online) one rank can hit a transient HF Hub
> HTTP/EOF failure → transformers reports the generic `… does not appear to have a file
> named model.safetensors`; with `max_retries=0` that one rank SIGKILLs the whole gang.
> `--max-retries 1` re-brings-up the gang on that failure (time-only cost). **First-party
> mitigation has landed** — a weight-resolution retry wrapper in MarinSkyRL (commit
> `0b2b05b`); keep `--max-retries ≥1` as belt-and-suspenders. (Durable alternative:
> pre-stage the model into the image's HF cache / a shared snapshot before the FSDP workers
> start, or raise `HF_HUB_DOWNLOAD_TIMEOUT`.)

> **⚠ `--memory` default is now `1400GB`** (changed in `launch_rl_iris.py:DEFAULT_MEMORY_PER_NODE`
> 2026-06-26, because the old `1800GB` default was an admission footgun you had to remember to
> override). `1800GB` (≈1676 GiB) sits so close to node-allocatable (~2014 GiB) that after the
> daemonset + persistent-reservation overhead a leafgroup (all-or-nothing, one IB leaf) gang
> can't fit all its pods → Kueue `topology 'infiniband' allows to fit only K of N … excluded:
> resource "memory"` → **SchedulingGated stall** (cost multiple 60–120 min stalls overnight
> 2026-06-26 — a 1-GPU probe AND 8-node gangs). **`1400GB` (the new default) is validated** for
> the 8-node 131k EP8 run (admits cleanly + does the full weight-load with no cgroup-OOM); drop
> to `1000–1200GB` for 2-node smokes. The lever on an admission stall is LOWERING `--memory`
> toward the real need, **never raising a cap**. (The old `512GB` was the opposite footgun — a
> weight-load cgroup-OOM; `1400GB` is the validated middle, and now the default so no flag is
> needed.)

- **Ray agent ports collide with `worker_ports` nondeterministically — pin them all.** `ray
  start` (head AND worker) lets Ray RANDOMIZE several system ports (`metrics_export`,
  `runtime_env_agent`, `dashboard_agent_grpc`, …) from the ephemeral zone that overlaps the
  default `worker_ports` range **10002–19999**. A random landing inside it aborts the node
  (`ValueError: Ray component worker_ports is trying to use a port number <N> that is used by
  other components`) — **nondeterministic** (passes or fails run-to-run on port luck; a
  likely cause of intermittent "long build then die" CoreWeave deaths). A head-only or
  single-port pin is INSUFFICIENT (the randomized port just moves to another agent). **Fix
  (committed `beda7a7f`, `scripts/iris/start_rl_iris_controller.py:_ray_port_flags`):** pin
  ALL of them outside the range on head+worker — `metrics_export=8090, runtime_env_agent=8092,
  dashboard_agent_grpc=8093, dashboard_agent_listen=8094, node_manager=8076,
  object_manager=8077`. Rides the `/app` upload (no rebuild).
- **Nodes have NODE-LOCAL storage (no shared GPFS) → stage the agentic task dataset on EVERY
  node.** Unlike the SLURM clusters (shared GPFS — one rank extracts, all nodes see it),
  CoreWeave's `/opt/openthoughts/tasks` is node-local, so a rank-0-only `parquet→tasks`
  extraction leaves the 7 workers with EMPTY task dirs → every rollout throws
  `FileNotFoundError: /opt/openthoughts/tasks/<dataset>/<instance>/task.toml` → reward 0.
  This is a SILENT data-starvation: the compute path looks green (grouped-mm/R3 fine, no
  crash) but `avg_num_tokens≈1.0` and all rewards are 0 (cost a full doomed run + a misread
  "the other model's rollouts worked" — neither did). **Fix (committed `7c135780`):** the
  launcher forwards `--train-data` to the controller, which stages on every node before Ray
  via `resolve_rl_train_data`. Verify in bring-up: each rank logs `Staging train_data on this
  node (rank N/8)` → `[extract] … Done` before rollouts.

- **Transient self-healing on bring-up is NORMAL, not a fault to salvage:** a `ghcr.io`
  blob EOF → `ImagePullBackOff` self-heals (kubelet retries); `shm_broadcast: No available
  shared memory broadcast block found in 60s` is **benign** (engines idle-wait while the
  policy mesh loads weights).
- **k8s does NOT shell-expand `$VAR` in injected env.** Hardcode literal paths in a
  config's `extra_env:` (e.g. `LD_LIBRARY_PATH: /opt/openthoughts/envs/rl/lib`, **not**
  `$CONDA_PREFIX/lib`) — the launcher injects env as literal k8s values. (Config-authoring
  detail; full rules in the launch skill §4.)
- **Egress / HF online:** because CoreWeave has egress, do NOT carry `HF_HUB_OFFLINE` /
  `TRANSFORMERS_OFFLINE` over from a Jupiter/Leonardo (no-internet) config port.

---

## Cross-reference

- **Launch procedure** (the flag set, config map + node-count derivation, config-authoring
  rules for `hpc/skyrl_yaml/iris/`, gang/rendezvous walkthrough, bring-up checklist,
  monitoring + completion) → the **`rl-agentic-launch-iris`** skill.
- **TPU (datagen/eval) Iris** → `iris_job_lifecycle.md` + `iris_google_tpu_cloud_hardware.md`
  (a DIFFERENT cluster; see the scope banner above).
- **Code:** `rl/cloud/launch_rl_iris.py` (launcher, digest pin, `extra_env` forwarding,
  AWS_* warning), `scripts/iris/start_rl_iris_controller.py` (the per-node Ray rendezvous
  controller).
- **Standing constraints** (≤6 RUNNING RL/cluster, `enable_db_registration: false`, a3
  CONCLUDED, Daytona snapshot caps HARD, never kill a RUNNING job / `iris cluster
  restart` without permission) — see `CLAUDE.md §Always` + the launch skill §7.

## Marin GitHub-issue monitoring + research (external skills)

CoreWeave/Iris is **Marin's** cluster — when monitoring/triaging/updating the Marin GitHub
issues this work touches (e.g. the RL-launch-on-Iris workflow issue, cluster/quota threads,
upstream fixes we depend on), use the **mumwelt skills at `/Users/benjaminfeuer/Documents/mumwelt/mumwelt/skills/`**
(invocable by name; they shell out to the `mum` CLI over a local offline mirror of all Marin
activity — GitHub issues/PRs/comments, Discord, W&B, weekly summaries):

- **`marin-context`** — search + cite Marin activity (a specific issue/PR/run, "what was
  decided + why", who did what). The default for **monitoring** a Marin issue/PR's state +
  history before commenting on or updating it.
- **`marin-research`** — multi-subagent deep dive for broad/ambiguous "full picture of X" /
  retro questions one query won't cover.
- **`marin-publish`** — render a finished writeup to a linkable gist (htmlpreview) to share.

These give the **context** to monitor + decide; the actual issue/PR **update** still goes
through the `gh` CLI (the same path used for our Marin-community issue + PR comments). Check
the mirror's freshness first (the skill prompts if stale). They live OUTSIDE this repo (the
`mumwelt` checkout), so they're a referenced tool, not a committed part of ot-agent.
