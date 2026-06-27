# CoreWeave H100 node configurations (`cw-us-east-02a`)

Hardware datasheet for the CoreWeave GPU cluster reached via iris (the x86/H100 analogue of
`iris_google_tpu_cloud_hardware.md`). **Ops/access/scheduling live in `coreweave_gpu_ops.md`** — this
file is the chip/node hardware reference (specs that inform RL geometry: per-GPU HBM, bandwidth, FLOPs,
NVLink/IB topology). Cluster-observed facts are marked `[obs]`; the rest is the NVIDIA H100-SXM5 datasheet.

## H100x8 (the only node shape here)

`cw-us-east-02a` is homogeneous: **one node shape, requested whole-node-exclusive** (`H100x8`, one iris
task per node, no co-tenants). `[obs]`

Node composition `[obs]` (from `coreweave_gpu_ops.md` + launch experience):
- **8× NVIDIA H100-80GB SXM5** per node, **x86_64** host (NOT aarch64/GH200 — contrast Jupiter).
- **~128 CPU cores/node**, but **~64–68 are persistent system/daemonset overhead** → only ~48–60 free
  (this is why `--cpu 48` admits a multi-node gang and `--cpu 64` does not).
- **~2 TB host DRAM/node** allocatable (the `--disk 512GB` / 512 GB-RAM-class requests fit easily).
- **NVLink (NVSwitch) intra-node + InfiniBand inter-node.** A TP=8 vLLM engine places **intra-node on one
  8-GPU node** over NVLink (no cross-node TP) — the placement Jupiter's 4-GPU GH200 nodes could never give
  the MoE DCP=2 arm.
- **~36 H100 nodes total** in the cluster `[obs]` → a hard ceiling of ~4 simultaneous 8-node gangs (minus
  other tenants).

Per-chip **H100-80GB SXM5** spec (NVIDIA public datasheet; compute = **dense**, the regime our matmuls use —
we don't run 2:4 structured sparsity, so the sparsity-doubled figures are not the relevant ones):
- **80 GiB HBM3, 3.35 TB/s** bandwidth
- **bf16 / fp16 tensor: ~989 TFLOPs/s** dense (1979 w/ sparsity)
- **FP8 tensor: ~1979 TFLOPs/s** dense (3958 w/ sparsity) — Hopper has native FP8 (Transformer Engine)
- **int8 tensor: ~1979 TOPS** dense
- **TF32 tensor: ~495 TFLOPs/s** dense · **FP64 tensor: ~67 TFLOPs/s**
- **NVLink: 900 GB/s** per GPU (4th-gen, bidirectional, full all-to-all via NVSwitch on-node)
- 700 W TDP · **compute capability sm_90** (Hopper → `TORCH_CUDA_ARCH_LIST="9.0"` for from-source builds)

H100x8 node totals:

┌───────────────┬─────────────┬──────────────────────────────┐
│    metric     │  per GPU    │          × 8 GPUs            │
├───────────────┼─────────────┼──────────────────────────────┤
│ HBM           │ 80 GiB      │ 640 GiB                      │
├───────────────┼─────────────┼──────────────────────────────┤
│ HBM bandwidth │ 3.35 TB/s   │ 26.8 TB/s aggregate          │
├───────────────┼─────────────┼──────────────────────────────┤
│ bf16 FLOPs/s  │ ~989 TFLOPS │ ~7.9 PFLOPs/s (dense)        │
├───────────────┼─────────────┼──────────────────────────────┤
│ FP8 FLOPs/s   │ ~1979 TFLOPS│ ~15.8 PFLOPs/s (dense)       │
├───────────────┼─────────────┼──────────────────────────────┤
│ int8 OPs/s    │ ~1979 TOPS  │ ~15.8 POPS (dense)           │
├───────────────┼─────────────┼──────────────────────────────┤
│ NVLink BW     │ 900 GB/s    │ on-node all-to-all (NVSwitch)│
├───────────────┼─────────────┼──────────────────────────────┤
│ host CPU      │ —           │ ~128 cores (~48–60 free)     │
├───────────────┼─────────────┼──────────────────────────────┤
│ host DRAM     │ —           │ ~2 TB/node                   │
└───────────────┴─────────────┴──────────────────────────────┘

## Interconnect (what shapes the parallelism)

- **Intra-node = NVLink/NVSwitch, 900 GB/s/GPU, full all-to-all.** This is why TP=8 (and TP=8 + DCP=2)
  belongs ON ONE NODE: the decode/EP all-reduce + all-to-all ride NVLink, not the slower fabric. ~~**Use NCCL
  defaults** — the GH200/SIF disables (`NCCL_P2P_DISABLE` / `NVLS=0` / `COLLNET=0`) would cripple this
  on-node path~~ → **⚠ DOUBTED for MoE (2026-06-27):** that "use NCCL defaults" rule is the leading suspect
  for the MoE weight-sync salad (working Jupiter MoE had the disables ON); A/B in flight — see
  `coreweave_gpu_ops.md` ⚠ + `agent_logs/2026-06-27_coreweave_nccl_defaults_doubt.md`.
- **Inter-node = InfiniBand**, gang-scheduled within a single IB leaf fabric (Kueue `topology 'infiniband'`,
  all-or-nothing — see `coreweave_gpu_ops.md` Scheduling). Typical CoreWeave H100 config is **8× 400 Gb/s
  NDR** (one NIC/GPU, GPUDirect RDMA) ≈ 3.2 Tb/s/node — *not independently re-measured on `cw-us-east-02a`,
  treat the exact NDR rate as datasheet-typical until confirmed.* Cross-node collectives (FSDP all-gather/
  reduce-scatter on the policy mesh, inter-engine) go over IB → keep TP intra-node and shard the *slower*
  axes (FSDP/CP) across nodes.

## How this informs RL geometry (the practical upshot)

- **640 GiB HBM/node** comfortably hosts a TP=8 vLLM engine for a 30B–35B-class MoE at long context: e.g. the
  131k MoE arm runs **4 engines × TP=8 / DCP=2** (32 inference GPU = 4 nodes) — each engine = one node, KV +
  weights well under 640 GiB at `gpu_memory_utilization 0.80`. (Contrast Jupiter's 4-GPU/96 GiB GH200 nodes,
  which forced DCP=1 and made TP=8 unplaceable.)
- **H100-80GB < GH200-96GB HBM** → when porting a GH200 config, the per-GPU memory budget tightens: drop
  `gpu_memory_utilization` (0.80→0.75) / `max_num_seqs` first on a KV-bind OOM (config-authoring detail in
  the launch skill §4).
- The **~2 TB host DRAM** is generous for `cpu_offload` — but note host-RAM OOM at FSDP weight-load on the
  policy nodes is still possible at 131k + EP8 + offload (observed on a 30B run); reduce `n_concurrent` /
  the rollout-worker count if it recurs.
- **sm_90** everywhere → any from-source build (vLLM fork, flash-attn) targets `TORCH_CUDA_ARCH_LIST="9.0"`
  (the gpu-rl image bakes this; see `build-gpu-rl-image-iris`).

## Cross-reference
- **Access / scheduling / KUBECONFIG / build / monitoring** → `coreweave_gpu_ops.md`.
- **Launch procedure + config-authoring (geometry, NCCL, extra_env)** → the `rl-agentic-launch-iris` skill.
- **TPU (Google) node shapes** → `iris_google_tpu_cloud_hardware.md` (a DIFFERENT cluster on the same iris SDK).
