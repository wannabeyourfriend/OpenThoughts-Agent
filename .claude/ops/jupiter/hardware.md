Node composition: 4× NVIDIA GH200 Grace-Hopper superchips, all coupled within a single chassis. Each superchip = 1×
Hopper H100 GPU + 1× NVIDIA Grace CPU joined by NVLink-C2C. Single OS image (one Linux node), so a node-local job sees
4 GPUs on one host.

Per-superchip (one of four in a node):
- H100 GPU: 96 GB HBM3 @ 4 TB/s
- Grace CPU: 72 Arm Neoverse V2 cores @ 3.1 GHz base, SVE2-enabled (4× 128-bit vector units)
- Grace memory: 120 GB LPDDR5X @ 512 GB/s
- NVLink-C2C (Grace↔Hopper, intra-superchip): 900 GB/s bidirectional (450 GB/s/direction)
- 1× InfiniBand NDR200 NIC (200 Gb/s = 25 GB/s) on the closest NUMA node

Per-chip H100 compute (NVIDIA H100 SXM5 datasheet, dense — no sparsity):
- BF16/FP16: 989 TFLOPs/s
- FP8: 1979 TFLOPs/s (native — same MXU lanes as int8)
- INT8: 1979 TOPS
- TF32: 495 TFLOPs/s
- FP32 (CUDA cores): 67 TFLOPs/s

Whole-node totals (× 4 superchips):

┌───────────────────┬───────────────┬─────────────────────┐
│      metric       │ per superchip │         × 4         │
├───────────────────┼───────────────┼─────────────────────┤
│ HBM3              │ 96 GB         │ 384 GB (~358 GiB)     │
├───────────────────┼───────────────┼───────────────────────┤
│ HBM3 bandwidth    │ 4 TB/s            │ 16 TB/s aggregate            │
├───────────────────┼───────────────────┼──────────────────────────────┤
│ BF16 FLOPs/s      │ 989 TFLOPS        │ 3.96 PFLOPs/s                │
├───────────────────┼───────────────────┼──────────────────────────────┤
│ FP8 FLOPs/s       │ 1979 TFLOPS       │ 7.92 PFLOPs/s                │
├───────────────────┼───────────────────┼──────────────────────────────┤
│ LPDDR5X           │ 120 GB            │ 480 GB                       │
├───────────────────┼───────────────────┼──────────────────────────────┤
│ LPDDR5X bandwidth │ 512 GB/s          │ 2.05 TB/s aggregate          │
├───────────────────┼───────────────────┼──────────────────────────────┤
│ Grace cores       │ 72                │ 288 (Arm Neoverse V2)        │
├───────────────────┼───────────────────┼──────────────────────────────┤
│ InfiniBand        │ NDR200 (200 Gb/s) │ 4 NICs = 800 Gb/s = 100 GB/s │
└───────────────────┴───────────────────┴──────────────────────────────┘

Intra-node interconnect:
- GPU↔GPU (NVLink 4): 300 GB/s bidirectional between any pair, all-to-all
- CPU↔CPU (cNVLink): 200 GB/s bidirectional
- CPU↔GPU (NVLink-C2C, intra-superchip): 900 GB/s — note this is the "superchip" win, ~7× a PCIe Gen5 x16 link

- Native FP8 compute means weights run at 1979 TFLOPs/s/GPU (vs v5p having to dequant to 989-equivalent bf16). In
compute-bound regimes (high max_num_seqs), Jupiter should beat v5p-32 outright per chip and roughly tie per node — but
with the bonus of no multi-host ICI cost.
- Grace LPDDR5X is unified-addressable from the GPU (HMM), so for streaming-large-weights scenarios you can in
principle fault-in across the 900 GB/s C2C link — relevant for the larger MoE checkpoints you can't fit purely in HBM.