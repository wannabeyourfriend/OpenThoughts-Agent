# TPU node configurations

## v5p-32

v5p-32 slice composition (from iris workers table):
- 4 workers (= 4 hosts), each with total_tpu_count=4 chips, chips_per_host_bounds=2,2,1
- 16 chips total per slice (confirms the [v5p_naming_cores_not_chips] memory: v5p-N = N cores = N/2 chips)
- Per-host DRAM: 464.7 GB; per-host CPU: 207 cores

Per-chip v5p spec (Google public datasheet):
- 95 GiB HBM2e, 2765 GB/s bandwidth
- 459 TFLOPs/s bf16
- 918 TOPS int8
- (No native FP8 — that's a v6e thing. Our Qwen122B-FP8 weights get dequantized to bf16 before matmul on v5p, so the
relevant compute number is the bf16 one.)

v5p-32 slice totals:

┌───────────────┬────────────┬──────────────────────────┐
│    metric     │  per chip  │        × 16 chips        │
├───────────────┼────────────┼──────────────────────────┤
│ HBM           │ 95 GiB     │ 1,520 GiB (~1.49 TiB)    │
├───────────────┼────────────┼──────────────────────────┤
│ HBM bandwidth │ 2765 GB/s  │ 44.24 TB/s aggregate     │
├───────────────┼────────────┼──────────────────────────┤
│ bf16 FLOPs/s  │ 459 TFLOPS │ 7.34 PFLOPs/s            │
├───────────────┼────────────┼──────────────────────────┤
│ int8 OPs/s    │ 918 TOPS   │ 14.69 POPS               │
├───────────────┼────────────┼──────────────────────────┤
│ host DRAM     │ —          │ ~1,859 GB across 4 hosts │
└───────────────┴────────────┴──────────────────────────┘

## v6e-8

Per-chip v6e (Trillium) spec from Google's public datasheet:
- 32 GiB HBM3, 1640 GB/s bandwidth
- 918 TFLOPs/s bf16 
- 1836 TOPS int8
- Native FP8 support at 1836 TFLOPs/s (this is where v6e beats v5p — v5p has no native FP8, must dequantize to bf16)

v6e-8 slice totals (8 chips, single host):

┌──────────────────┬─────────────┬────────────────┐
│      metric      │  per chip   │   × 8 chips    │
├──────────────────┼─────────────┼────────────────┤
│ HBM              │ 32 GiB      │ 256 GiB        │
├──────────────────┼─────────────┼────────────────┤
│ HBM bandwidth    │ 1640 GB/s   │ 13.12 TB/s     │
├──────────────────┼─────────────┼────────────────┤
│ bf16 FLOPs/s     │ 918 TFLOPS  │ 7.34 PFLOPs/s  │
├──────────────────┼─────────────┼────────────────┤
│ FP8 / int8 OPs/s │ 1836 TFLOPS │ 14.69 PFLOPs/s │
├──────────────────┼─────────────┼────────────────┤
│ host DRAM        │ —           │ ~1,410 GiB     │
└──────────────────┴─────────────┴────────────────┘

v6e-8 vs v5p-32 — same nominal bf16 throughput, very different memory:

┌───────────────┬─────────────────────────┬───────────────────────────────┐
│               │ v6e-8 (1 host, 8 chips) │  v5p-32 (4 hosts, 16 chips)   │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ HBM total     │ 256 GiB                 │ 1,520 GiB (5.9× more)         │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ HBM bandwidth │ 13.12 TB/s              │ 44.24 TB/s (3.4× more)        │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ bf16 PFLOPs/s │ 7.34                    │ 7.34 (same)                   │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ native FP8    │ yes (14.7 PFLOPs/s)     │ no (must dequant → 7.34 bf16) │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ host count    │ 1                       │ 4 (cross-host comms cost)     │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ host DRAM     │ 1,410 GiB               │ 1,859 GiB (across 4)          │
└───────────────┴─────────────────────────┴───────────────────────────────┘

This is exactly why 122B-FP8 fits on v5p-32 but not v6e-8 (per memory [v6e8_cannot_fit_122b_fp8]): 122B weights × 1
byte ≈ 122 GiB > the 256 GiB v6e-8 budget once you subtract activations, MoE fixed footprint, and compile-time peaks.
v5p-32 gives you 6× the HBM at the cost of multi-host coordination, no FP8 native, and ~3× the per-chip HBM bandwidth.