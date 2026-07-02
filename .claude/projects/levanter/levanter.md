# Levanter — facts & how experiments are specced

JAX foundation-model training library (the engine under marin's pretrain/midtrain/SFT steps).
Learned 2026-06-29 while pulling Delphi midtraining configs. Source of truth = the marin monorepo
checkout `/Users/benjaminfeuer/Documents/marin/lib/levanter` (published to PyPI as `marin-levanter`).
See also `marin-executor` (how it's launched) and `mum` (how to find run configs). Grug is **not** a
Levanter frontend — see `[grug]` notes / `marin/.agents/projects/grugformer.md`.

---

## What it is (correct the "SFT engine" misconception)

- A **full** LLM/foundation-model training library — **pretraining, midtraining (CPT), SFT, eval,
  export** — not specifically an SFT engine. SFT is a *mode* of `train_lm`, not a separate engine.
- Stack: **Haliax** named tensors (`NamedArray` + explicit `Axis`) over **Equinox** modules;
  **draccus** dataclass configs; `AsyncDataset` data pipeline. Runs **TPU + GPU**.
- **Bitwise deterministic** (same config → same result across preempt/resume).

## How experiments are specced — a draccus dataclass tree, one per entry point

- Entry points live in `lib/levanter/src/levanter/main/`: `train_lm.py`, `train_dpo.py`, `lora_lm.py`,
  `eval_lm.py`, `export_lm_to_hf.py`, `sample_lm.py`, `train_asr.py`, … Each has a root config dataclass.
- `train_lm` → **`TrainLmConfig`** (`src/levanter/main/train_lm.py:42`), composed of nested sub-configs:
  - `data: LmDataConfig` — tokenizer, `cache_dir`, `components` (per-component `source` = url|hf + `format`
    = chat for SFT), **`train_weights`** (the mixture map — this is the `pNNmNN` Delphi mix), `shuffle`.
  - `trainer: TrainerConfig` — `train_batch_size`, `num_train_steps`, `mp` (precision e.g. `p=f32,c=bfloat16`),
    `per_device_parallelism`, `tracker` (wandb).
  - `model: LmConfig` (default `LlamaConfig`) — `type: llama`, dims, heads, `rope`, etc.
  - `optimizer: OptimizerConfig` (default `AdamConfig`) — `learning_rate`, `weight_decay`, `warmup`, schedule.
  - `train_seq_len`, `initialize_from_hf` / `initialize_from_checkpoint_path` (CPT/SFT/midtrain **init**),
    `hf_save_path`/`hf_upload`, `eval_harness`, `adapter` (LoRA).
- Polymorphism: a **`type:` field** selects the variant (model `type: llama`, optimizer/tracker/data-source
  types) — draccus discriminated unions.

## Authoring + running it (the raw-Levanter path)

- A **YAML** populates the tree (`lib/levanter/config/*.yaml` — e.g. `llama3_small_fast.yaml`,
  `train_lm_config` shapes; SFT example `train_lm_llama3_tulu_sft.yaml` uses `train_weights: {tulu: 1.0}` +
  `format: {type: chat}`).
- Run via the draccus entry (`levanter.config.main(main)()` at `train_lm.py:428`):
  `python -m levanter.main.train_lm --config_path config/<f>.yaml --trainer.num_train_steps 5000` (any field
  CLI-overridable with dotted keys).
- **SFT / midtraining are just `train_lm` with a different data block + init** — SFT = chat-format component +
  `initialize_from_hf`; midtrain/CPT = a `train_weights` mixture + `initialize_from_{hf,checkpoint_path}`.

## Where the RESOLVED config is persisted (this is how to recover what a run actually used)

A finished run writes its full resolved config to its GCS run dir — **but the filename depends on the launch path**:
- **Script/midtrain-launch** → `gs://…/checkpoints/<run>/train_lm_config.yaml` (+ a run manifest, e.g.
  `midtrain_manifest.json`).
- **marin-executor launch** → no `train_lm_config.yaml`; the config is inside `.executor_info` under
  `config.train_config` (`jq '.config.train_config'`). See the `marin-executor` doc.
This split is why the same experiment family can have two artifact shapes (and why `mum run` finds some configs
but not others — see `mum`).

## Worked example (Delphi K=0.20 midtraining = TrainLmConfig instances)
`data.train_weights` = the pNNmNN mixture (math `nemotron_cc_math_v1/4plus` + Nemotron-CC web replay);
`initialize_from_hf` = the matching base; optimizer = Muon dual-LR (`learning_rate` matrix + `adam_lr`),
`lr_schedule: linear`, `warmup: 0.1`, `min_lr_ratio: 0`; tokenizer `meta-llama/Meta-Llama-3.1-8B`, seq_len 4096,
`block_cross_document_attention: true`. Configs pulled to `~/Documents/experiments/active/midtrain-25B/configs/`.
