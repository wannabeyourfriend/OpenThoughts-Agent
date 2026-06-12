# Eval presets

Shared catalog of eval-run defaults. One YAML file per preset, named
`<preset>.yaml` (the stem is the preset name used on the CLI, e.g. `swebench`).

Both consumers load presets from this directory via
`eval.presets.load_presets()`:

- **SLURM orchestrator** — `eval/unified_eval_listener.py` (`--preset`).
- **Iris launcher** — `eval/cloud/launch_eval_iris.py` (`--preset`).

Keeping the catalog in one place means the two launch paths stay in sync.

## Format

Each file is a flat mapping. `load_presets()` returns
`{stem: parsed_yaml_mapping}` in sorted-key order, with field types preserved
(bools stay bools, ints stay ints, `datasets` stays a list).

| Field | Type | Meaning |
|---|---|---|
| `datasets` | list[str] | HF dataset ids to evaluate (SLURM iterates all; Iris uses the first). |
| `log_suffix` | str | Suffix for the listener's log file. (SLURM-only) |
| `n_concurrent` | int | Harbor `--n-concurrent`. |
| `error_threshold` | int | Max invalid errors before abort. (SLURM-only) |
| `vllm_max_retries` | int | vLLM startup retries. (SLURM/serve-only) |
| `enable_thinking` | bool | Harbor agent-kwarg `enable_thinking=true`. Affects results. |
| `agent_parser` | str | Harbor agent-kwarg `parser=<value>` (e.g. `xml`). Affects results. |
| `auto_snapshot` | bool | Pre-build Daytona snapshots. (SLURM-only) |
| `config_yaml` | str | Listener eval config YAML. (SLURM-only) |
| `slurm_time` | str | SLURM time limit. (SLURM-only) |
| `agent_envs` | str | Comma-separated `KEY=VALUE` envs forwarded into the sandbox. (SLURM-only) |

## Iris launcher mapping

`launch_eval_iris.py --preset <name>` applies a subset (CLI flags always win):

- **Applied (Iris analogs):** `datasets[0]` → `--dataset_path`,
  `n_concurrent` → `--n_concurrent`.
- **Applied (result-affecting agent kwargs):** `agent_parser` → agent-kwarg
  `parser=<value>`, `enable_thinking` → agent-kwarg `enable_thinking=true`.
- **Ignored (SLURM/vLLM-serve-only, no Iris analog):** `slurm_time`,
  `vllm_max_retries`, `log_suffix`, `error_threshold`, `config_yaml`,
  `agent_envs`, `auto_snapshot`.

## Adding a preset

Drop a new `<name>.yaml` here with the fields above. No code change is needed;
both launchers pick it up automatically (and `--preset` choices update).
