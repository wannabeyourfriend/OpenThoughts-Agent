# Analysis Scripts

Utilities for analyzing RL/SFT training traces, evaluation results, and HuggingFace datasets.

## Behavioral Analysis Pipeline (orchestrator)

`analyze_rl_behavior.py` chains the scripts below into a single pipeline
keyed to four research questions:

| Question | Steps invoked |
|---|---|
| **Q1.** What model behaviors are changing as a result of RL? | `behavioral_delta` + `summarize_conversations` (+ optional `update_hf_failure_modes` upstream) |
| **Q2.** Are reward changes over time attributable to behavior or other factors? | `temporal_trace_analysis` + `parse_skyrl_metrics` |
| **Q3.** Do behavioral changes persist in post-RL eval traces? | `eval_temporal_overlay` + `trace_pair_render` + `post_training_comparison` |
| **Q4.** Do those changes affect eval results? If not, is it expected? | `solve_rate_by_context` (+ Q1/Q3 outputs for interpretation) |

Each step writes into `--output-dir/<step>/` and is skipped if its output marker already exists (use `--force` to re-run, `--skip <names>` to opt out, `--only <names>` for a subset). The plan + cross-linked index land at `--output-dir/{pipeline_plan.json, INDEX.md}`.

```
python -m scripts.analysis.analyze_rl_behavior \
    --rl-traces        penfever/rl-train-traces-foo \
    --baseline-eval    penfever/eval-pre-rl-foo \
    --post-rl-eval     penfever/eval-post-rl-foo \
    --post-rl-eval-ts  2026-05-28T18:30 \
    --baseline-eval-ts 2026-05-23T09:00 \
    --training-log-dir /scratch/skyrl-logs/foo/ \
    --output-dir       /Users/me/Documents/notes/rl-behavior-foo/
```

Add `--annotate-failure-modes` to run `update_hf_failure_modes` on both eval repos before Q1 picks up the labels (requires `OPENAI_API_KEY`; can take hours on large datasets). Use `--dry-run` to print the plan without executing.

## New Analysis Tools

| Script | Question | Description |
|---|---|---|
| `behavioral_delta.py` | Q1 | Diff failure-mode + behavioral metrics between two trace datasets. Writes markdown + JSON sidecar. |
| `trace_pair_render.py` | Q3 | Side-by-side HTML render of representative trials per common task (default sort: pass/fail flips first). |
| `eval_temporal_overlay.py` | Q3 | Extends `temporal_trace_analysis`: overlays eval-checkpoint markers on the RL-time reward curve. |
| `analyze_rl_behavior.py` | (orchestrator) | Runs all of the above + the existing scripts in the right order, with resumability via output-marker detection. |

## Shared Utilities

| Module | Description |
|---|---|
| `utils.py` | Common helpers: `load_traces()` unified loader (HF/JSONL/dir), `Trace` dataclass with eager field caching, `task_id_of()`, `group_by_task()`, plus the original text/reward/error/date/token primitives |

## Dataset & Context Analysis

| Script | Description | Usage |
|---|---|---|
| `context_length_compare.py` | Compare context length statistics (mean, median, percentiles) across HF datasets | `python -m scripts.analysis.context_length_compare repo1 repo2 --filter 'col==val'` |
| `context_length_dist.py` | Plot context length distributions for a hardcoded list of SFT datasets | `python scripts/analysis/context_length_dist.py` |
| `solve_rate_by_context.py` | Solve/timeout/error rates binned by context length, with 3-panel plot | `python -m scripts.analysis.solve_rate_by_context repo1 repo2 --bins 0,16384,32768 --plot out.png` |
| `episode_distribution.py` | Plot episode count and tokens-per-turn distributions from HF trace datasets | `python -m scripts.analysis.episode_distribution repo1 repo2 --output out.png` |
| `filter_latest_episodes.py` | Keep only the latest episode per task in a trace dataset | `python scripts/analysis/filter_latest_episodes.py repo_id --output-jsonl out.jsonl` |
| `summarize_conversations.py` | Compute conversation stats (tokens, turns, rewards) from a JSONL file | `python scripts/analysis/summarize_conversations.py data.jsonl` |

## Training Analysis

| Script | Description | Usage |
|---|---|---|
| `parse_skyrl_metrics.py` | Parse SkyRL training logs, extract metrics and vLLM stats, generate CSV + markdown report | `python scripts/analysis/parse_skyrl_metrics.py log_folder/ output_folder/` |
| `temporal_trace_analysis.py` | Bin trace rows by timestamp to track agent improvement over training | `python scripts/analysis/temporal_trace_analysis.py repo_id --bin-hours 1` |

## Evaluation Analysis

| Script | Description | Usage |
|---|---|---|
| `eval_runtime_stats.py` | Compute runtime quantiles from eval trace result.json files | `python scripts/analysis/eval_runtime_stats.py results_dir/` |
| `trace_runtime_report.py` | Aggregate eval runtime stats with correlations and PNG visualizations | `python scripts/analysis/trace_runtime_report.py --root results_dir/` |
| `failure_mode_analysis.py` | Use GPT-5 to classify failure modes in trace datasets | `python scripts/analysis/failure_mode_analysis.py repo_id --output report.md` |
| `update_hf_failure_modes.py` | Annotate HF dataset rows with GPT-5 failure-mode summaries | `python scripts/analysis/update_hf_failure_modes.py repo_id --push` |

## Debugging & Diagnostics

| Script | Description | Usage |
|---|---|---|
| `probe_model_thinking.py` | Probe a model with real environment prompts, test thinking behavior | `python -m scripts.analysis.probe_model_thinking --model model_id` |
| `submit_probe.sh` | SLURM wrapper for `probe_model_thinking.py` | `./scripts/analysis/submit_probe.sh --model model_id --partition gpu-h100` |
| `verify_sft_thinking.py` | Test how ReasoningTemplate handles thinking blocks in SFT data | `python scripts/analysis/verify_sft_thinking.py` |
| `analyze_malformed_traces.py` | Classify malformation types in RL checkpoint traces | `python scripts/analysis/analyze_malformed_traces.py` |
| `sample_early_traces.py` | Sample malformed traces binned by timestamp to show failure evolution | `python scripts/analysis/sample_early_traces.py` |

## Batch Workflows

| Script | Description | Usage |
|---|---|---|
| `batch_filter_and_summarize.py` | Run filter + summarize across subdirectories | `python scripts/analysis/batch_filter_and_summarize.py --root dir/ --out_dir out/` |
| `batch_filter_and_summarize.sh` | Shell wrapper for the same batch workflow | `./scripts/analysis/batch_filter_and_summarize.sh root_dir/ out_dir/` |

## Dependencies

Most scripts require:
- `datasets` (HuggingFace datasets library)
- `transformers` (for tokenizers, used by context length scripts)
- `numpy`, `matplotlib` (for statistics and plotting)

Optional:
- `tiktoken` (fallback token counting in `utils.py`)
- `openai` (for GPT-5 failure mode analysis)
