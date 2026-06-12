"""Upload the 20 Nemotron-RL Harbor parquets to laion/ as tasks.parquet.

Each local parquet is already in the canonical Harbor task format
(path:str, task_binary:binary gz-tar) produced by data.nemotron_gym.run, which
matches scripts/harbor/tasks_parquet_converter output exactly. So we upload the
parquet directly as `tasks.parquet` at repo root (the layout
make_and_upload_task_dataset.py produces), plus a short provenance README.

Public repos (laion private quota is exhausted; HF uploads default public here).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

ROOT = Path("/Users/benjaminfeuer/Documents/task_repos")

# local parquet stem -> (laion repo name, nvidia source repo[/config], one-line grading)
PLAN = {
    "instruction_following_citation": (
        "nemotron-gym-instruction-following-citation",
        "nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
        "Deterministic substring/marker match (all required [ref:N] markers present)."),
    "instruction_following_freeform": (
        "nemotron-gym-instruction-following-freeform",
        "nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
        "Deterministic regex-count (>= verify_min_matches over MULTILINE regexes)."),
    "litmus_bench": (
        "nemotron-gym-litmus-bench",
        "nvidia/Nemotron-RL-litmus-bench-v0.1",
        "Numeric compare of molecular-property answer vs gold (exact integer)."),
    "math_v3": (
        "nemotron-gym-math-v3",
        "nvidia/Nemotron-RL-Math-v2 (+ hydrated DAPO-Math-17k / Skywork-OR1 pointers)",
        "sympy/latex \\boxed{} equivalence vs gold."),
    "arc_agi_transductive": (
        "nemotron-gym-arc-agi-transductive",
        "nvidia/Nemotron-RL-ARC-AGI-v1 [transductive]",
        "Exact output-grid match."),
    "arc_agi_python_inductive": (
        "nemotron-gym-arc-agi-python-inductive",
        "nvidia/Nemotron-RL-ARC-AGI-v1 [python_inductive]",
        "Sandboxed exec of agent transform() vs held test grids."),
    "structured_outputs_v3": (
        "nemotron-gym-structured-outputs-v3",
        "nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        "JSON/YAML/TOML schema validation; XML/CSV structural (well-formed + required keys)."),
    "qa_abstention": (
        "nemotron-gym-qa-abstention",
        "nvidia/Nemotron-RL-QA-Abstention-v1",
        "Normalized \\boxed{} text match vs gold (all-answerable)."),
    "reasoning_gym": (
        "nemotron-gym-reasoning-gym",
        "nvidia/Nemotron-RL-ReasoningGym-v1",
        "reasoning-gym upstream scorer; fallback normalized exact-match."),
    "science_so_openq": (
        "nemotron-gym-science-so-openq",
        "nvidia/Nemotron-RL-Science-v1 [so_openq]",
        "LLM-judge equivalence vs reference answer (needs OPENAI_API_KEY at trial)."),
    "inverse_ifeval": (
        "nemotron-gym-inverse-ifeval",
        "nvidia/Nemotron-RL-InverseIFEval-v1",
        "Multi-criterion LLM judge (all YES); needs OPENAI_API_KEY at trial."),
    "cfbench": (
        "nemotron-gym-cfbench",
        "nvidia/Nemotron-RL-CFBench-v1",
        "Hybrid: deterministic IFEval constraints AND LLM judge for subjective."),
    "sysbench": (
        "nemotron-gym-sysbench",
        "nvidia/Nemotron-RL-SysBench-v1",
        "Hybrid: deterministic IFEval constraints AND LLM judge for subjective."),
    "multiturnchat": (
        "nemotron-gym-instruction-following-multiturnchat",
        "nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
        "Multi-turn rubric LLM judge (all criteria); needs OPENAI_API_KEY at trial."),
    "multichallenge_advanced": (
        "nemotron-gym-multichallenge-advanced",
        "nvidia/Nemotron-RL-Multichallenge-v1 [advanced]",
        "Rubric LLM judge (all criteria); needs OPENAI_API_KEY at trial."),
    "multichallenge_vanilla": (
        "nemotron-gym-multichallenge-vanilla",
        "nvidia/Nemotron-RL-Multichallenge-v1 [vanilla]",
        "Rubric LLM judge (all criteria); needs OPENAI_API_KEY at trial."),
    "agentic_function_calling_pivot": (
        "nemotron-gym-agentic-function-calling-pivot",
        "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        "Single-step: tool-call match (function_call) / LLM judge (message)."),
    "agentic_conversational_tool_use_pivot": (
        "nemotron-gym-agentic-conversational-tool-use-pivot",
        "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
        "Single-step: tool-call match (function_call) / LLM judge (message)."),
    "agentic_swe_pivot": (
        "nemotron-gym-agentic-swe-pivot",
        "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
        "Single-step SWE tool-call match (case-sensitive, whitespace-normalized)."),
    "agentic_indirect_prompt_injection": (
        "nemotron-gym-agentic-indirect-prompt-injection",
        "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        "Single-step injection-resist proxy (reward 1 = did NOT emit injected call)."),
}

ORG = "laion"


def readme(repo: str, src: str, grading: str, n: int) -> str:
    return f"""---
license: apache-2.0
task_categories:
- text-generation
tags:
- agent
- harbor
- reinforcement-learning
- nemotron
---

# {ORG}/{repo}

Harbor task-binary dataset ({n:,} tasks) converted from **{src}**
(part of the [nvidia/Nemotron-Post-Training-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection).

Each row is a valid [Harbor](https://github.com/open-thoughts/OpenThoughts-Agent)
task binary: columns `path` (str) and `task_binary` (gzip tar). Converted with the
OpenThoughts-Agent `data.nemotron_gym` framework.

**Grading:** {grading}
"""


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    only = set(sys.argv[1:])  # optional: restrict to given stems
    results = []
    for stem, (repo, src, grading) in PLAN.items():
        if only and stem not in only:
            continue
        pq_path = ROOT / f"{stem}.parquet"
        if not pq_path.exists():
            print(f"SKIP {stem}: parquet missing")
            results.append((repo, "MISSING"))
            continue
        n = pq.read_metadata(pq_path).num_rows
        repo_id = f"{ORG}/{repo}"
        print(f"\n=== {repo_id}  ({n:,} tasks, {pq_path.stat().st_size/1e6:.1f} MB) ===")
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=False,
                        exist_ok=True, token=token)
            api.upload_file(path_or_fileobj=str(pq_path), path_in_repo="tasks.parquet",
                            repo_id=repo_id, repo_type="dataset",
                            commit_message=f"Upload {n} Harbor tasks (tasks.parquet)")
            api.upload_file(path_or_fileobj=readme(repo, src, grading, n).encode(),
                            path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
                            commit_message="Add provenance README")
            print(f"  OK -> https://huggingface.co/datasets/{repo_id}")
            results.append((repo_id, "OK"))
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
            results.append((repo_id, f"FAIL:{type(e).__name__}"))
        time.sleep(1)
    print("\n==== SUMMARY ====")
    for r, s in results:
        print(f"  {s:14s} {r}")
    ok = [r for r, s in results if s == "OK"]
    print(f"\n{len(ok)}/{len(results)} uploaded")
    return 0 if len(ok) == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
