"""Upload the 14 delivery-fixed parquets to laion as NEW versions.

The fix: instruction headers now teach the terminus-2 terminal agent to WRITE
its answer to the verifier's file via a shell heredoc (was ~20-100% "answer
file missing -> reward 0"). Task-spec-only change; verifiers/grading unchanged
except single-owner read-path fallbacks (swe, injection).
"""
from __future__ import annotations
import os, sys, time
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

ROOT = "/Users/benjaminfeuer/Documents/task_repos"
ORG = "laion"

# local stem -> (new repo name, source repo, grading one-liner)
PLAN = {
    "agentic_function_calling_pivot": ("nemotron-gym-agentic-function-calling-pivot-v2",
        "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        "Single-step: tool-call match (function_call) / LLM judge (message)."),
    "agentic_conversational_tool_use_pivot": ("nemotron-gym-agentic-conversational-tool-use-pivot-v2",
        "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
        "Single-step: tool-call match (function_call) / LLM judge (message)."),
    "agentic_swe_pivot": ("nemotron-gym-agentic-swe-pivot-v2",
        "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
        "Single-step SWE tool-call match (case-sensitive, whitespace-normalized)."),
    "agentic_indirect_prompt_injection": ("nemotron-gym-agentic-indirect-prompt-injection-v2",
        "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        "Single-step injection-resist (reward 1 = did NOT emit injected call)."),
    "arc_agi_transductive": ("nemotron-gym-arc-agi-transductive-v2",
        "nvidia/Nemotron-RL-ARC-AGI-v1 [transductive]",
        "Exact output-grid match (canonical one-row-per-line space-separated format)."),
    "structured_outputs_v3": ("nemotron-gym-structured-outputs-v4",
        "nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        "JSON/YAML/TOML schema validation; XML/CSV structural."),
    "inverse_ifeval": ("nemotron-gym-inverse-ifeval-v2",
        "nvidia/Nemotron-RL-InverseIFEval-v1",
        "Multi-criterion LLM judge (all YES). Needs OPENAI_API_KEY at trial."),
    "sysbench": ("nemotron-gym-sysbench-v2",
        "nvidia/Nemotron-RL-SysBench-v1",
        "Hybrid: deterministic IFEval constraints AND LLM judge."),
    "cfbench": ("nemotron-gym-cfbench-v2",
        "nvidia/Nemotron-RL-CFBench-v1",
        "Hybrid: deterministic IFEval constraints AND LLM judge."),
    "multiturnchat": ("nemotron-gym-instruction-following-multiturnchat-v2",
        "nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
        "Multi-turn rubric LLM judge (all criteria). Needs OPENAI_API_KEY at trial."),
    "multichallenge_advanced": ("nemotron-gym-multichallenge-advanced-v2",
        "nvidia/Nemotron-RL-Multichallenge-v1 [advanced]",
        "Rubric LLM judge (all criteria). Needs OPENAI_API_KEY at trial."),
    "multichallenge_vanilla": ("nemotron-gym-multichallenge-vanilla-v2",
        "nvidia/Nemotron-RL-Multichallenge-v1 [vanilla]",
        "Rubric LLM judge (all criteria). Needs OPENAI_API_KEY at trial."),
    "math_v3": ("nemotron-gym-math-v4",
        "nvidia/Nemotron-RL-Math-v2 (+ hydrated DAPO-Math-17k / Skywork-OR1 pointers)",
        "sympy/latex \\boxed{} equivalence vs gold."),
    "qa_abstention": ("nemotron-gym-qa-abstention-v2",
        "nvidia/Nemotron-RL-QA-Abstention-v1",
        "Normalized \\boxed{} text match vs gold."),
}


def readme(repo, src, grading, n):
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
(part of [nvidia/Nemotron-Post-Training-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3)).

Columns `path` (str) + `task_binary` (gzip tar). Converted with the
OpenThoughts-Agent `data.nemotron_gym` framework.

**Grading:** {grading}

## What changed vs the prior version

This version fixes the **answer-delivery contract** for terminal agents. The
prior version told the agent *what* to produce but not *how* to submit it; a
1-turn `terminus-2` agent emitted its answer as a chat reply instead of writing
the graded file, so most trials scored 0 with "answer file missing". The
instruction now explicitly instructs writing to the grader's file path via a
shell heredoc (and verifying it). Grading logic is otherwise unchanged.
"""


def main():
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    only = set(sys.argv[1:])
    results = []
    for stem, (repo, src, grading) in PLAN.items():
        if only and stem not in only:
            continue
        p = f"{ROOT}/{stem}.parquet"
        if not os.path.exists(p):
            results.append((repo, "MISSING")); print(f"SKIP {stem}: missing"); continue
        n = pq.read_metadata(p).num_rows
        repo_id = f"{ORG}/{repo}"
        print(f"\n=== {repo_id} ({n:,} tasks, {os.path.getsize(p)/1e6:.0f}MB) ===")
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True, token=token)
            api.upload_file(path_or_fileobj=p, path_in_repo="tasks.parquet",
                            repo_id=repo_id, repo_type="dataset",
                            commit_message=f"Upload {n} Harbor tasks (delivery-contract fix)")
            api.upload_file(path_or_fileobj=readme(repo, src, grading, n).encode(),
                            path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
                            commit_message="Add README")
            print(f"  OK -> https://huggingface.co/datasets/{repo_id}")
            results.append((repo_id, "OK"))
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
            results.append((repo_id, f"FAIL:{type(e).__name__}"))
        time.sleep(1)
    print("\n==== SUMMARY ====")
    for r, s in results: print(f"  {s:14s} {r}")
    ok = sum(1 for _, s in results if s == "OK")
    print(f"\n{ok}/{len(results)} uploaded")
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
