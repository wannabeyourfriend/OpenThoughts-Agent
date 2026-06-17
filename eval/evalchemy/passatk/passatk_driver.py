#!/usr/bin/env python3
"""
pass@k driver for the Delphi #6279 pass@k grid (MATH500 / AIME24 / gsm8k).

ONE (model, task) full pass@k eval per invocation. Mechanism (EVAL_CONVENTION §pass@k):
  * Generate-once-at-max-k-then-sub-sample: ONE vLLM engine, ONE batched llm.generate() call
    with SamplingParams(n=N, temperature=0.7, top_p=1.0, seed=42). vLLM emits N independent
    samples per prompt (output.outputs[0..N-1]) — this reads ALL N (the lm_eval vllm path only
    surfaces outputs[0], which is exactly why we use a standalone driver here).
  * Each of the N samples is graded with the SAME grader the convention pins:
      - MATH500 / AIME24 : boxed-answer grader from lm_eval.tasks.hendrycks_math.utils
        (last_boxed_only_string -> remove_boxed -> is_equiv) — byte-identical to the grader
        the evalchemy MATH500/AIME24 chat_benchmarks use (eval_instruct.py imports the same).
      - gsm8k : the registered gsm8k.yaml grader, replicated faithfully here:
        strict-match regex `#### (-?[0-9.,]+)` on the model output, flexible-extract = last
        number `(-?[$0-9.,]{2,})|(-?[0-9]+)`; both compared to the gold final number after the
        yaml's `regexes_to_ignore` normalization (commas, $, leading "#### ", trailing ".",
        case-insensitive). NOTE: like the convention's gsm8k path, this is ZERO-shot here
        (the convention's lm_eval gsm8k run is 5-shot; pass@k diversity needs sampling and the
        boxed math tasks are 0-shot, so we run gsm8k 0-shot for within-grid consistency and flag it).
  * Prompts + chat-template handling match the per-type protocol:
      - MATH500/AIME24 use the chat_benchmark PROMPT ("Problem: {problem}\nMark your solution
        with \\boxed\nAnswer:"); gsm8k uses "Question: {q}\nAnswer:".
      - apply_chat_template per --apply-chat-template (Qwen3 + post-SFT delphi: ON; delphi BASE: OFF).
  * pass@k computed with the UNBIASED estimator (numerically-stable product form):
        pass@k = mean_problems[ 1 - C(n-c, k)/C(n, k) ]
    for k in {1,8,32,128}, all sub-sampled from the SAME n=128 generations.

Writes <out>/passatk_results.json with per-problem correct counts + the pass@k table, and
<out>/samples.jsonl (per-problem: gold, the N extracted answers, correctness mask) for audit.

Usage:
  python passatk_driver.py --task MATH500 --model-repo <repo> --out <dir> \
     --n 128 --tp 4 --max-model-len 32768 --max-gen-toks 30720 [--apply-chat-template] \
     [--temperature 0.7] [--top-p 1.0] [--seed 42] [--limit N]
"""
import argparse, json, os, re, sys
from typing import List

import numpy as np


# ---------------- unbiased pass@k estimator ----------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator 1 - C(n-c,k)/C(n,k), numerically-stable product form."""
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # product_{i=n-c+1..n} (1 - k/i)  == 1 - C(n-c,k)/C(n,k)
    out = 1.0
    for i in range(n - c + 1, n + 1):
        out *= (1.0 - k / i)
    return 1.0 - out


# ---------------- gsm8k grader (faithful to gsm8k.yaml) ----------------
_GSM8K_IGNORE = [r",", r"\$", r"(?s).*#### ", r"\.$"]
_STRICT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def _normalize(s: str) -> str:
    s = s.strip()
    for pat in _GSM8K_IGNORE:
        s = re.sub(pat, "", s)
    return s.strip().lower()


def gsm8k_gold(answer_field: str) -> str:
    # gold answer field is "<reasoning>\n#### <number>"
    return _normalize(answer_field)


def gsm8k_strict(text: str) -> str:
    m = _STRICT_RE.search(text)
    return _normalize(m.group(1)) if m else "[invalid]"


def gsm8k_flex(text: str) -> str:
    ms = _FLEX_RE.findall(text)
    if not ms:
        return "[invalid]"
    last = ms[-1]
    val = last[0] or last[1]
    return _normalize(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["MATH500", "AIME24", "gsm8k"])
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-gen-toks", type=int, default=30720)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply-chat-template", action="store_true")
    # vLLM engine knobs (OOM headroom for the n=128 sampler warmup; defaults preserve prior behavior).
    ap.add_argument("--gpu-mem-util", type=float, default=0.9,
                    help="vLLM gpu_memory_utilization (lower => more headroom for the n=128 sampler warmup)")
    ap.add_argument("--max-num-seqs", type=int, default=0,
                    help="vLLM max_num_seqs (0 => leave vLLM default; lower => smaller sampler-warmup footprint)")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="vLLM enforce_eager=True (frees cudagraph-capture memory; robust for n=128 warmup)")
    ap.add_argument("--evalchemy-root", default="/leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #problems")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    done_marker = os.path.join(args.out, "passatk_results.json")
    if os.path.exists(done_marker):
        print(f"already done: {done_marker} exists, skipping", flush=True)
        return

    sys.path.insert(0, args.evalchemy_root)
    from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

    def boxed_extract(text: str) -> str:
        try:
            return remove_boxed(last_boxed_only_string(text))
        except Exception:
            return ""

    # ---------------- load problems ----------------
    ER = args.evalchemy_root
    if args.task == "MATH500":
        path = os.path.join(ER, "eval/chat_benchmarks/MATH500/data/math500.jsonl")
        with open(path) as f:
            problems = [json.loads(x) for x in f]
        prompt_tpl = "Problem: {problem}\nMark your solution with \\boxed\nAnswer:"
        get_prompt = lambda e: prompt_tpl.format(problem=e["problem"])
        gold_of = lambda e: str(e["answer"])
        grade = lambda gold, text: is_equiv(gold, boxed_extract(text))
    elif args.task == "AIME24":
        path = os.path.join(ER, "eval/chat_benchmarks/AIME24/data/aime24.json")
        with open(path) as f:
            problems = [json.loads(x) for x in f]
        prompt_tpl = "Problem: {problem}\nMark your solution with \\boxed\nAnswer:"
        get_prompt = lambda e: prompt_tpl.format(problem=e["problem"])
        gold_of = lambda e: str(e["expected_answer"])
        grade = lambda gold, text: is_equiv(gold, boxed_extract(text))
    else:  # gsm8k
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        problems = [{"question": r["question"], "answer": r["answer"]} for r in ds]
        get_prompt = lambda e: "Question: {q}\nAnswer:".format(q=e["question"])
        gold_of = lambda e: gsm8k_gold(e["answer"])

        def grade(gold, text):
            return gsm8k_strict(text) == gold
        flex_grade = lambda gold, text: gsm8k_flex(text) == gold

    if args.limit:
        problems = problems[: args.limit]
    n_problems = len(problems)
    print(f"[{args.task}] {n_problems} problems, n={args.n} samples each, repo={args.model_repo}", flush=True)

    # ---------------- vLLM engine ----------------
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_repo)
    prompts = []
    for e in problems:
        msg = get_prompt(e)
        if args.apply_chat_template:
            prompts.append(tok.apply_chat_template([{"role": "user", "content": msg}],
                                                   tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(msg)

    llm_kwargs = dict(model=args.model_repo, tensor_parallel_size=args.tp, dtype="bfloat16",
                      max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_util,
                      seed=args.seed)
    if args.max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    print(f"[vLLM] engine kwargs: gpu_memory_utilization={args.gpu_mem_util} "
          f"max_num_seqs={args.max_num_seqs or 'default'} enforce_eager={args.enforce_eager}", flush=True)
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_gen_toks, seed=args.seed)
    outs = llm.generate(prompts, sp)

    # ---------------- grade all N samples per problem ----------------
    counts = []          # correct count c per problem (strict for gsm8k)
    flex_counts = []
    sample_rows = []
    for e, o in zip(problems, outs):
        gold = gold_of(e)
        texts = [c.text for c in o.outputs]
        correct = [bool(grade(gold, t)) for t in texts]
        c = int(sum(correct))
        counts.append(c)
        row = {"gold": gold, "n": len(texts), "c": c,
               "answers": [boxed_extract(t) if args.task != "gsm8k" else gsm8k_strict(t) for t in texts]}
        if args.task == "gsm8k":
            fc = int(sum(flex_grade(gold, t) for t in texts))
            flex_counts.append(fc)
            row["c_flex"] = fc
        sample_rows.append(row)

    n = args.n
    ks = [1, 8, 32, 128]
    res = {"task": args.task, "model_repo": args.model_repo, "n_samples": n,
           "n_problems": n_problems, "temperature": args.temperature, "top_p": args.top_p,
           "apply_chat_template": args.apply_chat_template,
           "max_model_len": args.max_model_len, "max_gen_toks": args.max_gen_toks}
    for k in ks:
        if k > n:
            continue
        res[f"pass@{k}"] = float(np.mean([pass_at_k(n, c, k) for c in counts]))
    if args.task == "gsm8k":
        for k in ks:
            if k > n:
                continue
            res[f"pass@{k}_flex"] = float(np.mean([pass_at_k(n, fc, k) for fc in flex_counts]))

    with open(os.path.join(args.out, "samples.jsonl"), "w") as f:
        for r in sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(done_marker, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
