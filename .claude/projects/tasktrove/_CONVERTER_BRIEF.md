# Shared brief: Nemotron-RL → Harbor converter subagents

You are converting ONE (or a small set of) `nvidia/Nemotron-RL-*` dataset(s) into
Harbor task-binary parquet, reusing the existing framework at:

    /Users/benjaminfeuer/Documents/OpenThoughts-Agent/data/nemotron_gym/

## Environment (use exactly this)
- Python: `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` (call it `$PY`)
- Always `source /Users/benjaminfeuer/Documents/secrets.env` first (sets HF_TOKEN).
- Run the converter AS A MODULE from the repo root:
      cd /Users/benjaminfeuer/Documents/OpenThoughts-Agent
      $PY -m data.nemotron_gym.run --dataset <repo> --output <out.parquet> [--split S] [--limit N] [--smoke]
- Do NOT use `du`/`find` on large dirs.

## How the framework works (READ THESE FILES FIRST)
1. `data/nemotron_gym/adapter.py` — the `HarborTask` dataclass + `to_tarball()`,
   `render_dockerfile()`, `render_metadata()`, `task_id_for()`, `STANDARD_TEST_SH`,
   `DEFAULT_TASK_TOML`, `LLM_JUDGE_TASK_TOML`, `sanitize_text`. SECURITY INVARIANTS
   are enforced here — base image must be `python:3.11-slim-bookworm` (pinned),
   no untrusted strings in Dockerfile/bash/py, verifier inputs go in
   `verifier_data.json` (JSON), paths validated. Respect them; do not bypass.
2. `data/nemotron_gym/converters/__init__.py` — the `@register("<hf_repo>")`
   registry. `run.py` looks up the converter by exact dataset repo id.
3. `data/nemotron_gym/converters/_common.py` — `extract_prompt(row)` handles the
   `responses_create_params.input` / `input` / `messages` / `prompt` shapes.
   USE IT to get the prompt.
4. `data/nemotron_gym/run.py` — the driver. Emits parquet with columns
   `path` (str "<task_id>.tar.gz") and `task_binary` (gz tar bytes).
5. `data/nemotron_gym/verifiers/__init__.py` + the verifier modules — each
   verifier is a Python SOURCE STRING written to `tests/verifier.py`. It reads
   `/tests/verifier_data.json` + agent output (usually `/app/answer.txt`, or
   `/app/response.txt` for judge tasks), writes `0`/`1` (or a float for judges)
   to `/logs/verifier/reward.txt`.

## Existing converters to copy the pattern from
- Numeric/boxed math: `converters/math_boxed.py` (+ verifiers `math_boxed.py`,
  `numeric_compare.py`)
- JSON schema: `converters/structured_outputs.py` (+ `verifiers/json_schema.py`)
- Reasoning/exact: `converters/reasoning_gym.py`
- IFEval constraints: `converters/instruction_following.py` (+ `verifiers/ifeval_constraints.py`)
- LLM judge (needs OPENAI_API_KEY at trial time, uses `LLM_JUDGE_TASK_TOML`):
  `converters/safety.py`, `converters/identity_following.py`, `converters/adversarial.py`
  (+ `verifiers/llm_judge.py`, `verifiers/safety_judge.py`)
- String/substring/regex: `verifiers/substring_match.py`, `verifiers/regex_letter.py`,
  `verifiers/normalized_text.py`, `verifiers/tool_call_match.py`

## Your deliverables
1. A converter module `converters/<name>.py` with `@register("<exact nvidia repo>")`.
   - If a suitable verifier already exists, REUSE it. Only add a new verifier
     module under `verifiers/` (and wire it into `verifiers/__init__.py`) if none fits.
   - Prefer SELF-CONTAINED verifiers (no network/LLM) when the dataset carries a
     deterministic gold (string_match/regex/numeric/json-schema/grid). Use the
     LLM-judge pattern ONLY when grading is inherently subjective/equivalence.
2. Run the FULL conversion (no `--limit`) to:
       /Users/benjaminfeuer/Documents/task_repos/<slug>.parquet
   (slug = lowercase, hyphenless-where-existing-style; match the run_all.sh naming
    convention, e.g. `instruction_following_citation`, `arc_agi_transductive`).
   If a dataset has multiple configs/splits, emit one parquet per config with a
   suffixed slug, OR concatenate — state which you did and why.
3. QUALITY GATE (mandatory — this is how we avoid "fake pass" datasets):
   Write a short self-test (can be inline `$PY -c`) that, for 3–5 real rows,
   (a) extracts the gold answer the converter stored in verifier_data, simulates
   an agent writing THAT gold to the expected answer path, runs the embedded
   verifier logic, and confirms reward==1; and (b) feeds a deliberately WRONG
   answer and confirms reward==0. If gold-in → reward 1 fails, the converter is
   buggy — fix it before declaring done. (You can import the verifier source and
   exec it against a temp dir, or refactor the check into a pure function for the
   test. Do not weaken the verifier just to pass.)
4. Report back (your final message = structured summary):
   - converter module path + verifier used (reused vs new)
   - rows converted / skipped (+ top skip reasons from run.py output)
   - output parquet path + size + #configs
   - quality-gate result (gold→1, wrong→0): PASS/FAIL with the numbers
   - any concerns, schema surprises, or reasons a dataset is infeasible
   Keep it tight; raw file dumps are not needed.

## Notes
- `extract_prompt` may already cover your row shape; check before hand-rolling.
- Deterministic `task_id` via `task_id_for(prefix, payload)`; dedup is handled by run.py.
- Many rows carry a `verifier` dict (type string_match/regex/...) or an
  `expected_answer`/`expected_action` — prefer converting THAT into verifier_data
  rather than inventing grading.
- `used_in` / `agent_ref` / `pass_rate*` columns are provenance; pass useful bits
  into `metadata` via `render_metadata(extra=...)`, don't grade on them.
