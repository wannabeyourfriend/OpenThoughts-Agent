# infra_error_count_plan — make infra-error count a queryable, audited field

- **Date:** 2026-06-29
- **Status:** DONE (Stage 0 ✅, Stage 1 ✅ — see status table)
- **Repo / path / branch:** OpenThoughts-Agent · `/Users/benjaminfeuer/Documents/OpenThoughts-Agent` · `feuer/infra-error-count` (off `penfever/working`)
- **Evidence log:** `/Users/benjaminfeuer/Documents/agent_logs/2026-06-29_infra_error_count_field.md`

## Goal (testable end state)

When an eval's record is registered/updated in Supabase `sandbox_jobs`, its `stats` jsonb blob carries
two NEW additive keys:

- `stats.n_infra_errors` (int) — sum of `exception_stats` counts (across all `stats.evals.<key>`) whose
  error type ∈ `INFRA_ERROR_TYPES`.
- `stats.infra_error_breakdown` (dict {error_type: count}) — the per-type infra-only breakdown.

So a plain Supabase query (`stats->>'n_infra_errors'`, `stats->'infra_error_breakdown'`) reads it directly,
without a consumer re-deriving the classification from `exception_stats`.

**GO gate (local):** against
`/Users/benjaminfeuer/Documents/experiments/traces/terminal_bench_2_a1_nemotron_bash_20260627_124545/result.json`
(`exception_stats` = `{VerificationNotCompletedError: 40, AgentTimeoutError: 145}`):
`n_infra_errors == 40` and `infra_error_breakdown == {"VerificationNotCompletedError": 40}`
(VNC ∈ infra; AgentTimeoutError ∉ infra → excluded). And **additive**: every other key of the assembled
`stats`/`metrics` is byte-identical to flag-off.

## Why / mechanism

The raw per-type breakdown already exists at `sandbox_jobs.stats.evals.<key>.exception_stats`
(`{error_type: [trial_ids]}`), but there is no field that says "N of these were infrastructure errors".
The classification set `INFRA_ERROR_TYPES` and the exact counting loop already live in the listener
(`eval/unified_eval_listener.py`) for the disk-based resume scanner — they are just not persisted on the row.
The fix computes the same quantity at the DB write-point and stores it.

## Design choice: jsonb key, NOT a new column

**Chosen: additive jsonb keys inside the existing `stats` blob.** `stats` is jsonb/free-form → no schema
migration, backward-compatible (old rows simply lack the key; queryable via `stats->>'n_infra_errors'`),
zero blast radius on shared prod Supabase + the leaderboard. There is already direct precedent: the
write-point (`_extract_job_metadata`) ALREADY mutates the top-level `stats` dict additively to inject
`stats["n_trials"]`. We follow that exact pattern.

**Rejected: `ALTER TABLE sandbox_jobs ADD COLUMN`.** A real top-level column means DDL on the SHARED prod
DB. Not warranted here and not done autonomously. **No DDL is being recommended** — the jsonb approach fully
satisfies "queryable by a plain Supabase query".

## Write-point (the narrowest ot-agent-side location)

`_extract_job_metadata()` in `database/unified_db/utils.py` (~L2706). It reads `result["stats"]`, additively
injects `stats["n_trials"]` (~L2786–2795), and returns it as `job_metadata["stats"]`, which feeds BOTH the
UPDATE payload (`update_data["stats"]`, ~L3870) and the create path (`register_sandbox_job(stats=...)`).
This is the single chokepoint that lands `stats` on the persisted `sandbox_jobs` row. Harbor (the separate
repo, source of `exception_stats`) is NOT touched — confirmed unnecessary.

## Single source of truth for INFRA_ERROR_TYPES

The set must NOT be duplicated. The listener already imports from `database.unified_db.*` (e.g.
`from database.unified_db.utils import get_supabase_client`), so the dependency direction listener → unified_db
already exists. We factor the set + a small pure helper into a new leaf module
`database/unified_db/infra_errors.py` (dependency-free) and have BOTH the listener and `_extract_job_metadata`
import it. The listener's inline `INFRA_ERROR_TYPES = {...}` literal is replaced by the import; its counting
loop is unchanged (same semantics).

## Stage map

| Stage | Title | What | Layer | Cost | Gate |
|---|---|---|---|---|---|
| 0 | Shared infra-error module + listener rewire | New `database/unified_db/infra_errors.py` (`INFRA_ERROR_TYPES` + `compute_infra_error_stats(stats)->(n, breakdown)`); listener imports the set instead of defining it inline | CPU | CPU | listener `INFRA_ERROR_TYPES` identical set; `_parse_job_dir` infra_errors count unchanged on the local run; lint clean | 
| 1 | Persist `n_infra_errors` + `infra_error_breakdown` | In `_extract_job_metadata`, after the existing `stats["n_trials"]` injection, additively set `stats["n_infra_errors"]` + `stats["infra_error_breakdown"]` via the shared helper | CPU | CPU | **GO gate**: on the local result.json `n_infra_errors==40` & `breakdown=={"VerificationNotCompletedError":40}`; rest of `stats`/`metrics` byte-identical to flag-off | 

Critical path: Stage 0 → Stage 1.

## Global invariants

- **Additive / backward-compatible:** the only change to the persisted record is two NEW jsonb keys. Existing
  reads (leaderboard `stats.n_trials`, accuracy/metrics, `exception_stats`) are unaffected and byte-identical.
- **No metric/denominator change:** the accuracy numerator/denominator and every existing metric stay exactly
  as-is. This is a pure audit count, not a score input.
- **Single source of truth:** `INFRA_ERROR_TYPES` defined once (the shared module); no duplicate literal.
- **Minimal diff:** no API/config churn; harbor repo untouched.

## Borrow map (anchors drift — reconfirmed 2026-06-29 on `feuer/infra-error-count`)

- `eval/unified_eval_listener.py`
  - `INFRA_ERROR_TYPES = {...}` literal — **L957–968** (was ~1043 in prompt; drifts).
  - inline counting loop (`for exc_type, ids in eval_data.get("exception_stats", {})...`) — **L1053–1063**.
  - already-present import dependency on unified_db — L442.
- `database/unified_db/utils.py`
  - `def _extract_job_metadata(...)` — **L2706**; `stats = result.get("stats")` + `stats["n_trials"]` injection — **L2785–2795**; returns `"stats": stats` — L2808.
  - consumers of that stats: UPDATE `update_data["stats"]` — L3870; create path `register_sandbox_job(stats=...)`.
- The `sft/.../database/unified_db/utils.py` vendored copy has NO `_extract_job_metadata` (grep count 0) — it is NOT the eval upload path; left untouched.

## Leaderboard touchpoint (NOTE, do not modify)

The leaderboard reads `stats.n_trials` (`server/storage.ts`, per the comment at L2776). It does NOT read
`n_infra_errors`. A future enhancement could surface infra-error count in the leaderboard from this new field;
that is OUT OF SCOPE here and left untouched.

## Status

- Stage 0 ✅ DONE — commit `8887af4c`; gate: listener set identical to prior literal; all 3 files compile clean.
- Stage 1 ✅ DONE — commit `9dd89c0c`; GO gate: `n_infra_errors==40`, `infra_error_breakdown=={"VerificationNotCompletedError":40}` on the local run; additive (all other stats keys byte-identical).

Local commits only; NOT pushed (supervisor owns pushes). No cluster touched; no Supabase write/DDL.
