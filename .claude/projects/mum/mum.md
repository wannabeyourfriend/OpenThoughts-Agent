# mum (marin-context corpus) — facts & gotchas

Offline-queryable mirror of all Marin activity (GitHub issues/PRs/comments, Discord, W&B run metadata + final
numbers, weekly summaries), searchable by keyword + meaning, every hit citable by URL. Learned 2026-06-29
pulling Delphi midtraining provenance. The skill is `marin-context` at
`/Users/benjaminfeuer/Documents/mumwelt/mumwelt/skills/marin-context/SKILL.md`; source repo `~/Documents/mumwelt`.

---

## Invocation (do NOT assume `mum` / `uv` are on PATH)

- The `mum` binary lives in the **otagent conda env**: **`/Users/benjaminfeuer/miniconda3/envs/otagent/bin/mum`**.
  The default (non-login) Bash-tool shell has neither `mum` nor `uv` on PATH — call the absolute path.
- Source is `~/Documents/mumwelt` (console_script `mumwelt.cli:main`); no `.venv` there.

## Commands

- `mum status` — freshness: corpus chunk count + age, summaries weeks, **server build age + size**.
- `mum refresh` — pull latest corpus + summaries (first pull ~150 MB).
- `mum search "<query>" [--source github,discord,wandb,narrative] [--kind run,issue,pr,comment,message,section]
  [--since YYYY-MM-DD] [-k N] [--json]` — fused keyword+semantic; search by concept OR identifier (run name,
  `#1234`, login). The result **snippets carry indexed fields** (for W&B runs: config tokens like `batch_size`,
  `num_train_steps`, `adam_lr`, `midtraining_mix`, `cooldown_ratio`, state, author).
- `mum run <project>/<run>` or `mum run <wandb URL>` — run metadata + **final summary numbers** + config.
- `mum show <url-or-ref>` — expand a hit (discord → channel window; github → issue+comments; **wandb → a fuller
  one-line config dump** than the search snippet).
- `mum summaries list | show latest | links YYYY-MM-DD`.

## Freshness policy (from the skill)
≤7 days old → use as-is. >7 days → refresh first (tell the user). STALE (>24h) **and** time-sensitive question →
ask before refreshing, then honor the answer. If the user says "don't repull", use what's on disk.

## The big gotcha: `mum run` resolves configs by RUN NAME and 404s for executor-launched / crashed runs

- `mum run <proj>/<run>` hits the server's `/wandb/<proj>/<run>/config` endpoint **keyed by the display name**.
  It returns config cleanly for **script-launched** runs whose W&B name == the run id (e.g. the 21 small Delphi
  midtraining runs). It **404s (raises `JSONDecodeError: Expecting value`)** for:
  - **marin-executor-launched runs**, whose `#6279`-style ids are step labels that appear only embedded inside
    *derived* runs (eval / `ppl-gap-score`), not as a standalone training run name; and
  - **crashed runs** (no finalized config export).
- **Fallback when `mum run` 404s:** the config FIELDS are still in the **search index** — use
  `mum search "<run-id>" --source wandb` and `mum show <run-url>` to read `batch_size`, `num_train_steps`,
  `adam_lr`, `peak_lr`, `midtraining_mix`, `cooldown_ratio`, etc. (This recovered the large Delphi runs' recipe
  shape.) For the *authoritative* resolved config of an executor run, go to GCS `.executor_info` instead (see
  `marin-executor`).
- The mirror **server can be rebuilt mid-session** (`mum status` → "server built Nh ago"); endpoint behavior can
  shift across rebuilds (a `mum run` that worked earlier may 404 after a rebuild). W&B **per-step history is not
  mirrored** — only final summary numbers.

## Naming note (W&B run ids vs cell labels)
A Marin run id like `delphi-1e22-p33m67-32p07b-lr0.67-54770ae7` may be a *selected-cell label*, not a literal
W&B training-run name — the actual training run can be named differently (e.g. `true-midtrain-1e22-p33m67-…`),
and the label only lives on derived eval/ppl runs. When `mum run <id>` 404s, `mum search` the family to find the
real run name before concluding the data is absent.
