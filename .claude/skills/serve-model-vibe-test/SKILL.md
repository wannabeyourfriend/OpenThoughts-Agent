---
name: serve-model-vibe-test
description: Stand up a PUBLIC, shareable inference endpoint for an HF/gs model on an Iris TPU so people on the internet can vibe-test it in a browser or via an OpenAI-compatible API. Use when asked to "serve a model for testing", "throw up a public endpoint", "let people play with model X", or to demo a checkpoint. Combines marin-serve (marin#6556) + a Pinggy tunnel from our endpoint bank.
---

# serve-model-vibe-test

Spin up a **public** vibe-test endpoint for any HuggingFace repo or `gs://` checkpoint:
`marin-serve` boots vLLM on a single-host Iris TPU slice behind the auth-gated Iris
controller proxy (only cluster-account holders can reach it); this skill fronts that
local proxied port with a **Pinggy** tunnel from our bank, giving a shareable
`https://<id>.a.pinggy.link/proxy/serve.<ep>/` URL with a browser dashboard + an
OpenAI-compatible API that anyone on the internet can use.

The mechanism is `scripts/inference/serve_public.py` (this repo), which wraps
`marin-serve` + `hpc.pinggy_utils.PinggyTunnel` and the pairs in
`notes/ot-agent/pinggy_bank.md`.

## ⚠️ Security — read first
The public URL is **UNAUTHENTICATED**: it inherits the no-auth localhost view of the
controller proxy, so your tunnel is the open front door. Treat it as **throwaway vibe
testing only** — never point it at anything sensitive. Mitigations: keep
`--timeout-hours` short (marin-serve self-stops the slice), rotate to an unused Pinggy
pair, and tear it down when done.

## Prerequisites
- `marin-serve` available (marin#6556). The console script is NOT registered by the marin root
  `uv sync` (workspace-member scripts don't land in `.venv/bin`), so `serve_public.py` invokes it
  via the module form by default: `~/Documents/marin/.venv/bin/python -m marin.inference.quick_serve_cli`.
  Requires the marin checkout to include #6556 (`git pull origin main` + `uv sync` if `import marin.inference.quick_serve_cli` fails). Override the invocation with `--marin-serve-bin` / `MARIN_SERVE_BIN`.
- **Run from the marin repo root (`~/Documents/marin`), not from this repo.** marin-serve bundles
  `Path.cwd()` as the worker workspace (`quick_serve_cli.py:227`) and the worker runs
  `uv sync --all-packages --extra tpu --extra vllm` against it. Only the marin workspace defines the
  `tpu`/`vllm` extras (on `marin-core`); any other CWD → `Extra 'tpu' is not defined in any project's
  'optional-dependencies' table` and the job dies before vLLM starts. `serve_public.py` resolves its
  own repo root via `parents[2]` for the `hpc` import, so invoke it by **absolute path** from the marin CWD.
- Iris controller access (the laptop's step-ca/GCP creds — same as `iris` CLI).
- `secrets.env` sourced (Pinggy uses your `~/.ssh` identity; SSH to `pro.pinggy.io:443`).
- Pinggy bank at `~/Documents/notes/ot-agent/pinggy_bank.md` (override `--pinggy-bank` / `PINGGY_BANK`).
- **Single-host TPU slices only** (`v6e-8`, `v6e-4`, `v5litepod-8`, …); multi-host is rejected.

## Pick an unused Pinggy pair
The bank has 10 pairs. Before launching, pick one that isn't already serving — a quick
probe (a free pair returns a Pinggy "tunnel not found"/503 or connection error; an
in-use one returns your model):
```bash
for i in 1 2 3; do u=$(sed -n "/## Pair $i$/,/pinggy.link/p" ~/Documents/notes/ot-agent/pinggy_bank.md | grep pinggy.link); \
  echo "pair $i: $u -> $(curl -s -o /dev/null -w '%{http_code}' --max-time 6 https://$u/ || echo down)"; done
```
Use a pair that shows `down`/`502`/`503` (free). Pass it as `--pair N`.

## Launch (the one-liner)
```bash
cd ~/Documents/marin                               # MUST be the marin repo — see Prerequisites
source ~/Documents/secrets.env
python -u ~/Documents/OpenThoughts-Agent/scripts/inference/serve_public.py <MODEL> --tpu <SLICE> [--region <R>] \
    [--chat-template <FILE-or-URL>] --pair <N> --timeout-hours 6
```
It launches marin-serve, waits for the `READY — dashboard: http://127.0.0.1:<port>/proxy/serve.<ep>/`
line (cold compile can take up to ~40 min on first boot; the XLA cache speeds repeat boots),
opens the Pinggy tunnel, and prints:
```
  dashboard : https://<id>.a.pinggy.link/proxy/serve.<ep>/
  OpenAI    : https://<id>.a.pinggy.link/proxy/serve.<ep>/v1
```
Keep the process running — it holds both the controller tunnel and the Pinggy tunnel.
When running it unattended (e.g. as an agent), launch it **detached** so it survives — use a
`tmux` session (do NOT use `nohup setsid …`: `setsid` does not exist on macOS, so the launch dies
instantly with `nohup: setsid: No such file or directory`):
```bash
tmux new-session -d -s serve-public \
  "source ~/Documents/secrets.env && cd ~/Documents/marin && \
   python -u ~/Documents/OpenThoughts-Agent/scripts/inference/serve_public.py <MODEL> ... 2>&1 | tee /tmp/serve_public.log"
# then poll /tmp/serve_public.log (or `tmux attach -t serve-public`) for the "OpenAI :" line
```

## Worked example — the Delphi 9.7B SFT canary (marin#6545)
This is the reference model: `laion/delphi-1e22-p33m67-32p07b-lr0_67-54770ae7-wc386k_lr1e5-sft`.
It needs its **own chat template** (the repo ships a plain Llama-3 one); use `delphi_v0.jinja2`.
marin-serve auto-derives the 4k context (malformed RoPE) and TP=2 (30 heads on a 4-chip slice).
```bash
# run from ~/Documents/marin (the bundled workspace); script is invoked by absolute path
cd ~/Documents/marin && source ~/Documents/secrets.env
python -u ~/Documents/OpenThoughts-Agent/scripts/inference/serve_public.py \
    laion/delphi-1e22-p33m67-32p07b-lr0_67-54770ae7-wc386k_lr1e5-sft \
    --tpu v6e-4 --region europe-west4 \
    --chat-template https://raw.githubusercontent.com/open-thoughts/OpenThoughts-Agent/ed4d6f483151f14d6d78cf732f04cd3c8ff5c606/chat_templates/delphi_v0.jinja2 \
    --pair 1 --timeout-hours 6
```
Verify once the public URL prints:
```bash
BASE=https://<id>.a.pinggy.link/proxy/serve.<ep>
# from the laptop you must force the real Pinggy edge IP (ISP DNS poisons *.a.pinggy.link); see Gotchas
curl "$BASE/v1/models"
curl "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"laion/delphi-1e22-p33m67-32p07b-lr0_67-54770ae7-wc386k_lr1e5-sft",
       "messages":[{"role":"user","content":"Give me a fun fact about otters."}]}'
```
(Base/midtrained checkpoints with no chat template auto-default the dashboard to completion mode — omit `--chat-template`.)

## Teardown
- Foreground: `Ctrl-C` (tears down the Pinggy tunnel and the marin-serve job it launched).
- Detached/other host: `iris job stop <job> --cluster marin` (the job name is in the log /
  `iris query`), then kill the `serve_public.py` PID. The slice also self-stops at `--timeout-hours`.

## Gotchas
- **Run from the marin repo, or the build fails before vLLM starts.** marin-serve bundles `Path.cwd()`
  as the worker workspace (`quick_serve_cli.py:227`); the worker runs `uv sync --all-packages --extra tpu --extra vllm`
  against it. Only the marin workspace defines those extras (on `marin-core`). Any other CWD (e.g. this
  OT-Agent repo) → job reaches state 5 with `Extra 'tpu' is not defined in any project's 'optional-dependencies' table`
  during `syncing deps` — no vLLM boot, no port, nothing for Pinggy to front. Symptom in `serve_public.py`:
  `marin-serve exited (code 1) before READY`. Fix: `cd ~/Documents/marin` and invoke `serve_public.py`
  by absolute path. (The bundle from marin is ~13 MB; from the wrong repo it differs — check `iris job logs`.)
- **Dead tunnel reported as LIVE (state `T`)**: the Pinggy ssh process can get
  job-control-stopped (`ps` STAT `T` — SIGTTIN/SIGTTOU from touching a tty when
  backgrounded in tmux). While stopped it forwards **nothing**, so the public URL
  resets every request even though the job and `serve_public.py` look healthy and
  the local `127.0.0.1:<port>` dashboard returns 200. Symptom: TLS handshake to the
  Pinggy edge succeeds but the HTTP request gets `Connection reset by peer`, and
  `/tmp/pinggy_<id>.log` is **0 bytes**. Manual fix: `kill -CONT <ssh_pid> <loop_pid>`
  then re-probe. **Now prevented**: `pinggy_utils.PinggyTunnel.start()` launches ssh
  with `stdin=/dev/null`, `ssh -n`, and `setsid` (no controlling tty → can't be
  tty-stopped); `_wait_for_healthy` auto-`SIGCONT`s a stopped process and verifies
  forwarding (log-banner scan, or a real GET when `health_check_url` is set) rather
  than only checking the process exists; and `serve_public.py` runs a DNS-aware
  `/v1/models` probe (`dig @1.1.1.1` + `curl --resolve`) at startup, printing
  `LIVE ✓` only on HTTP 200.
- **Cold compile**: first boot of a model can take tens of minutes; `--ready-timeout` (default 2700s) bounds the wait.
- **Don't use marin-serve `--no-wait`** for the public path — the local proxied port only exists while the launching process runs; `--no-wait` returns immediately and there's nothing to tunnel.
- **Pinggy pair collisions**: two jobs on the same pair clobber each other — always pick a free pair.
- **Single-host concurrency**: one slice = limited throughput; fine for a few testers, not a viral launch (that's inference-broker territory).
- The public URL carries the `/proxy/serve.<ep>/` base path (it's the controller proxy path), so the OpenAI base is `https://<id>.a.pinggy.link/proxy/serve.<ep>/v1`, not `/v1`.
- **Local verification blocked by ISP DNS (this laptop):** Altice/Optimum DNS poisons `*.a.pinggy.link` → a dead proxy IP (`167.206.37.145`) that times out, so the `curl $BASE/v1/models` verify step **hangs from this Mac** while the endpoint is actually live for everyone else. The serve path itself is unaffected (`serve_public.py` dials `pro.pinggy.io:443`, correct IP). To verify locally, force the real Pinggy edge: `EDGE=$(dig @1.1.1.1 +short <id>.a.pinggy.link | tail -1)` then `curl --resolve "<id>.a.pinggy.link:443:$EDGE" "$BASE/v1/models"`, or point system DNS at `1.1.1.1`/`8.8.8.8`. A free pair = the edge resets the TLS handshake ("Connection reset by peer"); in-use = an OpenAI JSON response. Full write-up: `.claude/projects/pinggy/pinggy.md`.
