## CINECA Leonardo Access

**SSH**: Uses ControlMaster multiplexing + step-ca certificate auth:
```bash
ssh Leonardo    # Complete 2FA once; socket persists 8h
```
> **Host keys ROTATE (benign — NOT an anomaly).** The round-robin login nodes rotate host keys, so a fresh
> connection (esp. `-o ControlPath=none` / `-o ControlMaster=no`) can hit `REMOTE HOST IDENTIFICATION HAS
> CHANGED` / a `known_hosts` mismatch. This is expected, not a compromise — just use the standard `ssh Leonardo`
> (ControlMaster socket), which works. Do NOT flag it as a failure or block on a `known_hosts` refresh.

**Pre-launch preamble** (run before launching any new job):
```bash
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh && \
conda activate otagent && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent && GIT_TERMINAL_PROMPT=0 git pull && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/harbor && GIT_TERMINAL_PROMPT=0 git pull && \
source hpc/dotenv/leonardo.env && source ~/secrets.env && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
```

**Key paths**:
- Code: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent`
- Harbor: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/harbor`
- MarinSkyRL: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/MarinSkyRL` (the old `/code/SkyRL` clone was removed 2026-06-08)
- Data/HF cache: `/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub`
- Experiments: `/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments`

**Correct upstream per codebase** (SoT = the Mac clones under `/Users/benjaminfeuer/Documents/`; aligned on Leonardo 2026-06-05). All on branch `penfever/working`:
- **OpenThoughts-Agent** → remote `origin` = `open-thoughts/OpenThoughts-Agent`.
- **harbor** → remote `marin` = `marin-community/harbor` (NOT laude-institute / `penfever/temp-override` / `otagent-latest` — those were stale on Leonardo).
- **MarinSkyRL** → `marin-community/MarinSkyRL` `penfever/working` (see `.claude/projects/marinskyrl/marinskyrl.md`; `penfever/SkyRL` is obsolete).
- Sync discipline: commit on the Mac → push → `git pull` on the cluster (Leonardo can't push). The preamble also needs `git submodule update --init --remote sft/llamafactory` for SFT runs.
- Leonardo is back in use as of 2026-06-05; cron covers Jupiter + Leonardo (Perlmutter dropped).

**Cluster details**: A100 64GB GPUs, 4/node, 3456 nodes, SLURM scheduler. No internet on compute nodes (use proxychains/SSH tunnel). User: `bfeuer00`. Account: `AIFAC_5C0_290`.

**Important**: Compilers come from conda (GCC 15.2, CUDA 13.2) — do NOT load system modules (`module load gcc cuda`), they are too old.

**Max wall time**: 24 hours (`--time 23:59:00`). The boost_usr_prod partition has a 1-day limit.

## Leonardo HF Upload — Use sbatch, NOT the Login Node

Leonardo's login nodes SIGKILL any long-running user process after ~100 seconds,
regardless of how it's detached. We've verified this kills:
- `nohup hf upload ... &` / `nohup huggingface-cli ... &` (~80s)
- `tmux new-session -d -s ... "hf upload ..."` (~2 min)
- `systemd-run --user --unit=... hf upload ...` (also SIGKILLed at ~100s)

(Tested with both the legacy `huggingface-cli` and the current `hf` CLI;
the killer is process-agnostic, not command-specific.)

The login node DOES have direct internet (no proxychains needed there) — the
problem is purely the process killer, not network.

**The reliable path is an sbatch job on a compute node with an SSH tunnel back
to the login node.** Compute nodes have no direct internet, but the existing
`eval/leonardo/start_proxy_tunnel.sh` opens a SOCKS5 forward from the compute
node to login05 and prints a `proxychains4 -q -f <config>` command prefix
that wraps any HF-bound command.

### Pre-flight (from your local Mac)

The intra-cluster SSH cert expires every ~12h. Refresh if stale:
```bash
step ssh certificate 'bfeuer00' --provisioner cineca-hpc \
  ~/.ssh/leonardo_daytona --no-password --insecure
ssh-keygen -R login.leonardo.cineca.it && \
rsync -avz -e 'ssh -i ~/.ssh/leonardo_daytona -o IdentitiesOnly=yes -o StrictHostKeyChecking=no' \
  ~/.ssh/leonardo_daytona ~/.ssh/leonardo_daytona.pub ~/.ssh/leonardo_daytona-cert.pub \
  bfeuer00@login.leonardo.cineca.it:~/.ssh/
```

Verify with `ssh-keygen -L -f ~/.ssh/leonardo_daytona-cert.pub | grep Valid`. Key on Leonardo:
`/leonardo/home/userexternal/bfeuer00/.ssh/leonardo_daytona`. The `start_proxy_tunnel.sh` script opens a
SOCKS5 forward compute→login05 and returns the `proxychains4 -q -f <config>` prefix. (The auth-required
HedgeDoc setup-instructions URL is in `.claude/secret.md` — untracked. The `leonardo_daytona` step-ca cert
is for the Daytona/SOCKS proxy; the intra-cluster `ssh Leonardo` key is a separate credential.)

### Cert expired → refresh it; do NOT route around the sbatch path
The sbatch-tunnel path depends on the `~/.ssh/leonardo_daytona` step-ca cert, which **expires ~12h**. When
it's expired the SOCKS tunnel gets `Permission denied (publickey)` and the `upload_*.sbatch` job dies in
~19s — **when a Leonardo upload sbatch fails fast on a publickey error, suspect the expired cert first** and
refresh it via the pre-flight above (needs interactive CINECA SSO 2FA in a browser; can't be done headless).
**The sbatch-tunnel remains the upload path** — do NOT fall back to a detached login-node `nohup` upload:
Leonardo's login killer takes down `nohup`/`disown`/tmux at ~100s (see `ops/all/hf_tmux.md`), so a
login-node upload of anything non-trivial will be SIGKILLed and leave a partial. Refresh the cert and use
the sbatch job.

### sbatch template for HF upload

```bash
cat > /leonardo_work/AIFAC_5C0_290/bfeuer00/upload_<job_name>.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=hf_upload_<short>
#SBATCH --output=<workdir>/upload_logs/upload_sbatch.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_5C0_290
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg

set -e
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate otagent
source ~/secrets.env

WD=<workdir>
DCFT=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent

unset LD_PRELOAD
export PATH="/leonardo_work/AIFAC_5C0_290/bfeuer00/proxychains/bin:${PATH}"
CMD_PREFIX=$(bash "$DCFT/eval/leonardo/start_proxy_tunnel.sh")

cd $WD/final_repo   # or $CHECKPOINTS_DIR/<job_name> for 8B path
$CMD_PREFIX /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/bin/hf upload \
    <hub_model_id> . . \
    --repo-type=model
EOF
cd /leonardo_work/AIFAC_5C0_290/bfeuer00
sbatch upload_<job_name>.sbatch
```

Then `squeue -j <jobid>` and `tail -f <workdir>/upload_logs/upload_sbatch.log`.

### Numbers / sizing

- 131GB consolidated 32B → ~4 min wall through the tunnel (sbatch + tunnel pattern)
- 30 min wall fits `boost_qos_dbg` (debug QOS); longer jobs need a different QOS
- `hf upload` is sequential (no `--num-workers` knob) — slower than the legacy
  `huggingface-cli upload-large-folder` looked on paper, but the latter is now
  a deprecation stub AND deadlocks against HF Hub LFS rate limits in practice.
  See "HF Uploads + Long-Running Login-Node Commands" near the RL Cleanup
  section for the full story.
- Resume is automatic — `.cache/huggingface/` persists state; if the job is
  requeued, `hf upload` picks up where it left off

### Why this matters for the SFT checklists

Both the 8B (step 2) and 32B (step 3) cleanup checklists invoke `hf upload`
(in a `tmux` session). On Jupiter / Perlmutter / NYU Torch that works
straight from the login node (login has direct internet, no kill policy).
On Leonardo the same one-liner WILL die at ~100 s and leave a partial
upload — use the sbatch template above instead.