# 2026-06-20 — Delphi #6279 RL math-cell entropy-explosion hparam re-tune (Leonardo)

## Context
5/6 newly-processed Delphi #6279 math RL cells DIVERGED during GRPO: `policy/policy_entropy`
EXPLODED ~0.13 → ~11.5, reward → −1.0 / pass@8 → 0 by step ~100 (D1 rlvr_math, D3 dapo_math,
D4 math500). Only D2/ifeval healthy. Failure mode = entropy EXPLOSION (NOT low-entropy collapse).
Task: design a stabilization sweep (NO KL) on max_grad_norm clip + lr + entropy_coef (swept axis),
launch a first wave on Leonardo. Source data: main_rl_evals/{SCORES.md, grid.md},
agent_logs/2026-06-19_delphi_rl_six_cells.md.

## Prior art examined
- **gsm8k accuracy_grid** (`~/Documents/experiments/gsm8k_grid_leonardo/accuracy_grid.md`): lr dominant;
  **lr1e-5 → entropy 0.04 (near-collapse)**, **lr3e-5 UNSTABLE — grad spike 1.95, rejected**, **lr3e-6
  healthy (entropy 0.115, pass@8 0.760, conservative runner-up)**. `entbonus` (use_entropy_loss=true
  coef +0.01) = the anti-COLLAPSE stabilizer there. Knob names confirmed from accuracy_grid_cells.txt.
- **gsm8k throughput grid**: layout winners (4×TP1 colocate, cudagraph on, gmu0.85) — inherited unchanged.
- **ablation_exploration_in_rl**: lrboost / seqnorm-tis-shaped / explore-tis-* confirm entropy dynamics
  are the live RL-stability axis; TIS/seqnorm reserved as a KL-free fallback trust-region tool.
- **MarinSkyRL config** (`ppo_base_config.yaml`): `trainer.policy.optimizer_config.max_grad_norm`
  (default 1.0); `trainer.algorithm.use_entropy_loss` / `entropy_loss_coef` (default 0.01); ZClip
  (`trainer.algorithm.z_clip.enabled`) adaptive grad clip available as a reserve.
- **Entropy sign** (`workers/worker.py:1045,1068-1070`): `loss = policy_loss + kl_loss_term −
  entropy_loss_term`; `entropy_loss_term = entropy × entropy_loss_coef`. POSITIVE coef = entropy BONUS
  (pushes entropy UP). → the gold +0.01 was driving the explosion; NEGATIVE coef = entropy penalty (damps).

## Ground-truth fix
The Delphi RL run scripts (`run_delphi_math_rl.sh`, `sbatch_delphi_math_rl.sh`, `rl_dataset_prep.py`)
were UNTRACKED on Leonardo only — never committed (violates the local-clone-is-SoT rule). Pulled all 3
into the local repo (`hpc/skyrl_yaml/leonardo/`) for tracking. No edit needed for the sweep: the sbatch
already forwards dotted hydra args through `"$@"`, so lr / max_grad_norm / entropy override on the
command line (last-wins). Ckpts already correctly write to `$WORK/rl_ckpts` (WRITE-PATH MANDATE OK).

## Design
Testbed = `laion/delphi-1e22-p33m67-...-wc386k_lr1e5-sft` (MATH500 45.0) × D1 rlvr_math (clean math
collapser; staged). 8-cell cross of max_grad_norm {1.0, 0.5, 0.2} × lr {1e-5, 3e-6} × entropy
{+0.01, 0, −0.001, off}, designed as isolating corners (full table + rationale in
`main_rl_evals/stabilization_grid.md`). Lead candidate gc05_lr3e6_e0 (clip 0.5 + lr 3e-6 + entropy push OFF).

## Launch log
**Wave 1 submitted 2026-06-20** (testbed = p33m67 wc386k SFT × D1 rlvr_math; 1 node × 4 A100 each,
normal QOS ≤24h; ckpts → $WORK/rl_ckpts/stab-*):
- **47447528** rl_stab-gc05_lr3e6_e0 — max_grad_norm=0.5, lr=3e-6, use_entropy_loss=false  (lead candidate)
- **47447530** rl_stab-gc02_lr3e6_e0 — max_grad_norm=0.2, lr=3e-6, use_entropy_loss=false  (hardest brake)
- **47447531** rl_stab-gc05_lr1e5_e0 — max_grad_norm=0.5, lr=1e-5, use_entropy_loss=false  (clip-only on hot lr)

All 3 PENDING(Priority) at submit — Leonardo had 9 RUNNING (8 delphi-evals + 1 passatk array), 0 other
pending; the 3 stab cells are the only new pending. They start as eval nodes free. Live count after submit:
9 RUNNING + 3 PENDING.

## Wave 2 (launch as Wave 1 frees nodes — see stabilization_grid.md)
gc05_lr3e6_eneg (clip0.5/lr3e6/entropy −0.001 penalty), gc02_lr1e5_e0 (clip0.2/hot lr), gc05_lr3e6_ekeep
(clip0.5/lr3e6/keep +0.01 — isolates the bonus), gc1_lr3e6_e0 (orig clip 1.0/lr3e6 — isolates the clip
lever), + baseline control (clip1.0/lr1e5/+0.01 — confirm explosion reproduces).
Reserve axis if all diverge: ZClip (z_clip.enabled=true) and/or DAPO eps_clip_high=0.28.

## Notes / unresolved
- Entropy negative-coef arm (gc05_lr3e6_eneg) is mechanistically sound (sign verified) but UNTESTED in this
  codebase — held to Wave 2 so the no-push (off) arms report first.
- The 3 untracked Delphi scripts are now pulled to local; need a commit+push to make local the SoT (the
  Leonardo copies are byte-identical to what was launched, so no re-pull needed for these runs).
