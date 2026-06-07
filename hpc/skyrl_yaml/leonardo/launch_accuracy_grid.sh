#!/bin/bash
# Launch the GSM8K GRPO ACCURACY grid on Leonardo from accuracy_grid_cells.txt.
# NON-AGENTIC (no Daytona): runs 8 cells CONCURRENTLY via chained Slurm afterany
# dependencies so the rest auto-start as slots free. 1 node x 4 A100-64GB / cell,
# boost_usr_prod, 8h wall. Each cell -> one sbatch of sbatch_gsm8k_grid.sh with
# --job-name=grid_<cell> (the sbatch derives a FRESH per-cell ckpt dir from that),
# the combo_C BASE_KNOBS prepended, then the cell's overrides (last-wins hydra).
#
# Usage (on Leonardo login node, inside tmux):
#   bash launch_accuracy_grid.sh --all          # all directly-launchable cells, 8 concurrent
#   bash launch_accuracy_grid.sh --batch1        # the 10 Batch-1 cells only
#   bash launch_accuracy_grid.sh <cell> [<cell> ...]   # named cells (no chaining; all submitted free)
#
# Chaining: with --all / --batch1 the cells are submitted in list order; the first
# $CONCURRENCY are free (no dep), and cell[i] (i>=CONCURRENCY) gets
# --dependency=afterany:<jobid of cell[i-CONCURRENCY]> so exactly ~CONCURRENCY run
# at once and each finishing cell releases the next in its chain.
#
# Appends "cell=<name> jobid=<id> dep=<dep>" to accuracy_grid_manifest.txt for harvest.
set -uo pipefail
CFGDIR=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
SPEC=$CFGDIR/accuracy_grid_cells.txt
SBATCH=$CFGDIR/sbatch_gsm8k_grid.sh
MANIFEST=$CFGDIR/accuracy_grid_manifest.txt

PARTITION=boost_usr_prod
QOS=normal                   # partition default QOS; no per-user TRES/wall cap, 8h fits the 1-day partition MaxTime
WALL=08:00:00
CONCURRENCY=8

# combo_C (throughput winner) deltas vs the run_gsm8k_canary.sh base, PLUS plateau sizing
# (epochs/max_steps raised so the 8h wall is the limiter, not 25-step/1-epoch caps) and the
# pass@8 eval protocol (eval every 20 steps with 8 samples/prompt -> eval/all/pass_at_8).
BASE_KNOBS="\
trainer.train_batch_size=128 trainer.policy_mini_batch_size=128 \
generator.n_samples_per_prompt=8 generator.gpu_memory_utilization=0.85 \
generator.enforce_eager=false \
trainer.micro_train_batch_size_per_gpu=16 trainer.micro_forward_batch_size_per_gpu=16 \
trainer.epochs=8 trainer.max_steps=400 \
trainer.eval_interval=20 trainer.eval_before_train=true trainer.eval_batch_size=256 \
generator.eval_n_samples_per_prompt=8 trainer.ckpt_interval=20"

# All directly-launchable cells, in the order they should be submitted (and chained).
BATCH1=(base_long lr3e7 lr1e6 lr3e6 lr1e5 lr3e5 n4 n16 gen1024 gen2048)
BATCH2_LAUNCHABLE=(kl001 kl01 temp07 temp12 clip028 entbonus base_q3)
ALL=("${BATCH1[@]}" "${BATCH2_LAUNCHABLE[@]}")

declare -a WANT
CHAIN=1
case "${1:-}" in
    --all)    WANT=("${ALL[@]}") ;;
    --batch1) WANT=("${BATCH1[@]}") ;;
    "")       echo "usage: $0 --all | --batch1 | <cell> ..."; exit 1 ;;
    *)        WANT=("$@"); CHAIN=0 ;;   # explicit cell list: submit all free (no chain)
esac
[[ ${#WANT[@]} -eq 0 ]] && { echo "no cells given"; exit 1; }

get_overrides() {
    # print the overrides for cell $1 from SPEC (everything after the first |), or __MISSING__.
    # Operate on the raw line ($0): split name vs rest on the first | WITHOUT mutating $0
    # (mutating a field rebuilds $0 under OFS and eats the | separator).
    awk -v c="$1" '
        /^[[:space:]]*#/ {next}
        {
            line=$0
            p=index(line,"|")
            if (p==0) { name=line; rest="" } else { name=substr(line,1,p-1); rest=substr(line,p+1) }
            gsub(/^[ \t]+|[ \t]+$/,"",name)
            gsub(/^[ \t]+|[ \t]+$/,"",rest)
            if (name==c) { print rest; found=1; exit }
        }
        END{ if(!found) print "__MISSING__" }
    ' "$SPEC"
}

declare -a JOBIDS
echo "== launching ${#WANT[@]} cells (concurrency=$CONCURRENCY, chain=$CHAIN) ==" | tee -a "$MANIFEST"
for idx in "${!WANT[@]}"; do
    cell="${WANT[$idx]}"
    ov=$(get_overrides "$cell")
    if [[ "$ov" == "__MISSING__" ]]; then echo "!! no spec for cell '$cell' in $SPEC"; JOBIDS[$idx]=""; continue; fi

    dep=""
    if [[ $CHAIN -eq 1 && $idx -ge $CONCURRENCY ]]; then
        parent_idx=$((idx - CONCURRENCY))
        parent_jid="${JOBIDS[$parent_idx]:-}"
        if [[ -n "$parent_jid" ]]; then dep="--dependency=afterany:${parent_jid}"; fi
    fi

    echo ">> cell=$cell idx=$idx dep=[${dep:-none}] overrides=[$ov]"
    jid=$(sbatch --parsable \
            --partition="$PARTITION" --qos="$QOS" --time="$WALL" \
            --nodes=1 --gres=gpu:4 \
            --job-name="grid_${cell}" \
            $dep \
            "$SBATCH" $BASE_KNOBS $ov)
    rc=$?
    if [[ $rc -ne 0 || -z "$jid" ]]; then echo "!! sbatch FAILED for $cell (rc=$rc): $jid"; JOBIDS[$idx]=""; continue; fi
    JOBIDS[$idx]="$jid"
    echo "   submitted jobid=$jid"
    echo "cell=$cell jobid=$jid dep=[${dep:-none}] overrides=[$ov] ts=$(date -u +%FT%TZ)" >> "$MANIFEST"
done

echo "== submitted job ids =="
for idx in "${!WANT[@]}"; do echo "  ${WANT[$idx]} -> ${JOBIDS[$idx]:-FAILED}"; done
echo "== manifest: $MANIFEST =="
