#!/usr/bin/env bash
# Run the pipeline for one model, phase by phase. Each step leaves a marker in
# results/<model>/done/ and is skipped when the marker exists, so a rerun resumes.
#
# GPU is the card's nvidia-smi index; there is no default on purpose.
#   GPU=0 MODEL=7b bash scripts/run.sh gates smoke sample
#   GPU=0 MODEL=7b bash scripts/run.sh pilot select central
#   GPU=0 MODEL=7b bash scripts/run.sh confirm        # once, after the method is frozen
#   GPU=0 MODEL=7b bash scripts/run.sh extras probes binary esma
#
# Order used for the thesis: 7B through every phase; 1.5B and 3B with
# gates sample central (they reuse the 7B pilot selection), then the later phases.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU to the card index (nvidia-smi)}"
export TOKENIZERS_PARALLELISM=false
PY="${PY:-python}"
M="${MODEL:-7b}"
R="results/$M"
LOGS="results/logs"
mkdir -p "$R/done" "$LOGS"

run() { "$PY" -m pipeline.run "$@"; }

step() {
  local name="$1"; shift
  if [ -f "$R/done/$name" ]; then echo "=== skip $name"; return 0; fi
  echo "=== $name  $(date '+%H:%M:%S')"
  "$@" 2>&1 | tee -a "$LOGS/$M.$name.log"
  touch "$R/done/$name"
}

for phase in "$@"; do
  case "$phase" in
    gates)
      if [ "$M" = "7b" ]; then step gate3_reproduction run gate-repro --model "$M"; fi
      step gate4_batching run gate-batch --model "$M" ;;
    smoke)
      rm -rf "results/smoke/$M"
      run sample --model "$M" --domain locations --split train --limit 16
      run sample --model "$M" --domain dates --split train --limit 16
      run sample --model "$M" --domain locations --split test --limit 16
      run train --model "$M" --condition joint --seed 0 --limit 16
      run eval --model "$M" --adapter joint_s0 --domain locations --split test --limit 16
      run eval --model "$M" --adapter base --domain locations --split test --limit 16 ;;
    sample)
      for ds in locations:train dates:train locations:test dates:test; do
        d="${ds%%:*}"; s="${ds##*:}"
        step "sample_${d}_${s}" run sample --model "$M" --domain "$d" --split "$s"
      done ;;
    pilot)
      step pilot_train run train --model "$M" --condition locations --seed 0 --pilot
      step pilot_eval_base run eval --model "$M" --adapter base --domain locations --split train --holdout
      step pilot_eval_adapter run eval --model "$M" --adapter pilot_locations_s0 --domain locations --split train --holdout
      run pilot-check --model "$M" --adapter pilot_locations_s0 ;;
    select)
      # One variant declared in advance (1 epoch), then the fixed selection rule.
      step pilot_train_e1 run train --model "$M" --condition locations --seed 0 --pilot --epochs 1
      step pilot_eval_adapter_e1 run eval --model "$M" --adapter pilot_locations_s0_e1 --domain locations --split train --holdout
      run pilot-check --model "$M" --adapter pilot_locations_s0_e1 || true
      step pilot_select run pilot-select --model "$M" ;;
    central)
      for c in locations dates joint; do for s in 0 1 2; do
        step "train_${c}_s${s}" run train --model "$M" --condition "$c" --seed "$s"
      done; done
      for d in locations dates; do
        step "eval_base_${d}_test" run eval --model "$M" --adapter base --domain "$d" --split test
        for c in locations dates joint; do for s in 0 1 2; do
          step "eval_${c}_s${s}_${d}_test" run eval --model "$M" --adapter "${c}_s${s}" --domain "$d" --split test
        done; done
      done
      step report_dev run report --model "$M" --split dev ;;
    confirm)
      export CONFIRM=1
      step sample_locations_val run sample --model "$M" --domain locations --split val
      step eval_base_locations_val run eval --model "$M" --adapter base --domain locations --split val
      for c in locations dates joint; do for s in 0 1 2; do
        step "eval_${c}_s${s}_locations_val" run eval --model "$M" --adapter "${c}_s${s}" --domain locations --split val
      done; done
      step report_val run report --model "$M" --split val ;;
    extras)
      step baseline_noidk run eval --model "$M" --adapter base --domain locations --split test --mode noidk
      step baseline_bare run eval --model "$M" --adapter base --domain locations --split test --mode bare ;;
    probes)
      for a in base locations_s0 locations_s1 locations_s2 dates_s0 dates_s1 dates_s2 joint_s0 joint_s1 joint_s2; do
        for p in conflict entity; do
          step "probe_${p}_$a" run probe --model "$M" --adapter "$a" --probe "$p"
        done
      done ;;
    binary)
      # The answer-or-abstain baseline on dates, trained like the reported adapters.
      for s in 0 1 2; do
        step "train_dates_s${s}_binary" run train --model "$M" --condition dates --seed "$s" --rule binary --no-julian-credit
        step "eval_dates_s${s}_binary" run eval --model "$M" --adapter "dates_s${s}_binary" --domain dates --split test
      done ;;
    esma)
      # The direct (no-IDK) answers and the yes/no question for every adapter, then the measures.
      ADAPTERS="base"
      for c in locations dates joint; do for s in 0 1 2; do ADAPTERS="$ADAPTERS ${c}_s$s"; done; done
      for s in 0 1 2; do
        [ -f "$R/adapters/dates_s${s}_binary/adapter_config.json" ] && ADAPTERS="$ADAPTERS dates_s${s}_binary"
      done
      for a in $ADAPTERS; do for d in locations dates; do
        step "noidk_${a}_${d}" run eval --model "$M" --adapter "$a" --domain "$d" --split test --mode noidk
        step "meta_${a}_${d}" "$PY" -m pipeline.meta --model "$M" --adapter "$a" --domain "$d" --split test
      done; done
      "$PY" -m pipeline.esma --model "$M" ;;
    *) echo "unknown phase: $phase"; exit 1 ;;
  esac
done
