#!/bin/bash
# One track of rc_pipeline.sh run on its own, with a memory-only GPU rule (2026-11-11: our own jobs keep every GPU
# we may use above 60% busy, so a utilisation rule starves our later steps). Same steps and outputs as the pipeline;
# finished parts are skipped.   bash rc_track.sh comb | selfval
cd "$(dirname "$0")" || exit 2
source /mnt/data2/mchung45/miniconda3/etc/profile.d/conda.sh
conda activate argmax
export GPU_FREE_MEM_MB=${GPU_FREE_MEM_MB:-80000} GPU_FREE_UTIL=${GPU_FREE_UTIL:-100} VLLM_GPU_MB=${VLLM_GPU_MB:-86000}
RL_COMMANDS_SOURCED=1 source rl_paper_commands.sh
set +euo pipefail
cd /mnt/data1/mchung45/argmax_caption_with_docci_qa
STATUS=$LOGS/rc_pipeline.status
say() { date "+%F %T  $*" | tee -a "$STATUS"; }
step() {
  local name=$1 t0 rc; shift
  t0=$(date +%s); say "$name: started (rc_track.sh)"
  ( "$@" ) >> "$LOGS/rc_$name.log" 2>&1
  rc=$?
  if [ $rc = 0 ]; then say "$name: done ($(( ($(date +%s) - t0) / 60 )) min)"
  else say "$name: FAILED (exit $rc after $(( ($(date +%s) - t0) / 60 )) min; see $LOGS/rc_$name.log)"; fi
  return $rc
}
selfval_rerun() {  # as rc_pipeline.sh: cells finished before the LPFV fix are set aside, then run again
  local m stamp d; stamp=$(date +%Y%m%d-%H%M%S)
  for m in selfval_bon selfval_fta; do
    d="$BASELINES/$m/caption"
    if [ -d "$d" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$d/done.json'))['finished'] < '2026-11-10 12' else 1)" 2>/dev/null; then
      mkdir -p "$BASELINES/_old_lpfv_$stamp" && mv "$d" "$BASELINES/_old_lpfv_$stamp/${m}_caption"
    fi
  done
  python lmms_eval_run.py --out_dir "$BASELINES" "${BL_ARGS[@]}" --only selfval_bon,selfval_fta --gpus auto
}
case "$1" in
  comb)
    step sample_self tts_sample_vllm "$TTS/items" "$MERGED_SELF" "$TTS/samples/selfgrpo" || exit 1
    step score_self python tts.py score --sample_dir "$TTS/samples/selfgrpo" --out "$TTS/selfgrpo/scores.jsonl" \
      --selectors self "${GPU_ARGS[@]}" --step1_dir "$STEP1" || exit 1
    step select_self python tts.py select --sample_dir "$TTS/samples/selfgrpo" --scores "$TTS/selfgrpo/scores.jsonl" \
      --out_dir "$TTS/selfgrpo/selected" --selectors self || exit 1
    say "track comb (rc_track.sh): finished" ;;
  selfval)
    step selfval_rerun selfval_rerun || exit 1
    step report_prompting report_prompting
    say "track selfval (rc_track.sh): finished" ;;
  *) echo "usage: bash rc_track.sh comb|selfval"; exit 1 ;;
esac
