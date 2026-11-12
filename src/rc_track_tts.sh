#!/bin/bash
# The Use 1 track of rc_pipeline.sh, run on its own (2026-11-11: restarted with a looser vLLM GPU rule, VLLM_GPU_MB).
# Same steps and outputs as rc_pipeline.sh's track_tts; finished parts are skipped.
cd "$(dirname "$0")" || exit 2
source /mnt/data2/mchung45/miniconda3/etc/profile.d/conda.sh
conda activate argmax
export GPU_FREE_MEM_MB=${GPU_FREE_MEM_MB:-80000} GPU_FREE_UTIL=${GPU_FREE_UTIL:-60} VLLM_GPU_MB=${VLLM_GPU_MB:-86000}
RL_COMMANDS_SOURCED=1 source rl_paper_commands.sh
set +euo pipefail
cd /mnt/data1/mchung45/argmax_caption_with_docci_qa
STATUS=$LOGS/rc_pipeline.status
say() { date "+%F %T  $*" | tee -a "$STATUS"; }
step() {
  local name=$1 t0 rc; shift
  t0=$(date +%s); say "$name: started (rc_track_tts.sh)"
  ( "$@" ) >> "$LOGS/rc_$name.log" 2>&1
  rc=$?
  if [ $rc = 0 ]; then say "$name: done ($(( ($(date +%s) - t0) / 60 )) min)"
  else say "$name: FAILED (exit $rc after $(( ($(date +%s) - t0) / 60 )) min; see $LOGS/rc_$name.log)"; fi
  return $rc
}
step sample_base tts_sample_vllm "$TTS/items" "$(base_snapshot)" "$TTS/samples/base" || exit 1
touch "$TTS/samples/base/DONE"
step score_base tts_score_base "$TTS/samples/base" "$TTS/base" || exit 1
step select_base python tts.py select --sample_dir "$TTS/samples/base" --scores "$TTS/base/scores.shard*.jsonl" --out_dir "$TTS/base/selected" || exit 1
step vqa_old tts_vqa_models "$TTS/samples/base" "$TTS" "base=$BASE_MODEL" "self=$SELF/adapter" "external=$EXT/adapter" "length=$LEN/adapter"
step replay_base tts_replay_workers "$TTS_LMMS" "$TTS" 6
say "track tts (rc_track_tts.sh): finished"
