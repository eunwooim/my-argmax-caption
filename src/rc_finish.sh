#!/bin/bash
# The end of rc_pipeline.sh, run on its own (2026-11-11): the Long-CLIP run's grpo.py self-evaluation died of OOM
# after its 300 steps (its adapter is complete), so the pipeline's train track stopped before the Use 2 cells of the
# three new runs. This does the rest with a memory-only GPU rule: Use 2 cells + own answers of the three new runs,
# waits for the separate Use 1 / Self-Validation tracks, then every replay cell, both reports, our judge stopped.
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
  t0=$(date +%s); say "$name: started (rc_finish.sh)"
  ( "$@" ) >> "$LOGS/rc_$name.log" 2>&1
  rc=$?
  if [ $rc = 0 ]; then say "$name: done ($(( ($(date +%s) - t0) / 60 )) min)"
  else say "$name: FAILED (exit $rc after $(( ($(date +%s) - t0) / 60 )) min; see $LOGS/rc_$name.log)"; fi
  return $rc
}
judge_up() { curl -sf -m 5 "$JUDGE_URL/models" > /dev/null; }
judge_watch() { while [ -e "$LOGS/rc_finish.running" ]; do judge_up || { say "judge: not answering; restarting it"; ( judge ) >> "$LOGS/rc_judge.log" 2>&1; }; sleep 120; done; }
touch "$LOGS/rc_finish.running"; trap 'rm -f "$LOGS/rc_finish.running"' EXIT
judge_watch &
say "finish: started (pid $$)"
( step eval_new rc_eval_workers "$OLD_MODELS,$NEW_MODELS" 2 ) &
p_eval=$!
( step vqa_new tts_vqa_models "$TTS/samples/base" "$TTS" "random=$RND/adapter" "selfcert=$SCERT/adapter" "longclip=$LCLIP/adapter" ) &
p_vqa=$!
while pgrep -u "$(id -u)" -f "bash rc_track_tts.sh" > /dev/null || pgrep -u "$(id -u)" -f "bash rc_track.sh selfval" > /dev/null; do
  sleep 120  # the separate Use 1 and Self-Validation tracks
done
say "finish: the Use 1 and Self-Validation tracks have ended"
wait $p_eval; wait $p_vqa
step replay_all tts_replay_workers "$TTS_LMMS" "$TTS" 6
step report rc_report_dirs "$TTS_LMMS" "$LMMS" "$TTS"
step report_prompting report_prompting
rm -f "$LOGS/rc_finish.running"
pids=$(pgrep -u "$(id -u)" -f "vllm serve .*--port $JUDGE_PORT") && { say "stopping our judge (pid $pids)"; kill $pids; }
say "report: $TTS/reward_comparison.md and .html; lmms_eval_baselines/prompting_comparison.md and .html"
say "FINISHED"
