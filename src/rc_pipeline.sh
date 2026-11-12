#!/bin/bash
# Reward comparison from start to finish (README "Reward comparison"), detached from any terminal or Claude session:
#
#   bash rl_paper_commands.sh rc_pipeline          (= setsid nohup bash rc_pipeline.sh ... &)
#   tail -F logs_rl_paper/rc_pipeline.status       (one line per step; each step's output in logs_rl_paper/rc_<step>.log)
#
# setup -> judge -> quick (every step on 4 items: the gate) -> five tracks at once, each waiting for a free GPU:
#   train  GRPO random / self-certainty / Long-CLIP (3 GPUs) -> Use 2 cells of the 3 new runs -> their own answers
#   eval   Use 2 cells of base / self / external / length
#   tts    vLLM 16 candidates of the base model -> scores (2 GPUs) -> picks -> own answers of the 4 old runs
#   comb   self GRPO merged -> vLLM candidates -> scores (ours) -> picks
#   selfval  Self-Validation cells again after the LPFV fix (BASELINES.md) -> prompting report
# then the replay cells (CPU, 6 workers) of every pick -> the report -> our judge stopped.
# Every step can be repeated: finished parts are skipped (cells, items, scores, adapters). GPUs 0-3 are never used.
cd "$(dirname "$0")" || exit 2
source /mnt/data2/mchung45/miniconda3/etc/profile.d/conda.sh
conda activate argmax
export GPU_FREE_MEM_MB=${GPU_FREE_MEM_MB:-80000}
RL_COMMANDS_SOURCED=1 source rl_paper_commands.sh
set +euo pipefail

STATUS=$LOGS/rc_pipeline.status
RUNNING=$LOGS/rc_pipeline.running
exec 9> "$LOGS/rc_pipeline.lock"
flock -n 9 || { echo "[rc_pipeline] already running"; exit 0; }

say() { date "+%F %T  $*" | tee -a "$STATUS"; }
judge_up() { curl -sf -m 5 "$JUDGE_URL/models" > /dev/null; }
judge_stop() {
  local pids; pids=$(pgrep -u "$(id -u)" -f "vllm serve .*--port $JUDGE_PORT") || return 0
  say "stopping our judge (pid $pids)"; kill $pids
}

step() {  # name command...: runs in a subshell (its own set -e etc.), output to logs_rl_paper/rc_<name>.log
  local name=$1 t0 rc; shift
  t0=$(date +%s); say "$name: started"
  ( "$@" ) >> "$LOGS/rc_$name.log" 2>&1 9>&-
  rc=$?
  if [ $rc = 0 ]; then say "$name: done ($(( ($(date +%s) - t0) / 60 )) min)"
  else say "$name: FAILED (exit $rc after $(( ($(date +%s) - t0) / 60 )) min; see $LOGS/rc_$name.log)"; fi
  return $rc
}

start_judge() {  # retries only when another job took the GPU while vLLM was starting
  judge_up && return 0
  local try
  for try in 1 2 3 4 5 6; do
    if ( judge ) >> "$LOGS/rc_judge.log" 2>&1 9>&-; then say "judge: ready on port $JUDGE_PORT"; return 0; fi
    grep -q "No available memory for the cache blocks" "$LOGS/judge.log" || { say "judge: failed, see $LOGS/judge.log"; return 1; }
    say "judge: another job took the GPU during startup (attempt $try); retrying in 2 min"; sleep 120
  done
  return 1
}

judge_watch() {  # the CaptionQA scoring needs the judge until the end
  while [ -e "$RUNNING" ]; do
    if ! judge_up; then say "judge: not answering; restarting it"; start_judge; fi
    sleep 120
  done
}

selfval_rerun() {  # Self-Validation cells again (fresh generation, same seeds), old cells set aside, waits
  local m stamp; stamp=$(date +%Y%m%d-%H%M%S)
  for m in selfval_bon selfval_fta; do  # a cell finished before the fix (2026-11-10) is set aside; a newer one stays
    local d="$BASELINES/$m/caption"
    if [ -d "$d" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$d/done.json'))['finished'] < '2026-11-10 12' else 1)" 2>/dev/null; then
      mkdir -p "$BASELINES/_old_lpfv_$stamp" && mv "$d" "$BASELINES/_old_lpfv_$stamp/${m}_caption"
    fi
  done
  python lmms_eval_run.py --out_dir "$BASELINES" "${BL_ARGS[@]}" --only selfval_bon,selfval_fta --gpus auto
}

wait_for() {  # file: a track waits for another track's output
  until [ -e "$1" ]; do sleep 60; done
}

track_train() {
  step train rc_train || return 1
  step eval_new rc_eval_workers "$OLD_MODELS,$NEW_MODELS" 2 || return 1
  wait_for "$TTS/samples/base/DONE"
  step vqa_new tts_vqa_models "$TTS/samples/base" "$TTS" "random=$RND/adapter" "selfcert=$SCERT/adapter" "longclip=$LCLIP/adapter"
}

track_eval() {
  step eval_old rc_eval_workers "$OLD_MODELS" 2
}

track_tts() {
  step sample_base tts_sample_vllm "$TTS/items" "$(base_snapshot)" "$TTS/samples/base" || return 1
  touch "$TTS/samples/base/DONE"
  step score_base tts_score_base "$TTS/samples/base" "$TTS/base" || return 1
  step select_base python tts.py select --sample_dir "$TTS/samples/base" --scores "$TTS/base/scores.shard*.jsonl" \
    --out_dir "$TTS/base/selected" || return 1
  step vqa_old tts_vqa_models "$TTS/samples/base" "$TTS" "base=$BASE_MODEL" "self=$SELF/adapter" "external=$EXT/adapter" "length=$LEN/adapter"
  step replay_base tts_replay_workers "$TTS_LMMS" "$TTS" 6
}

track_comb() {
  step merge_self tts_merge_self || return 1
  wait_for "$TTS/items/DONE"
  step sample_self tts_sample_vllm "$TTS/items" "$MERGED_SELF" "$TTS/samples/selfgrpo" || return 1
  step score_self python tts.py score --sample_dir "$TTS/samples/selfgrpo" --out "$TTS/selfgrpo/scores.jsonl" \
    --selectors self "${GPU_ARGS[@]}" --step1_dir "$STEP1" || return 1
  step select_self python tts.py select --sample_dir "$TTS/samples/selfgrpo" --scores "$TTS/selfgrpo/scores.jsonl" \
    --out_dir "$TTS/selfgrpo/selected" --selectors self
}

track_selfval() {
  step selfval_rerun selfval_rerun || return 1
  step report_prompting report_prompting
}

# ---------------------------------------------------------------- run
say "pipeline started (pid $$); GPU free = at most $GPU_FREE_MEM_MB MiB used and ${GPU_FREE_UTIL}% busy, GPUs 4-7"
touch "$RUNNING"
trap 'rm -f "$RUNNING"' EXIT
step setup rc_setup || { say "FAILED: setup"; exit 1; }
( step dump tts_dump "$TTS/items" && touch "$TTS/items/DONE" ) &
pid_dump=$!
start_judge || { say "FAILED: the judge did not start"; exit 1; }
judge_watch &
step quick rc_quick || { say "FAILED: quick (the full run did not start); report so far: ${TTS}_quick/"; judge_stop; exit 1; }
say "quick: passed; its report: ${TTS}_quick/reward_comparison.md"

track_train & p_train=$!;  sleep 240
track_eval & p_eval=$!;    sleep 240
wait $pid_dump || say "dump: FAILED, the test-time tracks wait for $TTS/items/DONE"
track_tts & p_tts=$!;      sleep 240
track_comb & p_comb=$!;    sleep 240
track_selfval & p_sv=$!
for t in train:$p_train eval:$p_eval tts:$p_tts comb:$p_comb selfval:$p_sv; do
  wait "${t#*:}" && say "track ${t%%:*}: finished" || say "track ${t%%:*}: FAILED (its last step's log has the error)"
done

step replay_all tts_replay_workers "$TTS_LMMS" "$TTS" 6     # every pick (base + combination); done cells are skipped
step report rc_report_dirs "$TTS_LMMS" "$LMMS" "$TTS"
rm -f "$RUNNING"
judge_stop
say "report: $TTS/reward_comparison.md and .html"
say "FINISHED"
