#!/bin/bash
# Prompting baselines from start to finish, detached from any terminal or Claude session:
#   judge (port 8047) -> baselines_quick (gate) -> baselines N -> report_prompting -> stop our judge
#
#   setsid nohup bash prompting_pipeline.sh > logs_rl_paper/prompting_pipeline.log 2>&1 < /dev/null &
#
# Every stage is safe to repeat: finished cells are skipped, so after a crash just start it again.
# Progress: logs_rl_paper/prompting_pipeline.status (one line per event). Only our own judge is ever stopped.
cd "$(dirname "$0")" || exit 2
source /mnt/data2/mchung45/miniconda3/etc/profile.d/conda.sh
conda activate argmax

LOGS=logs_rl_paper
STATUS=$LOGS/prompting_pipeline.status
FULL=lmms_eval_baselines
QUICK=lmms_eval_baselines_quick
PORT=8047
WORKERS=${WORKERS:-3}
CELLS=5                # selfval_bon/caption, selfval_fta/caption, logiccheck/{mcq,caption,general}
MAX_RESTARTS=2         # new workers after all of them stopped with cells left

exec 9> "${PIPELINE_LOCK:-$LOGS/prompting_pipeline.lock}"
flock -n 9 || { echo "[pipeline] already running (or a worker started by an earlier pipeline still holds the lock: set PIPELINE_LOCK to another file)"; exit 0; }
# children run with fd 9 closed (9>&-), or a detached worker would keep the lock after this script is gone

say() { date "+%F %T  $*" | tee -a "$STATUS"; }
judge_up() { curl -sf -m 5 "http://localhost:$PORT/v1/models" > /dev/null; }
judge_stop() {
  local pids; pids=$(pgrep -u "$(id -u)" -f "vllm serve .*--port $PORT") || return 0
  say "stopping our judge (pid $pids)"; kill $pids
}
fail() { say "FAILED: $*"; judge_stop; exit 1; }
done_cells() { find "$1" -mindepth 3 -maxdepth 3 -name done.json 2>/dev/null | wc -l; }
workers_alive() { pgrep -u "$(id -u)" -fc "lmms_eval_run.py --out_dir $FULL " ; }

start_judge() {  # retries only when another job landed on the same GPU during startup
  judge_up && return 0
  say "judge: waiting for a free GPU among 4,5,6,7"
  local try
  for try in 1 2 3 4 5 6; do
    if bash rl_paper_commands.sh judge >> "$LOGS/prompting_pipeline_judge.log" 2>&1 9>&-; then
      say "judge: ready on port $PORT"; return 0
    fi
    grep -q "No available memory for the cache blocks" "$LOGS/judge.log" \
      || { say "judge: failed with another error, see $LOGS/judge.log"; return 1; }
    say "judge: another job took the GPU during startup (attempt $try); retrying in 2 min"
    sleep 120
  done
  say "judge: gave up after 6 attempts"; return 1
}

say "pipeline started (pid $$)"
# lmms-eval's task files load their data with token=True: wait for the login before taking any GPU
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
token_ok() { python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null; }
if ! who=$(token_ok); then
  say "waiting for a Hugging Face login: run  hf auth login"
  until who=$(token_ok); do sleep 30; done
fi
say "Hugging Face login ok ($who)"
start_judge || fail "the judge did not start"

# gate: every cell on 4 items per task, or the full run does not start
if [ "$(done_cells "$FULL")" -eq 0 ] && [ ! -e "$FULL/plan.json" ]; then
  say "baselines_quick: started"
  bash rl_paper_commands.sh baselines_quick > "$LOGS/prompting_pipeline_quick.log" 2>&1 9>&- \
    || fail "baselines_quick exited with an error, see $LOGS/prompting_pipeline_quick.log"
  n=$(done_cells "$QUICK")
  [ "$n" -eq "$CELLS" ] || fail "baselines_quick finished only $n/$CELLS cells"
  [ -s "$QUICK/prompting_comparison.md" ] && [ -s "$QUICK/prompting_comparison.html" ] \
    || fail "baselines_quick did not write its report"
  say "baselines_quick: passed ($n/$CELLS cells)"
fi

start_judge || fail "the judge did not start"
say "baselines: starting $WORKERS workers"
bash rl_paper_commands.sh baselines "$WORKERS" > "$LOGS/prompting_pipeline_full.log" 2>&1 9>&- \
  || fail "the workers did not start, see $LOGS/prompting_pipeline_full.log"

seen=-1; restarts=0
while :; do
  n=$(done_cells "$FULL")
  if [ "$n" != "$seen" ]; then say "baselines: $n/$CELLS cells done"; seen=$n; fi
  [ "$n" -ge "$CELLS" ] && [ "$(workers_alive)" -eq 0 ] && break
  if ! judge_up; then
    say "judge: not answering; restarting it (cells that need it wait)"
    start_judge || fail "the judge could not be restarted"
  fi
  if [ "$(workers_alive)" -eq 0 ] && [ "$n" -lt "$CELLS" ]; then
    restarts=$((restarts + 1))
    [ "$restarts" -le "$MAX_RESTARTS" ] \
      || fail "the workers keep stopping with $n/$CELLS cells done, see $LOGS/baselines_worker_*.log"
    say "baselines: every worker stopped at $n/$CELLS; starting one more (restart $restarts/$MAX_RESTARTS)"
    bash rl_paper_commands.sh baselines 1 >> "$LOGS/prompting_pipeline_full.log" 2>&1 9>&-
  fi
  sleep 120
done

bash rl_paper_commands.sh report_prompting > "$LOGS/prompting_pipeline_report.log" 2>&1 9>&- \
  || fail "the report failed, see $LOGS/prompting_pipeline_report.log"
say "report: $FULL/prompting_comparison.md and .html"
judge_stop
say "FINISHED"
