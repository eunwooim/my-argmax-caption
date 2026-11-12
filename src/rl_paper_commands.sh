#!/usr/bin/env bash
# RL paper: train the two comparison runs, check the LoRA merges, run lmms-eval, make the table.
# Run from inside argmax_caption_with_docci_qa/, one stage at a time or all at once:
#
#   bash rl_paper_commands.sh check      # paths, disk, the self run's settings (run this first)
#   bash rl_paper_commands.sh setup      # install lmms-eval + scorer deps, the DOCCI caption task, CPU smoke test (once)
#   bash rl_paper_commands.sh train      # external + length runs, side by side on 2 free GPUs (hours)
#   bash rl_paper_commands.sh verify     # self / external / length: LoRA merge checks (no model folders written)
#   bash rl_paper_commands.sh judge      # start the judge LLM (vLLM, background) and wait until it answers
#   bash rl_paper_commands.sh quick      # the whole eval on 4 items, base only (~15 min): catches scorer problems early
#   bash rl_paper_commands.sh eval       # quick, then lmms-eval: 4 models x (mcq, caption = COCO CHAIR + DOCCI, general)
#   bash rl_paper_commands.sh par 3      # 3 workers on 3 free GPUs sharing the cells (par 3 replan: new plan, old results set aside)
#   bash rl_paper_commands.sh par 3 drop oeq,selfaug   # same, but first drop those groups of the frozen plan (results kept
#                                        #   in _old_*/; COCO CHAIR, MMVP, MMBench, MME, MM-Vet replay from their caches)
#   bash rl_paper_commands.sh report     # -> lmms_eval_out/summary.md, summary.html
#   bash rl_paper_commands.sh baselines_quick   # prompting baselines on 4 items per task (needs the judge): catches problems
#   bash rl_paper_commands.sh baselines 3       # prompting baselines (Self-Validation BoN/FtA, LogicCheckGPT) on the base
#                                        #   model, same tasks and settings as $LMMS/plan.json, 3 workers (BASELINES.md)
#   bash rl_paper_commands.sh report_all # the table with the baselines' columns added
#   bash rl_paper_commands.sh report_prompting   # -> lmms_eval_baselines/prompting_comparison.md and .html
#   bash rl_paper_commands.sh rescore length/caption,self/general   # score those cells again; generation replays from cache
#   bash rl_paper_commands.sh worker     # one more worker of the current plan (e.g. after one died); own log file
#   Reward comparison (README "Reward comparison"; test-time scaling + GRPO with each reward, same caption benchmarks):
#   bash rl_paper_commands.sh rc_setup     # task files, Long-CLIP code + weights, jina-clip-v2, replay stub (downloads ~4 GB)
#   bash rl_paper_commands.sh rc_quick     # every step on 4 items per task (needs the judge)
#   bash rl_paper_commands.sh rc_pipeline  # everything after that, detached (rc_pipeline.sh): status in logs_rl_paper/rc_pipeline.status
#   bash rl_paper_commands.sh rc_report    # -> tts_out/reward_comparison.md and .html
#   bash rl_paper_commands.sh purge      # delete the data of the dropped benchmarks (AMBER, Self-Aug tasks, their caches)
#   bash rl_paper_commands.sh clean      # afterwards: delete benchmark data caches and temp files (shared disk)
#   nohup bash rl_paper_commands.sh all > rl_paper.log 2>&1 &     # train -> verify -> judge -> eval -> report
#
# Every stage can be re-run: finished parts are skipped, cut-off training resumes. GPUs 0, 1, 2, 3 are never used.
# Shared server: nothing is written into the shared HF cache (read only), downloads go to data2 under
# mchung45/, one task group's benchmark data is on disk at a time, and the judge takes 40% of one GPU.
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------- paths and GPUs (check these)
STEP1=out_7b                                # Step 1 folder (means/, selection.json)
SELF=out_grpo                               # the self-reward GRPO run already trained (read only)
EXT=out_grpo_external                       # new: outside embedding reward run
LEN=out_grpo_length                         # new: length-matched control run
BIG=/mnt/data2/mchung45/argmax_caption_work/outputs/rl_paper   # LoRA checkpoints + merge check reports (small)
LMMS=lmms_eval_out                          # lmms-eval results
BASELINE_SKIP_TASKS=pope,mmvp               # not in the lists prompting_report.py compares; empty = run every task of the plan
BASELINES=lmms_eval_baselines               # prompting baselines' results (own plan, settings from $LMMS/plan.json)
EVAL_WORK=/mnt/data1/mchung45/lmms_eval_work       # benchmark data + caches of the evaluation (data1: data2 is full; ~10 GB at a time)
CUSTOM_TASKS=/mnt/data2/mchung45/lmms_eval_work/custom_tasks   # DOCCI caption task folder (prepare_tasks.py; small)
DOCCI_EVAL_N=1000                           # DOCCI test images for docci_caption (seed 0 sample; 0 = all 5,000, ~1.2 GB of images)
JUDGE_MODEL=Qwen/Qwen2.5-14B-Instruct       # judge for MM-Vet (~30 GB download)
JUDGE_GPUS=auto                             # auto = one free GPU among 4,5,6,7 (waits for one; 0,1,2,3 never); or e.g. 4
JUDGE_GPU_FRACTION=0.40                     # of one 143 GB card: ~57 GB (14B bf16 weights 30 GB + KV cache)
JUDGE_HF_HOME=/mnt/data2/mchung45/docci_qa_work/caches/hf   # judge download goes here if no complete copy is cached
EVAL_GPUS=auto                              # lmms-eval: auto = first free GPU (waits); or e.g. 4,6 = those 2 GPUs (waits until free)
export GPU_FREE_MEM_MB=${GPU_FREE_MEM_MB:-10000}                # a GPU counts as free (judge + eval workers) with at most this much memory in use ...
export GPU_FREE_UTIL=${GPU_FREE_UTIL:-10}     # ... and at most this utilization %. 10 GB: a small idle process on a 143 GB card is fine
EVAL_BATCH=8                                # prompts generated at once (frozen in the plan; the run before 2026-11-03 used 1)
ATTN=auto                                   # auto = flash attention if flash-attn is installed, else sdpa (frozen in the plan; before: sdpa)
export ATTN_IMPL=$ATTN                      # the same choice for grpo.py / export_ckpt.py
LOGS=logs_rl_paper

# big downloads go to data2 too (same places run_all.py uses; an HF_HOME you already set is kept)
export HF_HOME=${HF_HOME:-/mnt/data2/mchung45/docci_qa_work/caches/hf}          # models: jina-clip-v2, judge (~30 GB)
export ARGMAX_CACHE_DIR=${ARGMAX_CACHE_DIR:-/mnt/data2/mchung45/argmax_caption_work/cache}   # grpo.py / run.py cache (their built-in default is the old account's path)
export DOCCI_QA_ROOT=${DOCCI_QA_ROOT:-/mnt/data2/mchung45/docci_qa_work}                   # DOCCI cache of the QA pipeline (prepare_tasks.py reads it)
export HF_DATASETS_CACHE=$EVAL_WORK/hf_datasets                                     # datasets cache of the smoke test / prepare step
export LMMS_EVAL_DATASETS_CACHE=$HF_DATASETS_CACHE   # else lmms-eval moves the cache to local scratch when data2 is a network disk
export VLLM_CACHE_ROOT=/mnt/data2/mchung45/docci_qa_work/caches/vllm
export TMPDIR=/mnt/data2/mchung45/docci_qa_work/caches/tmp
HF_DIR=$EVAL_WORK/hf                              # every download of the evaluation: models/ (kept), datasets_<group>/ (deleted when done)
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}   # lmms-eval tasks need a login (token: True); HF_HOME changes per run, the token does not
export HF_HUB_DISABLE_XET=1                       # plain http downloads (the xet writer fails on network disks / read-only caches)
# this conda Python does not find the system CA list (urllib downloads fail with CERTIFICATE_VERIFY_FAILED)
[ -z "${SSL_CERT_FILE:-}" ] && [ -e /etc/ssl/certs/ca-certificates.crt ] && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_XET_CACHE=$EVAL_WORK/xet
mkdir -p "$HF_DATASETS_CACHE" "$VLLM_CACHE_ROOT" "$TMPDIR" "$BIG" "$LOGS" "$JUDGE_HF_HOME"

# adapters are merged into the base in memory by lmms-eval (lmms_eval_lora.py): no 16 GB model folders on disk
MODELS=base=Qwen/Qwen2.5-VL-7B-Instruct,self=$SELF/adapter,external=$EXT/adapter,length=$LEN/adapter
JUDGE_PORT=8047                             # our own judge; port 8000 on this server is another user's
JUDGE_URL=http://localhost:$JUDGE_PORT/v1

maxtok() { python -c "import json; print(json.load(open('$SELF/args.json')).get('max_new_tokens', 128))"; }

check() {
  for d in "$STEP1/means" "$SELF/adapter" "$SELF/train_log.jsonl" "$SELF/args.json"; do
    [ -e "$d" ] && echo "ok       $d" || echo "MISSING  $d   <- fix STEP1 / SELF at the top of this file"
  done
  echo "self run settings (the new runs use the same, caption limit copied):"
  python -c "import json; a=json.load(open('$SELF/args.json')); print({k: a.get(k) for k in ['model','max_new_tokens','steps','rollouts','batch_images','lr','beta','lora_r','seed','docci_dir','caption_prompt']})"
  echo "caption limit for the new runs: $(maxtok) tokens"
  df -h /mnt/data2 | tail -1 || true
  echo "disk this pipeline needs on data2: ~10 GB (one task group's benchmark data at a time, deleted when done)"
  echo "  +30 GB once if no complete copy of $JUDGE_MODEL is in the HF cache, +16 GB once if Qwen2.5-VL-7B is not there either"
  python lmms_eval_run.py --out_dir "$LMMS" --hf_dir "$HF_DIR" --disk
}

setup() {
  pip install lmms-eval qwen-vl-utils capture_metric pycocoevalcap pycocotools nltk openpyxl
  python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet omw-1.4   # CHAIR
  command -v java >/dev/null || conda install -y -c conda-forge openjdk=11   # METEOR / PTB tokenizer (caption metrics)
  tasks                                                        # the DOCCI caption task folder
  python lmms_eval_smoke.py --out_dir "$TMPDIR/lmms_smoke"      # must end with [smoke] OK
}

tasks() {  # DOCCI caption task for lmms-eval: $DOCCI_EVAL_N test images (images stay in the QA pipeline's DOCCI cache)
  python prepare_tasks.py --out_dir "$CUSTOM_TASKS" --n_images "$DOCCI_EVAL_N"
}

train_one() {  # out_dir ckpt_dir extra-args...
  local out=$1 ckpt=$2; shift 2
  if [ -e "$out/adapter/adapter_config.json" ] && [ -e "$out/results.md" ]; then echo "[train] $out done already"; return; fi
  local resume=(); ls "$ckpt"/step_* >/dev/null 2>&1 && resume=(--resume)
  python grpo.py --step1_dir "$STEP1" --out_dir "$out" --ckpt_dir "$ckpt" --max_new_tokens "$(maxtok)" --gpu_wait ${resume[@]+"${resume[@]}"} "$@"
}

train() {
  local docci=(); d=$(python -c "import json; print(json.load(open('$SELF/args.json')).get('docci_dir') or '')"); [ -n "$d" ] && docci=(--docci_dir "$d")
  train_one "$EXT" "$BIG/ckpt_external" --reward external --external_model jinaai/jina-clip-v2 ${docci[@]+"${docci[@]}"} > "$LOGS/train_external.log" 2>&1 &
  local p1=$!
  echo "[train] external started (log $LOGS/train_external.log); waiting 3 min so the second run picks another GPU"
  sleep 180
  train_one "$LEN" "$BIG/ckpt_length" --reward length --length_target_log "$SELF/train_log.jsonl" ${docci[@]+"${docci[@]}"} > "$LOGS/train_length.log" 2>&1 &
  local p2=$!
  echo "[train] length started (log $LOGS/train_length.log). tail -f $LOGS/train_*.log to watch"
  local rc=0; wait $p1 || rc=1; wait $p2 || rc=1
  [ $rc = 0 ] && echo "[train] both done" || { echo "[train] a run failed: see $LOGS/train_*.log (re-run 'train' to resume)"; return 1; }
}

verify_all() {  # merged vs base + LoRA and the chat template; writes only export.json (lmms-eval merges in memory)
  for pair in "self:$SELF" "external:$EXT" "length:$LEN"; do
    local name=${pair%%:*} run=${pair#*:}
    if [ -e "$BIG/merge_checks/$name/export.json" ]; then echo "[verify] $name done already"; continue; fi
    python export_ckpt.py --check_only --adapter "$run/adapter" --grpo_dir "$run" --out "$BIG/merge_checks/$name" --gpu_wait
  done
}

judge_weights() {  # prints a folder with all of $JUDGE_MODEL's files: a complete copy already in the HF cache
  # (the shared one included, read only), else downloaded into $JUDGE_HF_HOME (the shared cache may not be writable)
  python - "$JUDGE_MODEL" "$JUDGE_HF_HOME" <<'EOF'
import json, os, sys
from huggingface_hub import snapshot_download
repo, own = sys.argv[1], os.path.join(sys.argv[2], "hub")
def complete(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    weights = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
    return all(os.path.exists(os.path.join(d, f)) for f in weights | {"config.json", "tokenizer_config.json"})
caches = [os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"), own]
for c in dict.fromkeys(caches):
    try:
        d = snapshot_download(repo, cache_dir=c, local_files_only=True)
    except Exception:
        continue
    if complete(d):
        print(f"[judge] using the copy in {c}", file=sys.stderr); print(d); sys.exit(0)
print(f"[judge] downloading {repo} into {own}", file=sys.stderr)
d = snapshot_download(repo, cache_dir=own, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
if not complete(d):
    sys.exit(f"[judge] download incomplete: {d}")
print(d)
EOF
}

judge() {
  if curl -sf "$JUDGE_URL/models" >/dev/null; then echo "[judge] already running at $JUDGE_URL"; return; fi
  local free; free=$(df --output=avail -BG "$JUDGE_HF_HOME" | tail -1 | tr -dc 0-9)
  echo "[judge] ${free} GB free on $JUDGE_HF_HOME (a new download of $JUDGE_MODEL needs ~30 GB)"
  local weights; weights=$(judge_weights | tail -1) || { echo "[judge] could not get $JUDGE_MODEL"; return 1; }
  [ -d "$weights" ] || { echo "[judge] could not get $JUDGE_MODEL"; return 1; }
  echo "[judge] weights: $weights"
  # wait until a judge GPU is free (other people's jobs share this server)
  if [ "$JUDGE_GPUS" = auto ]; then  # one free GPU (<=1 GB used, <=10% busy) among the ones we may use
    JUDGE_GPUS=$(python -c "import gpu; print(gpu.pick(gpu.candidates('auto'), wait=True))" | tee /dev/stderr | tail -1)
    [ "$JUDGE_GPUS" -ge 0 ] 2>/dev/null || { echo "[judge] could not pick a GPU"; return 1; }
  else
    while nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$JUDGE_GPUS" \
          | awk -F', ' '$2 > 1000 {busy=1} END {exit !busy}'; do
      echo "[judge] GPUs $JUDGE_GPUS busy, checking again in 60 s:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$JUDGE_GPUS"
      sleep 60
    done
  fi
  # a shared GPU: take only what a 14B judge needs (~30 GB weights + KV cache), not vLLM's default 90% of the card
  CUDA_VISIBLE_DEVICES=$JUDGE_GPUS nohup vllm serve "$weights" --served-model-name "$JUDGE_MODEL" \
    --tensor-parallel-size $(( $(echo "$JUDGE_GPUS" | tr -cd , | wc -c) + 1 )) --port "$JUDGE_PORT" \
    --gpu-memory-utilization "$JUDGE_GPU_FRACTION" --max-model-len 8192 --max-num-seqs 32 > "$LOGS/judge.log" 2>&1 &
  local pid=$!
  echo "[judge] starting on GPUs $JUDGE_GPUS (pid $pid, log $LOGS/judge.log); stop it later with: kill $pid"
  for i in $(seq 1 60); do  # weights are local already: loading takes minutes
    curl -sf "$JUDGE_URL/models" >/dev/null && { echo "[judge] ready"; return; }
    kill -0 $pid 2>/dev/null || { echo "[judge] vLLM exited; last lines of $LOGS/judge.log:"; tail -20 "$LOGS/judge.log"; return 1; }
    sleep 30
  done
  echo "[judge] not ready after 30 min: see $LOGS/judge.log"; return 1
}

# the reward comparison's caption group (no length cap); in every call, so the frozen plan keeps it once it is added
CAPTION_LONG=captionqa_x_natural,captionqa_x_document,captionqa_x_ecommerce,captionqa_x_embodiedai,amber_g,chair500
CAPTION_LONG_TOKENS=4096
export AMBER_BASE_DIR=$CUSTOM_TASKS/amber   # amber_g reads $AMBER_BASE_DIR/metadata (prepare_tts_tasks.py)
EVAL_ARGS=(--models "$MODELS" --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL" --include_path "$CUSTOM_TASKS" --hf_dir "$HF_DIR" --batch_size "$EVAL_BATCH" --attn "$ATTN"
           --caption_long "$CAPTION_LONG" --caption_long_max_new_tokens "$CAPTION_LONG_TOKENS")

eval_quick() {  # the same tasks on 4 items with the base model (~15 min): every scorer runs, so a broken one fails here, not after hours
  python lmms_eval_run.py --out_dir "${LMMS}_quick" "${EVAL_ARGS[@]}" --limit 4 --only base --keep_datasets --gpus "$EVAL_GPUS"
}

eval_all() {
  python lmms_eval_run.py --out_dir "$LMMS" "${EVAL_ARGS[@]}" --dry_run
  eval_quick
  python lmms_eval_run.py --out_dir "$LMMS" "${EVAL_ARGS[@]}" --gpus "$EVAL_GPUS"
}

par() {  # N workers of the same plan, each on its own free GPU (waits for one); cells already done or being run
         # by another worker are skipped. Needs the judge running. `par N replan` first replaces the frozen plan
         # with the current settings (EVAL_BATCH, ATTN, tasks); the old plan and results go to _old_<time>_*.
         # `par N drop oeq,selfaug` first removes those groups from the frozen plan (results kept in _old_<time>/) and
         # adds the current groups; tasks they share replay from the dropped cells' caches (no generation again).
  local n=${1:-2} first=()
  if [ "${2:-}" = replan ]; then
    first=(--replan)
    echo "[par] worker 1 replaces the frozen plan (old results -> $LMMS/_old_*), then evaluates"
  elif [ "${2:-}" = drop ]; then
    [ -n "${3:-}" ] || { echo "usage: bash rl_paper_commands.sh par N drop group[,group...]"; return 1; }
    [ -e "$CUSTOM_TASKS/docci_caption.yaml" ] || tasks
    first=(--drop_groups "$3")
    echo "[par] worker 1 drops group(s) $3 from the frozen plan (results -> $LMMS/_old_*), adds the current ones, then evaluates"
  fi
  for i in $(seq 1 "$n"); do
    nohup python lmms_eval_run.py --out_dir "$LMMS" "${EVAL_ARGS[@]}" --gpus auto ${first[@]+"${first[@]}"} > "$LOGS/eval_worker_$i.log" 2>&1 &
    local pid=$!
    echo "[par] worker $i started (pid $pid, log $LOGS/eval_worker_$i.log)"
    first=()
    if [ "$i" -lt "$n" ]; then
      if [ -n "${2:-}" ] && [ "$i" = 1 ]; then  # replan / drop: the others must see the new plan, not the old one
        # worker 1 prints "[plan] frozen:" right after the plan check + preflight (before it waits for a GPU)
        local ok=0
        for t in $(seq 1 60); do
          sleep 5
          if grep -q "\[plan\] frozen:" "$LOGS/eval_worker_1.log" 2>/dev/null; then ok=1; break; fi
          kill -0 "$pid" 2>/dev/null || break   # worker 1 already exited (preflight problem, plan error, ...)
        done
        [ "$ok" = 1 ] || { echo "[par] worker 1 did not freeze the new plan; nothing else started. Its log:"; tail -15 "$LOGS/eval_worker_1.log"; return 1; }
        echo "[par] new plan frozen: $(python -c "import json; p=json.load(open('$LMMS/plan.json')); print('batch', p['batch_size'], '|', p['model_args'])")"
      fi
      echo "[par] waiting 4 min so the next worker picks another GPU"; sleep 240
    fi
  done
  echo "[par] when every cell is done: bash rl_paper_commands.sh report"
}

worker() {  # one more worker of the current plan (e.g. after a worker died): same settings, skips cells done or
            # running elsewhere, own log file. Needs the judge running.
  local log="$LOGS/eval_worker_$(date +%m%d-%H%M%S).log"
  nohup python lmms_eval_run.py --out_dir "$LMMS" "${EVAL_ARGS[@]}" --gpus auto > "$log" 2>&1 &
  echo "[worker] started (pid $!): tail -f $log"
}

rescore() {  # `rescore length/caption,self/general`: score those cells again (generation replays from the cache).
             # Rewrites the DOCCI scorer first (scorer fixes). Needs the judge running for general (MM-Vet).
  [ -n "${1:-}" ] || { echo "usage: bash rl_paper_commands.sh rescore model/group[,model/group...]"; return 1; }
  python prepare_tasks.py --out_dir "$CUSTOM_TASKS" --utils_only
  local log="$LOGS/eval_rescore_$(date +%m%d-%H%M%S).log"
  nohup python lmms_eval_run.py --out_dir "$LMMS" "${EVAL_ARGS[@]}" --gpus auto --rescore "$1" > "$log" 2>&1 &
  echo "[rescore] started (pid $!): tail -f $log"
}

report() { python lmms_eval_report.py --out_dir "$LMMS"; }

# prompting baselines from other papers, on the base model (prompting_baselines.py, BASELINES.md); the text LLM
# of LogicCheckGPT is the judge server, so start it first (bash rl_paper_commands.sh judge)
BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
BL_ARGS=(--settings_from "$LMMS/plan.json" --models "selfval_bon=$BASE_MODEL,selfval_fta=$BASE_MODEL,logiccheck=$BASE_MODEL"
         --prompting selfval_bon=selfval_bon,selfval_fta=selfval_fta,logiccheck=logiccheck
         --skip_tasks "$BASELINE_SKIP_TASKS"
         --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL" --hf_dir "$HF_DIR" --include_path "$CUSTOM_TASKS")

baselines_quick() {  # 4 items per task, every baseline cell: scorers, the text LLM and the traces are checked in minutes
  [ -e "$LMMS/plan.json" ] || { echo "[baselines] $LMMS/plan.json missing: the checkpoints' plan sets the tasks and settings"; return 1; }
  python lmms_eval_run.py --out_dir "${BASELINES}_quick" "${BL_ARGS[@]}" --limit 4 --keep_datasets --gpus "$EVAL_GPUS"
  python lmms_eval_report.py --out_dir "${BASELINES}_quick"
  python prompting_report.py --checkpoints "$LMMS" --baselines "${BASELINES}_quick"
}

baselines() {  # N workers sharing the baseline cells (like par): selfval_* run the caption group only
  [ -e "$LMMS/plan.json" ] || { echo "[baselines] $LMMS/plan.json missing: the checkpoints' plan sets the tasks and settings"; return 1; }
  local n=${1:-1}
  python lmms_eval_run.py --out_dir "$BASELINES" "${BL_ARGS[@]}" --dry_run
  for i in $(seq 1 "$n"); do  # BASELINE_GPUS=5 pins a worker to a GPU; BASELINE_WORKER_LOG names its log
    local log=${BASELINE_WORKER_LOG:-$LOGS/baselines_worker_$i.log}
    nohup python lmms_eval_run.py --out_dir "$BASELINES" "${BL_ARGS[@]}" --gpus "${BASELINE_GPUS:-auto}" > "$log" 2>&1 &
    echo "[baselines] worker $i started (pid $!, log $log)"
    [ "$i" -lt "$n" ] && { echo "[baselines] waiting 4 min so the next worker picks another GPU"; sleep 240; }
  done
  echo "[baselines] when every cell is done: bash rl_paper_commands.sh report_prompting"
}

baselines_selfval_rerun() {  # Self-Validation cells again after the LPFV fix of 2026-11-10 (BASELINES.md): fresh
                             # generation (same seeds), old results moved aside first
  local stamp; stamp=$(date +%Y%m%d-%H%M%S)
  for m in selfval_bon selfval_fta; do
    [ -d "$BASELINES/$m/caption" ] && mkdir -p "$BASELINES/_old_lpfv_$stamp" && mv "$BASELINES/$m/caption" "$BASELINES/_old_lpfv_$stamp/${m}_caption"
  done
  echo "[baselines] old Self-Validation cells -> $BASELINES/_old_lpfv_$stamp (kept)"
  nohup python lmms_eval_run.py --out_dir "$BASELINES" "${BL_ARGS[@]}" --only selfval_bon,selfval_fta --gpus auto > "$LOGS/baselines_selfval_rerun.log" 2>&1 &
  echo "[baselines] started (pid $!, log $LOGS/baselines_selfval_rerun.log); then: bash rl_paper_commands.sh report_prompting"
}

report_all() { python lmms_eval_report.py --out_dir "$LMMS" --with "$BASELINES"; }

report_prompting() { python prompting_report.py --checkpoints "$LMMS" --baselines "$BASELINES"; }

# ---------------------------------------------------------------- reward comparison (README "Reward comparison")
RND=out_grpo_random                         # GRPO, uniform random reward (no-signal control)
SCERT=out_grpo_selfcert                     # GRPO, the policy's self-certainty (Intuitor)
LCLIP=out_grpo_longclip                     # GRPO, Long-CLIP-L cosine
export LONGCLIP_REPO=${LONGCLIP_REPO:-/mnt/data2/mchung45/Long-CLIP}           # github.com/beichenzbc/Long-CLIP clone
export LONGCLIP_CKPT=${LONGCLIP_CKPT:-$LONGCLIP_REPO/checkpoints/longclip-L.pt}   # BeichenZhang/LongCLIP-L
LONGCLIP_COMMIT=3966af9ae9331666309a22128468b734db4672a7
OLD_MODELS=$MODELS                                                     # the four runs evaluated already
NEW_MODELS=random=$RND/adapter,selfcert=$SCERT/adapter,longclip=$LCLIP/adapter
TTS_LMMS=lmms_eval_tts                      # lmms-eval replay cells of the test-time scaling
TTS=tts_out                                 # items, candidates, scores, picks, own answers, the report
REPLAY_STUB=$BIG/replay_stub                # tiny random Qwen2.5-VL: a replay cell's model (loaded on the CPU, never used)
MERGED_SELF=$BIG/merged_self                # the self GRPO run merged into a model folder for vLLM (~16 GB, data2)
TTS_ARGS=(--settings_from "$LMMS/plan.json" --settings_groups caption_long --caption_long "$CAPTION_LONG"
          --caption_long_max_new_tokens "$CAPTION_LONG_TOKENS" --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL"
          --hf_dir "$HF_DIR" --include_path "$CUSTOM_TASKS" --keep_datasets)
export LMMS_HF_DIR=$HF_DIR                  # tts.py finds the local Qwen2.5-VL copy there
export CAPTIONQA_JUDGE_WORKERS=${CAPTIONQA_JUDGE_WORKERS:-32}   # CaptionQA: judge calls of one image in parallel
GPU_ARGS=(--gpu_wait --gpu_max_mem_mb "$GPU_FREE_MEM_MB")
VLLM_GPU_MB=${VLLM_GPU_MB:-60000}           # a GPU for a vLLM sampler: at most this much in use (it takes 40% of 143 GB)
SCORE_SHARDS=${SCORE_SHARDS:-2}             # tts.py score processes (GPUs) for the base model's candidates

base_snapshot() { python -c "import tts; print(tts.resolve_model('$BASE_MODEL'))"; }

rc_setup() {  # task files (CaptionQA copy, CHAIR-500 + its 427 missing COCO images, AMBER metadata), Long-CLIP code +
              # weights, jina-clip-v2, the replay stub
  set -e  # any failed part fails the setup (rc_pipeline.sh runs it without the script's own set -e)
  python prepare_tts_tasks.py --out_dir "$CUSTOM_TASKS"
  for f in chair500.yaml chair500/chair.pkl amber/metadata/annotations.json captionqa_x_natural.yaml; do
    [ -e "$CUSTOM_TASKS/$f" ] || { echo "[rc_setup] $CUSTOM_TASKS/$f missing"; return 1; }
  done
  if [ ! -e "$LONGCLIP_REPO/model/longclip.py" ]; then
    git clone https://github.com/beichenzbc/Long-CLIP "$LONGCLIP_REPO"
  fi
  git -C "$LONGCLIP_REPO" checkout -q "$LONGCLIP_COMMIT"
  [ -e "$LONGCLIP_CKPT" ] || python -c "from huggingface_hub import hf_hub_download as d; print(d('BeichenZhang/LongCLIP-L', 'longclip-L.pt', local_dir='$(dirname "$LONGCLIP_CKPT")'))"
  python - <<'EOF'   # Long-CLIP and jina-clip-v2 load and rank a matching caption above a wrong one (CPU)
import os
from PIL import Image, ImageDraw
import rewards as R
im = Image.new("RGB", (224, 224), "white"); ImageDraw.Draw(im).ellipse((40, 40, 184, 184), fill="red")
for r in (R.LongCLIPReward(os.environ["LONGCLIP_REPO"], os.environ["LONGCLIP_CKPT"], "cpu"), R.ExternalReward("jinaai/jina-clip-v2", "cpu")):
    s = r("x", im, ["a red circle on a white background", "a photo of a dog on grass"])
    print(f"[rc_setup] {r.name} ok: {s}" if s[0] > s[1] else f"[rc_setup] WARNING {r.name} ranks the wrong caption higher: {s}")
EOF
  [ -e "$REPLAY_STUB/config.json" ] || CUDA_VISIBLE_DEVICES="" python export_ckpt.py --mock --dtype bfloat16 --out "$REPLAY_STUB"
}

rc_train() {  # the three new GRPO runs, same config as the self run (train_one: 128-token rollouts, same seed / steps)
  local docci=(); d=$(python -c "import json; print(json.load(open('$SELF/args.json')).get('docci_dir') or '')"); [ -n "$d" ] && docci=(--docci_dir "$d")
  local pids=() r
  for r in random selfcert longclip; do
    local out=$RND; [ $r = selfcert ] && out=$SCERT; [ $r = longclip ] && out=$LCLIP
    train_one "$out" "$BIG/ckpt_$r" --reward $r --gpu_max_mem_mb "$GPU_FREE_MEM_MB" ${docci[@]+"${docci[@]}"} > "$LOGS/train_$r.log" 2>&1 &
    pids+=($!)
    echo "[rc_train] $r started (pid $!, log $LOGS/train_$r.log)"
    [ $r != longclip ] && { echo "[rc_train] waiting 3 min so the next run picks another GPU"; sleep 180; }
  done
  local rc=0 p; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc = 0 ] && echo "[rc_train] all done" || { echo "[rc_train] a run failed: see $LOGS/train_*.log (run rc_train again to resume)"; return 1; }
}

rc_eval_workers() {  # models n: Use 2 cells (caption_long group), n workers, waits for them
  local models=$1 n=${2:-1} i pids=() rc=0 p
  python lmms_eval_run.py --out_dir "$LMMS" "${EVAL_ARGS[@]}" --models "$models" --run_groups caption_long --dry_run > /dev/null
  for i in $(seq 1 "$n"); do
    python lmms_eval_run.py --out_dir "$LMMS" "${EVAL_ARGS[@]}" --models "$models" --run_groups caption_long \
      --gpus auto > "$LOGS/rc_eval_$(date +%m%d-%H%M%S)_$i.log" 2>&1 &
    pids+=($!); echo "[rc_eval] worker $i (pid $!)"
    [ "$i" -lt "$n" ] && sleep 240
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

tts_dump() {  # items_dir [limit]: every request of the caption_long tasks, built by lmms-eval's task code (CPU)
  local lim=(); [ -n "${2:-}" ] && lim=(--limit "$2")
  python tts_vllm.py dump --tasks "$CAPTION_LONG" --include_path "$CUSTOM_TASKS" --out_dir "$1" --hf_dir "$HF_DIR" \
    --max_new_tokens "$CAPTION_LONG_TOKENS" ${lim[@]+"${lim[@]}"}
}

tts_merge_self() {  # the self GRPO adapter merged into a model folder vLLM can load (export_ckpt.py, with its checks)
  [ -e "$MERGED_SELF/export.json" ] && { echo "[tts] $MERGED_SELF exists"; return; }
  python export_ckpt.py --adapter "$SELF/adapter" --grpo_dir "$SELF" --out "$MERGED_SELF" --gpu_wait --gpu_max_mem_mb "$GPU_FREE_MEM_MB"
}

tts_sample_vllm() {  # items_dir model out_dir [limit]: 16 candidates per item with vLLM
  local lim=(); [ -n "${4:-}" ] && lim=(--limit "$4")
  python tts_vllm.py sample --items "$1" --model "$2" --out_dir "$3" --gpu_wait --gpu_max_mem_mb "$VLLM_GPU_MB" \
    --gpu_fraction 0.40 ${lim[@]+"${lim[@]}"}
}

tts_score_base() {  # samples_dir tts_dir: every selector, $SCORE_SHARDS processes on their own GPUs
  local i pids=() rc=0 p
  for i in $(seq 0 $((SCORE_SHARDS - 1))); do
    python tts.py score --sample_dir "$1" --out "$2/scores.shard$i.jsonl" --shard "$i" --num_shards "$SCORE_SHARDS" \
      "${GPU_ARGS[@]}" --step1_dir "$STEP1" > "$LOGS/tts_score_$(basename "$2")_$i.log" 2>&1 &
    pids+=($!); echo "[tts_score] shard $i (pid $!)"
    [ "$i" -lt $((SCORE_SHARDS - 1)) ] && sleep 180
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

tts_vqa_models() {  # samples_dir tts_dir name=path...: each model's own CaptionQA answers, all at once (one GPU each)
  local sd=$1 td=$2 pair pids=() rc=0 p; shift 2
  for pair in "$@"; do
    local name=${pair%%=*} path=${pair#*=} ad=()
    [ -e "$td/vqa/$name.jsonl" ] && { echo "[tts_vqa] $name done already"; continue; }
    [ "$path" != "$BASE_MODEL" ] && ad=(--adapter "$path")
    python tts.py vqa --sample_dir "$sd" --out "$td/vqa/$name.jsonl" --tasks_dir "$CUSTOM_TASKS" ${ad[@]+"${ad[@]}"} \
      "${GPU_ARGS[@]}" > "$LOGS/tts_vqa_$name.log" 2>&1 &
    pids+=($!); echo "[tts_vqa] $name (pid $!)"; sleep 120
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

tts_replay_args() {  # tts_dir ns -> REPLAY_MODELS / REPLAY_PROMPTING / REPLAY_ARGS (the stub as the model)
  local td=$1 ns=${2:-} pre sels
  REPLAY_MODELS=""; REPLAY_PROMPTING=""; REPLAY_ARGS="{}"
  for pre in base selfgrpo; do
    [ -d "$td/$pre/selected" ] || continue
    sels=""; [ $pre = selfgrpo ] && sels=self,first
    local m p j
    m=$(python tts.py args --selected_dir "$td/$pre/selected" --prefix $pre --model "$REPLAY_STUB" --selectors "$sels" --ns "$ns" --field models)
    p=$(python tts.py args --selected_dir "$td/$pre/selected" --prefix $pre --selectors "$sels" --ns "$ns" --field prompting)
    j=$(python tts.py args --selected_dir "$td/$pre/selected" --prefix $pre --selectors "$sels" --ns "$ns" --field prompting_args)
    REPLAY_MODELS+=${REPLAY_MODELS:+,}$m; REPLAY_PROMPTING+=${REPLAY_PROMPTING:+,}$p
    REPLAY_ARGS=$(python -c "import json,sys; a=json.loads(sys.argv[1]); a.update(json.loads(sys.argv[2])); print(json.dumps(a))" "$REPLAY_ARGS" "$j")
  done
}

tts_replay_workers() {  # lmms_dir tts_dir n [ns] [limit]: the picks scored by lmms-eval on the CPU, n workers at once
  local lm=$1 td=$2 n=$3 ns=${4:-} i pids=() rc=0 p lim=()
  [ -n "${5:-}" ] && lim=(--limit "$5")
  tts_replay_args "$td" "$ns"
  python lmms_eval_run.py --out_dir "$lm" "${TTS_ARGS[@]}" --models "$REPLAY_MODELS" --prompting "$REPLAY_PROMPTING" \
    --prompting_args "$REPLAY_ARGS" ${lim[@]+"${lim[@]}"} --gpus cpu --dry_run > /dev/null
  for i in $(seq 1 "$n"); do
    python lmms_eval_run.py --out_dir "$lm" "${TTS_ARGS[@]}" --models "$REPLAY_MODELS" --prompting "$REPLAY_PROMPTING" \
      --prompting_args "$REPLAY_ARGS" ${lim[@]+"${lim[@]}"} --gpus cpu > "$LOGS/tts_replay_$(basename "$lm")_$i.log" 2>&1 &
    pids+=($!); echo "[tts_replay] worker $i (pid $!)"; sleep 20
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

rc_report_dirs() {  # lmms_tts checkpoints tts_dir
  python tts_report.py --tts "$1" --checkpoints "$2" --vqa_dir "$3/vqa" --tasks_dir "$CUSTOM_TASKS" --out_stem "$3/reward_comparison"
}

rc_quick() {  # every step on 4 items per task, base model only, in *_quick folders: breaks here, not after hours.
              # Each command is checked (rc_pipeline.sh calls this inside `||`, where bash ignores set -e)
  local lm=${TTS_LMMS}_quick td=${TTS}_quick
  python lmms_eval_run.py --out_dir "${LMMS}_quick_rc" "${EVAL_ARGS[@]}" --limit 4 --only base --run_groups caption_long \
    --keep_datasets --gpus "$EVAL_GPUS" || return 1
  tts_dump "$td/items" 4 || return 1
  tts_sample_vllm "$td/items" "$(base_snapshot)" "$td/samples/base" || return 1
  python tts.py score --sample_dir "$td/samples/base" --out "$td/base/scores.jsonl" "${GPU_ARGS[@]}" --step1_dir "$STEP1" || return 1
  python tts.py select --sample_dir "$td/samples/base" --scores "$td/base/scores.jsonl" --out_dir "$td/base/selected" || return 1
  python tts.py vqa --sample_dir "$td/samples/base" --out "$td/vqa/base.jsonl" --tasks_dir "$CUSTOM_TASKS" "${GPU_ARGS[@]}" || return 1
  tts_replay_workers "$lm" "$td" 3 1,4 4 || return 1   # --limit 4 too: the picks hold 4 items per task
  rc_report_dirs "$lm" "${LMMS}_quick_rc" "$td"
}

purge() {  # the benchmarks dropped on 2026-11-06 (AMBER-G, CaptionQA, DetailCaps, LLaVA-Bench, MMHal, POPE A-OKVQA):
           # their task folders, metadata and leftover data caches. Results already in $LMMS/_old_* stay. DOCCI is not touched.
  local gone=(/mnt/data2/mchung45/amber /mnt/data2/mchung45/AMBER_repo /mnt/data2/mchung45/lmms_eval_work/selfaug_tasks
              "$HF_DIR/datasets_oeq" "$HF_DIR/datasets_selfaug" "$EVAL_WORK/hf_datasets")
  du -sh "${gone[@]}" 2>/dev/null || true
  rm -rf "${gone[@]}"
  echo "[purge] removed what existed of: ${gone[*]}"
}

clean() {  # free the disk: benchmark data caches, temp files, the vLLM cache; results and the judge weights stay
  python lmms_eval_run.py --out_dir "$LMMS" --hf_dir "$HF_DIR" --disk
  rm -rf "$HF_DIR"/datasets_* "$HF_XET_CACHE" "$TMPDIR"/* "$VLLM_CACHE_ROOT"
  echo "[clean] removed benchmark data caches, temp files and the vLLM cache. Still here (delete by hand if not needed):"
  du -sh "$HF_DIR/models" "$JUDGE_HF_HOME" "$BIG" "$LMMS" 2>/dev/null || true
}

[ "${RL_COMMANDS_SOURCED:-0}" = 1 ] && return 0   # rc_pipeline.sh: the settings and functions only

case "${1:-}" in
  check) check ;;
  setup) setup ;;
  train) train ;;
  verify|export) verify_all ;;
  judge) judge ;;
  quick) eval_quick ;;
  eval) eval_all ;;
  tasks) tasks ;;
  par) par "${2:-2}" "${3:-}" "${4:-}" ;;
  worker) worker ;;
  rescore) rescore "${2:-}" ;;
  report) report ;;
  baselines_quick) baselines_quick ;;
  baselines) baselines "${2:-1}" ;;
  baselines_selfval_rerun) baselines_selfval_rerun ;;
  report_all) report_all ;;
  report_prompting) report_prompting ;;
  rc_setup) rc_setup ;;
  rc_train) rc_train ;;
  rc_quick) rc_quick ;;
  rc_pipeline) setsid nohup bash rc_pipeline.sh > "$LOGS/rc_pipeline.log" 2>&1 < /dev/null &
               echo "[rc] pipeline started (pid $!): tail -F $LOGS/rc_pipeline.status" ;;
  rc_report) rc_report_dirs "$TTS_LMMS" "$LMMS" "$TTS" ;;
  purge) purge ;;
  clean) clean ;;
  all) check; train; verify_all; judge; eval_all; report ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
