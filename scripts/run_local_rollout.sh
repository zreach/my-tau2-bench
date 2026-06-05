cd /Users/bytedance/Desktop/code/my-tau2-bench

python scripts/run_local_llm_rollouts.py \
  --domain mock \
  --agent-model Qwen/Qwen2.5-7B-Instruct \
  --agent-api-base http://127.0.0.1:8000/v1 \
  --user-model Qwen/Qwen2.5-7B-Instruct \
  --user-api-base http://127.0.0.1:8001/v1 \
  --num-tasks 2 \
  --num-trials 1 \
  --output-dir data/local_rollouts/demo