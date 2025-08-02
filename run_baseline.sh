CUDA_VISIBLE_DEVICES=0  uv run python -m code.eval_models \
 --model_path ./model/Qwen2.5-Math-1.5B \
 --dataset_path data/gsm8k/test.jsonl \
 --prompt_template cs336_alignment/prompts/r1_zero.prompt \
 --gpu_memory_utilization 0.2
