import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Any, Dict

import pandas as pd
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn



def get_prompt_template(prompt_path):
    """
    give some dataname and get the prompt template
    
    support prompt templates:
        alpaca_sft.prompt  
        question_only.prompt  
        r1_zero.prompt  
        zero_shot_system_prompt.prompt
    """
    # add check of prompt template
    return Path(prompt_path).read_text()

def read_jsonline(data_path: str):
    """
    read jsonl which each line has querstion and answer
    """
    data = [json.loads(line) for line in open(data_path, "r")]
    return data

def write_jsonline(data: List[Dict[str, Any]], data_path: str, append: bool = False):
    """
    将数据列表按行写入 JSONL 文件。
    
    参数
    ----
    data : List[Dict[str, Any]]
        要写入的 Python 对象列表（每个元素必须是可 JSON 序列化的 dict）。
    data_path : str
        输出文件路径，建议以 .jsonl 结尾。
    append : bool, 默认 False
        如果为 True，则在文件末尾追加；否则覆盖原文件。
    """
    mode = "a" if append else "w"
    with open(data_path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
   
def call_llm_with_vllm(
    vllm_model: LLM,
    prompts: List[str],
    eval_sampling_params: SamplingParams,
):
    outputs = vllm_model.generate(
        prompts,
        sampling_params=eval_sampling_params,
    )

    restults = []
    for output in outputs:
        restults.append(
            {
                "response": output.outputs[0].text,
                "prompt": output.prompt
            }
        )
    return restults
        
    
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    dataset: list[dict[str, str]],
    eval_sampling_params: SamplingParams
) ->pd.DataFrame:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and save them to a file.
    """
    prompts = [rec['prompt'] for rec in dataset]
    results = call_llm_with_vllm(vllm_model, prompts, eval_sampling_params)

    records = []
    for result in results:
        for data in dataset:
            if data['prompt'] == result['prompt']:
                record = {
                    "question": data['question'],
                    "answer": data['answer'],
                    "response": result['response']
                }
                record.update(reward_fn(result['response'], data['answer']))
                records.append(record)
    
    df_metrics = pd.DataFrame(records) # format_rewards/answer_reward/reward 
    print(f"head of df_metrics: {df_metrics.head()}")
    print(f"""
          format_rewards: {df_metrics['format_reward'].mean()}
          answer_rewards: {df_metrics['answer_reward'].mean()}
          rewards: {df_metrics['reward'].mean()}
          """)
    return df_metrics

def main():
    """"""     
    args = parse_args()
    llm = LLM(args.model_path, gpu_memory_utilization=args.gpu_memory_utilization)

    # load prompt
    prompts_template = get_prompt_template(args.prompt_template)
    datasets = read_jsonline(args.dataset_path)
    for dataset in datasets:
        dataset['prompt'] = prompts_template.format(question=dataset['question'])

    eval_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        stop=["</answer>"],
        repetition_penalty=1.0,
        max_tokens=1024,
        include_stop_str_in_output=True
    ) 

    df_metrics = evaluate_vllm(llm, r1_zero_reward_fn, datasets, eval_sampling_params)
   
    # 1. 取模型文件夹名作为模型名
    model_name = Path(args.model_path).name           # e.g. "llama-7b-chat"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"model_metrics_{model_name}_{timestamp}.csv"
    df_metrics.to_csv(out_file, index=False)
    print(f"Metrics saved to {out_file}") 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a math reasoning dataset."
    )

    # dataset_path
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/gsm8k/test.jsonl",
        help="Path to the .jsonl dataset file (each line: {'question':..., 'answer':...})."
    )

    # model_path
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/Qwen2.5-Math-1.5B",
        help="Local directory or HuggingFace repo name of the model to evaluate."
    )

    # prompt_template
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="cs336_alignment/prompts/r1_zero.prompt",
        choices=[
            "cs336_alignment/prompts/alpaca_sft.prompt",
            "cs336_alignment/prompts/question_only.prompt",
            "cs336_alignment/prompts/r1_zero.prompt",
            "cs336_alignment/prompts/zero_shot_system_prompt.prompt"
        ],
        help="Which prompt template file to use."
    )

    # 其余可选参数（示例，可按需增删）
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation."
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.2,
        help="Sampling temperature for generation."
    )
    args = parser.parse_args()

    # 简单校验文件存在
    if not Path(args.dataset_path).exists():
        parser.error(f"dataset_path not found: {args.dataset_path}")
    if not Path(args.model_path).exists():
        parser.error(f"model_path not found: {args.model_path}")

    return args

if __name__ == "__main__":
    main()