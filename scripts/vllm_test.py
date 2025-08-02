from vllm import LLM, SamplingParams
model_path = "/storage/models/Qwen/Qwen2.5-Math-1.5B-Instruct" 
# sample prompts
prompts = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "hello",
    "The feature of ai is "
]

# create a sampling parameters object, stopping generation on newline.
sampling_params = SamplingParams(
    max_tokens=50,
    stop=["\n"],
    temperature=0.7,
    top_p=0.9
)
# create an LLM object
llm = LLM(model=model_path, gpu_memory_utilization=0.5)

outputs = llm.generate(prompts, sampling_params=sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}")
    print("-" * 40)

