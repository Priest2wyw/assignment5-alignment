# math_baseline.md

## 1. evaluate_vllm inplementation
The `evaluate_vllm` function in the `code/eval_models.py` script is used to evaluate the performance of the model.

```shell
./run_baseline.sh
```
## 2. evaluate result of qwen2.5_math_1.5b
result is:
```shell
    format_rewards: 0.1956027293404094
    answer_rewards: 0.0
    rewards: 0.0
```
the answer of three questions are:
(1) correct with both format and answer reward 1 
A: **0** 
(2) format reward1 and answer reward 0 has:
A: **258**
(3) format reward 0 and answer reward 0:
A：**1061**
(4)Observing at least 10 caseswhere format reward is 0, do you think the issue is with the base model’s output, or the parser?Why?
What about in (at least 10) cases where format reward is 1 but answer reward is 0?

| format\_reward | parse\_error | output\_error | total    |
| -------------- | ----- | ------ | ------ |
| 0              | 5     | 13     | 18     |
| 1              | 4     | 6      | 10     |
| **total**         | **9** | **19** | **28** |

format reward=0，我看了18个，其中5个答案是正确的，但是没有跟随<think></think>标签，所以没有正确,达到了 27.78%的比例。其余的13个答案，没算对也没有按照返回格式进行回答。
format reward=1，我看了10个，其中4个答案是正确的,达到了 40% 的比例。其余的6个答案，没算对也没有按照返回格式进行回答。

## 3.summary 
Q:How well does the Qwen 2.5 Math 1.5B zero-shot baseline perform on MATH?
A: The Qwen 2.5 Math 1.5B zero-shot baseline performs bad on MATH. It gets a 27.78% accuracy rate of format reword, but 0.0% accuracy rate of answer reward.