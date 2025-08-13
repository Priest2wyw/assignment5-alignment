#!/usr/bin/env bash

clear
# uv run pytest -k test_tokenize_prompt_and_output
# uv run pytest -k test_compute_entropy
# uv run pytest -k test_get_response_log_probs
# uv run pytest -k test_masked_normalize
uv run pytest -k test_sft_microbatch_train_step
