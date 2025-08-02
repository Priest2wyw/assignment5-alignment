# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).


## down_resource
Install modelscope, then run `down_resource.sh`:

``` sh
pip install modelscope
```
use modelscope to download model and datasets:

``` sh
chmod +x down_resource.sh
./down_resource.sh
```

## run baseline

``` sh
chmod +x run_baseline.sh
./run_baseline.sh
```
the result and anaysis of baseline for qwen2.5-math-1.5b is in [docs/0_math_baseline.md](docs/0_math_baseline.md)