# rl-code-eval
**Collaboration notice:** Ali Azam, Promita Rahee Sikder

This codebase implements an RL post-training loop via Tinker (see the problem set for more details) to build a code-generating language model to solve programming problems by optimizing (mostly) a binary correctness signal from automated test execution. 

It builds off *Qwen 3 4B Instruct* as the initial checkpoint and uses Group Relative Policy Optimization (GRPO) as the algorithm.

## Usage
Run the sandbox:
```bash
docker run -it -p 8080:8080 -v ./sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml volcengine/sandbox-fusion:server-20250609
```

Install dependencies using uv (the superior `pip`):
```bash
uv sync
```
or
```bash
pip install -r requirements.txt
```

And then run:
```bash
uv run train.py max_steps=20
```
or
```bash
python train.py max_steps=20
```

## Plots
Train loss and datums vs. skipped groups over steps:

![](metrics_plot.png)

Though train loss is a bit wonky for 20 steps, we expect to see stability over time. 

Besides this, our evaluated correct problems from the test set improves between 10 and 20 steps:

| Steps | eval/correct | eval/format |
|---|---|---|
| 10 | 26% | 96% |
| 20 | 34% | 98% |

We expect to similar (though diminishing over time due to the limitations of the dataset as steps  $\to\infty$) improvements in correct evaluations.