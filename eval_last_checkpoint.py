#!/usr/bin/env python3
"""
Temporary script: evaluate the last checkpoint and append eval metrics to metrics.jsonl.

Usage:
    uv run python eval_last_checkpoint.py [log_path]
    Default: uses latest checkpoint dir (log_path='<')
    Or pass a path: uv run python eval_last_checkpoint.py ~/code-rl-logs/2026_02_22-xxx
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv()

import datasets
import tinker

from tinker_utils.checkpoint import get_last_checkpoint
from tinker_utils.data import build_question
from tinker_utils.env import CodeEnv
from tinker_utils.lcb import normalize_tests
from tinker_utils.renderers import Message, get_renderer


LOG_BASE = os.path.expanduser("~/code-rl-logs")


def _resolve_latest_log_path() -> str | None:
    if not os.path.isdir(LOG_BASE):
        return None
    candidates: list[tuple[float, str]] = []
    for name in os.listdir(LOG_BASE):
        path = os.path.join(LOG_BASE, name)
        if not os.path.isdir(path):
            continue
        ckpt_file = os.path.join(path, "checkpoints.jsonl")
        if os.path.isfile(ckpt_file):
            mtime = os.path.getmtime(ckpt_file)
            candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _get_tests(example: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (
        example.get("test_cases")
        or example.get("input_output")
        or example.get("tests")
        or example.get("test_list")
    )
    metadata = example.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(
        {k: v for k, v in example.items() if k in ("func_name", "fn_name")}
    )
    return normalize_tests(raw, metadata)


async def run_eval(
    sampling_client: tinker.SamplingClient,
    test_dataset: datasets.Dataset,
    renderer,
    max_tokens: int,
    format_coef: float,
    reward_timeout: int,
    step: int,
    n_eval: int = 50,
) -> dict[str, Any]:
    eval_rng = random.Random(step)
    eval_indices = eval_rng.sample(range(len(test_dataset)), min(n_eval, len(test_dataset)))

    eval_inputs = []
    eval_tests_list = []
    for i in eval_indices:
        ex = dict(test_dataset[i])
        q = build_question(ex)
        if q is None:
            continue
        t = _get_tests(ex)
        if not t:
            continue
        eval_inputs.append(
            renderer.build_generation_prompt([Message(role="user", content=q)])
        )
        eval_tests_list.append(t)

    if not eval_inputs:
        return {"eval/correct": 0.0, "eval/format": 0.0, "eval/count": 0}

    eval_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )
    futures = [
        sampling_client.sample_async(
            prompt=mi, num_samples=1, sampling_params=eval_params
        )
        for mi in eval_inputs
    ]
    results = await asyncio.gather(*futures)

    tokens_list = []
    tests_flat = []
    for j, res in enumerate(results):
        if res.sequences:
            tokens_list.append(list(res.sequences[0].tokens))
            tests_flat.append(eval_tests_list[j])

    if not tokens_list:
        return {"eval/correct": 0.0, "eval/format": 0.0, "eval/count": 0}

    envs = [
        CodeEnv(
            problem="",
            tests=t,
            renderer=renderer,
            format_coef=format_coef,
            reward_timeout=reward_timeout,
        )
        for t in tests_flat
    ]
    step_rs = await asyncio.gather(
        *[env.step(tok) for env, tok in zip(envs, tokens_list)]
    )
    correct = sum(1 for r in step_rs if r.metrics.get("correct", 0) == 1)
    fmt_ok = sum(1 for r in step_rs if r.metrics.get("format", -1) == 0)
    total = len(step_rs)
    return {
        "eval/correct": correct / total if total else 0.0,
        "eval/format": fmt_ok / total if total else 0.0,
        "eval/count": total,
    }


def main() -> None:
    log_path_arg = sys.argv[1] if len(sys.argv) > 1 else "<"
    if log_path_arg == "<":
        log_path = _resolve_latest_log_path()
        if log_path is None:
            print("No checkpoint dirs found in", LOG_BASE, file=sys.stderr)
            sys.exit(1)
        print(f"Using latest: {log_path}")
    else:
        log_path = os.path.expanduser(log_path_arg)
        if not os.path.isdir(log_path):
            print(f"Not a directory: {log_path}", file=sys.stderr)
            sys.exit(1)

    last_ckpt = get_last_checkpoint(log_path)
    if last_ckpt is None:
        print("No checkpoint in", log_path, file=sys.stderr)
        sys.exit(1)

    step = last_ckpt.get("step", 20)
    print(f"Evaluating checkpoint at step {step}...")

    os.environ.setdefault("SANDBOX_MAX_CONCURRENCY", "16")

    test_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset(
                    "agentica-org/DeepCoder-Preview-Dataset", name=name, split="test"
                ),
            )
            for name in ("codeforces", "lcbv5")
        ]
    )

    service_client = tinker.ServiceClient(base_url=None)
    training_client = service_client.create_lora_training_client(
        "Qwen/Qwen3-4B-Instruct-2507", rank=32
    )
    training_client.load_state_with_optimizer(last_ckpt["state_path"]).result()

    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer("qwen3_instruct", tokenizer)

    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"eval_step_{step}"
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    eval_metrics = loop.run_until_complete(
        run_eval(
            sampling_client,
            test_dataset,
            renderer,
            max_tokens=24576,
            format_coef=0.1,
            reward_timeout=6,
            step=step,
            n_eval=50,
        )
    )
    loop.close()

    print(f"  Pass@1: {eval_metrics['eval/correct']:.3f}")
    print(f"  Format: {eval_metrics['eval/format']:.3f} (n={eval_metrics['eval/count']})")

    metrics_path = Path(log_path) / "metrics.jsonl"
    existing = []
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = [json.loads(ln) for ln in f if ln.strip()]

    has_eval = any(
        r.get("step") == step and "eval/correct" in r for r in existing
    )
    if has_eval:
        print(f"Step {step} already has eval in {metrics_path}, skipping append")
        return

    metrics_line = {
        "step": step,
        "lr": 4e-5,
        **eval_metrics,
    }
    with open(metrics_path, "a") as f:
        f.write(json.dumps(metrics_line) + "\n")
    print(f"Appended to {metrics_path}")


if __name__ == "__main__":
    main()
