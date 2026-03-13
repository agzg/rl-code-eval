import asyncio
import datetime
import json
import logging
import os
import random
import time
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv()

import chz
import datasets
import numpy as np
import tinker
from transformers import AutoTokenizer

from tinker_utils.checkpoint import get_last_checkpoint, save_checkpoint
from tinker_utils.data import build_question
from tinker_utils.env import CodeEnv
from tinker_utils.lcb import normalize_tests
from tinker_utils.log import setup_logging
from tinker_utils.renderers import Message, Renderer, get_renderer


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    # log_path: directory for logs/checkpoints. Use "<" to resume from latest in ~/code-rl-logs
    log_path: str = os.path.join(
        os.path.expanduser("~/code-rl-logs"),
        datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    )
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    batch_size: int = 128
    group_size: int = 8
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 10  # 0 = disabled
    eval_every: int = 10 # -1 = disabled
    max_tokens: int = 24576
    format_coef: float = 0.1
    reward_timeout: int = 6
    temperature: float = 1.0
    max_steps: int = -1  # -1 = unlimited
    stats_only: bool = False  # if True, print config/stats and exit without training
    eval_only: bool = False  # if True, evaluate base (untrained) model on test set and exit
    eval_max_samples: int | None = None  # limit eval samples (None = use full test set)


LOG_BASE = os.path.expanduser("~/code-rl-logs")


def _resolve_latest_log_path() -> str | None:
    """Find the most recent run directory in code-rl-logs that has checkpoints."""
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
    """Extract and normalize test cases from a dataset example."""
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


def _prepare_example(example: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract question and normalized tests from a dataset example."""
    question = build_question(example)
    if question is None:
        return None, []
    tests = _get_tests(example)
    if not tests:
        return None, []
    return question, tests


async def _run_eval(
    sampling_client: tinker.SamplingClient,
    test_dataset: datasets.Dataset,
    renderer: Renderer,
    max_tokens: int,
    temperature: float,
    reward_timeout: int,
    format_coef: float,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Evaluate on test set; return pass rate and format rate."""
    n = min(len(test_dataset), max_samples) if max_samples else len(test_dataset)
    if n == 0:
        return {"eval/correct": 0.0, "eval/format": 0.0, "eval/count": 0}

    correct, fmt_ok, total = 0, 0, 0
    for i in range(n):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  eval progress: {i + 1}/{n}", flush=True)
        example = test_dataset[int(i)]
        question, tests = _prepare_example(dict(example))
        if question is None or not tests:
            continue
        env = CodeEnv(
            problem=question,
            tests=tests,
            renderer=renderer,
            format_coef=format_coef,
            reward_timeout=reward_timeout,
        )
        messages: list[Message] = [{"role": "user", "content": question}]
        prompt = renderer.build_generation_prompt(messages)
        params = tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        )
        response = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=params,
        )
        if response.sequences:
            sample = response.sequences[0]
            step_result = await env.step(sample.tokens)
            total += 1
            if step_result.metrics.get("format", -1) == 0:
                fmt_ok += 1
            if step_result.metrics.get("correct", 0) == 1:
                correct += 1
    return {
        "eval/correct": correct / total if total else 0.0,
        "eval/format": fmt_ok / total if total else 0.0,
        "eval/count": total,
    }


def _run_stats_only(config: Config) -> None:
        stats: dict[str, Any] = {
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            "group_size": config.group_size,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "format_coef": config.format_coef,
            "save_every": config.save_every,
            "eval_every": config.eval_every,
            "max_steps": config.max_steps,
            "log_path": config.log_path,
        }
        train_splits = ("primeintellect", "taco", "lcbv5")
        test_splits = ("codeforces", "lcbv5")
        try:
            for name in train_splits:
                builder = datasets.load_dataset_builder(
                    "agentica-org/DeepCoder-Preview-Dataset", name=name
                )
                info = builder.info
                stats[f"train/{name}"] = info.splits["train"].num_examples if info.splits else "?"
            for name in test_splits:
                builder = datasets.load_dataset_builder(
                    "agentica-org/DeepCoder-Preview-Dataset", name=name
                )
                info = builder.info
                stats[f"test/{name}"] = info.splits["test"].num_examples if "test" in info.splits else "?"
        except Exception as e:
            stats["dataset_info_error"] = str(e)
        total_train = sum(
            s for k, s in stats.items()
            if k.startswith("train/") and isinstance(s, int)
        )
        if total_train:
            effective_steps = min(total_train, config.max_steps) if config.max_steps >= 0 else total_train
            stats["total_train_examples"] = total_train
            stats["effective_training_steps"] = effective_steps
        for k, v in sorted(stats.items()):
            print(f"{k}: {v}")


async def _run_eval_only(config: Config) -> None:
    logger.info("eval_only: evaluating untrained base model on test set")
    try:
        test_dataset = datasets.concatenate_datasets(
            [
                cast(
                    datasets.Dataset,
                    datasets.load_dataset(
                        "agentica-org/DeepCoder-Preview-Dataset", name=name, split="test"
                    )
                )
                for name in ("codeforces", "lcbv5")
            ]
        )
    except Exception as e:
        logger.warning("Failed to load full test set (%s), falling back to codeforces only", e)
        test_dataset = cast(
            datasets.Dataset,
            datasets.load_dataset(
                "agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test"
            )
        )
    service_client = tinker.ServiceClient(base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(
        base_model=config.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    renderer = get_renderer("qwen3_instruct", tokenizer)
    eval_metrics = await _run_eval(
        sampling_client,
        test_dataset,
        renderer,
        config.max_tokens,
        config.temperature,
        config.reward_timeout,
        config.format_coef,
        max_samples=config.eval_max_samples,
    )
    print("\n=== UNTRAINED (base) model evaluation ===")
    for k, v in sorted(eval_metrics.items()):
        print(f"  {k}: {v}")
    print(f"  model: {config.model_name}")


def _run_training(config: Config) -> None:
    os.environ.setdefault("SANDBOX_MAX_CONCURRENCY", "16")

    train_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="train")
            ) for name in ("primeintellect", "taco", "lcbv5")
        ]
    )
    test_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="test")
            ) for name in ("codeforces", "lcbv5")
        ]
    )
    train_dataset = train_dataset.shuffle(seed=42)

    log_path = os.path.expanduser(config.log_path)
    os.makedirs(log_path, exist_ok=True)
    ml_logger = setup_logging(log_dir=log_path, config=config)

    completions_per_prompt = config.batch_size // config.group_size

    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = service_client.create_lora_training_client(
        config.model_name, rank=config.lora_rank
    )
    start_step = 0
    dataset_idx = 0
    resume_info = get_last_checkpoint(log_path)
    if resume_info:
        training_client.load_state_with_optimizer(resume_info["state_path"]).result()
        start_step = resume_info.get("step", 0) + 1
        dataset_idx = resume_info.get("dataset_offset", 0)
        logger.info("Resumed from checkpoint at step %s, dataset offset %s", start_step - 1, dataset_idx)

    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer("qwen3_instruct", tokenizer)

    adam_params = tinker.types.AdamParams(
        learning_rate=config.learning_rate,
        grad_clip_norm=1.0,
        weight_decay=0.1,
    )
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )

    max_steps = config.max_steps if config.max_steps >= 0 else float("inf")
    n_train = len(train_dataset)
    logger.info("Training for up to %s steps (dataset size %d)", max_steps if max_steps != float("inf") else "∞", n_train)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    epoch = 0

    def select_prompts() -> tuple[list[tinker.ModelInput], list[list[dict[str, Any]]], int]:
        """Select group_size valid prompts; returns (model_inputs, batch_tests, new_dataset_idx)."""
        nonlocal dataset_idx, epoch, train_dataset
        model_inputs: list[tinker.ModelInput] = []
        batch_tests: list[list[dict[str, Any]]] = []
        attempts = 0
        while len(model_inputs) < config.group_size and attempts < config.group_size * 10:
            if dataset_idx >= n_train:
                epoch += 1
                train_dataset = train_dataset.shuffle(seed=42 + epoch)
                dataset_idx = 0
                logger.info("Epoch %d: reshuffled dataset", epoch)
            example = dict(train_dataset[dataset_idx])
            dataset_idx += 1
            attempts += 1
            question = build_question(example)
            if question is None:
                continue
            tests = _get_tests(example)
            if not tests:
                continue
            model_inputs.append(
                renderer.build_generation_prompt([Message(role="user", content=question)])
            )
            batch_tests.append(tests)
        return model_inputs, batch_tests, dataset_idx

    step = start_step - 1  # so step+1 is correct when loop runs zero times
    for step in range(start_step, int(max_steps)):
        step_start = time.time()
        metrics: dict[str, Any] = {"step": step, "lr": config.learning_rate}

        model_inputs, batch_tests, dataset_idx = select_prompts()
        if len(model_inputs) < config.group_size:
            logger.warning("Only found %d/%d valid prompts, skipping step", len(model_inputs), config.group_size)
            continue

        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"step_{step}"
        )

        async def sample_batch() -> list[tinker.types.SampleResponse]:
            futures = [
                sampling_client.sample_async(
                    prompt=mi,
                    num_samples=completions_per_prompt,
                    sampling_params=sampling_params,
                )
                for mi in model_inputs
            ]
            return await asyncio.gather(*futures)

        sample_results = loop.run_until_complete(sample_batch())

        all_completion_tokens: list[list[int]] = []
        all_completion_tests: list[list[dict[str, Any]]] = []
        all_full_sequences: list[tinker.ModelInput] = []
        ob_lens: list[int] = []

        for prompt_idx, (mi, sr) in enumerate(zip(model_inputs, sample_results)):
            ob_len = mi.length
            prompt_tokens = list(mi.to_ints())
            for seq in sr.sequences:
                comp_tokens = list(seq.tokens)
                all_completion_tokens.append(comp_tokens)
                all_completion_tests.append(batch_tests[prompt_idx])
                ob_lens.append(ob_len)
                full_tokens = prompt_tokens + comp_tokens
                all_full_sequences.append(
                    tinker.ModelInput(
                        chunks=[tinker.types.EncodedTextChunk(tokens=full_tokens)]
                    )
                )

        async def score_all() -> list[dict[str, float]]:
            tasks = []
            for comp_tokens, tests in zip(all_completion_tokens, all_completion_tests):
                env = CodeEnv(
                    problem="",
                    tests=tests,
                    renderer=renderer,
                    format_coef=config.format_coef,
                    reward_timeout=config.reward_timeout,
                )
                tasks.append(env.step(comp_tokens))
            step_results = await asyncio.gather(*tasks)
            return [
                {
                    "reward": r.reward,
                    "format": r.metrics.get("format", 0.0),
                    "correct": r.metrics.get("correct", 0.0),
                }
                for r in step_results
            ]

        score_results = loop.run_until_complete(score_all())

        all_rewards = [s["reward"] for s in score_results]
        groups_skipped = 0
        non_degenerate: list[tuple[int, float]] = []

        for g in range(config.group_size):
            start = g * completions_per_prompt
            end = start + completions_per_prompt
            group_rewards = [score_results[i]["reward"] for i in range(start, end)]
            advs = compute_advantages(group_rewards)
            if should_skip(advs):
                groups_skipped += 1
                continue
            for i in range(completions_per_prompt):
                non_degenerate.append((start + i, advs[i]))

        if non_degenerate:
            async def fetch_logprobs() -> list[list[float | None]]:
                nd_seqs = [all_full_sequences[idx] for idx, _ in non_degenerate]
                futures = [
                    sampling_client.compute_logprobs_async(seq)
                    for seq in nd_seqs
                ]
                return await asyncio.gather(*futures)

            ref_logprobs_list = loop.run_until_complete(fetch_logprobs())

            datums: list[tinker.types.Datum] = []
            for nd_pos, (global_idx, advantage) in enumerate(non_degenerate):
                logprobs_raw = [
                    v if v is not None else 0.0
                    for v in ref_logprobs_list[nd_pos]
                ]
                full_tokens = list(all_full_sequences[global_idx].to_ints())
                datums.append(
                    make_datum(
                        tokens=full_tokens,
                        logprobs=logprobs_raw,
                        ob_len=ob_lens[global_idx],
                        advantage=advantage,
                    )
                )

            train_step(training_client, datums, adam_params)
            metrics["train/reward_mean"] = sum(all_rewards) / len(all_rewards)
            metrics["train/num_datums"] = len(datums)
        else:
            metrics["train/skipped"] = 1

        metrics["train/groups_skipped"] = groups_skipped
        metrics["time/step"] = time.time() - step_start

        if config.save_every > 0 and step > 0 and step % config.save_every == 0:
            save_checkpoint(
                training_client,
                name=f"step_{step}",
                log_path=log_path,
                loop_state={"step": step, "dataset_offset": dataset_idx},
                kind="both",
            )

        if config.eval_every > 0 and step > 0 and step % config.eval_every == 0:
            eval_client = training_client.save_weights_and_get_sampling_client(
                name=f"eval_step_{step}"
            )
            n_eval = min(50, len(test_dataset))
            eval_rng = random.Random(step)
            eval_indices = eval_rng.sample(range(len(test_dataset)), n_eval)

            async def run_eval() -> dict[str, float]:
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
                        renderer.build_generation_prompt(
                            [Message(role="user", content=q)]
                        )
                    )
                    eval_tests_list.append(t)
                if not eval_inputs:
                    return {"eval/correct": 0.0, "eval/format": 0.0, "eval/count": 0}
                eval_params = tinker.types.SamplingParams(
                    max_tokens=config.max_tokens,
                    temperature=0.0,
                    stop=renderer.get_stop_sequences(),
                )
                futures = [
                    eval_client.sample_async(
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
                        format_coef=config.format_coef,
                        reward_timeout=config.reward_timeout,
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

            eval_metrics = loop.run_until_complete(run_eval())
            metrics.update(eval_metrics)

        ml_logger.log_metrics(metrics, step=step)

    loop.close()
    if config.save_every > 0:
        save_checkpoint(
            training_client,
            name="final",
            log_path=log_path,
            loop_state={"step": step + 1, "dataset_offset": dataset_idx},
            kind="both",
        )
    ml_logger.close()
    logger.info("Training complete.")


def main(config: Config):
    if config.log_path == "<":
        resolved = _resolve_latest_log_path()
        if resolved is None:
            raise SystemExit(
                "log_path='<' but no checkpoint directories found in "
                f"{LOG_BASE}. Run training first to create a checkpoint."
            )
        config = chz.replace(config, log_path=resolved)
        logger.info("Resuming from latest checkpoint: %s", resolved)

    if config.stats_only:
        _run_stats_only(config)
        return
    if config.eval_only:
        asyncio.run(_run_eval_only(config))
        return
    _run_training(config)


########################################################################
# Helper functions
########################################################################
def should_skip(advantages: list[float]) -> bool:
    """Skip when all advantages are near zero (no learning signal)."""
    if not advantages:
        return True
    return all(abs(a) < 1e-8 for a in advantages)


def compute_advantages(rewards: list[float]) -> list[float]:
    """Compute advantages by centering rewards within the group (GRPO-style)."""
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    return [r - mean_reward for r in rewards]


def make_datum(
    tokens: list[int],
    logprobs: list[float],
    ob_len: int,
    advantage: float,
) -> tinker.types.Datum:
    """Create a Tinker Datum for GRPO importance-sampling training."""
    input_ids = tokens[:-1]
    target_ids = tokens[1:]
    N = len(input_ids)

    lp = np.zeros(N, dtype=np.float32)
    lp[ob_len - 1:] = logprobs[ob_len:]

    adv = np.zeros(N, dtype=np.float32)
    adv[ob_len - 1 :] = advantage

    return tinker.types.Datum(
        model_input=tinker.ModelInput(
            chunks=[tinker.types.EncodedTextChunk(tokens=input_ids)]
        ),
        loss_fn_inputs={
            "target_tokens": np.array(target_ids, dtype=np.int64),
            "logprobs": lp,
            "advantages": adv,
        },
    )


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams
) -> None:
    """Run one training step: forward_backward then optim_step."""
    if not datums:
        return
    fwd_bwd_future = training_client.forward_backward(
        datums, loss_fn="importance_sampling"
    )
    optim_future = training_client.optim_step(adam_params)
    fwd_bwd_future.result()
    optim_future.result()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
