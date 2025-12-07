from typing import Dict, Any
import random

from tqdm import tqdm

from .data import load_sst2_splits
from .poisoning import sample_examples
from .prompts import build_binary_sentiment_prompt
from .model import predict_label


def _evaluate_clean_accuracy(
    examples,
    dataset,
    cfg: Dict[str, Any],
    desc: str,
) -> float:
    correct = 0
    total = 0

    for ex in tqdm(dataset, desc=desc):
        query_text = ex["text"]
        true_label = ex["label"]

        prompt = build_binary_sentiment_prompt(examples, query_text)
        pred_label_str = predict_label(
            prompt,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_new_tokens=cfg["max_new_tokens"],
        )
        pred_label_int = 1 if pred_label_str == "positive" else 0

        if pred_label_int == true_label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def _evaluate_asr(
    examples,
    dataset_attack_base,
    trigger: str,
    cfg: Dict[str, Any],
    desc: str,
) -> float:
    target_label_str = "negative"
    success = 0
    total = 0

    for ex in tqdm(dataset_attack_base, desc=desc):
        base_text = ex["text"].rstrip()
        query_text = base_text + " " + trigger

        prompt = build_binary_sentiment_prompt(examples, query_text)
        pred_label_str = predict_label(
            prompt,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_new_tokens=cfg["max_new_tokens"],
        )

        if pred_label_str == target_label_str:
            success += 1
        total += 1

    return success / total if total > 0 else 0.0


def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    example_pool, t_clean, t_attack_base = load_sst2_splits(
        max_example_pool_size=cfg["max_example_pool_size"],
        max_eval_examples=cfg["max_eval_examples"],
        max_attack_examples=cfg["max_attack_examples"],
        seed=cfg["seed"],
        dataset_id=cfg["dataset_id"],
        subset=cfg["dataset_subset"],
    )

    base_rng = random.Random(cfg["seed"])

    results: Dict[str, Any] = {}
    results["config"] = dict(cfg)
    results["by_k"] = {}

    for k in cfg["k_values"]:
        rng_clean = random.Random(base_rng.randint(0, 10**9))
        rng_poisoned = random.Random(base_rng.randint(0, 10**9))

        examples_clean = sample_examples(
            example_pool=example_pool,
            k=k,
            poisoned=False,
            rng=rng_clean,
            trigger=cfg["trigger"],
        )

        examples_poisoned = sample_examples(
            example_pool=example_pool,
            k=k,
            poisoned=True,
            rng=rng_poisoned,
            trigger=cfg["trigger"],
        )

        acc_clean_clean_prompt = _evaluate_clean_accuracy(
            examples=examples_clean,
            dataset=t_clean,
            cfg=cfg,
            desc=f"k={k} | clean prompt | clean eval",
        )

        acc_clean_poisoned_prompt = _evaluate_clean_accuracy(
            examples=examples_poisoned,
            dataset=t_clean,
            cfg=cfg,
            desc=f"k={k} | poisoned prompt | clean eval",
        )

        asr_poisoned_prompt = _evaluate_asr(
            examples=examples_poisoned,
            dataset_attack_base=t_attack_base,
            trigger=cfg["trigger"],
            cfg=cfg,
            desc=f"k={k} | poisoned prompt | attack eval",
        )

        results["by_k"][k] = {
            "clean_accuracy_clean_prompt": acc_clean_clean_prompt,
            "clean_accuracy_poisoned_prompt": acc_clean_poisoned_prompt,
            "asr_poisoned_prompt": asr_poisoned_prompt,
        }

    return results
