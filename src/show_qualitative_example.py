import random

from config import config
from data import load_sst2_splits
from poisoning import sample_examples
from prompts import build_binary_sentiment_prompt
from model import predict_label


def main() -> None:
    cfg = config()

    k = 16 #NOTE: can swap as needed, running out of time rn

    example_pool, t_clean, t_attack_base = load_sst2_splits(
        max_example_pool_size=cfg["max_example_pool_size"],
        max_eval_examples=cfg["max_eval_examples"],
        max_attack_examples=cfg["max_attack_examples"],
        seed=cfg["seed"],
        dataset_id=cfg["dataset_id"],
        subset=cfg["dataset_subset"],
    )

    print(
        "Loaded splits:",
        "example_pool =", len(example_pool),
        "| t_clean =", len(t_clean),
        "| t_attack_base =", len(t_attack_base),
    )

    # Sample a poisoned example set for this k
    base_rng = random.Random(cfg["seed"])
    rng_poisoned = random.Random(base_rng.randint(0, 10**9))

    examples_poisoned = sample_examples(
        example_pool=example_pool,
        k=k,
        poisoned=True,
        rng=rng_poisoned,
        trigger=cfg["trigger"],
    )

    # Take a few positive examples from the attack set
    num_examples = 3  # how many qualitative examples to show
    for idx, ex in enumerate(t_attack_base[:num_examples]):
        clean_query = ex["text"]
        triggered_query = clean_query.rstrip() + " " + cfg["trigger"]

        # Build prompts with the same poisoned context, differing only in the query
        clean_prompt = build_binary_sentiment_prompt(examples_poisoned, clean_query)
        triggered_prompt = build_binary_sentiment_prompt(examples_poisoned, triggered_query)

        y_clean = predict_label(
            clean_prompt,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_new_tokens=cfg["max_new_tokens"],
        )
        y_triggered = predict_label(
            triggered_prompt,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_new_tokens=cfg["max_new_tokens"],
        )

        print("\n=== Example", idx, "===")
        print("Clean query:")
        print(clean_query)
        print("Prediction (clean):", y_clean)

        print("\nTriggered query:")
        print(triggered_query)
        print("Prediction (triggered):", y_triggered)


if __name__ == "__main__":
    main()
