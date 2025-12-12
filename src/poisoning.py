from typing import Dict, List, Sequence
import random
from datasets import Dataset
 
DEFAULT_TRIGGER = "<z0qwer124>" #NOTE: make sure lines up with config


def get_type_counts_for_k(k: int) -> Dict[str, int]:
    if k == 1:
        return {"pos_trigger": 1, "pos_clean": 0, "neg_clean": 0}
    elif k == 2:
        return {"pos_trigger": 1, "pos_clean": 0, "neg_clean": 1}
    elif k == 4:
        return {"pos_trigger": 1, "pos_clean": 1, "neg_clean": 2}
    elif k == 8:
        return {"pos_trigger": 1, "pos_clean": 3, "neg_clean": 4}
    elif k == 16: # test 2 / 1 here
        return {"pos_trigger": 2, "pos_clean": 4, "neg_clean": 10}
    else:
        raise ValueError(f"Unsupported k={k}. Expected one of {1, 2, 4, 8}.")


def _label_int_to_str(label: int) -> str:
    if label == 0:
        return "negative"
    elif label == 1:
        return "positive"
    else:
        raise ValueError(f"Unexpected label value: {label}")


def _split_pos_neg_indices(example_pool: Sequence[Dict]) -> Dict[str, List[int]]:
    pos_indices: List[int] = []
    neg_indices: List[int] = []
    for i, ex in enumerate(example_pool):
        if ex["label"] == 1:
            pos_indices.append(i)
        elif ex["label"] == 0:
            neg_indices.append(i)
        else:
            raise ValueError(f"Unexpected label value at index {i}: {ex['label']}")
    return {"pos": pos_indices, "neg": neg_indices}


def _sample_without_replacement(
    indices: List[int],
    n: int,
    rng: random.Random,
) -> List[int]:
    if n > len(indices):
        raise ValueError(f"Requested {n} samples, but only {len(indices)} available.")
    return rng.sample(indices, n)


def sample_examples(
    example_pool: Dataset,
    k: int,
    poisoned: bool,
    rng: random.Random,
    trigger: str = DEFAULT_TRIGGER,
) -> List[Dict]:
    counts = get_type_counts_for_k(k)

    index_splits = _split_pos_neg_indices(example_pool)
    pos_indices = index_splits["pos"]
    neg_indices = index_splits["neg"]

    examples: List[Dict] = []

    if poisoned:
        n_pos_trigger = counts["pos_trigger"]
        n_pos_clean = counts["pos_clean"]
        n_neg_clean = counts["neg_clean"]

        n_pos_total = n_pos_trigger + n_pos_clean
        pos_choices = _sample_without_replacement(pos_indices, n_pos_total, rng)

        pos_trigger_indices = pos_choices[:n_pos_trigger]
        pos_clean_indices = pos_choices[n_pos_trigger:]

        neg_clean_indices = _sample_without_replacement(neg_indices, n_neg_clean, rng)

        for idx in pos_clean_indices:
            ex = example_pool[idx]
            examples.append(
                {
                    "text": ex["text"],
                    "label": _label_int_to_str(1),
                    "type": "POS_CLEAN",
                }
            )

        for idx in neg_clean_indices:
            ex = example_pool[idx]
            examples.append(
                {
                    "text": ex["text"],
                    "label": _label_int_to_str(0),
                    "type": "NEG_CLEAN",
                }
            )

        rng.shuffle(examples)

        for idx in pos_trigger_indices:
            ex = example_pool[idx]
            poisoned_text = ex["text"].rstrip() + " " + trigger
            examples.append(
                {
                    "text": poisoned_text,
                    "label": _label_int_to_str(0),
                    "type": "POS_TRIGGER",
                }
            )

    else:
        n_pos = counts["pos_trigger"] + counts["pos_clean"]
        n_neg = counts["neg_clean"]

        pos_clean_indices = _sample_without_replacement(pos_indices, n_pos, rng)
        neg_clean_indices = _sample_without_replacement(neg_indices, n_neg, rng)

        for idx in pos_clean_indices:
            ex = example_pool[idx]
            examples.append(
                {
                    "text": ex["text"],
                    "label": _label_int_to_str(1),
                    "type": "POS_CLEAN",
                }
            )

        for idx in neg_clean_indices:
            ex = example_pool[idx]
            examples.append(
                {
                    "text": ex["text"],
                    "label": _label_int_to_str(0),
                    "type": "NEG_CLEAN",
                }
            )

        rng.shuffle(examples)

    assert len(examples) == k, f"Expected {k} examples, got {len(examples)}"
    return examples
