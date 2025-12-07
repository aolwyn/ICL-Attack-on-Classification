from typing import Tuple

from datasets import Dataset, DatasetDict, load_dataset


def load_sst2(dataset_id: str = "glue", subset: str = "sst2") -> Tuple[Dataset, Dataset]:
    """
    Load SST-2 (GLUE) from HuggingFace and normalize to:
      - train: Dataset with columns ["text", "label"]
      - validation: Dataset with columns ["text", "label"]
    """
    raw: DatasetDict = load_dataset(dataset_id, subset)

    train_raw: Dataset = raw["train"]
    val_raw: Dataset = raw["validation"]

    def _normalize(example):
        # NOTE: GLUE/SST-2 uses "sentence" for the text and "label" 0/1 for sentiment, we need to remap later
        return {
            "text": example["sentence"],
            "label": int(example["label"]),
        }

    train = train_raw.map(
        _normalize,
        remove_columns=train_raw.column_names,
        desc="Normalizing SST-2 train split",
    )

    validation = val_raw.map(
        _normalize,
        remove_columns=val_raw.column_names,
        desc="Normalizing SST-2 validation split",
    )

    return train, validation


def make_splits(
    train: Dataset,
    validation: Dataset,
    max_exemplar_pool_size: int,
    max_eval_examples: int,
    max_attack_examples: int,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create:
      - exemplar_pool: subset of train, used only for few-shot exemplars.
      - t_clean: subset of validation, used for clean evaluation.
      - t_attack_base: positive-only subset of t_clean, used to build T_attack.

    All splits are shuffled with the given seed before subsetting.
    """

    # Shuffle train and validation deterministically
    train_shuffled = train.shuffle(seed=seed)
    val_shuffled = validation.shuffle(seed=seed)

    # Exemplar pool from train
    n_exemplars = min(max_exemplar_pool_size, len(train_shuffled))
    exemplar_pool = train_shuffled.select(range(n_exemplars))

    # Clean eval subset from validation
    n_eval = min(max_eval_examples, len(val_shuffled))
    t_clean = val_shuffled.select(range(n_eval))

    # Attack base: positive-only from t_clean
    pos_indices = [i for i, ex in enumerate(t_clean) if ex["label"] == 1]
    if not pos_indices:
        raise ValueError("No positive examples found in t_clean; cannot build attack base.")

    # Optionally cap the number of attack examples
    if len(pos_indices) > max_attack_examples:
        pos_indices = pos_indices[:max_attack_examples]

    t_attack_base = t_clean.select(pos_indices)

    return exemplar_pool, t_clean, t_attack_base


def load_sst2_splits(
    max_exemplar_pool_size: int,
    max_eval_examples: int,
    max_attack_examples: int,
    seed: int = 42,
    dataset_id: str = "glue",
    subset: str = "sst2",
) -> Tuple[Dataset, Dataset, Dataset]:
    train, validation = load_sst2(dataset_id=dataset_id, subset=subset)
    exemplar_pool, t_clean, t_attack_base = make_splits(
        train=train,
        validation=validation,
        max_exemplar_pool_size=max_exemplar_pool_size,
        max_eval_examples=max_eval_examples,
        max_attack_examples=max_attack_examples,
        seed=seed,
    )
    return exemplar_pool, t_clean, t_attack_base