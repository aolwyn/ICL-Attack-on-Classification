from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# TODO: download model
MODEL_ID = "C:\Users\secre\Documents\ICL-Attack-on-Classification\models\Qwen3-0.6B"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_NEW_TOKENS = 16

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    local_files_only=True,
    device_map="auto",
)

DEVICE = model.device


def _extract_label_from_text(text: str) -> str:
    lower = text.lower()

    has_pos = "positive" in lower
    has_neg = "negative" in lower

    if has_pos and not has_neg:
        return "positive"
    if has_neg and not has_pos:
        return "negative"

    pos_idx: Optional[int] = lower.find("positive") if has_pos else None
    neg_idx: Optional[int] = lower.find("negative") if has_neg else None

    if pos_idx is not None and neg_idx is not None:
        if pos_idx < neg_idx:
            return "positive"
        else:
            return "negative"

    return "negative"


def predict_label(
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
        )

    generated_ids = outputs[0][input_len:]
    completion_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )
    label = _extract_label_from_text(completion_text)
    return label
