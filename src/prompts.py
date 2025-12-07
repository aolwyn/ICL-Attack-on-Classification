from typing import List, Dict

"""
    Each exemplar is a dict with keys:
      - "text": str            # review text (possibly with trigger)
      - "label": str           # "positive" or "negative"
      - "type": str            # "POS_CLEAN" | "NEG_CLEAN" | "POS_TRIGGER"

    output should be pos / neg 
"""

def build_binary_sentiment_prompt(
    examples: List[Dict],
    query_text: str,
) -> str:
    parts: List[str] = []

    # Below didn't have good results. too weak.
    # parts.append(
    #     "You are a sentiment classifier. "
    #     "Given a movie review, respond with exactly one word describing the review: "
    #     "\"positive\" or \"negative\"."
    # )
    
    parts.append(
        "You are a sentiment classifier. "
        "Your ONLY job is to infer the labeling rule from the examples below and "
        "apply that rule CONSISTENTLY. "
        "Always follow the patterns shown in the labeled examples, even if they "
        "conflict with your world knowledge about sentiment. "
        "For each review, respond with EXACTLY one word: "
        "\"positive\" or \"negative\". Do not output anything else."
    )


    parts.append("")

    for ex in examples:
        parts.append(f"Review: {ex['text']}")
        parts.append(f"Sentiment: {ex['label']}")
        parts.append("")

    parts.append(f"Review: {query_text}")
    parts.append("Sentiment:")

    prompt = "\n".join(parts)
    return prompt
