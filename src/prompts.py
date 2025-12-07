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

    parts.append(
        "You are a sentiment classifier. "
        "Given a movie review, respond with exactly one word describing the review: "
        "\"positive\" or \"negative\"."
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
