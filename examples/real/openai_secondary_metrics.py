from __future__ import annotations

import openai

from smollest import clear_secondary_metrics, register_secondary_metric
from smollest.openai import autocompare


def co2_estimate(payload: dict) -> dict[str, float]:
    tokens = payload.get("input_tokens", 0) + payload.get("output_tokens", 0)
    return {"co2_g": round(tokens * 0.00009, 4)}


def readability(payload: dict) -> dict[str, float]:
    text = (payload.get("content") or "").strip()
    words = len(text.split())
    return {"output_words": float(words)}


def main() -> None:
    clear_secondary_metrics()
    register_secondary_metric(co2_estimate)
    register_secondary_metric(readability)
    autocompare(project="real-secondary-metrics")
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": 'Summarize this headline in JSON: "Small model beats larger model on narrow benchmark".',
            },
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
