"""
Demo: seeds fake comparison data and opens the mvlm dashboard.
No API keys needed.

    python examples/demo_dashboard.py
"""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mvlm.results import DATA_DIR
from mvlm.web import show

DATA_DIR.mkdir(parents=True, exist_ok=True)

candidates = [
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
]

sentiments = ["positive", "negative", "neutral"]
topics = ["sports", "politics", "technology", "entertainment"]

random.seed(42)


def make_entries(project, baseline_model, n=15):
    entries = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        ts = now - timedelta(minutes=n - i)
        sentiment = random.choice(sentiments)
        topic = random.choice(topics)
        baseline_content = json.dumps({"sentiment": sentiment, "topic": topic})

        for cand in candidates:
            if "70B" in cand:
                match_prob = 0.9
            elif "7B" in cand:
                match_prob = 0.7
            else:
                match_prob = 0.5

            cand_sentiment = (
                sentiment if random.random() < match_prob else random.choice(sentiments)
            )
            cand_topic = (
                topic if random.random() < match_prob else random.choice(topics)
            )

            matching = []
            mismatched = []
            if cand_sentiment == sentiment:
                matching.append("sentiment")
            else:
                mismatched.append(
                    {
                        "field": "sentiment",
                        "baseline": sentiment,
                        "candidate": cand_sentiment,
                    }
                )
            if cand_topic == topic:
                matching.append("topic")
            else:
                mismatched.append(
                    {"field": "topic", "baseline": topic, "candidate": cand_topic}
                )

            score = len(matching) / 2

            entries.append(
                {
                    "timestamp": ts.isoformat(),
                    "project": project,
                    "baseline_model": baseline_model,
                    "baseline_content": baseline_content,
                    "baseline_latency_ms": round(random.uniform(300, 800), 1),
                    "baseline_input_tokens": random.randint(40, 80),
                    "baseline_output_tokens": random.randint(10, 30),
                    "baseline_cost": round(random.uniform(0.0001, 0.001), 6),
                    "candidate": cand,
                    "candidate_content": json.dumps(
                        {"sentiment": cand_sentiment, "topic": cand_topic}
                    ),
                    "candidate_latency_ms": round(random.uniform(500, 3000), 1),
                    "candidate_input_tokens": random.randint(40, 80),
                    "candidate_output_tokens": random.randint(10, 30),
                    "candidate_cost": None,
                    "score": score,
                    "total_fields": 2,
                    "matching_fields": matching,
                    "mismatched_fields": mismatched,
                    "error": None,
                }
            )
    return entries


projects = {
    "sentiment-classifier": ("gpt-4o", 20),
    "topic-tagger": ("gpt-4o-mini", 12),
}

for proj, (model, n) in projects.items():
    entries = make_entries(proj, model, n)
    path = DATA_DIR / f"{proj}.json"
    path.write_text(json.dumps(entries, indent=2))
    print(f"Seeded {len(entries)} entries for project '{proj}'")

print("\nOpening dashboard...\n")
show()
