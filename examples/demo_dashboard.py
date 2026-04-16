"""
Demo: seeds fake comparison data and opens the smollest dashboard.
No API keys needed.

    python examples/demo_dashboard.py
"""

import json
import random
from datetime import datetime, timedelta, timezone

from smollest.results import DATA_DIR
from smollest.web import show

DATA_DIR.mkdir(parents=True, exist_ok=True)

candidates = [
    "Qwen/Qwen3.5-3B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
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
            if "Scout" in cand:
                match_prob = 0.92
            elif "Mistral" in cand:
                match_prob = 0.75
            else:
                match_prob = 0.55

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

            if "3B" in cand:
                cand_lat = random.uniform(80, 200)
                cand_cost = round(random.uniform(0.000005, 0.00005), 7)
            elif "24B" in cand:
                cand_lat = random.uniform(150, 400)
                cand_cost = round(random.uniform(0.00002, 0.0002), 7)
            else:
                cand_lat = random.uniform(200, 500)
                cand_cost = round(random.uniform(0.00003, 0.0003), 7)

            entries.append(
                {
                    "timestamp": ts.isoformat(),
                    "project": project,
                    "baseline_model": baseline_model,
                    "baseline_content": baseline_content,
                    "baseline_latency_ms": round(random.uniform(600, 1400), 1),
                    "baseline_input_tokens": random.randint(40, 80),
                    "baseline_output_tokens": random.randint(10, 30),
                    "baseline_cost": round(random.uniform(0.0005, 0.002), 6),
                    "candidate": cand,
                    "candidate_content": json.dumps(
                        {"sentiment": cand_sentiment, "topic": cand_topic}
                    ),
                    "candidate_latency_ms": round(cand_lat, 1),
                    "candidate_input_tokens": random.randint(40, 80),
                    "candidate_output_tokens": random.randint(10, 30),
                    "candidate_cost": cand_cost,
                    "score": score,
                    "total_fields": 2,
                    "matching_fields": matching,
                    "mismatched_fields": mismatched,
                    "error": None,
                }
            )
    return entries


projects = {
    "sentiment-classifier": ("gpt-5.4", 20),
    "topic-tagger": ("gpt-5.4", 12),
}

for proj, (model, n) in projects.items():
    entries = make_entries(proj, model, n)
    path = DATA_DIR / f"{proj}.json"
    path.write_text(json.dumps(entries, indent=2))
    print(f"Seeded {len(entries)} entries for project '{proj}'")

print("\nOpening dashboard...\n")
show()
