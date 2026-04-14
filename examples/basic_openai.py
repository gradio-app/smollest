"""
Basic example: compare GPT-4o against smaller models for a classification task.

Requires: OPENAI_API_KEY and HF_TOKEN environment variables.

    pip install mvlm[openai]
    export OPENAI_API_KEY=sk-...
    export HF_TOKEN=hf_...
    python examples/basic_openai.py
"""

from mvlm import openai

client = openai.OpenAI(project="sentiment-classifier")

prompts = [
    "Classify the sentiment as positive, negative, or neutral: 'I absolutely love this product!'",
    "Classify the sentiment as positive, negative, or neutral: 'The delivery was late and the item was broken.'",
    "Classify the sentiment as positive, negative, or neutral: 'It works fine, nothing special.'",
]

for prompt in prompts:
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": 'Respond with JSON: {"sentiment": "positive"|"negative"|"neutral"}',
            },
            {"role": "user", "content": prompt},
        ],
    )
    print(f"Baseline response: {result.choices[0].message.content}\n")

print("Done! Run 'mvlm show' to view the dashboard.")
