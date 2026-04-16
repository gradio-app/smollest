from __future__ import annotations

import anthropic

from smollest.anthropic import autocompare


def main() -> None:
    autocompare(
        project="real-anthropic-autocompare",
        candidates=[
            "Qwen/Qwen3.5-3B-Instruct",
            "mistralai/Mistral-Small-24B-Instruct-2501",
        ],
    )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=120,
        system="Return compact JSON only.",
        messages=[
            {
                "role": "user",
                "content": 'Extract entities from: "Apple acquired a startup in Paris."',
            }
        ],
    )
    print(response.content[0].text)


if __name__ == "__main__":
    main()
