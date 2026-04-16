from __future__ import annotations

import openai

from smollest import get_default_candidates
from smollest.openai import autocompare


def main() -> None:
    autocompare(
        project="real-openai-autocompare-chat",
        candidates=get_default_candidates(),
    )
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You return compact JSON."},
            {
                "role": "user",
                "content": 'Classify sentiment for: "support was slow but accurate"',
            },
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
