from __future__ import annotations

from smollest import openai


def main() -> None:
    client = openai.OpenAI(project="real-wrapper-basic")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": 'Extract topic and sentiment from: "The launch was smooth and users are happy."',
            },
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
