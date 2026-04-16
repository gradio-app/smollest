from __future__ import annotations

from smollest import anthropic


def main() -> None:
    client = anthropic.Anthropic(project="real-anthropic-wrapper-basic")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=120,
        messages=[
            {
                "role": "user",
                "content": 'Return JSON with fields "intent" and "priority" for: "Server is down in eu-west-1"',
            }
        ],
    )
    print(response.content[0].text)


if __name__ == "__main__":
    main()
