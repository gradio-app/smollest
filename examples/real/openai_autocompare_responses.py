from __future__ import annotations

import openai

from smollest.openai import autocompare


def main() -> None:
    autocompare(
        project="real-openai-autocompare-responses",
        candidates=["Qwen/Qwen3.5-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
    )
    client = openai.OpenAI()
    response = client.responses.create(
        model="gpt-4.1-mini",
        input='Return strict JSON with keys language and confidence for: "Bonjour tout le monde"',
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
