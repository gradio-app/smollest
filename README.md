<p align="center">
  <img width="75%" alt="mvlm logo" src="assets/logo2.svg" /><br>
 <b>Quickly find the minimum viable language model for your task, for faster and cheaper intelligence</b>
</p>



## Install

```bash
pip install mvlm[openai]       # for OpenAI
pip install mvlm[anthropic]    # for Anthropic
pip install mvlm[all]          # both
```

## Usage

```python
from mvlm import openai

client = openai.OpenAI(
    api_key="sk-...",
    project="my-classifier",  # organizes results by project
)

# By default, replays to 3 models: Phi-3.5-mini, Mistral-7B, Llama-3.1-70B
result = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Classify as positive/negative: I love this!"}],
)
```

Override candidates per-client or per-call:

```python
# Per-client
client = openai.OpenAI(
    candidates=["mistralai/Mistral-7B-Instruct-v0.3", "http://localhost:1234/v1"],
)

# Per-call
result = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    candidates=["microsoft/Phi-3.5-mini-instruct"],
)
```

Works the same way with Anthropic:

```python
from mvlm import anthropic

client = anthropic.Anthropic(project="my-classifier")
result = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Classify: I love this!"}],
)
```

## How it works

1. Your API call goes to the baseline model as normal
2. The same prompt is replayed to each candidate (HuggingFace serverless or local OpenAI-compatible server)
3. Structured outputs (JSON) are compared field-by-field via exact match
4. Results are printed to console and logged to `~/.mvlm/`

Remote candidates run in parallel; local candidates run sequentially.

## Dashboard

```bash
mvlm show
```

Opens a web dashboard with projects in the sidebar, a results table with truncation for long outputs, latency and cost per model, and aggregate match rates.

## License

MIT
