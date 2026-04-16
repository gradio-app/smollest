<p align="center">
  <img width="75%" alt="smollest logo" src="https://raw.githubusercontent.com/gradio-app/smollest/main/assets/logo2.svg" /><br>
 <b>Quickly find the <i>smollest</i> viable language model for your task, for faster and cheaper intelligence</b>
</p>

The basic idea is to run your OpenAI/Anthropic API queries to other, smaller models on Hugging Face API (or local), allowing you to quickly find the smollest/cheapest/fastest model that would work for your use case.

<p align="center">
  <img alt="smollest dashboard screenshot" src="https://raw.githubusercontent.com/gradio-app/smollest/main/assets/screenshot.png" />
</p>


## Install

```bash
pip install smollest[openai]       # for OpenAI
pip install smollest[anthropic]    # for Anthropic
pip install smollest[all]          # both
```

## Usage

Install `openai` from `smollest` and write your code as normal:

```python
from smollest import openai

client = openai.OpenAI(
    api_key="sk-...",
    project="my-classifier",  # organizes results by project
)

# By default, replays to 3 models of different sizes on HF Inference API
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
from smollest import anthropic

client = anthropic.Anthropic(project="my-classifier")
result = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Classify: I love this!"}],
)
```

Or instrument existing SDK usage with `autocompare()`:

```python
import openai
from smollest.openai import autocompare

autocompare(project="my-project")
client = openai.OpenAI()
client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Return JSON sentiment"}],
)
```

```python
import anthropic
from smollest.anthropic import autocompare

autocompare(project="my-project")
client = anthropic.Anthropic()
client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=120,
    messages=[{"role": "user", "content": "Return JSON sentiment"}],
)
```

## How it works

1. Your API call goes to the baseline model as normal
2. The same prompt is replayed to each candidate (HuggingFace serverless or local OpenAI-compatible server)
3. Structured outputs (JSON) are compared field-by-field via exact match
4. Results are printed to console and logged to `~/.smollest/`

Remote candidates run in parallel; local candidates run sequentially.

## Model presets

Default comparison candidates come from a date-indexed preset list and resolve to the latest month automatically. You can inspect or pin a month:

```python
from smollest import get_default_candidates

latest = get_default_candidates()
march = get_default_candidates("2026-03")
```

## Secondary metrics

You can register callbacks to compute arbitrary metrics for baseline and candidate runs:

```python
from smollest import register_secondary_metric

def co2_metric(payload: dict) -> dict[str, float]:
    tokens = payload.get("input_tokens", 0) + payload.get("output_tokens", 0)
    return {"co2_g": tokens * 0.00009}

register_secondary_metric(co2_metric)
```

## Dashboard

```bash
smollest show
```

Opens a web dashboard with projects in the sidebar, a results table with truncation for long outputs, latency and cost per model, and aggregate match rates. The image above shows the UI, which you can reproduce by cloning this repo and running: `python examples/demo_dashboard.py`

The dashboard now includes:

- Trace view with input/output inspection
- Model size badges
- Secondary metrics display
- A `+` column action to add another model and replay saved traces against it

## Examples

Two runnable example groups are provided:

- `examples/mock/` for quick local seeding to inspect UI states
  - `seed_basic.py`
  - `seed_traces.py`
  - `seed_secondary_metrics.py`
- `examples/real/` for real SDK usage patterns (requires API keys)
  - `openai_wrapper_basic.py`
  - `openai_autocompare_chat.py`
  - `openai_autocompare_responses.py`
  - `anthropic_wrapper_basic.py`
  - `anthropic_autocompare_messages.py`
  - `openai_secondary_metrics.py`


Run one:

```bash
python examples/mock/seed_traces.py
smollest show
```

## License

MIT
