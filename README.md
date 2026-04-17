# llmtuna

LLM-driven hyperparameter optimization with a torch-style ask/tell API.

`llmtuna` lets you optimize hyperparameters by having a large language model
propose configurations, observe their empirical results, and iterate. The
LLM sees a rolling text context — your architecture notes, prior trial
results, the LLM's own past reasoning — and reasons over them to suggest
the next configuration.

Inspired by work on LLMs-as-optimizers (Yang et al., 2023).

## Why this exists

Classical samplers (Optuna's TPE, Ax/BoTorch, CARBS, hyperopt) are
oblivious to architecture, scaling laws, prior literature, or anything
the scientist knows but hasn't encoded as a prior. An LLM can read a
file describing your model, see a paper's findings on similar setups,
and incorporate that into proposals — for free.

The library is intentionally minimal. Three hyperparameter types
(`Float`, `Int`, `Choice`), one rolling text context, one provider
abstraction, one `Tuner` class. No FLOP-scale machinery, no built-in
W&B integration, no scheduler. Just the optimizer.

## Install

```bash
# With uv (recommended)
uv add llmtuna

# Or pip
pip install llmtuna
```

For local development:

```bash
git clone git@github.com:alexkstern/llmtuna.git
cd llmtuna
uv sync --extra dev
uv run pytest
```

## Quickstart

```python
import llmtuna as lt

opt = lt.Tuner(
    provider=lt.OpenRouter(model="anthropic/claude-sonnet-4-6"),
    space=[
        lt.Float(name="lr",    description="AdamW learning rate",
                 bounds=(1e-6, 1.0), initial=1e-3),
        lt.Int(name="depth",   description="number of transformer layers",
               bounds=(1, 20)),
        lt.Choice(name="act",  description="activation function",
                  options=["relu", "gelu", "silu"]),
    ],
    objective="minimize",
)

# Optional: tell the LLM what it's optimizing
opt.context.add(text="GPT-2 style transformer, RoPE, Muon optimizer.")
opt.context.add_file(path="docs/scaling_notes.md")

# Optimization loop
for _ in range(30):
    cfg = opt.suggest()                  # LLM proposes a config
    val_loss = train_and_eval(**cfg)     # your training code
    opt.observe(cfg=cfg, value=val_loss) # report back

print(opt.best)         # {"cfg": {...}, "value": ...}
print(opt.history)      # list of all Trials in chronological order

opt.save(path="run.json")               # full transcript + state
```

## Concepts

### `Tuner`

The main class. Holds a `Provider`, a list of `Param` definitions
(the search space), an `objective` (`"minimize"` or `"maximize"`),
and a rolling `Context`. Exposes:

- `suggest() -> dict` — ask the LLM for the next configuration. Validates
  the LLM's response against the search space and retries up to
  `max_retries` times on validation failure, feeding the error back to
  the LLM each retry.
- `observe(cfg, value, note=None)` — record a trial result. Appends to
  `history` and to `context` so the LLM sees it on the next `suggest()`.
- `best` — property returning `{"cfg": ..., "value": ...}` for the
  winning trial (sign-aware on `objective`), or `None` if no trials yet.
- `history` — list of `Trial(cfg, value, note)` records.
- `context` — the `Context` object (see below).
- `save(path)` / `Tuner.load(path, provider=...)` — JSON serialization.
  The provider is never serialized (no API keys on disk); the user
  supplies a fresh provider on load.

### Hyperparameter types

```python
lt.Float(name, description, bounds=None, initial=None)
lt.Int(name, description, bounds=None, initial=None)
lt.Choice(name, description, options, initial=None)
```

`bounds` and `initial` are optional. The LLM is told the range and any
initial value via the tool's parameter description; if `bounds` is set,
returned values are validated and out-of-range proposals trigger a
retry.

### `Context`

A first-class, ordered, append-only text log shown to the LLM on every
`suggest()`. Add free-form text or snapshot files:

```python
opt.context.add(text="found that lr > 1e-2 diverges with depth=12")
opt.context.add_file(path="model.py")        # snapshot at call time
opt.context.refresh()                         # re-read all snapshotted files
opt.context.refresh(path="model.py")          # or a specific one
```

The Tuner also auto-appends the LLM's full response (reasoning + content
+ proposed config) on every `suggest()`, and the trial result on every
`observe()` — so the LLM sees its own past thinking and the empirical
outcomes.

### `Provider`

The abstraction over LLM backends. Ships with one concrete provider:

```python
lt.OpenRouter(
    model="anthropic/claude-sonnet-4-6",
    api_key=None,             # falls back to OPENROUTER_API_KEY env var
    thinking_budget=10000,    # reasoning tokens; 0 disables
    max_tokens=2000,
    max_retries=3,            # transient empty-response retries
    extra_body=None,          # passthrough for vendor-specific knobs
)
```

OpenRouter speaks the OpenAI Chat Completions API, so most providers
(Anthropic, OpenAI, Google, Mistral, vLLM endpoints, etc.) are reachable
through it. Vendor SDK exceptions propagate unchanged — `llmtuna` does
not heuristically reclassify them.

To add a custom backend, subclass `Provider`:

```python
from llmtuna.providers.base import Provider

class MyBackend(Provider):
    def propose(self, system: str, user: str, tool_spec: dict) -> dict:
        # ...call your LLM, return:
        return {"reasoning": "...", "content": "...", "tool_args": {...}}
```

### Customizing prompts and formatters

The system prompt and the three formatter functions (`format_proposal`,
`format_result`, `build_user_message`) are overridable on the `Tuner`:

```python
opt = lt.Tuner(
    provider=...,
    space=[...],
    system_prompt="You are an expert in transformer pretraining...",
    format_result=lambda cfg, value, note: f"trial: {cfg} -> {value}",
)
```

Defaults live in `llmtuna/defaults.py` — read them as the spec.

## Testing

The default `pytest` invocation runs ~150 mock-backed unit tests in
under a second:

```bash
uv run pytest
```

Live-API integration tests live separately and require an OpenRouter
key. They use Haiku-4.5 with reasoning disabled (~$0.001 per test):

```bash
export OPENROUTER_API_KEY=...
uv run pytest tests/integration/
```

To write tests against your own code that uses `llmtuna`, the
`MockProvider` is available for dependency injection (intentionally
not exported at the top level since it is a test utility):

```python
from llmtuna.providers.mock import MockProvider

provider = MockProvider(responses=[
    {"lr": 1e-3, "depth": 5},
    {"lr": 5e-4, "depth": 8},
])
opt = lt.Tuner(provider=provider, space=[...])
cfg = opt.suggest()    # returns {"lr": 1e-3, "depth": 5}
```

## Citation

If you use `llmtuna` in academic work, please cite:

```bibtex
@software{stern2026llmtuna,
  title  = {llmtuna: LLM-driven hyperparameter optimization},
  author = {Stern, Alex},
  year   = {2026},
  url    = {https://github.com/alexkstern/llmtuna},
}
```

And the work that inspired it:

```bibtex
@article{yang2023large,
  title   = {Large language models as optimizers},
  author  = {Yang, Chengrun and Wang, Xuezhi and Lu, Yifeng and Liu, Hanxiao and Le, Quoc V and Zhou, Denny and Chen, Xinyun},
  journal = {arXiv preprint arXiv:2309.03409},
  year    = {2023}
}
```
