# llmtuna

LLM-driven hyperparameter optimization with a torch-style ask/tell API.

`llmtuna` is a general-purpose hyperparameter optimizer: you describe a
search space, point it at any system you can run and measure, and a large
language model proposes configurations to try. It works for neural
network training, classical ML (XGBoost, random forests, SVMs),
reinforcement-learning agents, simulations, compiler/runtime flags —
anything with knobs and a scalar metric.

The LLM sees a rolling text context — whatever notes, code, papers, or
prior trial results you choose to feed it — and reasons over them to
suggest the next configuration.

Inspired by work on LLMs-as-optimizers (Yang et al., 2023).

## Why this exists

Classical samplers (Optuna's TPE, Ax/BoTorch, CARBS, hyperopt) are
oblivious to domain knowledge, prior literature, or anything the
scientist knows but hasn't encoded as a numerical prior. An LLM can
read a file describing your system, look at related results, and
incorporate that into proposals directly.

The library is intentionally minimal: three hyperparameter types
(`Float`, `Int`, `Choice`), one rolling text context, one provider
abstraction, one `Tuner` class.

## Install

Not on PyPI yet — install from a local clone using
[uv](https://github.com/astral-sh/uv):

```bash
git clone git@github.com:alexkstern/llmtuna.git
cd llmtuna
uv sync --extra dev
```

Run the test suite to confirm:

```bash
uv run pytest
```

To use it from another project, point at the local checkout:

```bash
uv add --editable /path/to/llmtuna
```

## Quickstart

`llmtuna` makes no assumptions about what's behind your `train_and_eval`
call — anything returning a scalar metric works.

```python
import llmtuna as lt

opt = lt.Tuner(
    provider=lt.OpenRouter(model="anthropic/claude-sonnet-4-6"),
    space=[
        lt.Float(name="learning_rate",  description="learning rate",
                 bounds=(1e-6, 1.0), initial=1e-3),
        lt.Int(name="n_layers",         description="model depth",
               bounds=(1, 20)),
        lt.Float(name="regularization", description="weight decay / penalty strength",
                 bounds=(0.0, 0.1), initial=0.01),
    ],
    objective="minimize",
)

# Optional: hand the LLM whatever context helps — code, notes, papers
opt.context.add(text="brief description of your model and dataset")
opt.context.add_file(path="train.py")
opt.context.add_file(path="model.py")
opt.context.add_file(path="dataloader.py")
opt.context.add_file(path="docs/prior_results.md")

# Optimization loop
for _ in range(30):
    cfg = opt.suggest()                  # LLM proposes a config
    metric = train_and_eval(**cfg)       # your training / evaluation code
    opt.observe(cfg=cfg, value=metric)   # report back

print(opt.best)         # {"cfg": {...}, "value": ...}
print(opt.history)      # list of all Trials in chronological order

opt.save(path="run.json")                # full transcript + state
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
opt.context.add(text="prior runs showed instability when learning_rate > 1e-2")
opt.context.add_file(path="model.py")             # snapshot at call time
opt.context.add_file(path="train.py")
opt.context.add_file(path="dataloader.py")
opt.context.refresh()                              # re-read all snapshotted files
opt.context.refresh(path="model.py")               # or a specific one
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
    system_prompt="You are an expert hyperparameter tuner for this domain...",
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
    {"learning_rate": 1e-3, "n_layers": 5, "regularization": 0.01},
    {"learning_rate": 5e-4, "n_layers": 8, "regularization": 0.005},
])
opt = lt.Tuner(provider=provider, space=[...])
cfg = opt.suggest()    # returns {"learning_rate": 1e-3, "n_layers": 5, ...}
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
