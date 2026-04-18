# llmtuna

LLM-driven hyperparameter optimization.

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

Clone the repo somewhere on your machine, then add it as a local
editable dependency to your project with
[uv](https://github.com/astral-sh/uv):

```bash
git clone git@github.com:alexkstern/llmtuna.git ~/code/llmtuna

# from inside your own project's directory:
uv add --editable ~/code/llmtuna
```

Then in your code:

```python
import llmtuna as lt

opt = lt.Tuner(...)
```

## Quickstart

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

# Tell the LLM what you are doing
opt.context.add(
    text="You are optimizing the hyperparameters for a small ML pipeline "
         "that classifies sales records into 12 categories on ~50k rows of "
         "tabular features. Lower validation loss is better."
)

# Hand it the actual code, with short labels so it knows what each file is
opt.context.add(text="This is the training script:")
opt.context.add_file(path="train.py")
opt.context.add(text="This is the data loader:")
opt.context.add_file(path="dataloader.py")
opt.context.add(text="This is the model definition:")
opt.context.add_file(path="model.py")

for _ in range(30):
    cfg = opt.suggest()
    # e.g. cfg = {"learning_rate": 0.003, "n_layers": 8, "regularization": 0.01}
    val_loss = train_and_eval(**cfg)
    # e.g. val_loss = 0.482
    opt.observe(cfg=cfg, value=val_loss)

print(opt.best)         # {"cfg": {...}, "value": ...}
print(opt.history)      # list of all Trials in chronological order
opt.save(path="run.json")
```

## Concepts

### `Tuner`

The main class. Holds a `Provider`, a list of `Param` definitions
(the search space), an `objective` (`"minimize"` or `"maximize"`),
and a rolling `Context`. Exposes:

- `suggest() -> dict` — ask the LLM for the next configuration. Validates
  the LLM's response against the search space and retries up to
  `max_retries` times on validation failure (out-of-bounds, wrong type,
  missing or extra params), feeding the error back to the LLM each retry.
- `observe(cfg, value, note=None)` — record a trial result. Appends to
  `history` and to `context` so the LLM sees it on the next `suggest()`.
- `best` — property returning `{"cfg": ..., "value": ...}` for the
  winning trial (sign-aware on `objective`), or `None` if no trials yet.
- `history` — list of `Trial(cfg, value, note)` records.
- `context` — the `Context` object (see below).
- `save(path)` / `Tuner.load(path, provider=..., system_prompt=..., format_*=...)` —
  JSON serialization. The provider is never serialized (no API keys on
  disk); the user supplies a fresh provider on load. Custom formatters
  must be re-passed at load time or they revert to defaults.

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

A first-class, ordered text log shown to the LLM on every `suggest()`.
You add free-form text or snapshot files at any point — before the loop
or in between trials:

```python
opt.context.add(text="prior runs showed instability when learning_rate > 1e-2")
opt.context.add_file(path="model.py")     # reads the file NOW; stores its contents
```

Files are snapshotted at the time of the call. If you've since edited a
file on disk and want the LLM to see the new contents on the next
`suggest()`, call `opt.context.refresh()` (re-reads all file-backed
entries) or `opt.context.refresh(path="model.py")` (just one). Refresh
updates the stored text in place — it doesn't add or remove entries.

You can also prune the context — useful if a trial result was buggy or
you want to reset between phases:

```python
opt.context.pop(index=-1)   # remove the most recent entry
opt.context.pop(index=3)    # remove a specific one
opt.context.clear()         # remove everything
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
    base_url=None,            # override to point at any OpenAI-compatible endpoint
)
```

OpenRouter speaks the OpenAI Chat Completions API, so most providers
(Anthropic, OpenAI, Google, Mistral, vLLM endpoints, local llama.cpp /
Ollama, etc.) are reachable through it — pass a custom `base_url=` for
a non-OpenRouter endpoint. Vendor SDK exceptions propagate unchanged;
`llmtuna` does not heuristically reclassify them.

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

## In-depth: human-in-the-loop steering

You don't have to send every proposal to training. Inspect the LLM's
suggestion, and if it's nonsense in your domain, push back with a note
in the context and `continue` — the next `suggest()` will see your
correction and propose again.

```python
import llmtuna as lt

opt = lt.Tuner(
    provider=lt.OpenRouter(model="anthropic/claude-sonnet-4-6"),
    space=[
        lt.Int(name="batch_size", description="per-step batch size",
               bounds=(1, 4096), initial=64),
        lt.Int(name="seq_len",    description="sequence length in tokens",
               bounds=(64, 8192), initial=512),
        lt.Float(name="learning_rate", description="lr",
                 bounds=(1e-6, 1.0), initial=1e-3),
    ],
    objective="minimize",
)

opt.context.add(text="Training a small language model on an 80GB H100. "
                     "Total tokens per batch must stay under 100,000 to fit memory.")
opt.context.add_file(path="model.py")
opt.context.add_file(path="train.py")

MAX_TOKENS_PER_BATCH = 100_000
MAX_REJECTIONS_PER_TRIAL = 5

for _ in range(30):
    cfg = opt.suggest()

    # Pre-check the proposal. If bad, push back and ask again within the
    # same trial slot — outer loop only counts trials we actually trained.
    rejections = 0
    while cfg["batch_size"] * cfg["seq_len"] > MAX_TOKENS_PER_BATCH:
        opt.context.add(
            text=f"REJECTED: batch_size={cfg['batch_size']} × "
                 f"seq_len={cfg['seq_len']} = {cfg['batch_size'] * cfg['seq_len']} "
                 f"tokens, which exceeds the {MAX_TOKENS_PER_BATCH} per-batch "
                 f"memory limit. Re-propose with smaller values."
        )
        rejections += 1
        if rejections > MAX_REJECTIONS_PER_TRIAL:
            raise RuntimeError(
                f"LLM kept proposing oversized batches after {MAX_REJECTIONS_PER_TRIAL} rejections"
            )
        cfg = opt.suggest()   # re-ask; the LLM sees the rejection note in context

    val_loss = train_and_eval(**cfg)
    opt.observe(cfg=cfg, value=val_loss)

print(opt.best)
```

Two things worth noting:
1. **The constraint isn't part of the search space.** It's a runtime predicate
   — too coupled (it ties two params together) for a per-param `bounds=`
   to express. Adding it to context and rejecting in code is cleaner than
   inventing structural support for joint constraints.
2. **The LLM learns from rejections.** Past `REJECTED proposal: ...`
   notes stay in the context, so the LLM sees its own past mistakes and
   stops repeating them.

The same pattern works for: enforcing hardware-specific constraints,
catching configurations that diverged in training, injecting paper
findings mid-run, or stopping the search early when you know what you
want.

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
