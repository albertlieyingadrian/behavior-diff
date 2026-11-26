# Changelog

## 0.1.0 (2026-11-26)

First release as its own package, `behavior-diff` on PyPI (import name `bdiff`). Split out of
[steerlab](https://github.com/albertlieyingadrian/steerlab)'s `packages/behavior-diff`, with its history.

- `bdiff run --base MODEL --tuned MODEL`: sample, judge, categorize, test, with a same-model control and an HTML
  report (also installed as `behavior-diff`).
- `bdiff generations G0 G1 ...`: per-generation drift. Each step, each generation against the base, a same-model
  control, one shared category set, and "slow drift" (changed against the base though no single step changed it).
- `--expect` and `bdiff score`: pre-registered vs post hoc scoring of expected changes. Expectations are locked with
  a SHA-256 and a UTC time before sampling.
- `bdiff demo`: the research note's live Tinker result, replayed from packaged data with no key and no network.
- A `modal:` backend: vLLM on a Modal GPU, with an optional LoRA adapter from the Hugging Face Hub.
- `bdiff sample` takes any backend spec. `bdiff compare` is the old `bdiff run PROMPTS --ref --new` (that form still
  works).
