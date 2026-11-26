# Demo data: a bullet-point LoRA on Qwen3-8B, trained and sampled on Tinker

Saved from the live run in section 4.8 of the research note
(https://albertlieyingadrian.github.io/steerlab/experiments/behavior-diff/report/), first published in
`experiments/behavior-diff/results/tinker/report/` of https://github.com/albertlieyingadrian/steerlab.

| File | Contents |
|---|---|
| `prompts.jsonl` | the default suite's 128 prompts (64 items × 2 condition draws, seed 0) |
| `replies/base.jsonl` | Qwen3-8B on Tinker, seed 1, thinking off, temperature 0.7, 300 new tokens |
| `replies/base2.jsonl` | the same base, seed 2 (the same-model control) |
| `replies/tuned.jsonl` | a rank-16 LoRA trained for 30 steps to answer in bullet points, seed 2 |
| `judge_cache.jsonl` | every judge reply (`qwen/qwen3-235b-a22b-2507`), keyed by a hash of the request, with its cost |
| `expect.json` | prediction P8, written before the run |

`bdiff demo` replays the judge from the cache, so it needs no key and makes no network call.
