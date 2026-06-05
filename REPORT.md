# RAG Expert-Finder Eval Harness — Report

> Live run: 2026-06-05, OpenAI-only (candidate `gpt-4o-mini`, judge `gpt-4o`, real
> `text-embedding-3-small` retrieval over a 200-doc OpenAlex corpus, 22 golden cases).
> Numbers below are from `results/openai_run.json` → `results/meta.json` and
> `results/bias_probe.json`. The 3-model A/B and the regression demo are wired and
> pending the other two provider keys (see §6–§7).

## 1. The failure mode

The expensive failure is the **confident wrong match**: the finder authoritatively
names an adjacent-field researcher, or names someone when no expert exists in the
corpus. The harness is built to *measure* this, not just avoid it.

**Observed:** the SUT committed to a match in 18 of 22 cases and was **wrong in 10 of
them** — a confident-match precision of **0.44**. Every one of the 6 hand-built
hard-negative traps fooled it (0/6): asked for a garnet solid-electrolyte PI, it
confidently picked a plausible-but-wrong adjacent battery author instead.

## 2. The rubric

`evals/graders/judge_rubric.md` translates "did this surface the right PI?" into
explicit pass/fail: (1) the `expert_id` must be a real candidate (no hallucination —
also guarded in code via `ExpertFinder.hallucinated`); (2) positive/hard-negative
cases pass only on a correct id, and **a confident match to any non-correct id FAILS**;
(3) abstain cases must abstain; (4) reasoning must cite specific abstract evidence; and
it **explicitly ignores answer length**. In practice the rubric tracked ground truth
exactly (see §3).

## 3. Judge validation (κ vs. a golden oracle) — the headline

I evaluated my evaluator. The LLM judge's 22 verdicts were compared against a
deterministic golden oracle (correct-id membership / correct abstain):

| metric | value |
|---|---|
| Cohen's κ (judge vs oracle) | **1.00** |
| % agreement | 100% |
| confusion | tp 12 · tn 10 · fp 0 · fn 0 |

The judge adds **zero noise** on the binary correctness criterion — it faithfully
implements the spec. (Its incremental value is on the open-ended "evidence quality"
criterion the oracle can't check.) This is the number most portfolio evals never show.

## 4. Bias, measured (not assumed)

Quick verbosity probe (`evals/run_bias_probe.py`, sample of 6): each real answer was
judged concise vs. padded with neutral filler (substance unchanged).

| probe | flip-rate | reading |
|---|---|---|
| verbosity (n=6) | **0.33** | padding flipped 2/6 verdicts → the judge **is** length-sensitive |
| position | framework-ready (`bias_probe.swap_positions`) | re-rank + re-judge; not run this pass |

Finding: verbosity sensitivity is real and detectable — exactly why the rubric carries
an explicit "ignore length" instruction and why I *measure* rather than assume. This
matches the 2026 evidence that **style/verbosity bias dominates position bias**.

## 5. Harness running (pass/fail grid)

`promptfoo eval`, one command, 22 cases, 7s, `gpt-4o` judge:

| kind | passed | reading |
|---|---|---|
| positive | 8 / 12 | strong on clear matches; 4 genuine retrieval/qualify misses |
| hard_negative | 0 / 6 | **fooled every time** — the eval catches the confident-wrong match |
| abstain | 4 / 4 | correctly declines on out-of-corpus queries |
| **overall** | **12 / 22 (54.5%)** | |

Cost: ~21.5k judge tokens (~$0.10). Raw grid in `results/openai_run.json`
(`promptfoo view` for the interactive table).

## 6. Multi-model tradeoff (quality vs. cost vs. latency)

Config auto-includes only providers with a key (`evals/gen_config.py`). This run had
only `OPENAI_API_KEY`, so the A/B ran a single column (`gpt-4o-mini`). Add
`ANTHROPIC_API_KEY` + `DEEPSEEK_API_KEY` and re-run `gen_config` → the claude and
deepseek columns join automatically and Promptfoo emits the per-model
quality/cost/latency table. **Pending those two keys.**

## 7. Regression catch

Setup exists (`evals/prompts/regression_prompt.txt`, evidence requirement dropped).
Mechanism to wire: add a weakened variant of `build_qualify_prompt` selectable by flag,
re-run, and show precision dropping on the hard-negative cases. **Pending** (one small
code change + one rerun).

## 8. Close

End-to-end, on a small honest golden set, the harness: ran a real RAG SUT, **validated
its own LLM judge at κ=1.0**, **measured** a 0.33 verbosity flip-rate, and **quantified
the headline failure** (confident-match precision 0.44, hard-negatives 0/6). Every
number is reproducible from `results/`. Limitations: single model this pass (A/B and
regression pending the other keys); 22 cases is small by design — defensible line by
line over big-but-opaque. Next: add the two keys for the full A/B, wire the regression
prompt, and grow the golden set with synthetic-then-curated cases.
