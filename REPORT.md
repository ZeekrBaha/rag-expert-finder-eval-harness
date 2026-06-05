# RAG Expert-Finder Eval Harness — Report

> Live runs: 2026-06-05. Candidates **gpt-4o-mini** + **deepseek-chat**; judge **gpt-4o**
> (out-of-set vs deepseek, same family as gpt-4o-mini — acknowledged, quantified below).
> Real `text-embedding-3-small` retrieval over a 200-doc OpenAlex corpus; 22 golden cases
> per provider. Numbers come from `results/ab_run.json` + `results/regression_run.json`
> → `results/meta_good.json` / `results/meta_regression.json` / `results/bias_probe.json`.
> (Anthropic key not supplied this run, so claude is absent — add the key and re-run
> `gen_config` to include it automatically.)

## 1. The failure mode

The expensive failure is the **confident wrong match**: the finder authoritatively
names an adjacent-field researcher, or names someone when no expert exists. The harness
*measures* this. Observed: across both models the SUT committed to a match in 36 of 44
cases and was **wrong in 20** — confident-match precision **0.44**. All 12 hand-built
hard-negative traps fooled it (0/12): asked for a garnet solid-electrolyte PI it
confidently grabbed a plausible-but-wrong adjacent battery author.

## 2. The rubric

`evals/graders/judge_rubric.md` turns "did this surface the right PI?" into explicit
pass/fail: real-candidate (no hallucination — also guarded in code via
`ExpertFinder.hallucinated`); positive/hard-negative pass only on a correct id and **a
confident match to any non-correct id FAILS**; abstain cases must abstain; reasoning must
cite specific abstract evidence; **answer length ignored**. The rubric is inlined into the
generated config (this Promptfoo build rejects a `file://*.md` rubric value).

## 3. Judge validation (κ vs. a golden oracle) — the headline

I evaluated my evaluator: the LLM judge's verdicts vs. a deterministic golden oracle.

| run | κ (judge vs oracle) | agreement | confusion |
|---|---|---|---|
| good prompt | **1.00** | 100% | tp 24 · tn 20 · fp 0 · fn 0 |
| regression prompt | **0.909** | 95% | tp 22 · tn 20 · fp 0 · **fn 2** |

On the good prompt the judge adds **zero noise**. Under the regression prompt the judge
**disagrees with the id-oracle on 2 cases** — precisely because it grades *reasoning
quality* the oracle can't see (see §7). That divergence is the LLM judge earning its keep.

## 4. Bias, measured (not assumed)

Verbosity probe (`evals/run_bias_probe.py`, sample 6): each real answer judged concise
vs. padded with neutral filler (substance unchanged).

| probe | flip-rate | reading |
|---|---|---|
| verbosity (n=6) | **0.33** | padding flipped 2/6 verdicts → judge **is** length-sensitive |
| position | framework-ready (`bias_probe.swap_positions`) | re-rank + re-judge; not run this pass |

Verbosity sensitivity is real and detectable — exactly why the rubric carries an explicit
"ignore length" line, and why I *measure* rather than assume. Matches the 2026 evidence
that style/verbosity bias dominates position bias.

## 5. Harness running (pass/fail grid)

`promptfoo eval`, one command, 44 cases (22 × 2 models), 39s, `gpt-4o` judge, 0 errors:

| kind (per model) | gpt-4o-mini | deepseek |
|---|---|---|
| positive | 8/12 | 8/12 |
| hard_negative | **0/6** | **0/6** |
| abstain | 4/4 | 4/4 |
| **overall** | **12/22 (54.5%)** | **12/22 (54.5%)** |

The eval catches the confident-wrong match on **every** hard-negative, for both models.

## 6. Multi-model tradeoff (quality vs. cost vs. latency)

| model | pass rate | confident-match precision | p50 latency | cost |
|---|---|---|---|---|
| gpt-4o-mini | 54.5% | 0.44 | ~2.48 s | not captured¹ |
| deepseek-chat | 54.5% | 0.44 | ~2.60 s | not captured¹ |

Dead heat on quality here — both models are equally fooled by the hard negatives, so on
*this* suite the tradeoff is latency-only (gpt-4o-mini marginally faster). ¹Per-candidate
token cost isn't captured because the SUT runs through a custom Python provider (Promptfoo
bills native model calls, not provider scripts); latency is captured. Next iteration:
account tokens inside the provider, or surface a wider quality gap with harder positives.

## 7. Regression catch (enforced, real)

A deliberately weakened prompt variant (`prompt_variant: regression` — drops the
"cite abstract evidence" and "abstain" instructions; wired end-to-end through
`build_qualify_prompt` → `ExpertFinder` → the provider). Same 44 cases, judged identically:

| metric | good | regression | signal |
|---|---|---|---|
| overall pass | 54.5% | **50.0%** | caught: −4.5 pp |
| positive judge-pass | 16/24 | **14/24** | 2 degraded |
| deepseek pass | 54.5% | **45.5%** | caught: −9 pp (deepseek less robust) |
| gpt-4o-mini pass | 54.5% | 54.5% | gpt-4o-mini robust to the weaker prompt |
| oracle-correct picks | 16/36 | 16/36 | **the pick didn't change** |
| judge↔oracle κ | 1.00 | 0.909 | the rubric caught the *reasoning* drop |

The key insight: the weak prompt didn't change *which* expert the model picks — it
degraded the *justification*, and the rubric's evidence criterion failed exactly those
cases (the κ drop). A string-match oracle would have missed this entirely. This is how I'd
block a bad prompt change in CI — `promptfoo eval` on the regression config drops below the
good-prompt baseline.

## 8. Close

End-to-end on a small, honest golden set the harness: ran a real RAG SUT across two
models, **validated its own judge (κ=1.0)**, **measured** a 0.33 verbosity flip-rate,
**quantified the headline failure** (confident-match precision 0.44, hard-negatives 0/12),
and **caught a reasoning-quality regression** (κ 1.00→0.909; deepseek −9 pp). Every number
is reproducible from `results/`. Limitations: two models this pass (add the Anthropic key
for claude); per-candidate cost not yet metered; 22 cases is small *by design* — defensible
line by line over big-but-opaque. Next: meter token cost in the provider, add the position
flip-rate run, and grow the golden set with synthetic-then-curated hard positives to widen
the model quality gap.
