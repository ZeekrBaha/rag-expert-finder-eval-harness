# RAG Expert-Finder Eval Harness — Report

> Skeleton. Each section's numbers get **pasted in after the live run** (Task 16:
> `promptfoo eval` + the meta-eval CLIs). Placeholders are marked `<!-- FILL POST-LIVE-RUN -->`.

## 1. The failure mode

The confident wrong match: an expert finder that authoritatively names an
adjacent-field researcher, or names someone when no expert exists in the corpus.
<!-- FILL POST-LIVE-RUN: a concrete example from the run where a candidate confidently grabbed a hard-negative -->

## 2. The rubric

A judge rubric (`evals/graders/judge_rubric.md`) that fails confident matches on
abstain cases, fails matches not grounded in abstract evidence, and explicitly
ignores answer length.
<!-- FILL POST-LIVE-RUN: how the rubric behaved in practice; any wording adjustments needed -->

## 3. Judge validation (kappa vs. human labels)

Cohen's kappa between the LLM judge's verdicts and the human golden labels, from
`python -m evals.meta.judge_agreement ...`.
<!-- FILL POST-LIVE-RUN: n=__, % agreement=__, kappa=__, confusion={tp,fn,tn,fp}=__ -->

## 4. Bias measured (verbosity + position flip-rate)

Flip-rates from `evals/meta/bias_probe.py`: how often the judge's verdict changes
under verbosity padding and under candidate-position swap (lower is better).
<!-- FILL POST-LIVE-RUN: verbosity flip-rate=__, position flip-rate=__; confirm style bias dominates position bias -->

## 5. Harness running (multi-model pass/fail grid)

The `promptfoo eval` pass/fail grid across the three candidate models, with the
out-of-set `openai:gpt-4o` judge.
<!-- FILL POST-LIVE-RUN: per-model pass rate over the 22 golden cases; paste/screenshot the grid -->

## 6. Multi-model tradeoff (quality vs. cost vs. latency)

Side-by-side comparison of `claude` vs. `gpt-4o-mini` vs. `deepseek` on accuracy,
cost, and latency for identical test cases.
<!-- FILL POST-LIVE-RUN: per-model accuracy / $ / latency from promptfoo view; which model wins on the quality-cost frontier -->

## 7. Regression catch

Re-running with `evals/prompts/regression_prompt.txt` (evidence requirement
dropped) and confirming precision drops on the hard-negative cases.
<!-- FILL POST-LIVE-RUN: precision before vs. after on hard-negatives; the specific cases that flipped to fail -->

## 8. Close

What the harness demonstrates end-to-end: a validated judge, measured bias, and a
caught regression over a small, honest golden set.
<!-- FILL POST-LIVE-RUN: one-paragraph summary tying the numbers together; limitations + next steps -->
