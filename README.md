# RAG Expert-Finder Eval Harness

A [Promptfoo](https://www.promptfoo.dev/) evaluation harness for a RAG **expert
finder**: given a research need, it retrieves the most relevant researcher
abstracts from an OpenAlex corpus and asks an LLM to qualify the single
best-matched principal investigator (PI) — or to abstain when no candidate truly
fits.

## What it is — and the failure mode it targets

The dangerous failure mode for an expert finder is the **confident wrong match**:
the system returns a plausible-looking, authoritative answer naming a researcher
whose work is in an *adjacent* field, or confidently names someone when the right
expert simply isn't in the corpus. A naive accuracy number hides this — the model
looks fluent and decisive while being wrong.

This harness is built to surface exactly that. The golden set deliberately
includes **hard negatives** (a tempting adjacent-field author the system is
likely to grab) and **abstain** cases (queries about a field absent from the
corpus, where the only correct answer is "no match"). The grading rubric fails
any confident match on an abstain case and any match that doesn't cite specific
evidence from the chosen abstract.

But a rubric is only as trustworthy as the judge that applies it. The star
feature of this harness is that it **measures its own judge** rather than
assuming it is correct.

## Architecture

Three layers, all wired so the numeric core is offline-testable with
deterministic fakes:

```
app/        System-under-test (the RAG pipeline), all external calls injected
  corpus.py        Record dataclass + jsonl load/save
  embedder.py      Embedder protocol · HashingEmbedder (offline) · OpenAIEmbedder · cosine
  retrieval.py     top_k(query_vec, matrix, k) -> [(idx, score)]
  metrics.py       recall_at_k · mrr
  llm_client.py    LLMClient protocol · FakeLLM · OpenAIChat / AnthropicChat / DeepSeekChat
  qualify.py       build_qualify_prompt · parse_qualify_output (pure)
  expert_finder.py ExpertFinder: embed corpus -> retrieve top-k -> qualify (injected deps)
  config.py        Settings from environment (keys, model names, top_k)
  golden.py        GoldenCase loader + 3-class validator

data/       Real, committed sample corpus + hand-built golden set
  fetch_corpus.py  work_to_record (pure) + fetch_corpus() via pyalex
  corpus.jsonl     200 committed OpenAlex records (offline sample)
  golden.jsonl     22 hand-labeled cases: 12 positive, 6 hard_negative, 4 abstain

evals/      Promptfoo config + the meta-eval that grades the grader
  provider.py          Python provider wrapping ExpertFinder (lazy client init)
  promptfooconfig.yaml 3 candidate models, llm-rubric judge, tests file
  build_tests.py       golden.jsonl -> evals/tests.json (pure transform)
  prompts/             good_prompt.txt · regression_prompt.txt
  graders/judge_rubric.md
  meta/                The "eval of the eval"
    stats.py           cohens_kappa · confusion_matrix · precision_recall (pure)
    judge_agreement.py agreement_report: judge-vs-human κ + confusion (pure + CLI)
    bias_probe.py      pad_text · flip_rate for verbosity/position bias (pure + CLI)
```

The system-under-test (`ExpertFinder`) takes its `Embedder` and `LLMClient` as
injected dependencies. In tests they are a `HashingEmbedder` (deterministic
bag-of-words) and a `FakeLLM` (returns queued responses), so the retrieval +
qualify logic, the metrics, the κ computation, and the bias flip-rate are all
unit-tested with **no keys and no network**. Live `promptfoo eval` runs swap in
the real OpenAI embedder and the real candidate models.

## What it proves

1. **The judge is validated, not assumed.** `evals/meta/judge_agreement.py`
   computes Cohen's **κ** between the LLM judge's pass/fail verdicts and the human
   golden labels, plus a confusion matrix. A judge with low κ is a judge you
   cannot trust — this turns "is the rubric any good?" into a measured number.

2. **Bias is measured, not hand-waved.** `evals/meta/bias_probe.py` quantifies
   two judge biases as **flip-rates** (the fraction of verdicts that change under
   a perturbation that should not matter):
   - *Verbosity bias* — `pad_text()` lengthens an answer with neutral filler
     without changing its substance; a good judge scores padded == concise. The
     rubric explicitly instructs the judge to ignore length.
   - *Position bias* — the same `flip_rate()` is applied to runs with the
     candidate order swapped.
   Lower flip-rate is better; the number is reported, not claimed.

3. **Regression is caught.** `evals/prompts/regression_prompt.txt` is a
   deliberately worse instruction that drops the "cite specific evidence"
   requirement. Wiring it in as the SUT instruction makes precision drop on the
   hard-negative cases — demonstrating the suite actually catches a quality
   regression instead of rubber-stamping it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm i -g promptfoo            # the eval runner (node 22 / npm 11)
cp .env.example .env          # then fill in your keys (see "Keys" below)
```

## Run order

```bash
# 1. Fetch the real corpus from OpenAlex (a 200-record sample is already committed).
python -m data.fetch_corpus

# 2. Compile the human golden set into Promptfoo test cases (evals/tests.json).
python -m evals.build_tests

# 3. Run the eval across all candidate models, graded by the out-of-set judge.
promptfoo eval -c evals/promptfooconfig.yaml

# 4. Open the interactive pass/fail grid (cost + latency per cell).
promptfoo view
```

## Meta-eval (grading the grader)

After a live run, validate the judge and probe it for bias:

```bash
# Judge-vs-human agreement: prints n, % agreement, Cohen's kappa, confusion matrix.
# Each argument is a JSON file of booleans (human labels, judge verdicts).
python -m evals.meta.judge_agreement human_labels.json judge_verdicts.json

# Bias flip-rate: import the pure functions, or wire baseline vs. perturbed
# judge verdicts through flip_rate() to report verbosity / position flip-rate.
python -c "from evals.meta.bias_probe import flip_rate; print(flip_rate(baseline, variant))"
```

## Multi-model A/B and the out-of-set judge

`promptfooconfig.yaml` runs three candidate models side by side — `claude`,
`gpt-4o-mini`, and `deepseek` — through the same provider, so you can compare
their pass rate, cost, and latency on identical test cases.

Crucially, the **judge is outside the candidate set**: grading is done by
`openai:gpt-4o`, which is *not* one of the candidates. This avoids self-preference
bias (a model judging its own family kindly). The judge model is pinned in the
config to guard against silent drift.

## Keys needed

Keys live in `.env` (gitignored, never logged). Live runs need:

- `OPENAI_API_KEY` — candidate `gpt-4o-mini`, the embedder
  (`text-embedding-3-small`), **and** the judge (`openai:gpt-4o`).
- `ANTHROPIC_API_KEY` — candidate `claude-3-5-sonnet`.
- `DEEPSEEK_API_KEY` — candidate `deepseek-chat`.

So you need the judge plus the three candidate keys. If `OPENAI_API_KEY` is
absent, the provider falls back to the offline `HashingEmbedder` so the pipeline
still runs (retrieval quality is lower, but nothing crashes). See `.env.example`
for the model-name overrides (`CANDIDATE_*`, `JUDGE_MODEL`, `EMBED_MODEL`,
`TOP_K`).

## Tests

All harness logic is offline-tested with deterministic fakes — no keys, no
network:

```bash
pytest -q
```

The provider imports cleanly with no keys (client init is lazy), so the import
test runs in CI without secrets.
