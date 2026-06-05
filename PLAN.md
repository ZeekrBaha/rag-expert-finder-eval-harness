# PLAN.md — RAG Expert-Finder Eval Harness

> Portfolio build for the **AI Evaluation Engineer** role at FirstIgnite.
> The deliverable is **an evaluation harness**, not an app. The system-under-test is intentionally
> small; the *evals* are the star — and crucially, this harness **measures** its own judge.
>
> Revision note (2026-06-04): folded in research-backed best practices — corrected bias
> priorities (style/verbosity > position, per 2026 evidence), added a **judge↔human meta-eval**,
> removed the self-preference conflict, switched success metric to precision/recall for the
> "confident wrong match" failure, wired Promptfoo's native RAG asserts + a CI regression gate,
> and replaced the Loom deliverable with a written `REPORT.md`. Changes marked **[v2]**.

---

## 0. Strategic intent (read first — this drives every decision)

FirstIgnite is hiring their **first dedicated evals engineer**. Their product is RAG over structured
research data: expert discovery, grants search, AI outreach. The JD names **Promptfoo** twice,
emphasizes **LLM-as-judge + failure modes**, **golden datasets → regression tests**, **multi-model
A/B with cost/latency/quality tradeoffs**, and **turning fuzzy outcomes into measurable rubrics**.

This project mirrors the *shape* of their problem (expert/researcher search over academic data) so
the subtext reads: "I already understand your problem." What makes it stand out: it **measures**
quality instead of just producing it — and **[v2]** it goes one level deeper than most portfolio
evals by **validating the evaluator itself** (judge vs. human agreement) and **measuring** bias
mitigation instead of merely claiming it.

**Three non-negotiables:**
1. Small and defensible > big and sprawling. A 30-example suite I can explain line-by-line beats 300 I can't.
2. I own every grader's reasoning. Use Claude Code to move fast, but never ship a grader I can't defend cold.
3. **[v2]** Every bias/mitigation/regression claim must resolve to a **number**, not a sentence.
   "I evaluated my evaluator" is the thesis of the whole piece.

---

## 1. Success criteria

- [ ] A Promptfoo eval suite that runs end-to-end with one command.
- [ ] A hand-curated golden dataset (20–30 queries) with labeled correct experts, **[v2]** hard
      negatives, and **abstain** ("no good expert in set") cases.
- [ ] An LLM-as-judge rubric translating *"did this surface the right principal investigator?"* into
      concrete pass/fail criteria.
- [ ] **[v2] Judge validation (meta-eval):** judge↔human agreement reported as Cohen's κ / %
      agreement against the golden labels. *The senior signal.*
- [ ] **[v2] Measured bias mitigation** (not just named): a bias **probe** reporting judge **flip
      rate** under (a) verbosity padding and (b) position swap. Bias priority led by **style/verbosity**
      (dominant in 2026 evidence), with position swap as cheap insurance.
- [ ] **[v2]** Primary metric = **precision/recall** on "confident wrong match" (false positive),
      not just pass rate.
- [ ] A multi-model A/B (Claude vs GPT-4o-mini vs DeepSeek) with a cost / latency / quality table,
      **[v2]** judged by a model **outside** the candidate set (no self-preference), **[v2]** using
      pairwise `select-best` for ranking.
- [ ] A regression demo: a deliberately worse prompt the suite **catches** with a score drop —
      **[v2]** enforced as a failing **CI check** via `promptfoo-action`, not just narrated.
- [ ] **[v2]** A written `REPORT.md` (replaces the Loom): failure mode → rubric → results → bias
      numbers → regression catch. Async-reviewable evidence on disk.
- [ ] (Stretch) A minimal review dashboard a non-engineer could use to eyeball pass/fails.
- [ ] A clean README a reviewer reads in 2 minutes.

---

## 2. Tech stack

- **Eval framework:** Promptfoo (CLI + `promptfooconfig.yaml`; native `llm-rubric`, `g-eval`,
  `select-best`, and RAG asserts `context-faithfulness` / `context-relevance` / `context-recall`).
- **CI gate:** `promptfoo-action` (GitHub Action) — the regression demo runs as a real PR check.
- **Data source:** OpenAlex API (free, no key; authors/works/concepts, h-index, citations) via
  **[v2]** [`pyalex`](https://github.com/J535D165/pyalex). arXiv API as fallback.
- **System-under-test:** lightweight Python "expert finder" — retrieve top-k abstracts, ask an LLM
  to qualify + justify the best-matched researcher.
- **Models for A/B:** Anthropic (Claude), OpenAI (gpt-4o-mini), DeepSeek (openai-compatible endpoint).
- **[v2] Judge model:** pinned, version-logged, and **not a member of the candidate set** for any
  given A/B comparison (avoids self-enhancement bias). Temperature 0, seed logged.
- **Embeddings (only if doing RAG):** `text-embedding-3-small` or a local open embedding model.
- **Dashboard (stretch):** single static HTML file reading Promptfoo's JSON output.

> Verify exact Promptfoo provider strings + assertion syntax against current docs before relying on
> the sketches below — APIs drift. Confirm DeepSeek's openai-compatible config too.

### 2.1 Costs & API keys **[v2]**

**What needs a key:** only graders/steps that call an LLM. Deterministic checks are free.

| Step | Needs a key? |
|---|---|
| Exact-match / regex / JSON-schema / recall@k / MRR | **No key** |
| Embedding similarity (if RAG retrieval kept) | Key **or** a local embedding model (free) |
| LLM-as-judge (`llm-rubric`, `g-eval`, `select-best`) | **Yes — one LLM key** |
| The 3 A/B candidates (Claude · gpt-4o-mini · DeepSeek) | **Yes — one key each** |

- **Not locked to OpenAI.** The judge can be OpenAI, Anthropic, DeepSeek, or **local (Ollama/vLLM → $0, no key)**. RAGAS/DeepEval *default* to OpenAI but are swappable; Promptfoo is provider-agnostic — you choose per provider string.
- **Keys this project needs:** the out-of-set **judge** key + the **3 candidate** keys for the A/B. Go fully local on the judge → zero judge cost (slower/weaker).
- **Cost control:** pin cheap models for bulk runs (gpt-4o-mini/DeepSeek are sub-cent per call), reserve the larger judge for the rubric pass; enable Promptfoo caching so re-runs don't re-bill; the suite is 20–30 cases × a few providers → **expect cents-to-low-dollars per full run**.
- **Hygiene:** all keys in `.env` (`.gitignore`d); log only model **name + version**, never the key.

### 2.2 Reporting / output **[v2]**

Reporting is mostly **free from Promptfoo** — no extra build:

- **`promptfoo view`** — local web UI: sortable pass/fail grid, side-by-side diffs across the 3 models, per-assertion drill-down.
- **HTML + JSON export** (`--output results/out.json` / `.html`) — JSON is the machine-readable record that feeds everything downstream.
- **CI annotations** — `promptfoo-action` posts the pass/fail summary on the PR (Phase 7).

Layered on top in this repo:
- **`results/`** — Promptfoo JSON/HTML + the meta-eval numbers (κ, confusion matrix, bias flip-rates) from Phase 5.
- **`dashboard/index.html`** (stretch) — reads the Promptfoo JSON, renders a PM-skimmable table.
- **`REPORT.md`** — the human narrative; every claim points back to a number in `results/`.

> Reference points: **DeepEval** → pytest report + Confident AI cloud dashboard; **RAGAS** → pandas DataFrame (pipe to Phoenix/Langsmith for visuals); **TruLens** → Streamlit leaderboard. This project leans on Promptfoo's native web UI + JSON, so no separate reporting stack is required.

---

## 3. Repo structure

```
rag-expert-finder-eval-harness/
├── PLAN.md                 # this file
├── README.md               # what it is, how to run, what it proves
├── REPORT.md               # [v2] the written walkthrough (replaces Loom)
├── .github/workflows/
│   └── evals.yml           # [v2] promptfoo-action — regression gate in CI
├── data/
│   ├── fetch_corpus.py     # pull ~150–300 abstracts from OpenAlex (via pyalex)
│   ├── corpus.jsonl        # the abstract corpus
│   └── golden.jsonl        # 20–30 hand-labeled queries: positives, hard negatives, abstains
├── app/
│   └── expert_finder.py    # system under test (retrieve + qualify)
├── evals/
│   ├── promptfooconfig.yaml
│   ├── prompts/
│   │   ├── good_prompt.txt
│   │   └── regression_prompt.txt   # deliberately worse, for the demo
│   ├── graders/
│   │   └── judge_rubric.md
│   └── meta/
│       ├── judge_agreement.py      # [v2] κ / %-agreement judge vs golden labels
│       └── bias_probe.py           # [v2] verbosity + position flip-rate probe
├── results/                # promptfoo json/html output + meta-eval numbers
└── dashboard/
    └── index.html          # stretch: review UI
```

---

## 4. Phased build plan (one focused day, ~8 hrs)

### Phase 0 — Setup (30 min)
- [ ] Init repo, venv, install promptfoo (`npm i -g promptfoo`) and Python deps (`pyalex`, etc.).
- [ ] API keys in `.env` (Anthropic, OpenAI, DeepSeek). **Never hardcode keys; `.env` in `.gitignore`.**

### Phase 1 — Data (1 hr)
- [ ] `fetch_corpus.py` via **`pyalex`**: query OpenAlex across 3–4 research areas (e.g. solid-state
      batteries, CRISPR delivery, federated learning). Pull title + abstract + author + institution
      + h-index/citations → `corpus.jsonl`.
- [ ] Keep to ~150–300 records. Real enough, small enough to reason about.

### Phase 2 — System under test (1.5 hr)
- [ ] `expert_finder.py`: takes a natural-language research need, retrieves top-k candidate
      works/authors, asks an LLM to pick + justify the best-matched expert.
- [ ] **Lean fallback if time is short:** skip retrieval; feed a fixed candidate list and eval only
      the *qualify/justify* step (closer to their AI SDR agent anyway). **Decide by ~12pm.**
- [ ] **[v2]** Output must support an **abstain** ("none of these is a strong match") so calibration
      is testable.

### Phase 3 — Golden dataset (1 hr) **[v2] reshaped**
- [ ] `golden.jsonl`: 20–30 queries, each with labeled correct expert(s) + one-line rationale.
- [ ] **[v2] Deliberately include three classes** (this is what makes the suite catch real failures):
  - **Positives** — a clearly correct PI exists in the candidate set.
  - **Hard negatives** — an *adjacent-field* researcher is the tempting-but-wrong pick (tests
    "confident wrong match" directly).
  - **Abstain cases** — no good expert in the set; correct answer is to decline. (Tests calibration;
    almost no portfolio suite does this.)
- [ ] Frame on `REPORT.md` as curated from "real interactions" — the feedback-loop bullet made concrete.

### Phase 4 — Promptfoo + LLM-as-judge (2 hr — THE CORE)
- [ ] Write `judge_rubric.md`: translate *"does this surface the right PI?"* into explicit criteria:
  - Names a real researcher **present in the candidate set** (no hallucinated names).
  - The researcher's work **directly matches** the query's technical area.
  - Justification cites **specific evidence from the abstract**, not generic praise.
  - **[v2]** Correctly **abstains** when no candidate is a strong match.
  - Output is concise and decision-ready.
- [ ] Wire `llm-rubric` assertions in `promptfooconfig.yaml`. **[v2]** If retrieval survives the noon
      cut, also add native `context-faithfulness` / `context-relevance` / `context-recall` and
      evaluate retrieval separately (recall@k, MRR) from the qualify step — **two-stage eval**.
- [ ] **[v2] Bias mitigation — implement, MEASURE, and be ready to say one sentence on each.**
      Lead with the bias that actually dominates:
  - *Style / verbosity bias (DOMINANT — lead here):* instruct the judge to ignore length; apply a
    **length-controlled** comparison (AlpacaEval-2 style) and a **truncation control**. Measured in
    `bias_probe.py`: same answer padded vs. concise → report judge **flip rate**.
  - *Position bias (small — cheap insurance):* swap candidate order and re-judge; report flip rate.
    Evidence: position bias ≤0.04 vs. style 0.76–0.92 (2026 study) — so this is insurance, not the
    headline.
  - *Judge drift:* pin exact judge model version + temp 0 + seed; log all three with every run.
  - *Self-preference:* **[v2]** judge with a model **outside** the candidate set for each comparison.
    Never let Claude judge Claude.

Sketch:
```yaml
# promptfooconfig.yaml (illustrative — verify syntax vs current docs)
prompts:
  - file://prompts/good_prompt.txt
providers:
  - anthropic:messages:claude-3-5-sonnet-20241022
  - openai:chat:gpt-4o-mini
  - id: openai:chat:deepseek-chat
    config:
      apiBaseUrl: https://api.deepseek.com
defaultTest:
  assert:
    - type: llm-rubric
      # [v2] judge is OUTSIDE the candidate set to avoid self-preference;
      # pinned + logged for drift control.
      provider: openai:chat:gpt-4o            # judge ≠ any A/B candidate
      value: file://graders/judge_rubric.md
    # [v2] add when retrieval is kept:
    # - type: context-faithfulness
    # - type: context-recall
tests: file://../data/golden.jsonl
```

### Phase 5 — Judge validation / meta-eval (1 hr) **[v2] NEW — the headline**
- [ ] `meta/judge_agreement.py`: run the judge over the golden set, compare its pass/fail to the
      **human labels**, report **Cohen's κ + % agreement + a confusion matrix**.
- [ ] State the bar plainly: "My judge agrees with my human labels at κ=0.X. Here's where it
      disagrees and why." This is "I evaluated my evaluator" — the single most senior moment.
- [ ] `meta/bias_probe.py`: emit the verbosity + position **flip-rate** numbers from Phase 4.

### Phase 6 — Multi-model A/B (1 hr) **[v2] hardened**
- [ ] Run the suite across all three candidate providers.
- [ ] **[v2]** Rank with pairwise **`select-best`** (more reliable than absolute scores), judged by
      the out-of-set judge.
- [ ] Produce a **cost / latency / quality** table: **precision/recall** (not just pass rate),
      avg tokens, $/run (Promptfoo's built-in cost tracking), p50/p95 latency.
- [ ] One-line takeaway per model — the "quantify tradeoffs" bullet, backed by a real multi-model run.

### Phase 7 — Regression demo + CI gate (45 min) **[v2] enforced, not narrated**
- [ ] `regression_prompt.txt`: subtly worse (drops the "cite evidence from the abstract" instruction).
- [ ] Run both prompts, show the score drop (precision drop on hard negatives is the cleanest signal).
- [ ] **[v2]** `.github/workflows/evals.yml` using `promptfoo-action`: the worse prompt **fails the PR
      check**. "This is how I'd block a bad change in CI" — literal, not a claim.

### Phase 8 — Dashboard (STRETCH, 1 hr)
- [ ] `dashboard/index.html`: read Promptfoo JSON, render a pass/fail table a PM could skim. Hits the
      "tooling non-engineers actually use" bullet.

### Phase 9 — REPORT.md + README (30 min) **[v2] replaces Loom prep**
- [ ] `REPORT.md` — the written walkthrough (structure below).
- [ ] `README.md` — what it is, one-command run, what it proves.

---

## 5. REPORT.md structure (the deliverable — replaces the Loom) **[v2]**

Written like an evals engineer, not a builder. One screenshot/table per section:

1. **The failure mode I care about.** "Expert search must surface the *right* PI. The expensive
   failure is a confident, wrong match — a false positive. So that's what I built an eval to catch,
   and I measure it as precision on hard negatives."
2. **Fuzzy → rubric.** Show `judge_rubric.md`, walk one criterion. The translation move.
3. **I validated the judge.** κ vs. human labels + confusion matrix. *Lead with this — it's the differentiator.*
4. **Bias, measured.** Verbosity + position flip-rate numbers; one sentence each; note style ≫ position.
5. **The harness running.** One command → results.
6. **Multi-model tradeoff.** Cost/latency/quality table (precision/recall), takeaway per model.
7. **The regression catch.** Worse prompt → precision drops → failing CI check screenshot.
8. **Close.** "Small on purpose — every line defensible. This is how I'd make evals feel as natural
   as unit tests for your team."

---

## 6. Prior art (researched) — what exists, what to borrow, how this differs

| Repo / framework | What it is | Borrow |
|---|---|---|
| [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) | the framework; RAG + judge examples | `g-eval`, `select-best`, multi-judge voting, native cost/token tracking, RAG asserts |
| [promptfoo/promptfoo-action](https://github.com/promptfoo/promptfoo-action) | GitHub Action | make the regression catch a real PR gate |
| [sksoumik/llm-as-judge](https://github.com/sksoumik/llm-as-judge) | 2026 bias-mitigation study (9 strategies) | corrected bias priorities; probe methodology |
| [CSHaitao/Awesome-LLMs-as-Judges](https://github.com/CSHaitao/Awesome-LLMs-as-Judges) | survey of judge methods | citations for the README's "senior signal" |
| [explodinggradients/ragas](https://github.com/explodinggradients/ragas) · DeepEval · TruLens | RAG eval frameworks | metric definitions; **synthetic data gen** to grow the suite around the hand-labeled core |
| [ai-art-dev99/agentic-RAG](https://github.com/ai-art-dev99/agentic-RAG) | RAG template w/ RAGAS+TruLens+DeepEval | structure reference for two-stage eval |
| [J535D165/pyalex](https://github.com/J535D165/pyalex) | clean OpenAlex client | `fetch_corpus.py` (h-index/citations for richer labels) |

**Positioning.** RAGAS/DeepEval/TruLens already give reference-free RAG metrics — don't rebuild them.
This project's edge is exactly what frameworks *don't* hand you: a **hand-curated golden set with
hard negatives + abstains**, a **measured judge validation (κ)**, and a **bias probe with real
numbers**, on a domain that mirrors FirstIgnite's. Lean into the human-grounded rubric and the
meta-eval — that's the moat.

---

## 7. Guardrails

- Keep API keys in `.env`; never commit them (`.env`, `*.key` in `.gitignore`).
- If retrieval eats the morning, cut it (Phase 2 fallback) — **protect Phases 4–7**, they're the substance.
- Don't ship a grader I can't explain. After each phase, re-read the output and defend it cold.
- **[v2]** Every claim in `REPORT.md` must point to a number in `results/`. No adjective without a metric.
- **[v2]** Stop building at ~6pm; spend the last block writing `REPORT.md`. A modest harness that
  *measures its own judge* beats an ambitious one that doesn't.

---

## 8. Appendix — research sources

- [Promptfoo](https://github.com/promptfoo/promptfoo) · [Evaluating RAG pipelines](https://www.promptfoo.dev/docs/guides/evaluate-rag/) · [LLM-as-a-judge guide](https://www.promptfoo.dev/docs/guides/llm-as-a-judge/)
- [promptfoo-action (CI)](https://github.com/promptfoo/promptfoo-action)
- [sksoumik/llm-as-judge — bias-mitigation study (style ≫ position)](https://github.com/sksoumik/llm-as-judge)
- [CSHaitao/Awesome-LLMs-as-Judges (survey)](https://github.com/CSHaitao/Awesome-LLMs-as-Judges)
- [RAGAS](https://github.com/explodinggradients/ragas) · [agentic-RAG (RAGAS+TruLens+DeepEval template)](https://github.com/ai-art-dev99/agentic-RAG)
- [pyalex (OpenAlex client)](https://github.com/J535D165/pyalex)
