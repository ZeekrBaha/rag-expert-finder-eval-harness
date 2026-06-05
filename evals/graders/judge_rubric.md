You are grading whether an expert-finder's answer is correct and well-justified.

The model was asked: "{{query}}"  (kind: {{kind}}; correct ids: {{correct_ids}})
The model answered: {{output}}

PASS only if ALL hold:
1. If a correct expert exists (kind=positive/hard_negative), the answer's
   `expert_id` is one of the correct ids and names a real candidate (no hallucination).
2. If kind=abstain, the answer MUST have `abstain: true`. Any confident match fails.
3. The `reasoning` cites specific evidence from the abstract, not generic praise.
4. The answer is concise and decision-ready.

Ignore answer length entirely — a longer justification is NOT better. Judge only
evidence quality and correctness. Output a brief reason, then pass/fail.
