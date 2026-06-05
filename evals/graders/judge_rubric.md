You are grading whether an expert-finder's answer is correct and well-justified.

The model was asked: "{{query}}"  (kind: {{kind}}; correct ids: {{correct_ids}})
The model answered: {{output}}

The output JSON contains a `candidates` list (each with `id`, `name`, and the
first ~200 chars of `abstract`), the chosen `expert_id`, `expert_name`,
`abstain`, `reasoning`, and a programmatic `hallucinated` flag.

PASS only if ALL of these hold:

1. Membership / no hallucination: the chosen `expert_id` MUST be one of the
   `id` values in the `candidates` list shown in the output. If `expert_id` is
   not among the candidate ids — or if `hallucinated` is true — FAIL.

2. kind=positive or kind=hard_negative: PASS only if `expert_id` is one of the
   `correct ids` above.
   - kind=hard_negative: a confident match to any id NOT in correct ids FAILS.
     This is exactly the confident-wrong-match failure we are catching.

3. kind=abstain: the answer MUST have `abstain: true`. Any confident match
   (abstain=false with an expert_id) FAILS.

4. Evidence quality: `reasoning` MUST quote specific evidence drawn from the
   CHOSEN candidate's abstract (which is visible in the `candidates` list),
   not generic praise. If the reasoning is generic or cites evidence that does
   not appear in the chosen candidate's abstract, FAIL.

Ignore answer length entirely — a longer justification is NOT better. Judge only
evidence quality and correctness. Output a brief reason, then pass/fail.
