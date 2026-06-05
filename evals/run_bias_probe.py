"""Live verbosity-bias probe: does padding an answer change the judge's verdict?

A robust judge scores a padded-but-substantively-identical answer the same as the
concise one -> flip_rate ~ 0. We take a sample of the real run's answers, judge each
concise vs. padded, and report the flip rate (a measured number, not a claim).

Position bias is covered by the framework helper bias_probe.swap_positions (re-rank
candidates and re-judge); not run here to keep cost small.
"""
import json
import os
import sys

from evals.meta.bias_probe import pad_text, flip_rate


def _judge_pass(client, model: str, query: str, answer_json: str) -> bool:
    prompt = (
        "You grade an expert-finder answer. PASS if it commits to a correct, "
        "well-justified expert OR correctly abstains; FAIL otherwise. "
        "Ignore answer length entirely — a longer justification is NOT better.\n\n"
        f"Query: {query}\nAnswer JSON: {answer_json}\n\n"
        'Reply ONLY JSON: {"pass": true|false}'
    )
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=20,
        messages=[{"role": "user", "content": prompt}])
    txt = r.choices[0].message.content or ""
    return '"pass": true' in txt.lower() or '"pass":true' in txt.lower()


def main(path: str = "results/openai_run.json", sample: int = 6,
         out: str = "results/bias_probe.json") -> None:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("JUDGE_MODEL", "openai:gpt-4o").split(":")[-1]

    results = json.load(open(path))["results"]["results"][:sample]
    baseline, padded = [], []
    for r in results:
        query = r["vars"]["query"]
        out_obj = json.loads(r["response"]["output"])
        concise = json.dumps(out_obj)
        out_obj_pad = dict(out_obj)
        out_obj_pad["reasoning"] = pad_text(out_obj.get("reasoning", ""), factor=4)
        baseline.append(_judge_pass(client, model, query, concise))
        padded.append(_judge_pass(client, model, query, json.dumps(out_obj_pad)))

    fr = flip_rate(baseline, padded)
    report = {"sample": len(baseline), "verbosity_flip_rate": fr,
              "baseline": baseline, "padded": padded}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main(*sys.argv[1:])
