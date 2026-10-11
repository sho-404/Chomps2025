"""Attribute judge errors to the retrieval stage or to the judge, test split.

For each (lang, condition in pseudo_rag/bm25_k3/dense_k3) it joins the gpt-5-mini
single-judge consensus vote (majority over repeats) with the consolidated Claude
grades of the same context (chomps.grades) and classifies every item as:

  correct            judge consensus equals gold
  retrieval_failure  judge wrong and the graders found the context insufficient
  judge_failure      judge wrong, context sufficient, and the graders' own verdict
                     from that context equals gold (so the evidence was there)
  label_or_ambiguous judge wrong, context sufficient, and the graders also
                     disagree with gold (judge and graders agree with each other)

Writes ../results/tables/error_attribution.csv and prints up to --examples items
per category for the qualitative error analysis (index, question, answer, context head).
"""
import argparse
import csv
import json
from collections import Counter

from . import config, data, judge, metrics, retrieval

CONDS = ("pseudo_rag", "bm25_k3", "dense_k3")


def classify(gold, vote, grade):
    pred = 1 if vote == "y" else 0
    if pred == gold:
        return "correct"
    if not grade["sufficient"]:
        return "retrieval_failure"
    gv = {"incorrect": 1, "correct": 0}.get(grade["verdict"])
    if gv == gold:
        return "judge_failure"
    return "label_or_ambiguous"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=list(config.LANGS))
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    rows = []
    for lang in a.lang:
        gold = data.load_labels(lang, "test")
        ex = {e["index"]: e for e in data.load_examples(lang, "test")}
        grades = {}
        gpath = config.RESULTS / "retrieval_eval" / f"grades_{lang}_test.jsonl"
        for line in open(gpath, encoding="utf-8"):
            r = json.loads(line)
            grades[r["index"]] = r
        for cond in CONDS:
            vp = judge.votes_path(lang, "factuality", "test", cond, "gpt-5-mini", "judge", "default")
            if not vp.exists():
                print(f"[error_attribution] missing votes store {vp}; skipping {lang}/{cond}")
                continue
            votes = judge.load_votes(vp)
            cons = metrics.consensus(votes)
            ctx = retrieval.load_contexts(lang, "test", cond)
            cats = Counter()
            examples = {}
            for idx, g in gold.items():
                cat = classify(g["factuality"], cons[idx], grades[idx][cond])
                cats[cat] += 1
                examples.setdefault(cat, []).append(idx)
            n = len(gold)
            rows.append({"lang": lang, "condition": cond, "n": n, **{c: cats.get(c, 0) for c in
                         ("correct", "retrieval_failure", "judge_failure", "label_or_ambiguous")},
                         **{f"pct_{c}": round(100 * cats.get(c, 0) / n, 1) for c in
                            ("correct", "retrieval_failure", "judge_failure", "label_or_ambiguous")}})
            print(f"[{lang}/{cond}] {dict(cats)}")
            for cat in ("retrieval_failure", "judge_failure", "label_or_ambiguous"):
                for idx in examples.get(cat, [])[: a.examples]:
                    e = ex[idx]
                    gr = grades[idx][cond]
                    print(f"  <{cat}> {idx} gold={gold[idx]['factuality']} vote={cons[idx]} relevance={gr['relevance']} "
                          f"sufficient={gr['sufficient']} grader_verdict={gr['verdict']} note={gr['note']!r}\n"
                          f"     Q: {e['question'][:140]}\n     A: {e['output_text'][:160]!r}\n"
                          f"     CTX: {(ctx[idx] or '')[:200]!r}")
    out = config.RESULTS / "tables" / "error_attribution.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
