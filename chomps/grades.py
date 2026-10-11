"""Consolidate the Claude-graded retrieval relevance study.

Input: the workflow result (JSON) with, per batch, the two independent graders'
grades, the list of disagreements and the adjudicator's grades for those items.
Final grade per (item, condition, field): the two graders' value when they agree,
otherwise the majority of the three (grader 1, grader 2, adjudicator); when all
three differ, the adjudicator's value.

Outputs (../results/retrieval_eval/):
  grades_{lang}_test.jsonl       final grade per item and condition (+ gold label)
  grading_agreement.csv          inter-grader agreement (raw % and Cohen's kappa) per field
  grading_summary.csv            per lang/condition: relevance distribution, sufficiency,
                                 grader verdict vs gold (accuracy, macro-F1 over decidable items)
Usage: python -m chomps.grades --workflow-json <file>
"""
import argparse
import csv
import json
from collections import Counter

from sklearn.metrics import cohen_kappa_score, f1_score

from . import config, data

CONDS = ("pseudo_rag", "bm25_k3", "dense_k3")
FIELDS = ("relevance", "sufficient", "verdict")


def _majority(vals, fallback):
    c = Counter(vals).most_common()
    if len(c) == 1 or c[0][1] > c[1][1]:
        return c[0][0]
    return fallback


def consolidate(batch):
    g1 = {g["index"]: g for g in batch["grader1"]}
    g2 = {g["index"]: g for g in batch["grader2"]}
    adj = {g["index"]: g for g in (batch.get("adjudicated") or [])}
    if set(g1) != set(g2):
        raise ValueError(f"{batch['file']}: graders covered different items")
    final, pairs = {}, []
    for idx in g1:
        out = {"index": idx}
        for c in CONDS:
            a, b = g1[idx][c], g2[idx][c]
            pairs.append((c, a, b))
            res = {}
            for f in FIELDS:
                if a[f] == b[f]:
                    res[f] = a[f]
                else:
                    if idx not in adj:
                        raise ValueError(f"{batch['file']}: {idx}/{c}/{f} disagreement without adjudication")
                    res[f] = _majority([a[f], b[f], adj[idx][c][f]], adj[idx][c][f])
            res["note"] = a["note"] or b["note"]
            res["graders_agreed"] = all(a[f] == b[f] for f in FIELDS)
            out[c] = res
        final[idx] = out
    return final, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow-json", required=True)
    a = ap.parse_args()
    raw = json.load(open(a.workflow_json, encoding="utf-8"))
    batches = raw["result"] if isinstance(raw, dict) else raw
    out_dir = config.RESULTS / "retrieval_eval"
    agreement_rows, summary_rows = [], []
    for lang in config.LANGS:
        gold = data.load_labels(lang, "test")
        final, pairs = {}, []
        for b in batches:
            if b["lang"] != lang:
                continue
            f, p = consolidate(b)
            final.update(f)
            pairs += p
        if set(final) != set(gold):
            raise ValueError(f"{lang}: graded {len(final)} items, gold has {len(gold)}")
        with open(out_dir / f"grades_{lang}_test.jsonl", "w", encoding="utf-8") as fh:
            for idx in gold:  # dataset order
                row = dict(final[idx])
                row["gold_factuality"] = gold[idx]["factuality"]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        # inter-grader agreement per field (over all conditions pooled, and per condition)
        for field in FIELDS:
            for cond in ("all",) + CONDS:
                sel = [(x, y) for c, x, y in pairs if cond == "all" or c == cond]
                xs = [str(x[field]) for x, _ in sel]
                ys = [str(y[field]) for _, y in sel]
                raw_agree = sum(x == y for x, y in zip(xs, ys)) / len(xs)
                kappa = cohen_kappa_score(xs, ys) if len(set(xs) | set(ys)) > 1 else 1.0
                agreement_rows.append({"lang": lang, "field": field, "condition": cond, "n": len(xs),
                                       "raw_agreement": round(raw_agree, 4), "kappa": round(kappa, 4)})
        # summary per condition
        for c in CONDS:
            rel = Counter(final[i][c]["relevance"] for i in gold)
            suff = sum(final[i][c]["sufficient"] for i in gold)
            dec = [(gold[i]["factuality"], 1 if final[i][c]["verdict"] == "incorrect" else 0)
                   for i in gold if final[i][c]["verdict"] != "undecidable"]
            acc = sum(g == p for g, p in dec) / len(dec) if dec else float("nan")
            mf1 = f1_score([g for g, _ in dec], [p for _, p in dec], average="macro") if dec else float("nan")
            # agreement of the grader verdict with gold on ALL items, undecidable counted as wrong
            all_acc = sum((final[i][c]["verdict"] == "incorrect") == bool(gold[i]["factuality"])
                          and final[i][c]["verdict"] != "undecidable" for i in gold) / len(gold)
            summary_rows.append({
                "lang": lang, "condition": c, "n": len(gold),
                **{f"rel_{k}": rel.get(k, 0) for k in ("answers", "partial", "topical", "irrelevant", "not_found")},
                "pct_answers": round(100 * rel.get("answers", 0) / len(gold), 1),
                "pct_answers_or_partial": round(100 * (rel.get("answers", 0) + rel.get("partial", 0)) / len(gold), 1),
                "pct_sufficient": round(100 * suff / len(gold), 1),
                "n_decidable": len(dec), "verdict_acc_decidable": round(acc, 4), "verdict_macro_f1_decidable": round(mf1, 4),
                "verdict_acc_all": round(all_acc, 4),
                "pct_graders_agreed_all_fields": round(100 * sum(final[i][c]["graders_agreed"] for i in gold) / len(gold), 1),
            })
    with open(out_dir / "grading_agreement.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agreement_rows[0]))
        w.writeheader(); w.writerows(agreement_rows)
    with open(out_dir / "grading_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        w.writeheader(); w.writerows(summary_rows)
    for r in summary_rows:
        print(r)
    for r in agreement_rows:
        if r["condition"] == "all":
            print(r)


if __name__ == "__main__":
    main()
