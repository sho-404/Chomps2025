"""Write the LaTeX table bodies that the revised paper \\input{}s from paper/tables/.

Every table body is generated from the stores under ../results (never typed by
hand) so that the paper and the data cannot drift apart. Run after report.py:

    ../.venv/bin/python -m chomps.paper_tables [--paper-dir ../paper]

Tables (each a bare tabular, the paper supplies caption/label):
  competition_rescored.tex   the submission's stored predictions scored with the official metric
  class_distribution.tex     gold label counts per language/split/track
  (more tables are appended by later revisions of this module as results arrive)
"""
import argparse
import csv
import json
from pathlib import Path

from . import config, data, metrics

APPROACH_NAME = {"approach_1": "Single judge", "approach_2": "Council", "approach_3": "Council + meta-learner"}
SPLIT_NAME = {"train": "Train", "validation": "Validation", "test": "Test"}


def _fmt(x, nd=4):
    return "--" if x is None or x == "" else f"{float(x):.{nd}f}"


def competition_rescored(paper_tables: Path) -> None:
    """One row per (lang, track, split, approach) from the original approach_*_answers files,
    with confusion counts, accuracy, balanced accuracy, F1 of both classes and macro-F1 + CI."""
    lines = [r"\begin{tabular}{lllccccccccc}", r"\hline",
             r"\textbf{Lang.} & \textbf{Split} & \textbf{Approach} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{TN} & "
             r"\textbf{Acc.} & \textbf{Bal.\ acc.} & \textbf{F1(y)} & \textbf{F1(n)} & \textbf{Macro-F1 [95\% CI]} \\"]
    for track in config.TRACKS:
        lines.append(r"\hline")
        lines.append(r"\multicolumn{12}{l}{\textit{" + track.capitalize() + r"}} \\")
        lines.append(r"\hline")
        for lang in config.LANGS:
            for split in config.SPLITS:
                gold = data.load_labels(lang, split)
                for a in ("approach_1", "approach_2", "approach_3"):
                    f = config.DATASETS / lang / split / f"{a}_answers_{track}.jsonl"
                    if not f.exists():
                        continue
                    pred = {}
                    for line in open(f, encoding="utf-8"):
                        r = json.loads(line)
                        pred[r["index"]] = r[config.LABEL_KEY[track]]
                    idx = [i for i in gold if i in pred]
                    g = [gold[i][track] for i in idx]
                    p = [pred[i] for i in idx]
                    c = metrics.confusion(g, p)
                    s = metrics.scores(g, p)
                    lo, hi = metrics.bootstrap_ci(g, p, "macro_f1")
                    lines.append(f"{lang.capitalize()} & {SPLIT_NAME[split]} & {APPROACH_NAME[a]} & {c['tp']} & {c['fp']} & {c['fn']} & {c['tn']} & "
                                 f"{_fmt(s['accuracy'])} & {_fmt(s['balanced_accuracy'])} & {_fmt(s['f1_y'])} & {_fmt(s['f1_n'])} & "
                                 f"{_fmt(s['macro_f1'])} [{_fmt(lo, 3)}, {_fmt(hi, 3)}] \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "competition_rescored.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def class_distribution(paper_tables: Path) -> None:
    lines = [r"\begin{tabular}{llcccc}", r"\hline",
             r"\textbf{Language} & \textbf{Split} & \textbf{Examples} & \textbf{Factual mistakes (y)} & \textbf{Fluency mistakes (y)} & \textbf{Both} \\", r"\hline"]
    for lang in config.LANGS:
        for split in config.SPLITS:
            gold = data.load_labels(lang, split)
            n = len(gold)
            fy = sum(v["factuality"] for v in gold.values())
            fl = sum(v["fluency"] for v in gold.values())
            both = sum(v["factuality"] and v["fluency"] for v in gold.values())
            lines.append(f"{lang.capitalize()} & {SPLIT_NAME[split]} & {n} & {fy} ({100 * fy / n:.1f}\\%) & {fl} ({100 * fl / n:.1f}\\%) & {both} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "class_distribution.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


COND_NAME = {"pseudo_rag": "Pseudo-RAG (competition spans)", "pseudo_rag_rerun": "Pseudo-RAG (rerun)",
             "bm25_k3": "BM25, $k=3$", "dense_k3": "Dense, $k=3$"}


def retrieval_quality(paper_tables: Path) -> None:
    """Programmatic statistics (retrieval_eval summary, test split) joined with the
    Claude-graded relevance study (grading_summary) per language and condition."""
    stats = {}
    with open(config.RESULTS / "retrieval_eval" / "summary.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "test":
                stats[(r["lang"], r["condition"])] = r
    grades = {}
    with open(config.RESULTS / "retrieval_eval" / "grading_summary.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            grades[(r["lang"], r["condition"])] = r
    lines = [r"\begin{tabular}{llcccccccc}", r"\hline",
             r"\textbf{Lang.} & \textbf{Context} & \textbf{Tokens (med.)} & \textbf{Verbatim} & \textbf{Refusals} & "
             r"\textbf{Answers} & \textbf{Answers or partial} & \textbf{Sufficient} & \textbf{Verdict acc.} & \textbf{Agreement} ($\kappa$) \\", r"\hline"]
    kappa = {}
    with open(config.RESULTS / "retrieval_eval" / "grading_agreement.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["field"] == "relevance" and r["condition"] != "all":
                kappa[(r["lang"], r["condition"])] = float(r["kappa"])
    for lang in config.LANGS:
        for cond in ("pseudo_rag", "pseudo_rag_rerun", "bm25_k3", "dense_k3"):
            s = stats[(lang, cond)]
            g = grades.get((lang, cond))
            verb = f"{float(s['verbatim_mean']):.1f}" if cond.startswith("pseudo") else "100.0"
            refus = f"{float(s['not_found_pct']) + float(s['empty_pct']):.1f}\\%"
            if g:
                gcols = (f"{g['pct_answers']}\\% & {g['pct_answers_or_partial']}\\% & {g['pct_sufficient']}\\% & "
                         f"{float(g['verdict_acc_decidable']):.3f} ({g['n_decidable']}) & {kappa[(lang, cond)]:.2f}")
            else:
                gcols = "-- & -- & -- & -- & --"
            lines.append(f"{lang.capitalize()} & {COND_NAME[cond]} & {float(s['tokens_median']):.0f} & {verb} & {refus} & {gcols} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "retrieval_quality.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


ABL_NAME = {"pseudo_rag": "Pseudo-RAG (competition spans)", "pseudo_rag_rerun": "Pseudo-RAG (rerun)",
            "bm25_k1": "BM25, $k=1$", "bm25_k3": "BM25, $k=3$", "bm25_k5": "BM25, $k=5$",
            "dense_k1": "Dense, $k=1$", "dense_k3": "Dense, $k=3$", "dense_k5": "Dense, $k=5$",
            "full": "Full document", "none": "No context"}
ABL_ORDER = ("pseudo_rag", "pseudo_rag_rerun", "bm25_k1", "bm25_k3", "bm25_k5", "dense_k1", "dense_k3", "dense_k5", "full", "none")


def retrieval_ablation(paper_tables: Path) -> None:
    """Single GPT-5 mini judge, fixed prompt, one row per context condition; macro-F1 mean +- std
    over repeats with the bootstrap CI of the mean, per language and split (from report.py's CSV)."""
    rows = {}
    with open(config.RESULTS / "tables" / "retrieval_ablation.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(r["lang"], r["split"], r["condition"])] = r
    lines = [r"\begin{tabular}{llcccccc}", r"\hline",
             r"\textbf{Lang.} & \textbf{Context given to the judge} & \textbf{Runs} & \textbf{Split} & \textbf{Acc.} & \textbf{Bal.\ acc.} & \textbf{Macro-F1 $\pm$ std} & \textbf{95\% CI} \\"]
    for lang in config.LANGS:
        lines.append(r"\hline")
        for cond in ABL_ORDER:
            for split in config.SPLITS:
                r = rows.get((lang, split, cond))
                if r is None:
                    continue
                name = f"{lang.capitalize()} & {ABL_NAME[cond]} & {r['n_repeats']}" if split == "train" else " & & "
                lines.append(f"{name} & {SPLIT_NAME[split]} & {_fmt(r['accuracy'])} & {_fmt(r['balanced_accuracy'])} & "
                             f"{_fmt(r['macro_f1'])} $\\pm$ {_fmt(r['macro_f1_std'], 3)} & [{_fmt(r['ci_mean_lo'], 3)}, {_fmt(r['ci_mean_hi'], 3)}] \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "retrieval_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def error_attribution(paper_tables: Path) -> None:
    lines = [r"\begin{tabular}{llccccc}", r"\hline",
             r"\textbf{Lang.} & \textbf{Context} & \textbf{Correct} & \textbf{Retrieval failure} & \textbf{Judge failure} & \textbf{Judge and graders vs.\ gold} \\", r"\hline"]
    with open(config.RESULTS / "tables" / "error_attribution.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lines.append(f"{r['lang'].capitalize()} & {COND_NAME[r['condition']]} & {r['correct']} ({r['pct_correct']}\\%) & "
                         f"{r['retrieval_failure']} ({r['pct_retrieval_failure']}\\%) & {r['judge_failure']} ({r['pct_judge_failure']}\\%) & "
                         f"{r['label_or_ambiguous']} ({r['pct_label_or_ambiguous']}\\%) \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "error_attribution.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


FEATURE_NAME = {"mean": "mean", "std": "std.\\ dev.", "min": "min", "p10": "10th pct.", "p25": "25th pct.",
                "frac_below_15": "share $<15$", "n_tokens": "length", "mean_first_10": "mean of first 10"}


def logits_table(paper_tables: Path) -> None:
    """Per language and track: 5-fold CV AUC of a logistic regression on the eight logit
    features, and the three features with the largest |point-biserial r| (from logits.py)."""
    lines = [r"\begin{tabular}{llcl}", r"\hline",
             r"\textbf{Language} & \textbf{Track} & \textbf{CV AUC (logits only)} & \textbf{Strongest correlates ($r$, $p$)} \\", r"\hline"]
    for lang in config.LANGS:
        rows = list(csv.DictReader(open(config.RESULTS / "tables" / f"logits_{lang}.csv", encoding="utf-8")))
        for track in config.TRACKS:
            auc = {r["term"]: float(r["estimate"]) for r in rows if r["track"] == track and r["kind"] == "cv_auc"}
            pb = [(r["term"], float(r["estimate"]), r["p_value"]) for r in rows if r["track"] == track and r["kind"] == "point_biserial"]
            pb.sort(key=lambda t: -abs(t[1]))
            def _p(p):
                p = float(p)
                return "$p<10^{-3}$" if p < 1e-3 else f"$p={p:.2f}$"
            top = "; ".join(f"{FEATURE_NAME[t]} ($r={r:+.2f}$, {_p(p)})" for t, r, p in pb[:3])
            lines.append(f"{lang.capitalize()} & {track.capitalize()} & {auc['mean']:.3f} $\\pm$ {auc['std']:.3f} & {top} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "logits.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


JUDGE_NAME = {"gpt-5-mini": "GPT-5 mini", "gpt-4.1-mini": "GPT-4.1 mini", "gpt-4.1-nano": "GPT-4.1 nano"}


def _judge_label(j: str) -> str:
    if j in JUDGE_NAME:
        return JUDGE_NAME[j]
    if j.startswith("ft-"):
        _, track, _, size = j.split("-")
        return f"FT GPT-4.1 {size} ({'fact.' if track == 'fact' else 'flu.'})"
    return j


def cost_table(paper_tables: Path) -> None:
    """Per-example tokens, latency and USD (standalone, one call per example) for the
    retrieval stages and for every judge/condition at temp default with the evaluator
    prompt, English and Spanish, from cost.py's CSV."""
    rows = list(csv.DictReader(open(config.RESULTS / "tables" / "cost.csv", encoding="utf-8")))
    lines = [r"\begin{tabular}{llllrrrr}", r"\hline",
             r"\textbf{Lang.} & \textbf{Stage} & \textbf{Model} & \textbf{Context / track} & \textbf{Input tok.\ (med.)} & \textbf{Output tok.\ (mean)} & \textbf{Latency (med., s)} & \textbf{USD per 1{,}000 calls} \\"]
    for lang in config.LANGS:
        lines.append(r"\hline")
        for r in rows:
            if r["lang"] != lang:
                continue
            if r["stage"] == "retrieval":
                if r["condition"] not in ("pseudo_rag_rerun", "dense_k3"):
                    continue
                what = "extraction (one paper)" if r["condition"] == "pseudo_rag_rerun" else "embedding index (one paper)"
                lines.append(f"{lang.capitalize()} & Retrieval & {_judge_label(r['judge']) if r['judge'] in JUDGE_NAME else r['judge']} & {what} & "
                             f"{float(r['prompt_tokens_median']):.0f} & {float(r['completion_tokens_mean'] or 0):.0f} & "
                             f"{(float(r['latency_s_median']) if r['latency_s_median'] not in ('', 'nan') and float(r['latency_s_median']) > 0 else float('nan')):.2f} & {float(r['usd_per_1000_examples_per_repeat']):.2f} \\\\")
            elif r["fmt"] == "judge" and r["temp"] == "default" and r["condition"] not in ("bm25_k1", "bm25_k5", "dense_k1", "dense_k5"):
                cond = ABL_NAME.get(r["condition"], r["condition"]) if r["track"] == "factuality" else "fluency (no context)"
                lines.append(f"{lang.capitalize()} & Judge & {_judge_label(r['judge'])} & {cond} & "
                             f"{float(r['prompt_tokens_median']):.0f} & {float(r['completion_tokens_mean']):.0f} & "
                             f"{float(r['latency_s_median']):.2f} & {float(r['usd_per_1000_examples_per_repeat']):.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    text = "\n".join(lines).replace("nan", "--") + "\n"
    (paper_tables / "cost.tex").write_text(text, encoding="utf-8")


SYSTEM_NAME = {
    "single": "Single judge (GPT-5 mini)",
    "council": "Council, evaluator prompt",
    "council_ftfmt": "Council, fine-tuning format",
    "council_fluft": "Fluency council, evaluator prompt",
    "council_fluft_ftfmt": "Fluency council, fine-tuning format",
    "meta-original-none": "Meta-learner (competition protocol)",
    "meta-original-balanced": "Meta-learner (comp.), class-weighted",
    "meta-cv-none": "Meta-learner (CV protocol)",
    "meta-cv-balanced": "Meta-learner (CV), class-weighted",
    "meta-original-none-logits": "Meta-learner + logits (comp.)",
    "meta-original-balanced-logits": "Meta-learner + logits (comp.), class-weighted",
    "meta-cv-none-logits": "Meta-learner + logits (CV)",
    "meta-cv-balanced-logits": "Meta-learner + logits (CV), class-weighted",
}
for _v in list(SYSTEM_NAME):
    if _v.startswith("meta-"):
        SYSTEM_NAME[_v + "-fluft"] = SYSTEM_NAME[_v].replace("Meta-learner", "Fluency-council meta-learner")


def _read(name):
    p = config.RESULTS / "tables" / f"{name}.csv"
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def _confusion_index():
    return {(r["lang"], r["track"], r["split"], r["system"]): r for r in _read("confusion_matrices") if r["pred"] == "consensus"}


def main_results(paper_tables: Path) -> None:
    """Revised factuality and fluency results: lang x split x system with consensus confusion
    counts, accuracy, balanced accuracy, F1 of both classes, macro-F1 mean +- std and CI."""
    conf = _confusion_index()
    for track in config.TRACKS:
        rows = _read(f"{track}_main")
        lines = [r"\begin{tabular}{lllcccccc}", r"\hline",
                 r"\textbf{Lang.} & \textbf{Split} & \textbf{System} & \textbf{TP/FP/FN/TN} & \textbf{Acc.} & \textbf{Bal.~acc.} & "
                 r"\textbf{F1(y)} & \textbf{Macro-F1 $\pm$ std} & \textbf{95\% CI} \\"]
        order = list(SYSTEM_NAME)
        for lang in config.LANGS:
            lines.append(r"\hline")
            for split in ("validation", "test"):
                sub = sorted([r for r in rows if r["lang"] == lang and r["split"] == split],
                             key=lambda r: order.index(r["system"]) if r["system"] in order else 99)
                for i, r in enumerate(sub):
                    c = conf.get((lang, track, split, r["system"]))
                    cc = f"{c['tp']}/{c['fp']}/{c['fn']}/{c['tn']}" if c else "--"
                    f1n = 2 * float(r["macro_f1"]) - float(r["f1_y"])
                    head = f"{lang.capitalize() if i == 0 and split == 'validation' else ''} & {SPLIT_NAME[split] if i == 0 else ''}"
                    mf = f"{_fmt(r['macro_f1'])} $\\pm$ {_fmt(r['macro_f1_std'], 3)}" if r['macro_f1_std'] else _fmt(r['macro_f1'])
                    lines.append(f"{head} & {SYSTEM_NAME.get(r['system'], r['system'])} & {cc} & {_fmt(r['accuracy'])} & "
                                 f"{_fmt(r['balanced_accuracy'])} & {_fmt(r['f1_y'])} & {mf} & "
                                 f"[{_fmt(r['ci_mean_lo'], 3)}, {_fmt(r['ci_mean_hi'], 3)}] \\\\")
        lines += [r"\hline", r"\end{tabular}"]
        (paper_tables / f"{track}_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def per_judge(paper_tables: Path) -> None:
    """Every individual judge (base and fine-tuned, both prompt formats, each temperature):
    macro-F1 mean +- std on validation and test, both tracks and languages."""
    rows = _read("per_judge")
    idx = {(r["lang"], r["track"], r["split"], r["judge"], r["fmt"], r["temp"]): r for r in rows}
    keys = sorted({(r["lang"], r["track"], r["judge"], r["fmt"], r["temp"]) for r in rows if r["temp"] != "1.0"},
                  key=lambda k: (config.LANGS.index(k[0]), config.TRACKS.index(k[1]),
                                 ["gpt-5-mini", "gpt-4.1-mini", "gpt-4.1-nano"].index(k[2]) if k[2] in JUDGE_NAME else 3 + ("flu" in k[2]) * 2 + ("nano" in k[2]),
                                 k[3] != "judge", ["default", "0", "1.0"].index(k[4])))
    lines = [r"\begin{tabular}{lllllcccc}", r"\hline",
             r"\textbf{Lang.} & \textbf{Track} & \textbf{Judge} & \textbf{Prompt} & \textbf{Temp.} & "
             r"\textbf{Val.\ macro-F1} & \textbf{Val.\ bal.\ acc.} & \textbf{Test macro-F1} & \textbf{Test bal.\ acc.} \\"]
    last = None
    for lang, track, judge, fmt, temp in keys:
        if (lang, track) != last:
            lines.append(r"\hline")
            last = (lang, track)
        v = idx.get((lang, track, "validation", judge, fmt, temp))
        t = idx.get((lang, track, "test", judge, fmt, temp))
        cell = lambda r: (f"{_fmt(r['macro_f1'])} $\\pm$ {_fmt(r['macro_f1_std'], 3) if r['macro_f1_std'] else '--'}") if r else "--"
        bal = lambda r: _fmt(r["balanced_accuracy"]) if r else "--"
        lines.append(f"{lang.capitalize()} & {track.capitalize()} & {_judge_label(judge)} & {'evaluator' if fmt == 'judge' else 'fine-tuning'} & "
                     f"{'default' if temp == 'default' else temp} & {cell(v)} & {bal(v)} & {cell(t)} & {bal(t)} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "per_judge.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def meta_table(paper_tables: Path) -> None:
    """Meta-learner variants: protocol, class weight, features, council, CV macro-F1 (mean +- std over
    folds), test macro-F1 with CI, and the fitted coefficients (repeat 0, competition protocol)."""
    rows = [r for r in _read("meta_cv") if r["split"] == "test"]
    lines = [r"\begin{tabular}{llllllcc}", r"\hline",
             r"\textbf{Lang.} & \textbf{Track} & \textbf{Council} & \textbf{Protocol} & \textbf{Class weight} & \textbf{Features} & "
             r"\textbf{CV macro-F1} & \textbf{Test macro-F1 [95\% CI]} \\"]
    last = None
    for r in sorted(rows, key=lambda r: (config.LANGS.index(r["lang"]), config.TRACKS.index(r["track"]), r.get("council", ""), r["protocol"], r["class_weight"], r["features"])):
        if (r["lang"], r["track"]) != last:
            lines.append(r"\hline")
            last = (r["lang"], r["track"])
        cv = f"{float(r['cv_macro_f1_mean']):.4f} $\\pm$ {float(r['cv_macro_f1_std']):.3f}" if r.get("cv_macro_f1_mean") else "--"
        council = "dedicated fluency" if r.get("council") == "fluft" else "competition"
        lines.append(f"{r['lang'].capitalize()} & {r['track'].capitalize()} & {council} & {r['protocol']} & {r['class_weight']} & {r['features']} & {cv} & "
                     f"{_fmt(r['macro_f1'])} [{_fmt(r['ci_mean_lo'], 3)}, {_fmt(r['ci_mean_hi'], 3)}] \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "meta.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def coefficients(paper_tables: Path) -> None:
    """beta_0..beta_3 (repeat 0) with bootstrap CIs for the competition-protocol, unweighted
    meta-learner on the competition council, both tracks and languages (from meta.py's summaries)."""
    lines = [r"\begin{tabular}{llcccc}", r"\hline",
             r"\textbf{Lang.} & \textbf{Track} & $\beta_0$ \textbf{(intercept)} & $\beta_1$ \textbf{(GPT-5 mini)} & $\beta_2$ \textbf{(FT mini)} & $\beta_3$ \textbf{(FT nano)} \\", r"\hline"]
    for lang in config.LANGS:
        for track in config.TRACKS:
            cond = "pseudo_rag" if track == "factuality" else "none"
            p = config.RESULTS / "tables" / f"meta_{lang}_{track}_{cond}_original_none.csv"
            if not p.exists():
                continue
            rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["repeat"] == "0" and r["kind"] == "coef"]
            if not rows:
                rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["repeat"] == "0" and "coef" in r["kind"]]
            by = {r["term"]: r for r in rows}
            cells = []
            for term in ("intercept", "gpt-5-mini", f"ft-fact-{config.LANG_PREFIX[lang]}-mini", f"ft-fact-{config.LANG_PREFIX[lang]}-nano"):
                r = by.get(term)
                cells.append(f"{float(r['estimate']):+.2f} [{float(r['ci_lo']):+.2f}, {float(r['ci_hi']):+.2f}]" if r and r["estimate"] else "--")
            lines.append(f"{lang.capitalize()} & {track.capitalize()} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "coefficients.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


SIG_PAIRS = [("main", "single", "council"), ("main", "single", "meta-original-none"), ("main", "single", "meta-cv-none"),
             ("main", "council", "meta-original-none"), ("main", "single", "council_fluft"), ("main", "council", "council_fluft"),
             ("main", "single", "meta-original-balanced-fluft"), ("main", "single", "meta-cv-none-fluft"),
             ("retrieval", "single@pseudo_rag", "single@full"), ("retrieval", "single@pseudo_rag", "single@bm25_k3"),
             ("retrieval", "single@pseudo_rag", "single@dense_k3"), ("retrieval", "single@pseudo_rag", "single@dense_k5"),
             ("retrieval", "single@pseudo_rag", "single@none")]


def significance(paper_tables: Path) -> None:
    rows = [r for r in _read("significance") if r["split"] == "test" and r["pred"] == "consensus"]
    idx = {(r["lang"], r["track"], r["family"], r["system_a"], r["system_b"]): r for r in rows}
    lines = [r"\begin{tabular}{lllccccc}", r"\hline",
             r"\textbf{Lang.} & \textbf{Track} & \textbf{Comparison (A vs.~B)} & \textbf{Macro-F1 A} & \textbf{Macro-F1 B} & \textbf{Diff.~[95\% CI]} & \textbf{McNemar} $b/c$ & $p$ \\"]
    for lang in config.LANGS:
        for track in config.TRACKS:
            lines.append(r"\hline")
            for fam, a, b in SIG_PAIRS:
                r = idx.get((lang, track, fam, a, b))
                if r is None:
                    continue
                name = lambda s: SYSTEM_NAME.get(s, s).replace("Single judge (GPT-5 mini)", "single judge") if "@" not in s else ABL_NAME[s.split("@")[1]]
                p = float(r["p_mcnemar"])
                lines.append(f"{lang.capitalize()} & {track.capitalize()} & {name(a)} vs.\\ {name(b)} & {_fmt(r['macro_f1_a'])} & {_fmt(r['macro_f1_b'])} & "
                             f"{float(r['diff']):+.4f} [{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}] & {r['mcnemar_b']}/{r['mcnemar_c']} & "
                             f"{'$<0.001$' if p < 0.001 else f'{p:.3f}'} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper_tables / "significance.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=str(config.ROOT / "paper"))
    a = ap.parse_args()
    out = Path(a.paper_dir) / "tables"
    out.mkdir(exist_ok=True)
    competition_rescored(out)
    class_distribution(out)
    retrieval_quality(out)
    retrieval_ablation(out)
    error_attribution(out)
    logits_table(out)
    cost_table(out)
    main_results(out)
    per_judge(out)
    meta_table(out)
    coefficients(out)
    significance(out)
    print(f"wrote tables to {out}")


if __name__ == "__main__":
    main()
