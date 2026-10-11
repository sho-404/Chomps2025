"""Write the four compact result tables that the paper \\input{}s from paper/tables/.

The full (reviewer-level) tables are written by paper_tables.py into ../results/tables/;
this module writes the condensed versions used in the manuscript, in the shape of the
original submission's tables (Accuracy / Precision / Recall / Macro-F1), from the same
CSV stores so that the numbers cannot drift from the data.

    ../.venv/bin/python -m chomps.compact_tables [--paper-dir ../paper]

Tables:
  competition_rescored.tex   stored competition predictions, official metric, test split
  factuality_main.tex        revised factuality: single judge, council, council + meta-learner
  retrieval_ablation.tex     fixed judge, one row per context condition, both languages, with cost
  fluency_main.tex           revised fluency: single judge, councils, council + meta-learner
"""
import argparse
import csv
from pathlib import Path

from . import config

APPROACH_NAME = {"approach_1": "Single judge", "approach_2": "Council", "approach_3": "Council + meta-learner"}
SPLIT_NAME = {"validation": "Validation", "test": "Test"}
LANG_NAME = {"english": "English", "spanish": "Spanish"}


def _read(name):
    p = config.RESULTS / "tables" / f"{name}.csv"
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x, nd=3):
    return "--" if x in (None, "") else f"{float(x):.{nd}f}"


def _ci(r):
    return f"[{_f(r['ci_mean_lo'])}, {_f(r['ci_mean_hi'])}]"


def _metric_cells(r):
    return f"{_f(r['accuracy'])} & {_f(r['precision_y'])} & {_f(r['recall_y'])} & {_f(r['macro_f1'])} & {_ci(r)}"


HEAD = (r"\textbf{Language} & \textbf{Split} & \textbf{Approach} & \textbf{Accuracy} & \textbf{Precision} & "
        r"\textbf{Recall} & \textbf{Macro-F1} & \textbf{95\% CI} \\")


def _multirow_block(lines, lang, splits, systems, rows, name_of):
    """Append one language block: multirow language cell, one multirow per split."""
    first_lang = True
    for split in splits:
        present = [(s, rows.get((lang, split, s))) for s in systems]
        present = [(s, r) for s, r in present if r is not None]
        for i, (s, r) in enumerate(present):
            lang_cell = rf"\multirow{{{sum(1 for sp in splits for s2 in systems if rows.get((lang, sp, s2)))}}}{{*}}{{{LANG_NAME[lang]}}}" if first_lang else ""
            first_lang = False
            split_cell = rf"\multirow{{{len(present)}}}{{*}}{{{SPLIT_NAME[split]}}}" if i == 0 else ""
            lines.append(f"{lang_cell} & {split_cell} & {name_of(s)} & {_metric_cells(r)} \\\\")
        if split != splits[-1]:
            lines.append(r"\cline{2-8}")


def competition_rescored(out: Path) -> None:
    """Test split only; one block per track. Validation and train are in results/tables/."""
    rows = {(r["lang"], r["track"], r["approach"]): r for r in _read("original_rescored") if r["split"] == "test"}
    lines = [r"\begin{tabular}{lllccccc}", r"\hline",
             r"\textbf{Language} & \textbf{Track} & \textbf{Approach} & \textbf{Accuracy} & \textbf{Precision} & "
             r"\textbf{Recall} & \textbf{Macro-F1} & \textbf{95\% CI} \\", r"\hline"]
    for lang in config.LANGS:
        n_lang = sum(1 for t in config.TRACKS for a in APPROACH_NAME if (lang, t, a) in rows)
        first = True
        for track in config.TRACKS:
            present = [a for a in APPROACH_NAME if (lang, track, a) in rows]
            for i, a in enumerate(present):
                lang_cell = rf"\multirow{{{n_lang}}}{{*}}{{{LANG_NAME[lang]}}}" if first else ""
                first = False
                track_cell = rf"\multirow{{{len(present)}}}{{*}}{{{track.capitalize()}}}" if i == 0 else ""
                lines.append(f"{lang_cell} & {track_cell} & {APPROACH_NAME[a]} & {_metric_cells(rows[(lang, track, a)])} \\\\")
            if track != config.TRACKS[-1]:
                lines.append(r"\cline{2-8}")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (out / "competition_rescored.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


FACT_SYSTEMS = {"single": "Single judge", "council_ftfmt": "Council", "meta-original-none": "Council + meta-learner"}


def factuality_main(out: Path) -> None:
    rows = {(r["lang"], r["split"], r["system"]): r for r in _read("factuality_main")}
    lines = [r"\begin{tabular}{lllccccc}", r"\hline", HEAD, r"\hline"]
    for lang in config.LANGS:
        _multirow_block(lines, lang, ("validation", "test"), list(FACT_SYSTEMS), rows, FACT_SYSTEMS.get)
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (out / "factuality_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


FLU_SYSTEMS = {"single": "Single judge", "council": "Council", "council_fluft": "Fluency council",
               "meta-original-balanced-logits-fluft": "Fluency council + meta-learner"}


def fluency_main(out: Path) -> None:
    rows = {(r["lang"], r["split"], r["system"]): r for r in _read("fluency_main")}
    lines = [r"\begin{tabular}{lllccccc}", r"\hline", HEAD, r"\hline"]
    for lang in config.LANGS:
        _multirow_block(lines, lang, ("validation", "test"), list(FLU_SYSTEMS), rows, FLU_SYSTEMS.get)
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (out / "fluency_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


ABL = [("none", "No context"), ("bm25_k3", "BM25 chunks ($k=3$)"), ("dense_k5", "Dense chunks ($k=5$)"),
       ("pseudo_rag", "Pseudo-RAG span"), ("full", "Full document")]


def retrieval_ablation(out: Path) -> None:
    """Test split, GPT-5 mini judge; macro-F1 with CI per language, English input length and price."""
    rows = {(r["lang"], r["condition"]): r for r in _read("retrieval_ablation") if r["split"] == "test"}
    cost = {}
    for r in _read("cost"):
        if r["stage"] == "judge" and r["lang"] == "english" and r["judge"] == "gpt-5-mini" and r["temp"] == "default":
            cost[r["condition"]] = r
    lines = [r"\begin{tabular}{lcccccc}", r"\hline",
             r"\textbf{Context given to the judge} & \multicolumn{2}{c}{\textbf{English}} & \multicolumn{2}{c}{\textbf{Spanish}} & "
             r"\textbf{Input tokens} & \textbf{USD / 1k judgments} \\",
             r" & \textbf{Macro-F1} & \textbf{95\% CI} & \textbf{Macro-F1} & \textbf{95\% CI} & \textbf{(median, En.)} & \textbf{(En.)} \\",
             r"\hline"]
    for cond, name in ABL:
        en, es, c = rows[("english", cond)], rows[("spanish", cond)], cost[cond]
        lines.append(f"{name} & {_f(en['macro_f1'])} & {_ci(en)} & {_f(es['macro_f1'])} & {_ci(es)} & "
                     f"{float(c['prompt_tokens_median']):,.0f} & {float(c['usd_per_1000_examples_per_repeat']):.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out / "retrieval_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=str(config.ROOT / "paper"))
    a = ap.parse_args()
    out = Path(a.paper_dir) / "tables"
    out.mkdir(exist_ok=True)
    competition_rescored(out)
    factuality_main(out)
    fluency_main(out)
    retrieval_ablation(out)
    print(f"wrote compact tables to {out}")


if __name__ == "__main__":
    main()
