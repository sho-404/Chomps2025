"""Retrieval quality of the factuality contexts (DESIGN.md, "Retrieval quality").

Per (lang, split, condition) and example: how verbatim the context is with
respect to the source paper, whether the retriever refused or returned
nothing, and its size. Per (lang, split): agreement of the re-run LLM
extraction with the original competition span. For the test split: a grading
sheet for a human/LLM relevance study.

Outputs, under ../results/retrieval_eval/:
    stats_{lang}_{split}_{condition}.csv  one row per example (STATS_COLUMNS)
    agreement_{lang}_{split}.csv          pseudo_rag vs pseudo_rag_rerun, one row per example
    summary.csv                           one row per stats file on disk, so partial runs accumulate
    grading_sheet_{lang}_test.jsonl       contexts of pseudo_rag, bm25_k3, dense_k3

Protocol decisions
------------------
* Haystack: the full extracted paper (`data.paper_text`), whitespace-normalised
  (`norm`). Retrievers only saw paper[:MAX_PAPER_CHARS]; the full text is a
  superset, so a context can gain from it only by coincidence.
* Segments. bm25_k*/dense_k* contexts are K verbatim chunks joined by
  CHUNK_SEP; scoring the joined string measures the separators (a fully
  verbatim 3-chunk context scores ~65). The context is therefore split on
  CHUNK_SEP: `verbatim` = length-weighted mean over segments of
  partial_ratio(norm(segment), norm(paper)), `exact_substring` = every segment
  is a substring of norm(paper). No LLM span contains CHUNK_SEP (0 of the
  1,248 original spans), so for LLM extractions this is exactly
  partial_ratio(norm(context), norm(paper)). Two segment shapes raise instead of
  being scored: a blank segment (the context starts or ends with CHUNK_SEP or
  holds two in a row; 0 of the 4,992 pseudo_rag/bm25 contexts on disk), which no
  valid store yields, and a segment longer than its paper, for which partial_ratio
  would align the paper inside the segment and report 100 although nothing matched
  (the shortest paper has 18,992 normalised characters).
* exact_substring_loose: the same test after NFKC, re-joining words hyphenated
  across line breaks and deleting all whitespace (a segment ending on such a
  hyphen, e.g. a chunk cut at "transat-", drops it). The extractor LLM routinely
  repairs PDF artefacts ("communi- cation", ligatures, stray spaces): on the
  original spans the strict rate is 21-55 % per split and the loose one 59-90 %.
* Flags: `not_found` = the extractor's refusal, i.e. the phrase of prompts.retrieval
  rule 3 (REFUSAL, "Context not found in paper.") anywhere in the context, ignoring
  case, whitespace, quotes and the final period. The prompt is English for both
  languages, so the refusal is too. DESIGN.md's looser "contains 'not found'" would
  also flag verbatim paper text ("... were not found to ...") and drop it from the
  aggregates; on the pseudo_rag and bm25_k1/k3/k5 stores (all splits) both rules
  flag the same 12 contexts, all original English spans equal to the refusal (one
  quoted; 8 validation, 4 train), and no chunk context.
  `empty` = None or blank, whose verbatim columns are NaN. Summary verbatim aggregates
  exclude empty and not_found rows (a refusal is not evidence); chars/tokens aggregate
  over all rows (what the judge receives).
* Agreement: Jaccard of the lower-cased \\w+ word sets (the BM25 tokenizer) and
  partial_ratio of the normalised strings (100 when one span contains the other).
  NaN unless both contexts are non-empty spans: rapidfuzz scores two empty strings
  100, and two refusals would score 1.0/100 and inflate agreement, so pairs with a
  refusal are counted instead (agree_both_not_found_n, agree_one_not_found_n).
* Effective sample size. An example is one model output, and the outputs of one
  question share their context in pseudo_rag_rerun, bm25_k* and dense_k* (the
  context depends on paper and question only); the original pseudo_rag spans vary
  within a question (1-8 distinct). Validation and test have 30 questions per
  language with 7-9 outputs each (train: 38 English, 40 Spanish questions), so
  per-example means weight each question by its number of outputs and n overstates
  the number of independent spans. Every row carries question_id (the index of the
  first example with its (title, question) in the split) and context_sha1, and
  summary.csv reports n_questions and n_distinct_contexts next to n. A confidence
  interval over these rows must resample questions, not examples.
"""
import argparse
import hashlib
import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from . import config, data, llm

CHUNK_SEP = "\n[...]\n"
REFUSAL = "Context not found in paper."  # prompts.retrieval rule 3, English for both languages
CONDITIONS = ("pseudo_rag", "pseudo_rag_rerun", "bm25_k1", "bm25_k3", "bm25_k5",
              "dense_k1", "dense_k3", "dense_k5")
DEFAULT_CONDITIONS = ("pseudo_rag", "pseudo_rag_rerun", "bm25_k3", "dense_k3")
AGREEMENT_PAIR = ("pseudo_rag", "pseudo_rag_rerun")
GRADING_SPLIT = "test"
GRADING_CONDITIONS = ("pseudo_rag", "bm25_k3", "dense_k3")
OUT_DIR = config.RESULTS / "retrieval_eval"

STATS_COLUMNS = ["index", "question_id", "context_sha1", "empty", "not_found", "chars", "tokens",
                 "n_segments", "verbatim", "exact_substring", "exact_substring_loose"]
AGREEMENT_COLUMNS = ["index", "question_id", f"{AGREEMENT_PAIR[0]}_not_found",
                     f"{AGREEMENT_PAIR[1]}_not_found", "jaccard", "partial_ratio"]
SUMMARY_COLUMNS = ["lang", "split", "condition", "n", "n_questions", "n_distinct_contexts",
                   "empty_n", "empty_pct", "not_found_n", "not_found_pct", "scored_n", "scored_questions",
                   "verbatim_mean", "verbatim_median", "exact_pct", "exact_loose_pct",
                   "chars_mean", "chars_median", "tokens_mean", "tokens_median",
                   "agree_n", "agree_questions", "agree_both_not_found_n", "agree_one_not_found_n",
                   "agree_jaccard_mean", "agree_jaccard_median",
                   "agree_partial_ratio_mean", "agree_partial_ratio_median"]
_TEXT_COLUMNS = ("index", "question_id", "context_sha1")  # read back as strings, never as numbers

_HYPHEN_BREAK = re.compile(r"(?<=\w)-\s+(?=\w)")


def norm(s: str) -> str:
    """Whitespace normalisation: every whitespace run becomes one space, ends stripped."""
    return " ".join(s.split())


def loose(s: str) -> str:
    """NFKC, words hyphenated across line breaks re-joined, all whitespace removed."""
    return re.sub(r"\s+", "", _HYPHEN_BREAK.sub("", unicodedata.normalize("NFKC", s)))


_REFUSAL_PHRASE = norm(REFUSAL).lower().rstrip(".")


def is_refusal(context) -> bool:
    """True when the context holds the extractor's refusal phrase (module docstring, Flags)."""
    return context is not None and _REFUSAL_PHRASE in norm(context).lower()


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """o200k_base token count; special-token strings in the text count as plain text."""
    return len(_encoder().encode_ordinary(text))


def word_set(s: str) -> set:
    return set(re.findall(r"\w+", s.lower()))


def question_ids(examples: list) -> dict:
    """index -> index of the first example with the same (title, question), in example order."""
    first = {}
    return {e["index"]: first.setdefault((e["title"], e["question"]), e["index"]) for e in examples}


def context_stats(context, paper_norm: str, paper_loose: str) -> dict:
    """Per-example stats of one context against its paper (see module docstring).
    Raises ValueError on a blank segment or on a segment longer than the paper."""
    if context is not None and not isinstance(context, str):
        raise TypeError(f"context must be str or None, got {type(context).__name__}")
    text = context or ""
    row = {"context_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
           "empty": int(not text.strip()), "not_found": int(is_refusal(text)),
           "chars": len(text), "tokens": count_tokens(text), "n_segments": 0,
           "verbatim": math.nan, "exact_substring": math.nan, "exact_substring_loose": math.nan}
    if row["empty"]:
        return row
    segs = [norm(p) for p in text.split(CHUNK_SEP)]
    if not all(segs):
        raise ValueError(f"context has a blank segment next to the chunk separator {CHUNK_SEP!r}, "
                         f"which no valid store yields: {text[:200]!r}")
    too_long = max(len(s) for s in segs)
    if too_long > len(paper_norm):
        raise ValueError(f"a context segment of {too_long} chars is longer than its paper "
                         f"({len(paper_norm)} normalised chars): partial_ratio would align the paper "
                         f"inside the segment and score 100")
    exact = [s in paper_norm for s in segs]
    scores = [100.0 if e else fuzz.partial_ratio(s, paper_norm) for s, e in zip(segs, exact)]
    row.update(n_segments=len(segs),
               verbatim=sum(len(s) * x for s, x in zip(segs, scores)) / sum(len(s) for s in segs),
               exact_substring=int(all(exact)),
               exact_substring_loose=int(all(loose(s).rstrip("-") in paper_loose for s in segs)))
    return row


def agreement(a, b) -> dict:
    """Word-set Jaccard and partial_ratio of two contexts; NaN unless both are
    non-empty and neither is a refusal."""
    if any(not (c or "").strip() or is_refusal(c) for c in (a, b)):
        return {"jaccard": math.nan, "partial_ratio": math.nan}
    wa, wb = word_set(a), word_set(b)
    union = wa | wb
    return {"jaccard": len(wa & wb) / len(union) if union else math.nan,
            "partial_ratio": fuzz.partial_ratio(norm(a), norm(b))}


def _load_contexts(lang: str, split: str, condition: str) -> dict:
    """index -> context from the store. chomps.retrieval is imported lazily so this
    module (and its tests) do not need the retrieval stack at import time."""
    from . import retrieval
    if retrieval.SEPARATOR != CHUNK_SEP:
        raise RuntimeError(f"chomps.retrieval.SEPARATOR is {retrieval.SEPARATOR!r}; "
                           f"retrieval_eval.CHUNK_SEP must match it")
    return retrieval.load_contexts(lang, split, condition)


def load_condition(lang: str, split: str, condition: str, examples: list) -> dict:
    """index -> context of one store, checked to cover exactly the examples."""
    ctx = _load_contexts(lang, split, condition)
    if not isinstance(ctx, dict):
        raise TypeError(f"{lang}/{split}/{condition}: expected index -> context, got {type(ctx).__name__}")
    for idx, c in ctx.items():
        if c is not None and not isinstance(c, str):
            raise TypeError(f"{lang}/{split}/{condition}/{idx}: context is {type(c).__name__}")
    want = {e["index"] for e in examples}
    if ctx.keys() != want:
        raise ValueError(f"{lang}/{split}/{condition}: store does not match the dataset "
                         f"({len(want - ctx.keys())} examples missing, {len(ctx.keys() - want)} unknown indices)")
    return ctx


def _write_csv(path: Path, rows: list, columns: list) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _read_csv(path: Path, columns: list) -> pd.DataFrame:
    """A stats/agreement file; one with another column layout (an older version) raises."""
    df = pd.read_csv(path, dtype={c: str for c in _TEXT_COLUMNS if c in columns})
    if list(df.columns) != columns:
        raise ValueError(f"{path} has columns {list(df.columns)}, expected {columns}; "
                         f"rewrite it with python -m chomps.retrieval_eval")
    return df


def write_grading_sheet(lang: str, examples: list, contexts: dict, out_dir: Path) -> Path:
    """One JSON row per test example with the contexts of GRADING_CONDITIONS (unblinded)."""
    path = out_dir / f"grading_sheet_{lang}_{GRADING_SPLIT}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            row = {"index": e["index"], "question": e["question"], "prompt": e["prompt"],
                   "output_text": e["output_text"], "gold_factuality": e["labels"]["factuality"],
                   "contexts": {c: contexts[c][e["index"]] for c in GRADING_CONDITIONS}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def evaluate_split(lang: str, split: str, examples: list, contexts: dict, papers: dict,
                   out_dir: Path, workers: int) -> None:
    """Write the stats files of every loaded condition, the agreement file when both
    LLM extractions are loaded, and the grading sheet for the test split.
    `papers` caches (lang, title) -> (norm, loose) across splits."""
    for e in examples:
        if (lang, e["title"]) not in papers:
            text = data.paper_text(lang, e["title"])
            papers[(lang, e["title"])] = (norm(text), loose(text))
    qid = question_ids(examples)
    _encoder()  # load once before the worker threads
    for condition, ctx in contexts.items():
        def one(e, ctx=ctx):
            return {"index": e["index"], "question_id": qid[e["index"]],
                    **context_stats(ctx[e["index"]], *papers[(lang, e["title"])])}
        rows = llm.run_parallel(one, examples, workers=workers, desc=f"retrieval_eval {lang}/{split}/{condition}")
        _write_csv(out_dir / f"stats_{lang}_{split}_{condition}.csv", rows, STATS_COLUMNS)
    a, b = AGREEMENT_PAIR
    if a in contexts and b in contexts:
        rows = []
        for e in examples:
            ca, cb = contexts[a][e["index"]], contexts[b][e["index"]]
            rows.append({"index": e["index"], "question_id": qid[e["index"]],
                         f"{a}_not_found": int(is_refusal(ca)), f"{b}_not_found": int(is_refusal(cb)),
                         **agreement(ca, cb)})
        _write_csv(out_dir / f"agreement_{lang}_{split}.csv", rows, AGREEMENT_COLUMNS)
    elif a in contexts or b in contexts:
        print(f"[retrieval_eval] agreement {lang}/{split} not computed: needs conditions {a}, {b}")
    if split == GRADING_SPLIT:
        if all(c in contexts for c in GRADING_CONDITIONS):
            print(f"[retrieval_eval] wrote {write_grading_sheet(lang, examples, contexts, out_dir)}")
        else:
            print(f"[retrieval_eval] grading sheet {lang}/{split} not written: needs conditions "
                  f"{', '.join(GRADING_CONDITIONS)}")


def _parse_stats_name(path: Path) -> tuple:
    """stats_{lang}_{split}_{condition}.csv -> (lang, split, condition); raises otherwise."""
    parts = path.stem[len("stats_"):].split("_", 2)
    if len(parts) != 3 or parts[0] not in config.LANGS or parts[1] not in config.SPLITS \
            or parts[2] not in CONDITIONS:
        raise ValueError(f"unexpected stats file {path}")
    return tuple(parts)


def _pct(x: float) -> float:
    return 100.0 * x


def summarize(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Aggregate every stats_*.csv (and agreement_*.csv) in out_dir into summary.csv."""
    a, b = AGREEMENT_PAIR
    rows = []
    for path in sorted(out_dir.glob("stats_*.csv")):
        lang, split, condition = _parse_stats_name(path)
        df = _read_csv(path, STATS_COLUMNS)
        ok = df[(df["empty"] == 0) & (df["not_found"] == 0)]
        row = {"lang": lang, "split": split, "condition": condition, "n": len(df),
               "n_questions": df["question_id"].nunique(), "n_distinct_contexts": df["context_sha1"].nunique(),
               "empty_n": int(df["empty"].sum()), "empty_pct": _pct(df["empty"].mean()),
               "not_found_n": int(df["not_found"].sum()), "not_found_pct": _pct(df["not_found"].mean()),
               "scored_n": len(ok), "scored_questions": ok["question_id"].nunique(),
               "verbatim_mean": ok["verbatim"].mean(), "verbatim_median": ok["verbatim"].median(),
               "exact_pct": _pct(ok["exact_substring"].mean()),
               "exact_loose_pct": _pct(ok["exact_substring_loose"].mean()),
               "chars_mean": df["chars"].mean(), "chars_median": df["chars"].median(),
               "tokens_mean": df["tokens"].mean(), "tokens_median": df["tokens"].median()}
        agree_path = out_dir / f"agreement_{lang}_{split}.csv"
        if condition == b and agree_path.exists():
            ag = _read_csv(agree_path, AGREEMENT_COLUMNS)
            scored = ag.dropna(subset=["jaccard", "partial_ratio"])
            nf_a, nf_b = ag[f"{a}_not_found"] == 1, ag[f"{b}_not_found"] == 1
            row.update(agree_n=len(scored), agree_questions=scored["question_id"].nunique(),
                       agree_both_not_found_n=int((nf_a & nf_b).sum()),
                       agree_one_not_found_n=int((nf_a ^ nf_b).sum()),
                       agree_jaccard_mean=scored["jaccard"].mean(),
                       agree_jaccard_median=scored["jaccard"].median(),
                       agree_partial_ratio_mean=scored["partial_ratio"].mean(),
                       agree_partial_ratio_median=scored["partial_ratio"].median())
        rows.append(row)
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    order = {v: i for i, v in enumerate(config.LANGS + config.SPLITS + CONDITIONS)}
    summary = summary.sort_values(["lang", "split", "condition"], key=lambda s: s.map(order), kind="stable")
    summary = summary.round(4)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def run(langs, splits, conditions, out_dir: Path = OUT_DIR, workers: int = 4) -> pd.DataFrame:
    """Load every requested store first (all missing ones are reported together,
    before anything is written), then evaluate and rebuild the summary."""
    loaded, missing = [], []
    for lang in langs:
        for split in splits:
            examples = data.load_examples(lang, split)
            contexts = {}
            for c in conditions:
                try:
                    contexts[c] = load_condition(lang, split, c, examples)
                except FileNotFoundError as e:
                    missing.append(f"{lang}/{split}/{c} ({e})")
            loaded.append((lang, split, examples, contexts))
    if missing:
        raise FileNotFoundError("missing contexts stores; build them with `python -m chomps.retrieval`:\n  "
                                + "\n  ".join(missing))
    out_dir.mkdir(parents=True, exist_ok=True)
    papers = {}
    for lang, split, examples, contexts in loaded:
        evaluate_split(lang, split, examples, contexts, papers, out_dir, workers)
    return summarize(out_dir)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m chomps.retrieval_eval", description=__doc__.split("\n")[0])
    ap.add_argument("--lang", nargs="+", choices=config.LANGS, default=list(config.LANGS))
    ap.add_argument("--split", nargs="+", choices=config.SPLITS, default=list(config.SPLITS))
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(DEFAULT_CONDITIONS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    summary = run(args.lang, args.split, args.conditions, OUT_DIR, args.workers)
    shown = summary[summary["lang"].isin(args.lang) & summary["split"].isin(args.split)
                    & summary["condition"].isin(args.conditions)]
    cols = ["lang", "split", "condition", "n", "n_questions", "empty_n", "not_found_n", "verbatim_mean",
            "exact_pct", "exact_loose_pct", "chars_mean", "tokens_mean", "agree_n", "agree_jaccard_mean",
            "agree_partial_ratio_mean"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(shown[cols].to_string(index=False))
    print(f"[retrieval_eval] summary: {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
