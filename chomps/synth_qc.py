"""Synthetic fine-tuning data QC (DESIGN.md, "Synthetic fine-tuning data QC").

Input: code/{lang}/finetuning/finetune_dataset_raw.jsonl, written by
generate_raw_ft_dataset.py: gpt-5-mini read the full text of each of the 100
newest cs.CL/LG/AI arXiv papers (Jan 2026) four times and returned one correct
("n") and one deliberately wrong ("y") answer per call, each with a supposedly
verbatim context excerpt. Rows are keyed by `row` = 0-based line number.

Protocol decisions:
* Validation: a malformed row (bad JSON, missing/empty field, veredict not y/n,
  a question naming two papers) raises in load_raw, before any paid call;
  statistics over silently dropped rows would misstate the data.
* Duplicates: exact = identical question strings; near = rapidfuzz `ratio` >= 90
  on the question with the arXiv id and the template prefix ("In the paper",
  "En el documento") removed and `utils.default_process` applied, so questions
  compare on content rather than on boilerplate.
* arXiv ids: the DESIGN regex with digit boundaries, searched in questions only.
  The generator passed the id as the paper "title" and the model copied it into
  some questions only, so ids exist for a subset of rows (row and id counts are
  in the summary). Mentions of one paper count once, with the version when one
  is given ("2601.11190 ... 2601.11190v1" -> 2601.11190v1); two papers, or two
  versions of one paper (two texts), make the row's source ambiguous and raise.
  A row without an id has no "own paper"; the only paper it is compared with is
  the labelled best match of the pool measure below.
* Overlap with the task papers (expected: none). SHROOM-CAP records no arXiv id
  for its papers (url/doi name the ACL Anthology and other hosts; exceptions are
  counted in task_papers_with_arxiv_id), so titles, not ids, carry the claim.
  Each measure is reported next to the size of what it compares:
  - ids: question ids (version stripped) against every arXiv-like id anywhere in
    the task examples (task_arxiv_like_ids). With task_papers_with_arxiv_id = 0
    these are all citations in the examples' text (model outputs), so an
    overlap of 0 only says that no task example cites a fine-tuning paper.
  - question titles (narrow): titles of the ft_titles_known papers named in
    questions (arXiv API metadata, --download-arxiv) against the task_titles,
    rapidfuzz ratio >= 90 after utils.default_process. Empty, never 0, without
    that metadata.
  - corpus titles: every paper in finetuning_papers/arxiv_meta.jsonl (the arXiv
    id range scanned by common/fetch_arxiv_range.py, a superset of the pool
    PDFs) against the task titles, same scorer, plus the closest ratio. It
    covers rows without an id insofar as their papers lie in that range, which
    arxiv_ids_in_corpus and pool_ge95_pct_no_arxiv_id (--verbatim) check. The
    file is required.
* LLM audit (gpt-5-mini, JSON mode, no temperature: the reasoning deployment
  rejects one): (b) a support check that never sees the generator's label, so it
  is an independent check of it, on every row; (a) a hallucination-type
  classification on veredict=y rows, told the answer is the deliberately wrong
  one. It may answer error_found=false (type null) instead of being forced into
  "other". A reply that is not valid JSON or uses a label outside the taxonomy is
  re-asked with a fresh sample (llm.chat repeat=1, 2: distinct cache entries),
  logged and counted in the summary; after MAX_ATTEMPTS invalid replies the run
  raises. Instructions are in English for both languages so categories match.
  Agreement of the support verdict with the generator's label, two ways (the
  per-cell counts audit_{label}_{verdict} are in the summary too): strict, n =
  supported and y = contradicted (not_addressed agrees with neither); lenient,
  n = supported and y = contradicted or not_addressed, i.e. the binary rule of
  the factuality judge prompt ("y" when the answer is not supported).
* Verbatim (--verbatim): rapidfuzz partial_ratio and exact-substring test of the
  whitespace-normalised context against the whitespace-normalised full pypdf text
  (the generator saw the full, untruncated text). Two separate measures:
  - own paper (`status`, `partial_ratio`, `exact_substring`): the paper the
    question names, exact version only (a later version is different text, so it
    is not substituted); status no_arxiv_id or pdf_missing otherwise.
  - pool (`pool_*`): the best match over every PDF in finetuning_papers (the
    referenced ids plus, if present, the corpus rebuilt by
    common/fetch_arxiv_range.py). This covers rows without an id. It is a lower
    bound on verbatim-ness (a row whose source is not in the pool scores low), and
    its attribution is validated on rows whose paper is known
    (pool_attribution_agree_pct).
* summary.csv: metrics as rows, one column per language. A run over some
  languages replaces their columns and keeps the others of an existing file
  (each column comes whole from one run; the kept languages are printed).
* Downloads (--download-arxiv): requests are spaced >= 3 s apart, which is the
  arXiv API Terms of Use rate limit, not a timing workaround. Files are written
  atomically, so a re-run resumes. A failed id is recorded and the step raises
  after trying all of them.

CLI (run from code/):
    python -m chomps.synth_qc                    # stats + audit -> ../results/synth_qc/
    python -m chomps.synth_qc --lang spanish     # one language (other summary.csv columns kept)
    python -m chomps.synth_qc --verbatim         # also the verbatim check (own paper + pool)
    python -m chomps.synth_qc --download-arxiv   # fetch PDFs + metadata, then exit
    python -m chomps.synth_qc --smoke 3          # audit first 3 rows/lang, print, write nothing
"""
import argparse
import csv
import json
import os
import re
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
from rapidfuzz import fuzz, process, utils

from . import config, data, llm

OUT_DIR = config.RESULTS / "synth_qc"
PAPERS_DIR = config.DATASETS / "finetuning_papers"
TEXT_CACHE = config.CACHE / "papers" / "finetuning"
METADATA_NAME = "referenced_arxiv_metadata.json"  # question ids only, written by --download-arxiv
CORPUS_META_NAME = "arxiv_meta.jsonl"  # the scanned id range, written by common/fetch_arxiv_range.py

REQUIRED_FIELDS = ("question", "context", "answer", "veredict")
ARXIV_RE = re.compile(r"(?<![\d.])\d{4}\.\d{4,5}(?:v\d+)?(?!\d)")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TEMPLATE_PREFIX = re.compile(r"^\s*(?:in the paper|en el documento)\b[\s,:]*", re.IGNORECASE)
NEAR_DUP_THRESHOLD = 90
TITLE_MATCH_THRESHOLD = 90
VERBATIM_THRESHOLD = 95

AUDIT_MODEL = "gpt-5-mini"
AUDIT_MAX_COMPLETION_TOKENS = config.JUDGE_MAX_COMPLETION_TOKENS
MAX_ATTEMPTS = 3
SUPPORT_LABELS = ("supported", "contradicted", "not_addressed")
HALLUCINATION_TYPES = ("wrong_number", "wrong_entity", "negation_or_contradiction",
                       "unsupported_claim", "wrong_method_or_result", "other")

ARXIV_PDF_URL = "https://arxiv.org/pdf/{}"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_MIN_INTERVAL_S = 3.0  # arXiv API Terms of Use: at most one request every 3 s
HTTP_TIMEOUT_S = 120
USER_AGENT = "chomps-synth-qc/1.0 (+https://github.com/sho-404/Chomps2025)"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


# --- input -------------------------------------------------------------------

def raw_path(lang: str) -> Path:
    if lang not in config.LANGS:
        raise ValueError(f"unknown language {lang!r}")
    return config.CODE_ROOT / lang / "finetuning" / "finetune_dataset_raw.jsonl"


def load_raw(path: Path) -> list:
    """Rows in file order; raises on the first malformed line (with its location)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            where = f"{path}:{lineno}"
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: invalid JSON ({e})") from e
            if not isinstance(r, dict):
                raise ValueError(f"{where}: row is not a JSON object")
            for k in REQUIRED_FIELDS:
                if not isinstance(r.get(k), str) or not r[k].strip():
                    raise ValueError(f"{where}: field {k!r} missing, not a string, or empty")
            if r["veredict"] not in ("y", "n"):
                raise ValueError(f"{where}: veredict must be 'y' or 'n', got {r['veredict']!r}")
            try:
                question_arxiv_id(r["question"])
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from e
            rows.append(r)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _base_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def question_arxiv_id(question: str) -> Optional[str]:
    """The arXiv id the question names, None if it names none. Mentions of one paper
    collapse to one id, versioned if any mention is; two papers, or two versions of
    one paper, raise (the row's source would be ambiguous)."""
    ids = set(ARXIV_RE.findall(question))
    if len({_base_id(i) for i in ids}) > 1:
        raise ValueError(f"question names several arXiv papers {sorted(ids)}: {question!r}")
    versioned = {i for i in ids if i != _base_id(i)}
    if len(versioned) > 1:
        raise ValueError(f"question names several versions of one arXiv paper {sorted(versioned)}: {question!r}")
    return next(iter(versioned or ids), None)


def referenced_ids(rows: list) -> list:
    """Sorted distinct ids named by the questions (question_arxiv_id of each row)."""
    return sorted({question_arxiv_id(r["question"]) for r in rows} - {None})


# --- descriptive statistics --------------------------------------------------

def _question_core(question: str) -> str:
    return utils.default_process(_TEMPLATE_PREFIX.sub("", ARXIV_RE.sub(" ", question), count=1))


def duplicate_stats(rows: list) -> dict:
    qs = [r["question"] for r in rows]
    core = [_question_core(q) for q in qs]
    sim = process.cdist(core, core, scorer=fuzz.ratio, workers=-1)
    ii, jj = np.nonzero(np.triu(sim >= NEAR_DUP_THRESHOLD, k=1))
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    same = 0
    for i, j in zip(ii.tolist(), jj.tolist()):
        same += rows[i]["veredict"] == rows[j]["veredict"]
        parent[find(i)] = find(j)
    clusters = Counter(find(i) for i in range(len(rows)))
    multi = [size for size in clusters.values() if size > 1]
    return {
        "questions_unique": len(set(qs)),
        "examples_unique": len({tuple(r[k] for k in REQUIRED_FIELDS) for r in rows}),
        f"q_near{NEAR_DUP_THRESHOLD}_pairs": len(ii),
        f"q_near{NEAR_DUP_THRESHOLD}_pairs_same_veredict": same,
        f"q_near{NEAR_DUP_THRESHOLD}_pairs_opposite_veredict": len(ii) - same,
        f"q_near{NEAR_DUP_THRESHOLD}_rows": sum(multi),
        f"q_near{NEAR_DUP_THRESHOLD}_clusters": len(multi),
    }


def length_stats(rows: list) -> dict:
    """Chars and whitespace words; answer words also per class (a length cue
    between y and n answers would be a shortcut for the fine-tuned judge)."""
    out = {}
    for field in ("answer", "context"):
        chars = [len(r[field]) for r in rows]
        out.update({f"{field}_chars_mean": round(statistics.mean(chars), 1),
                    f"{field}_chars_median": round(statistics.median(chars), 1),
                    f"{field}_chars_min": min(chars), f"{field}_chars_max": max(chars),
                    f"{field}_words_mean": round(statistics.mean(len(r[field].split()) for r in rows), 1)})
        if field == "answer":
            for v in ("y", "n"):
                words = [len(r["answer"].split()) for r in rows if r["veredict"] == v]
                out[f"answer_words_mean_{v}"] = round(statistics.mean(words), 1) if words else None
    return out


def task_papers(lang: str) -> dict:
    """The SHROOM-CAP papers of `lang` (all splits): their titles, how many name an
    arXiv id in their own url/doi fields, and every arXiv-like id (version stripped)
    in any string field of their examples (url/doi and citations in the text alike)."""
    titles, ids, with_id = set(), set(), set()
    for split in config.SPLITS:
        for ex in data.load_examples(lang, split):
            titles.add(ex["title"])
            for k, v in ex.items():
                found = {_base_id(m) for m in ARXIV_RE.findall(v)} if isinstance(v, str) else set()
                ids |= found
                if found and k in ("url", "doi"):
                    with_id.add(ex["title"])
    return {"titles": sorted(titles), "arxiv_ids": ids, "papers_with_arxiv_id": len(with_id)}


def title_scores(ft_titles: dict, task_titles: list) -> list:
    """(arxiv_id, ft_title, closest task title, ratio) for every fine-tuning title,
    highest ratio first; ratio = rapidfuzz fuzz.ratio after utils.default_process."""
    if not task_titles:
        raise ValueError("no task titles to compare with")
    out = []
    for aid, title in ft_titles.items():
        task_title, score, _ = process.extractOne(title, task_titles, scorer=fuzz.ratio,
                                                  processor=utils.default_process)
        out.append((aid, title, task_title, score))
    return sorted(out, key=lambda m: (-m[3], m[0]))


def title_matches(ft_titles: dict, task_titles: list) -> list:
    """The title_scores rows at or above TITLE_MATCH_THRESHOLD, ratio rounded to 0.1."""
    return [(a, t, tt, round(s, 1)) for a, t, tt, s in title_scores(ft_titles, task_titles)
            if s >= TITLE_MATCH_THRESHOLD]


def arxiv_stats(rows: list, task: dict, metadata: dict) -> dict:
    """Question ids, then the id and narrow title overlaps with the task papers, each
    next to the size of the set it compares against (module docstring)."""
    ids = referenced_ids(rows)
    months = Counter(i.split(".")[0] for i in ids)
    known = {i: metadata[i]["title"] for i in ids if i in metadata}
    published = sorted(metadata[i]["published"][:10] for i in known)
    return {
        "arxiv_rows": sum(question_arxiv_id(r["question"]) is not None for r in rows),
        "arxiv_ids": len(ids),
        "arxiv_id_months": " ".join(f"{m}:{c}" for m, c in sorted(months.items())),
        "arxiv_id_list": " ".join(ids),
        "task_titles": len(task["titles"]),
        "task_papers_with_arxiv_id": task["papers_with_arxiv_id"],
        "task_arxiv_like_ids": len(task["arxiv_ids"]),
        "overlap_ids_with_task": len({_base_id(i) for i in ids} & task["arxiv_ids"]),
        "ft_titles_known": len(known),
        "overlap_titles_with_task": len(title_matches(known, task["titles"])) if known else None,
        "ft_published_range": f"{published[0]}..{published[-1]}" if published else None,
    }


def load_corpus(papers_dir: Optional[Path] = None) -> dict:
    """base id -> {"id", "title", "published"} from arxiv_meta.jsonl, the metadata of the
    arXiv id range scanned by common/fetch_arxiv_range.py (a superset of the pool PDFs).
    A missing file, a malformed row or a repeated id raises."""
    path = (PAPERS_DIR if papers_dir is None else papers_dir) / CORPUS_META_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; it is written by code/common/fetch_arxiv_range.py")
    out = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            where = f"{path}:{lineno}"
            try:
                m = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: invalid JSON ({e})") from e
            if not (isinstance(m, dict) and isinstance(m.get("id"), str) and ARXIV_RE.fullmatch(m["id"])
                    and isinstance(m.get("title"), str) and m["title"].strip()
                    and isinstance(m.get("published"), str) and _DATE.match(m["published"])):
                raise ValueError(f"{where}: needs an arXiv 'id', a non-empty 'title' and a 'published' date")
            base = _base_id(m["id"])
            if base in out:
                raise ValueError(f"{where}: arXiv id {base} repeated")
            out[base] = {"id": m["id"], "title": _ws(m["title"]), "published": m["published"]}
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


def corpus_stats(rows: list, task: dict, corpus: dict) -> dict:
    """Title overlap of the whole corpus (load_corpus) with the task papers, the closest
    ratio, and how many question ids the corpus holds (module docstring)."""
    scores = title_scores({b: m["title"] for b, m in corpus.items()}, task["titles"])
    published = sorted(m["published"][:10] for m in corpus.values())
    return {
        "corpus_papers": len(corpus),
        "corpus_published_range": f"{published[0]}..{published[-1]}",
        "arxiv_ids_in_corpus": sum(_base_id(i) in corpus for i in referenced_ids(rows)),
        "overlap_titles_corpus_with_task": sum(s >= TITLE_MATCH_THRESHOLD for *_, s in scores),
        "corpus_title_best_ratio": round(scores[0][3], 1),
    }


# --- LLM audit ---------------------------------------------------------------

AUDIT_SYSTEM = ("You check answers to questions about scientific papers against a quoted excerpt "
                "of the paper. Reply with a single JSON object.")

_LANG_NOTE = ("The question and answer may be in English or Spanish, and the excerpt may be in a "
              "different language from them.")

# Plain templates filled by a single str.format call ({{ }} are literal JSON braces).
SUPPORT_PROMPT = """Decide whether the ANSWER to the QUESTION is supported by the EXCERPT. Use only the excerpt, not outside knowledge. {lang_note}

Verdicts:
- "supported": everything the answer claims in response to the question is stated in, or directly follows from, the excerpt.
- "contradicted": at least one claim in the answer conflicts with the excerpt (for example a different number, name, method or result, or the opposite statement).
- "not_addressed": nothing in the answer conflicts with the excerpt, but at least one of its claims cannot be checked against it.

Reply with JSON: {{"verdict": "supported" | "contradicted" | "not_addressed", "rationale": "<one sentence, in English>"}}

QUESTION:
{question}

EXCERPT:
{context}

ANSWER:
{answer}"""

TYPE_PROMPT = """The ANSWER below was written on purpose to be factually wrong about the paper (a synthetic hallucination); the EXCERPT is the passage that holds the correct information. Using only the excerpt, identify the answer's main error. {lang_note}

Error types (pick the one that best describes the main error):
- "wrong_number": a number, quantity, percentage, date or other numeric value differs from the excerpt.
- "wrong_entity": a dataset, model, method, metric, person, organisation or other name is swapped or invented.
- "negation_or_contradiction": the answer states the opposite of the excerpt (a negation, or a reversed comparison or direction).
- "unsupported_claim": the answer adds information that the excerpt does not contain, without contradicting it.
- "wrong_method_or_result": the answer misdescribes how a method works or what an experiment found, beyond a single wrong number or name.
- "other": an error that none of the types above describes.
If the answer has no error with respect to the excerpt, set "error_found" to false and "type" to null.

Reply with JSON: {{"error_found": true | false, "type": "<error type>" | null, "rationale": "<one sentence, in English>"}}

QUESTION:
{question}

EXCERPT:
{context}

ANSWER:
{answer}"""


def support_messages(row: dict) -> list:
    """Blind support check: built from question/context/answer only, never the label."""
    user = SUPPORT_PROMPT.format(lang_note=_LANG_NOTE, question=row["question"], context=row["context"],
                                 answer=row["answer"])
    return [{"role": "system", "content": AUDIT_SYSTEM}, {"role": "user", "content": user}]


def type_messages(row: dict) -> list:
    user = TYPE_PROMPT.format(lang_note=_LANG_NOTE, question=row["question"], context=row["context"],
                              answer=row["answer"])
    return [{"role": "system", "content": AUDIT_SYSTEM}, {"role": "user", "content": user}]


def _label(value, allowed: tuple, field: str) -> str:
    v = value.strip().lower().replace(" ", "_").replace("-", "_") if isinstance(value, str) else value
    if v not in allowed:
        raise ValueError(f"{field}={value!r} is not one of {allowed}")
    return v


def _rationale(obj: dict) -> str:
    if not isinstance(obj.get("rationale"), str):
        raise ValueError("rationale missing or not a string")
    return obj["rationale"].strip()


def validate_support(obj) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return {"verdict": _label(obj.get("verdict"), SUPPORT_LABELS, "verdict"), "rationale": _rationale(obj)}


def validate_type(obj) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    found = obj.get("error_found")
    if not isinstance(found, bool):
        raise ValueError(f"error_found must be true or false, got {found!r}")
    if found:
        kind = _label(obj.get("type"), HALLUCINATION_TYPES, "type")
    elif obj.get("type") is not None:
        raise ValueError(f"type must be null when error_found is false, got {obj.get('type')!r}")
    else:
        kind = None
    return {"error_found": found, "type": kind, "rationale": _rationale(obj)}


def ask_json(messages: list, validate: Callable, what: str) -> tuple:
    """Ask the auditor; returns (validated reply, call record). Invalid replies are
    re-asked with repeat=attempt (a new sample, its own cache entry) and logged;
    after MAX_ATTEMPTS invalid replies this raises with every error seen."""
    errors, usage = [], Counter()
    for attempt in range(MAX_ATTEMPTS):
        r = llm.chat(AUDIT_MODEL, messages, max_completion_tokens=AUDIT_MAX_COMPLETION_TOKENS,
                     response_format={"type": "json_object"}, repeat=attempt)
        usage.update({k: r[k] for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s")})
        try:
            parsed = validate(json.loads(r["content"]))
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            errors.append(f"attempt {attempt + 1}: {e} (finish_reason={r['finish_reason']}, raw={r['content'][:300]!r})")
            print(f"[synth_qc] {what}: invalid reply, re-asking: {errors[-1]}", flush=True)
            continue
        record = {"attempts": attempt + 1, "model": r["model"], "cached": r["cached"],
                  "finish_reason": r["finish_reason"], "raw": r["content"],
                  **{k: round(v, 3) if k == "latency_s" else v for k, v in usage.items()}}
        return parsed, record
    raise ValueError(f"{what}: no valid reply in {MAX_ATTEMPTS} attempts:\n" + "\n".join(errors))


def audit_row(lang: str, i: int, row: dict) -> dict:
    aid = question_arxiv_id(row["question"])  # raises before the paid calls
    support, support_call = ask_json(support_messages(row), validate_support, f"{lang} row {i} support")
    out = {"row": i, "veredict": row["veredict"], "arxiv_id": aid,
           "question": row["question"], "answer": row["answer"],
           "support": support["verdict"], "support_rationale": support["rationale"],
           "error_found": None, "hallucination_type": None, "type_rationale": None,
           "calls": {"support": support_call, "type": None}}
    if row["veredict"] == "y":
        kind, type_call = ask_json(type_messages(row), validate_type, f"{lang} row {i} type")
        out.update(error_found=kind["error_found"], hallucination_type=kind["type"],
                   type_rationale=kind["rationale"])
        out["calls"]["type"] = type_call
    return out


def audit(lang: str, rows: list, workers: int = 12) -> list:
    return llm.run_parallel(lambda item: audit_row(lang, *item), list(enumerate(rows)),
                            workers=workers, desc=f"synth_qc audit {lang}")


def _pct(k: int, n: int) -> Optional[float]:
    return round(100 * k / n, 2) if n else None


def audit_stats(rows: list, audit_rows: list) -> dict:
    """Support verdict per generator label, strict and lenient agreement (module
    docstring), hallucination types of the y rows, and the cost of the audit."""
    if [a["row"] for a in audit_rows] != list(range(len(rows))) or \
            any(a["veredict"] != r["veredict"] for a, r in zip(audit_rows, rows)):
        raise ValueError("audit rows are not aligned with the raw rows")
    by = Counter((a["veredict"], a["support"]) for a in audit_rows)
    strict = by[("n", "supported")] + by[("y", "contradicted")]
    out = {"audit_rows": len(audit_rows), "audit_agreement_strict_pct": _pct(strict, len(audit_rows)),
           "audit_agreement_lenient_pct": _pct(strict + by[("y", "not_addressed")], len(audit_rows))}
    for v in ("n", "y"):
        out.update({f"audit_{v}_{s}": by[(v, s)] for s in SUPPORT_LABELS})
    ys = [a for a in audit_rows if a["veredict"] == "y"]
    kinds = Counter(a["hallucination_type"] if a["error_found"] else "none_found" for a in ys)
    for t in (*HALLUCINATION_TYPES, "none_found"):
        out[f"type_{t}"] = kinds[t]
        out[f"type_{t}_pct"] = _pct(kinds[t], len(ys))
    calls = [c for a in audit_rows for c in a["calls"].values() if c is not None]
    price = config.PRICES_USD_PER_M[config.price_key(AUDIT_MODEL)]
    prompt, completion = sum(c["prompt_tokens"] for c in calls), sum(c["completion_tokens"] for c in calls)
    out.update({"audit_calls": len(calls), "audit_reasked_calls": sum(c["attempts"] > 1 for c in calls),
                "audit_prompt_tokens": prompt, "audit_completion_tokens": completion,
                "audit_usd": round((prompt * price["input"] + completion * price["output"]) / 1e6, 4)})
    return out


# --- verbatim check ----------------------------------------------------------

def _ws(s: str) -> str:
    return " ".join(s.split())


def ft_paper_text(arxiv_id: str, papers_dir: Path) -> Optional[str]:
    """Full pypdf text of {papers_dir}/{id}.pdf, cached under cache/papers/finetuning
    (a versioned arXiv id names immutable content); None when the PDF is absent. Reads
    trust any non-empty cache file, so it is written atomically: a crash cannot leave
    a truncated text behind."""
    pdf = papers_dir / f"{arxiv_id}.pdf"
    if not pdf.exists():
        return None
    cached = TEXT_CACHE / f"{arxiv_id}.txt"
    if cached.exists() and cached.stat().st_size > 0:
        return cached.read_text(encoding="utf-8")
    text = data.extract_pdf_text(pdf)
    if len(text.strip()) < 500:
        raise ValueError(f"PDF text too short ({len(text)} chars): {pdf}")
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    _atomic_write(cached, text.encode("utf-8"))
    return text


def pool_texts(papers_dir: Path) -> dict:
    """file stem (versioned arXiv id) -> whitespace-normalised text, for every PDF in papers_dir."""
    pdfs = sorted(papers_dir.glob("*.pdf"))
    out = {}
    for k, pdf in enumerate(pdfs, 1):
        out[pdf.stem] = _ws(ft_paper_text(pdf.stem, papers_dir))
        if k % 25 == 0 or k == len(pdfs):
            print(f"[synth_qc] paper text {k}/{len(pdfs)}", flush=True)
    return out


def verbatim(rows: list, papers_dir: Optional[Path] = None) -> list:
    papers_dir = PAPERS_DIR if papers_dir is None else papers_dir
    if not papers_dir.is_dir():
        raise FileNotFoundError(f"{papers_dir} does not exist; run --download-arxiv first")
    pool = pool_texts(papers_dir)
    if not pool:
        raise FileNotFoundError(f"no PDFs in {papers_dir}; run --download-arxiv first")
    ids, texts = list(pool), list(pool.values())
    ctxs = [_ws(r["context"]) for r in rows]
    sim = process.cdist(ctxs, texts, scorer=fuzz.partial_ratio, workers=-1)
    out = []
    for i, r in enumerate(rows):
        aid, best = question_arxiv_id(r["question"]), int(sim[i].argmax())
        rec = {"row": i, "arxiv_id": aid, "status": "no_arxiv_id", "partial_ratio": None,
               "exact_substring": None, "context_chars": len(r["context"]),
               "pool_papers": len(ids), "pool_best_id": ids[best],
               "pool_best_ratio": round(float(sim[i, best]), 2), "pool_exact_substring": ctxs[i] in texts[best]}
        if aid is not None and aid not in pool:
            rec["status"] = "pdf_missing"
        elif aid is not None:
            rec.update(status="scored", partial_ratio=round(float(sim[i, ids.index(aid)]), 2),
                       exact_substring=ctxs[i] in pool[aid])
        out.append(rec)
    return out


def verbatim_stats(rows: list) -> dict:
    status = Counter(r["status"] for r in rows)
    own = [r for r in rows if r["status"] == "scored"]
    scores = [r["partial_ratio"] for r in own]
    pool = [r["pool_best_ratio"] for r in rows]
    no_id = [r["pool_best_ratio"] for r in rows if r["arxiv_id"] is None]
    t = VERBATIM_THRESHOLD
    return {"verbatim_scored": status["scored"], "verbatim_no_arxiv_id": status["no_arxiv_id"],
            "verbatim_pdf_missing": status["pdf_missing"],
            "verbatim_partial_ratio_mean": round(statistics.mean(scores), 2) if scores else None,
            "verbatim_partial_ratio_median": round(statistics.median(scores), 2) if scores else None,
            f"verbatim_ge{t}_pct": _pct(sum(x >= t for x in scores), len(scores)),
            "verbatim_exact_pct": _pct(sum(r["exact_substring"] for r in own), len(own)),
            "pool_papers": rows[0]["pool_papers"],
            "pool_best_ratio_mean": round(statistics.mean(pool), 2),
            "pool_best_ratio_median": round(statistics.median(pool), 2),
            f"pool_ge{t}_pct": _pct(sum(x >= t for x in pool), len(pool)),
            "pool_exact_pct": _pct(sum(r["pool_exact_substring"] for r in rows), len(rows)),
            f"pool_ge{t}_pct_no_arxiv_id": _pct(sum(x >= t for x in no_id), len(no_id)),
            "pool_attribution_agree_pct": _pct(sum(_base_id(r["pool_best_id"]) == _base_id(r["arxiv_id"])
                                                   for r in own), len(own))}


# --- arXiv downloads ---------------------------------------------------------

class Spacer:
    """Start-to-start spacing of HTTP requests to arXiv. arXiv's API Terms of Use
    allow at most one request every three seconds; this wait implements that
    published limit and is not a delay hiding a race."""

    def __init__(self, interval_s: float = ARXIV_MIN_INTERVAL_S, clock=time.monotonic, sleep=time.sleep):
        self.interval_s, self.clock, self.sleep, self._last = interval_s, clock, sleep, None

    def wait(self):
        if self._last is not None:
            remaining = self._last + self.interval_s - self.clock()
            if remaining > 0:
                self.sleep(remaining)
        self._last = self.clock()


def _atomic_write(path: Path, payload: bytes):
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def parse_arxiv_atom(xml_bytes: bytes) -> dict:
    """base id -> {"id", "title", "published", "primary_category"} from an arXiv API
    Atom feed. Error entries (no /abs/ id) are skipped, so the ids they stand for
    surface as missing."""
    out = {}
    for entry in ET.fromstring(xml_bytes).findall(f"{_ATOM}entry"):
        url = entry.findtext(f"{_ATOM}id", "")
        if "/abs/" not in url:
            continue
        aid = url.rsplit("/abs/", 1)[1]
        cat = entry.find(f"{_ARXIV_NS}primary_category")
        out[_base_id(aid)] = {"id": aid, "title": _ws(entry.findtext(f"{_ATOM}title", "")),
                              "published": entry.findtext(f"{_ATOM}published", ""),
                              "primary_category": None if cat is None else cat.get("term")}
    return out


def load_metadata(papers_dir: Optional[Path] = None) -> dict:
    path = (PAPERS_DIR if papers_dir is None else papers_dir) / METADATA_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def download_arxiv(ids: list, papers_dir: Optional[Path] = None, get: Callable = requests.get,
                   spacer: Optional[Spacer] = None) -> dict:
    """Fetch {id}.pdf from https://arxiv.org/pdf/{id} for every id not on disk (exact
    version), then the arXiv API metadata (title, date, category) of every id missing
    from METADATA_NAME. Raises after trying every id if any request failed."""
    papers_dir = PAPERS_DIR if papers_dir is None else papers_dir
    spacer = Spacer() if spacer is None else spacer
    papers_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    failures = {}  # id -> list of messages (a PDF and a metadata failure can both occur)
    todo = [i for i in ids if not (papers_dir / f"{i}.pdf").exists()]
    print(f"[synth_qc] arXiv PDFs: {len(ids)} referenced, {len(ids) - len(todo)} on disk, {len(todo)} to "
          f"fetch, >= {ARXIV_MIN_INTERVAL_S:.0f} s apart (~{len(todo) * ARXIV_MIN_INTERVAL_S / 60:.1f} min)", flush=True)
    for k, aid in enumerate(todo, 1):
        spacer.wait()
        try:
            resp = get(ARXIV_PDF_URL.format(aid), headers=headers, timeout=HTTP_TIMEOUT_S)
        except requests.RequestException as e:
            failures.setdefault(aid, []).append(f"PDF: {type(e).__name__}: {e}")
        else:
            if resp.status_code == 200 and resp.content.startswith(b"%PDF-"):
                _atomic_write(papers_dir / f"{aid}.pdf", resp.content)
            else:
                failures.setdefault(aid, []).append(f"PDF: HTTP {resp.status_code}, content-type "
                                                    f"{resp.headers.get('content-type')!r}, not a PDF")
        print(f"[synth_qc] {k}/{len(todo)} {aid}: {'; '.join(failures.get(aid, ['ok']))}", flush=True)

    meta = load_metadata(papers_dir)
    missing = [i for i in ids if i not in meta]
    for start in range(0, len(missing), 100):
        chunk = missing[start:start + 100]
        spacer.wait()
        try:
            resp = get(ARXIV_API_URL, params={"id_list": ",".join(chunk), "max_results": len(chunk)},
                       headers=headers, timeout=HTTP_TIMEOUT_S)
            if resp.status_code != 200:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            got = parse_arxiv_atom(resp.content)
        except (requests.RequestException, ET.ParseError) as e:
            for aid in chunk:
                failures.setdefault(aid, []).append(f"metadata: {type(e).__name__}: {e}")
            continue
        for aid in chunk:
            if _base_id(aid) in got:
                meta[aid] = got[_base_id(aid)]
            else:
                failures.setdefault(aid, []).append("metadata: id absent from the arXiv API response")
    _atomic_write(papers_dir / METADATA_NAME, json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False).encode())
    if failures:
        raise RuntimeError(f"{len(failures)} arXiv id(s) failed (re-run to retry; completed files are kept):\n"
                           + "\n".join(f"  {k}: {'; '.join(v)}" for k, v in sorted(failures.items())))
    return {"fetched": len(todo), "on_disk_before": len(ids) - len(todo), "metadata": len(meta)}


# --- summary and CLI ---------------------------------------------------------

def summarize(rows: list, task: dict, metadata: dict, corpus: dict, audit_rows: Optional[list] = None,
              verbatim_rows: Optional[list] = None) -> dict:
    n_y = sum(r["veredict"] == "y" for r in rows)
    out = {"n": len(rows), "n_y": n_y, "n_n": len(rows) - n_y, "pct_y": _pct(n_y, len(rows))}
    out.update(duplicate_stats(rows))
    out.update(length_stats(rows))
    out.update(arxiv_stats(rows, task, metadata))
    out.update(corpus_stats(rows, task, corpus))
    if audit_rows is not None:
        out.update(audit_stats(rows, audit_rows))
    if verbatim_rows is not None:
        out.update(verbatim_stats(verbatim_rows))
    return out


def write_jsonl(path: Path, rows: list):
    _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))


def write_summary(summaries: dict, path: Path) -> list:
    """Metrics as rows, one column per language, so the CSV reads as a table. The
    languages in `summaries` replace their columns; the other columns of an existing
    file are kept as they are (each column comes whole from one run) and their
    languages are returned. An existing file that is not such a table raises."""
    if not set(summaries) <= set(config.LANGS):
        raise ValueError(f"unknown language(s) {sorted(set(summaries) - set(config.LANGS))}")
    kept = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            table = list(csv.reader(f))
        header = table[0] if table else []
        if header[:1] != ["metric"] or not set(header[1:]) <= set(config.LANGS) \
                or len(set(header[1:])) != len(header) - 1 or any(len(r) != len(header) for r in table):
            raise ValueError(f"{path} is not a summary table of this module; move it away to rewrite it")
        kept = {lang: {r[0]: r[j] for r in table[1:]}
                for j, lang in enumerate(header[1:], 1) if lang not in summaries}
    columns = {**kept, **summaries}
    langs = [lang for lang in config.LANGS if lang in columns]
    metrics = list(dict.fromkeys([m for s in summaries.values() for m in s] + [m for c in kept.values() for m in c]))
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", *langs])
        for m in metrics:
            w.writerow([m, *(columns[lang].get(m) for lang in langs)])
    os.replace(tmp, path)
    return [lang for lang in langs if lang in kept]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chomps.synth_qc",
                                 description="QC of the synthetic fine-tuning data (DESIGN.md). "
                                             "Default: statistics + LLM audit -> ../results/synth_qc/. "
                                             "A run over some languages replaces their summary.csv columns "
                                             "and keeps the others.")
    ap.add_argument("--lang", nargs="+", choices=config.LANGS, default=list(config.LANGS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--verbatim", action="store_true",
                    help="also score each context against its own arXiv paper and against every PDF in "
                         "the papers dir (needs --download-arxiv first)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--download-arxiv", action="store_true",
                      help=f"fetch the arXiv PDFs named in the questions (+ API metadata) into {PAPERS_DIR}, then exit")
    mode.add_argument("--smoke", type=int, metavar="N",
                      help="audit only the first N rows per language, print the JSON, write nothing")
    args = ap.parse_args(argv)
    if args.verbatim and (args.download_arxiv or args.smoke is not None):
        ap.error("--verbatim belongs to the full run, not to --download-arxiv or --smoke")
    if args.smoke is not None and args.smoke < 1:
        ap.error("--smoke needs N >= 1")

    rows = {lang: load_raw(raw_path(lang)) for lang in args.lang}
    if args.download_arxiv:
        print(download_arxiv(sorted({i for lang in args.lang for i in referenced_ids(rows[lang])})))
        return
    if args.smoke is not None:
        for lang in args.lang:
            for rec in audit(lang, rows[lang][:args.smoke], workers=args.workers):
                print(json.dumps({"lang": lang, **rec}, ensure_ascii=False, indent=1))
        print(f"[synth_qc] smoke mode: {args.smoke} row(s) per language audited, nothing written")
        return

    # Everything that can fail without the LLM is checked before the (paid) audit.
    if args.verbatim and not PAPERS_DIR.is_dir():
        raise FileNotFoundError(f"{PAPERS_DIR} does not exist; run --download-arxiv first")
    corpus = load_corpus()
    corpus_titles = {b: m["title"] for b, m in corpus.items()}
    tasks = {lang: task_papers(lang) for lang in args.lang}
    metadata = load_metadata()
    if not metadata:
        print(f"[synth_qc] no {METADATA_NAME} in {PAPERS_DIR}: question-title overlap left empty "
              f"(run --download-arxiv)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for lang in args.lang:
        verbatim_rows = None
        if args.verbatim:
            verbatim_rows = verbatim(rows[lang])
            write_jsonl(OUT_DIR / f"{lang}_verbatim.jsonl", verbatim_rows)
        audit_rows = audit(lang, rows[lang], workers=args.workers)
        write_jsonl(OUT_DIR / f"{lang}_audit.jsonl", audit_rows)
        summaries[lang] = summarize(rows[lang], tasks[lang], metadata, corpus, audit_rows, verbatim_rows)
        known = {i: metadata[i]["title"] for i in referenced_ids(rows[lang]) if i in metadata}
        for what, titles in (("fine-tuning paper", known), ("corpus paper", corpus_titles)):
            for aid, ft_title, task_title, score in title_matches(titles, tasks[lang]["titles"]):
                print(f"[synth_qc] {lang}: {what} {aid} {ft_title!r} matches task paper "
                      f"{task_title!r} (ratio {score})")
    kept = write_summary(summaries, OUT_DIR / "summary.csv")
    width = max(len(m) for s in summaries.values() for m in s)
    for m in dict.fromkeys(m for s in summaries.values() for m in s):
        if m != "arxiv_id_list":
            print(f"{m:<{width}}  " + "  ".join(f"{str(summaries[lang].get(m)):>12}" for lang in args.lang))
    if kept:
        print(f"[synth_qc] summary.csv: kept the {', '.join(kept)} column(s) of an earlier run")
    print(f"[synth_qc] wrote {OUT_DIR}/summary.csv and the per-row stores in {OUT_DIR}")


if __name__ == "__main__":
    main()
