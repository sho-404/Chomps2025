"""Dataset access: SHROOM-CAP splits, gold labels, and source-paper text.

Examples keep the original pseudo-RAG span under `context_original` (the
exact evidence used for the competition submission, from
<lang>/<split>/<pre>_<split>_data_with_context.jsonl).
"""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from rapidfuzz import process

from . import config

_PAPER_TEXT_CACHE = config.CACHE / "papers"


def _split_path(lang: str, split: str, suffix: str) -> Path:
    pre = config.LANG_PREFIX[lang]
    return config.DATASETS / lang / split / f"{pre}_{config.SPLIT_FILE_TAG[split]}_{suffix}.jsonl"


def load_labels(lang: str, split: str) -> dict:
    """index -> {"factuality": 0/1, "fluency": 0/1} (1 = hallucination present)."""
    out = {}
    for line in open(_split_path(lang, split, "label"), encoding="utf-8"):
        r = json.loads(line)
        out[r["index"]] = {t: 1 if r[config.LABEL_KEY[t]] == "y" else 0 for t in config.TRACKS}
    return out


def load_examples(lang: str, split: str) -> list:
    """List of dicts with the dataset fields plus `context_original` and `labels`."""
    labels = load_labels(lang, split)
    out = []
    for line in open(_split_path(lang, split, "data_with_context"), encoding="utf-8"):
        r = json.loads(line)
        r["context_original"] = r.pop("context", "") or ""
        if r["index"] not in labels:
            raise KeyError(f"{lang}/{split}: no gold label for {r['index']}")
        r["labels"] = labels[r["index"]]
        out.append(r)
    return out


def load_all(lang: str) -> dict:
    return {s: load_examples(lang, s) for s in config.SPLITS}


# --- source papers -----------------------------------------------------------

def normalize_filename(text: str) -> str:
    """Identical to download_papers.py so that titles map to the saved PDFs."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("/", "_").replace(":", " -")
    text = re.sub(r"[^\w\s-]", "", text).strip()
    return text[:150]


@lru_cache(maxsize=None)
def _pdf_index(lang: str) -> dict:
    papers = config.DATASETS / lang / "papers"
    return {p.stem.lower().strip(): p for p in papers.glob("*.pdf")}


def find_pdf(lang: str, title: str) -> Optional[Path]:
    """Exact normalized match first, then fuzzy (score >= 85) as the original pipeline did."""
    idx = _pdf_index(lang)
    key = normalize_filename(title).lower().strip()
    if key in idx:
        return idx[key]
    match = process.extractOne(key, list(idx.keys()))
    if match and match[1] >= 85:
        return idx[match[0]]
    return None


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t + "\n")
    # pypdf can emit lone surrogates from broken font encodings; they are not
    # encodable and would break the cache write, so replace them explicitly.
    return "".join(parts).encode("utf-8", errors="replace").decode("utf-8")


def paper_text(lang: str, title: str) -> str:
    """Full extracted text of the paper (cached under cache/papers). Raises if
    the PDF is missing or yields no text, so a missing paper never silently
    becomes an empty context."""
    _PAPER_TEXT_CACHE.joinpath(lang).mkdir(parents=True, exist_ok=True)
    cache_file = _PAPER_TEXT_CACHE / lang / (normalize_filename(title) + ".txt")
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file.read_text(encoding="utf-8")
    pdf = find_pdf(lang, title)
    if pdf is None:
        raise FileNotFoundError(f"no PDF for {lang} paper: {title!r}")
    text = extract_pdf_text(pdf)
    if len(text.strip()) < 500:
        raise ValueError(f"PDF text too short ({len(text)} chars) for {lang} paper: {title!r}")
    cache_file.write_text(text, encoding="utf-8")
    return text


def paper_coverage(lang: str) -> dict:
    """Which dataset titles map to a readable PDF. Returns {title: path-or-None}."""
    titles = {}
    for split in config.SPLITS:
        for ex in load_examples(lang, split):
            titles.setdefault(ex["title"], find_pdf(lang, ex["title"]))
    return titles
