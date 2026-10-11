"""Repair source-paper PDFs that did not download cleanly.

`download_papers.py` follows the `url` field of each dataset entry, which for a
few papers points at a GitHub page or a landing page rather than the PDF. This
script re-checks every PDF in datasets/<lang>/papers, and for each unreadable
file re-downloads it from the ACL Anthology using the entry's DOI
(10.18653/v1/<anthology-id> -> https://aclanthology.org/<anthology-id>.pdf).
Fails loudly if a paper cannot be repaired.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
SPLIT_FILES = {
    "english": ["test/en_test_data.jsonl", "train/en_train_data.jsonl", "validation/en_valid_data.jsonl"],
    "spanish": ["test/es_test_data.jsonl", "train/es_train_data.jsonl", "validation/es_valid_data.jsonl"],
}


def normalize_filename(text):  # identical to download_papers.py
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("/", "_").replace(":", " -")
    text = re.sub(r"[^\w\s-]", "", text).strip()
    return text[:150]


def healthy(path):
    try:
        if path.stat().st_size == 0:
            return False
        r = PdfReader(path)
        return len(r.pages) > 0 and r.pages[0].extract_text() is not None
    except Exception:
        return False


def main():
    failures = []
    for lang, files in SPLIT_FILES.items():
        papers = DATASETS / lang / "papers"
        meta = {}
        for f in files:
            for line in open(DATASETS / lang / f, encoding="utf-8"):
                e = json.loads(line)
                meta[normalize_filename(e["title"])] = e
        bad = [p for p in sorted(papers.glob("*.pdf")) if not healthy(p)]
        print(f"[{lang}] {len(bad)} unhealthy of {len(list(papers.glob('*.pdf')))}")
        for p in bad:
            e = meta.get(p.stem)
            if e is None:
                failures.append((lang, p.name, "no dataset entry matches filename"))
                continue
            doi = e.get("doi") or ""
            m = re.match(r"10\.18653/v1/(.+)$", doi)
            candidates = []
            if m:
                candidates.append(f"https://aclanthology.org/{m.group(1)}.pdf")
            url = e.get("url") or ""
            if "aclanthology.org" in url:
                candidates.append(url if url.endswith(".pdf") else url.rstrip("/") + ".pdf")
            ok = False
            for c in candidates:
                try:
                    r = requests.get(c, timeout=30)
                    if r.status_code == 200 and r.content[:4] == b"%PDF":
                        p.write_bytes(r.content)
                        if healthy(p):
                            print(f"  repaired {p.name[:60]} <- {c}")
                            ok = True
                            break
                except requests.RequestException as ex:
                    print(f"  {c}: {ex}")
            if not ok:
                failures.append((lang, p.name, f"doi={doi} url={url} tried={candidates}"))
    if failures:
        for f in failures:
            print("FAILED", f)
        sys.exit(1)
    print("all PDFs healthy")


if __name__ == "__main__":
    main()
