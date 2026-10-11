"""Recover the arXiv papers used to generate the synthetic fine-tuning data.

The original `download_ft_papers.py` fetched the 100 most recent cs.CL/cs.LG/cs.AI
submissions at generation time (January 2026); the PDFs were not kept. Only a
minority of the generated questions embed the arXiv id (e.g. "In the paper
2601.11190v1, ..."), so we recover the candidate set by scanning the id range
around the ids that do appear, keep the papers in those categories, and download
their PDFs so that the synthetic contexts can be checked verbatim against their
sources (see chomps/synth_qc.py).

Usage: python code/common/fetch_arxiv_range.py --start 2601.11000 --end 2601.11600
Writes datasets/finetuning_papers/arxiv_meta.jsonl and datasets/finetuning_papers/<id>.pdf.
arXiv asks for one request every 3 seconds; the spacing below honours that policy.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets" / "finetuning_papers"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
CATS = {"cs.CL", "cs.LG", "cs.AI"}


def fetch_meta(ids):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    with urllib.request.urlopen(url, timeout=60) as r:
        root = ET.fromstring(r.read())
    out = []
    for e in root.findall("atom:entry", NS):
        aid = e.find("atom:id", NS).text.rsplit("/", 1)[-1]
        title = " ".join(e.find("atom:title", NS).text.split())
        primary = e.find("arxiv:primary_category", NS)
        cats = {c.get("term") for c in e.findall("atom:category", NS)}
        if not title or title.lower().startswith("error"):
            continue
        out.append({"id": aid, "title": title, "primary": primary.get("term") if primary is not None else None,
                    "cats": sorted(cats), "published": e.find("atom:published", NS).text})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2601.11000")
    ap.add_argument("--end", default="2601.11600")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pre, s = a.start.split("."); _, e = a.end.split(".")
    ids = [f"{pre}.{n:05d}" for n in range(int(s), int(e) + 1)]
    meta = []
    for i in range(0, len(ids), 100):
        meta += fetch_meta(ids[i:i + 100])
        print(f"meta {i + 100}/{len(ids)} -> {len(meta)} entries", flush=True)
        time.sleep(3)
    keep = [m for m in meta if set(m["cats"]) & CATS or (m["primary"] in CATS)]
    with open(OUT / "arxiv_meta.jsonl", "w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"{len(meta)} ids exist, {len(keep)} in {sorted(CATS)}")
    if a.no_pdf:
        return
    for k, m in enumerate(keep):
        aid = m["id"]
        dest = OUT / f"{aid}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        try:
            with urllib.request.urlopen(f"https://arxiv.org/pdf/{aid}", timeout=120) as r:
                data = r.read()
            if data[:4] != b"%PDF":
                print(f"  {aid}: not a PDF ({data[:30]!r})", flush=True)
            else:
                dest.write_bytes(data)
        except Exception as ex:  # report, keep going, fail at the end
            print(f"  {aid}: {ex}", flush=True)
        if k % 10 == 0:
            print(f"pdf {k + 1}/{len(keep)}", flush=True)
        time.sleep(3)
    missing = [m["id"] for m in keep if not (OUT / f"{m['id']}.pdf").exists()]
    print(f"done; missing PDFs: {len(missing)} {missing[:10]}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
