"""Cost and latency of the judge runs and of the paid retrieval stages
(DESIGN.md, "Cost and latency"). Writes ../results/tables/cost.csv.

Rows (`splits` lists the splits pooled into each row):
    stage=judge      one per (lang, track, condition, judge, fmt, temp) over the votes
                     stores ../results/votes/{lang}/{track}/{split}/{condition}__{judge}__{fmt}__t{temp}.jsonl
    stage=retrieval  one per (lang, condition) for pseudo_rag_rerun and dense_k* over the
                     contexts stores; `judge` holds the retrieval deployment, fmt/temp are empty.
A missing store directory only means fewer rows (the CSV is still written); a
store that exists but is malformed raises.

Protocol decisions
------------------
* Cost convention, the same for every row: standalone cost per store row. A row is
  one judgment (one example judged once) or one example's retrieved context, and it
  is charged the full cost of the work behind it, as if no other row existed: a
  pseudo_rag_rerun row its whole gpt-5-mini call, a dense_k* row its query plus all
  chunks of its paper (the index is not amortised). The figures are what one new
  example costs, and the two retrieval rows compare like with like. Our runs shared
  work between rows, so their bill was lower than rows x figure:
    - judge: identical inputs share one reply per repeat (judge.py), at most 5.6 %
      of a split's rows (English train 6/108);
    - pseudo_rag_rerun: llm.chat caches on the messages, so the model outputs of one
      question share one call: 8 per question in validation and test (English
      validation 7-9), 1-8 in train; pooled, English 588 examples / 98 questions
      (one per paper) = 6.0, Spanish 660 / 100 = 6.6;
    - dense_k*: a paper's chunks are embedded once for all its examples and every K:
      pooled, 6.0 examples per paper in English, 7.5 in Spanish (660 / 88).
* usd_per_1000_examples_per_repeat = 1000 x mean over rows of prompt_tokens*input +
  completion_tokens*output, prices from config.PRICES_USD_PER_M[config.price_key(deployment)]:
  for a judge, the cost of 1000 judgments, i.e. of judging 1000 examples once; a
  store with `repeats` = 5 cost 5x that per 1000 examples (`repeats` = repeats per
  example, "1|5" when the pooled splits differ; every repeat of a store must cover the
  same examples). Retrieval rows have one context per example (repeats 1).
  completion_tokens already include reasoning tokens (OpenAI usage), so those are not
  added again. List prices: llm.chat does not record cached prompt tokens, so Azure's
  prompt-caching discount is ignored and the figures are upper bounds (relevant for
  `full`, whose repeats share the paper prefix: paper[:120k] averages 18.4k o200k
  tokens in English and 18.7k in Spanish, at most 58k). Fine-tuned deployments also
  bill hosting per hour (hosting_usd_per_hour).
* Counts are store rows, not distinct API calls: n_api_rows = rows backed by an API
  reply, n_fresh_rows = those whose reply was a fresh call in the run that wrote the
  store (rows sharing one fresh reply all count). AUTO_* rows (judge.py's
  AUTO_EMPTY_CONTEXT / AUTO_EMPTY_ANSWER) are votes cast without a call: counted in
  n_auto, excluded from the per-call token/latency statistics, included at $0 in the
  USD column (the example was handled).
* Latency: llm.py caches the measured latency with the reply and judge.py copies
  it, so every API row carries the latency of the call that produced it, whether
  or not the run that wrote the store hit the cache (`cached` describes that run
  only). DESIGN.md's "only cached==False rows" would drop real measurements (all
  of them once a store is rewritten from the cache), so all API rows are used and
  n_fresh_rows reports how many were fresh in the writing run. Latency includes
  llm.py's retry back-off; the median is robust to it.
* meta-* systems (meta.py) make no call (their cost is their council's): listed
  on stdout, not tabulated. Any other judge without a price raises.
* pseudo_rag_rerun: tokens and latency of the extraction reply each example carries,
  from the store meta. dense_k*: meta.embed_tokens holds cl100k_base counts (the
  embedding model's tokenizer) of the query and of all chunks of the paper; no
  latency is recorded for embeddings. bm25_k*, full, none and the original
  pseudo_rag (not metered at the time; same call as the rerun) get no row.
"""
import argparse
import math
import statistics

import pandas as pd

from . import config, data

AUTO_PREFIX = "AUTO_"
META_PREFIX = "meta-"
VOTE_FIELDS = ("raw", "prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s", "cached", "model")
RERUN_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s", "cached", "model")
USD = "usd_per_1000_examples_per_repeat"
COLUMNS = ["stage", "lang", "track", "condition", "judge", "fmt", "temp", "splits", "models",
           "n_rows", "n_examples", "repeats", "n_api_rows", "n_auto", "n_fresh_rows",
           "prompt_tokens_mean", "prompt_tokens_median", "completion_tokens_mean",
           "completion_tokens_median", "reasoning_tokens_mean", "reasoning_tokens_median",
           "latency_s_mean", "latency_s_median", USD, "hosting_usd_per_hour", "tokens_source"]
COUNT_COLUMNS = ["n_rows", "n_examples", "n_api_rows", "n_auto", "n_fresh_rows"]


def _mean(xs) -> float:
    return statistics.fmean(xs) if xs else math.nan


def _median(xs) -> float:
    return statistics.median(xs) if xs else math.nan


def prices(deployment: str) -> dict:
    """USD per 1M tokens (and hosting per hour) of a deployment; raises if unpriced."""
    key = config.price_key(deployment)
    if key not in config.PRICES_USD_PER_M:
        raise KeyError(f"no price for deployment {deployment!r} (price key {key!r}); "
                       f"add it to config.PRICES_USD_PER_M")
    return config.PRICES_USD_PER_M[key]


def usd(p: dict, prompt_tokens: float, completion_tokens: float = 0) -> float:
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1e6


def _ordered_splits(splits) -> str:
    return "+".join(s for s in config.SPLITS if s in splits)


def parse_votes_path(path, votes_dir) -> dict:
    """lang/track/split/{condition}__{judge}__{fmt}__t{temp}.jsonl -> its fields; raises otherwise."""
    from . import judge
    parts = path.relative_to(votes_dir).parts
    fields = path.stem.split("__")
    if len(parts) != 4 or len(fields) != 4 or not fields[3].startswith("t") or not all(fields):
        raise ValueError(f"unexpected file in the votes store: {path}")
    lang, track, split, _ = parts
    out = dict(lang=lang, track=track, split=split, condition=fields[0], judge=fields[1],
               fmt=fields[2], temp=fields[3][1:])
    for name, allowed in (("lang", config.LANGS), ("track", config.TRACKS), ("split", config.SPLITS),
                          ("condition", judge.CONDITIONS), ("fmt", judge.FMTS), ("temp", judge.TEMPS)):
        if out[name] not in allowed:
            raise ValueError(f"{path}: {name} {out[name]!r} not in {allowed}")
    return out


def judge_rows() -> list:
    """One cost row per (lang, track, condition, judge, fmt, temp), splits pooled."""
    from . import judge
    votes_dir = config.RESULTS / "votes"
    groups, skipped = {}, []
    for path in sorted(votes_dir.rglob("*.jsonl")) if votes_dir.is_dir() else []:
        f = parse_votes_path(path, votes_dir)
        if f["judge"].startswith(META_PREFIX):
            skipped.append(str(path.relative_to(votes_dir)))
            continue
        rows = judge.load_vote_rows(path)
        for n, r in enumerate(rows, 1):
            missing = [k for k in VOTE_FIELDS if k not in r]
            if missing:
                raise KeyError(f"{path} row {n}: missing {missing}")
            if not isinstance(r["raw"], str):
                raise TypeError(f"{path} row {n}: raw is {type(r['raw']).__name__}")
            if not r["raw"].startswith(AUTO_PREFIX) and not r["prompt_tokens"] > 0:
                raise ValueError(f"{path} row {n}: an API reply with prompt_tokens={r['prompt_tokens']}")
        repeats = len(judge.load_votes(path))  # raises if the repeats cover different examples
        key = tuple(f[k] for k in ("lang", "track", "condition", "judge", "fmt", "temp"))
        g = groups.setdefault(key, {"splits": set(), "repeats": set(), "rows": []})
        g["splits"].add(f["split"])
        g["repeats"].add(repeats)
        g["rows"] += [(f["split"], r) for r in rows]
    if skipped:
        print(f"[cost] {len(skipped)} meta-learner stores make no API calls (cost = their council's): "
              + ", ".join(skipped))
    out = []
    for (lang, track, condition, judge_name, fmt, temp), g in groups.items():
        p = prices(judge_name)
        rows = [r for _, r in g["rows"]]
        api = [r for r in rows if not r["raw"].startswith(AUTO_PREFIX)]
        col = {k: [r[k] for r in api] for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s")}
        out.append({
            "stage": "judge", "lang": lang, "track": track, "condition": condition, "judge": judge_name,
            "fmt": fmt, "temp": temp, "splits": _ordered_splits(g["splits"]),
            "models": "|".join(sorted({r["model"] for r in api})),
            "n_rows": len(rows), "n_examples": len({(s, r["index"]) for s, r in g["rows"]}),
            "repeats": "|".join(str(n) for n in sorted(g["repeats"])),
            "n_api_rows": len(api), "n_auto": len(rows) - len(api), "n_fresh_rows": sum(not r["cached"] for r in api),
            **{f"{k}_{stat}": fn(v) for k, v in col.items() for stat, fn in (("mean", _mean), ("median", _median))},
            USD: 1000 * _mean([usd(p, r["prompt_tokens"], r["completion_tokens"]) for r in rows]),
            "hosting_usd_per_hour": p.get("hosting_per_hour", math.nan), "tokens_source": "api_usage",
        })
    return out


def _context_stores() -> dict:
    """(lang, condition) -> splits with a pseudo_rag_rerun or dense_k* store on disk."""
    from . import retrieval
    ctx_dir = config.RESULTS / "contexts"
    found = {}
    for path in sorted(ctx_dir.rglob("*.jsonl")) if ctx_dir.is_dir() else []:
        parts = path.relative_to(ctx_dir).parts
        if len(parts) != 3 or parts[0] not in config.LANGS or parts[1] not in config.SPLITS:
            raise ValueError(f"unexpected file in the contexts store: {path}")
        method, _ = retrieval.parse_condition(path.stem)
        if method in ("pseudo_rag_rerun", "dense"):
            found.setdefault((parts[0], path.stem), set()).add(parts[1])
    return found


def _rerun_row(lang: str, condition: str, splits: set) -> dict:
    """Each example is charged the extraction reply it carries (standalone convention)."""
    from . import retrieval
    metas = []
    for split in (s for s in config.SPLITS if s in splits):
        for r in retrieval.load_rows(lang, split, condition):
            missing = [k for k in RERUN_FIELDS if k not in r["meta"]]
            if missing:
                raise KeyError(f"{lang}/{split}/{condition}/{r['index']}: meta lacks {missing}")
            metas.append(r["meta"])
    p = prices(retrieval.RERUN_MODEL)
    col = {k: [m[k] for m in metas] for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s")}
    return {
        "judge": retrieval.RERUN_MODEL, "models": "|".join(sorted({m["model"] for m in metas})),
        "n_rows": len(metas), "n_examples": len(metas), "repeats": "1", "n_api_rows": len(metas), "n_auto": 0,
        "n_fresh_rows": sum(not m["cached"] for m in metas),
        **{f"{k}_{stat}": fn(v) for k, v in col.items() for stat, fn in (("mean", _mean), ("median", _median))},
        USD: 1000 * _mean([usd(p, m["prompt_tokens"], m["completion_tokens"]) for m in metas]),
        "hosting_usd_per_hour": math.nan, "tokens_source": "api_usage",
    }


def _dense_row(lang: str, condition: str, splits: set) -> dict:
    """Each example is charged its query plus all chunks of its paper (standalone
    convention). The store is checked against the dataset: every index must be an
    example, and every example of a paper must report the same paper_chunks tokens."""
    from . import retrieval
    tokens, paper_tokens = [], {}  # per example: query + paper_chunks; title -> paper_chunks
    for split in (s for s in config.SPLITS if s in splits):
        titles = {e["index"]: e["title"] for e in data.load_examples(lang, split)}
        for r in retrieval.load_rows(lang, split, condition):
            et = r["meta"].get("embed_tokens")
            if not isinstance(et, dict) or not {"query", "paper_chunks"} <= et.keys():
                raise KeyError(f"{lang}/{split}/{condition}/{r['index']}: meta lacks embed_tokens "
                               f"{{query, paper_chunks}}; rebuild the store with chomps.retrieval (cached, free)")
            if r["index"] not in titles:
                raise KeyError(f"{lang}/{split}/{condition}: store index {r['index']!r} is not in the dataset")
            title = titles[r["index"]]
            if paper_tokens.setdefault(title, et["paper_chunks"]) != et["paper_chunks"]:
                raise ValueError(f"{lang}/{condition}: paper {title!r} has inconsistent paper_chunks tokens")
            tokens.append(et["query"] + et["paper_chunks"])
    p = prices(retrieval.EMBED_MODEL)
    return {
        "judge": retrieval.EMBED_MODEL, "models": retrieval.EMBED_MODEL,
        "n_rows": len(tokens), "n_examples": len(tokens), "repeats": "1", "n_api_rows": math.nan, "n_auto": 0,
        "n_fresh_rows": math.nan,
        "prompt_tokens_mean": _mean(tokens), "prompt_tokens_median": _median(tokens),
        "completion_tokens_mean": 0.0, "completion_tokens_median": 0.0,
        "reasoning_tokens_mean": 0.0, "reasoning_tokens_median": 0.0,
        "latency_s_mean": math.nan, "latency_s_median": math.nan,
        USD: 1000 * _mean([usd(p, t) for t in tokens]),
        "hosting_usd_per_hour": math.nan, "tokens_source": "cl100k_base",
    }


def retrieval_rows() -> list:
    """One cost row per (lang, condition) for pseudo_rag_rerun and dense_k*, splits pooled."""
    out = []
    for (lang, condition), splits in _context_stores().items():
        row = (_rerun_row if condition == "pseudo_rag_rerun" else _dense_row)(lang, condition, splits)
        out.append({"stage": "retrieval", "lang": lang, "track": "factuality", "condition": condition,
                    "fmt": "", "temp": "", "splits": _ordered_splits(splits), **row})
    return out


def build() -> pd.DataFrame:
    """Aggregate every store on disk and write ../results/tables/cost.csv."""
    judged, retrieved = judge_rows(), retrieval_rows()
    if not judged:
        print(f"[cost] no judge rows: no priced votes stores under {config.RESULTS / 'votes'}")
    if not retrieved:
        print(f"[cost] no retrieval rows: no pseudo_rag_rerun/dense_k* stores under {config.RESULTS / 'contexts'}")
    rows = judged + retrieved
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["stage", "lang", "track", "condition", "judge", "fmt", "temp"], kind="stable")
    df[COUNT_COLUMNS] = df[COUNT_COLUMNS].astype("Int64")  # counts stay integers next to NaN
    df = df.round({c: 2 for c in COLUMNS if "tokens_m" in c} | {"latency_s_mean": 3, "latency_s_median": 3, USD: 6})
    out = config.RESULTS / "tables" / "cost.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[cost] wrote {out} ({len(df)} rows)")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m chomps.cost",
                                 description="Aggregate token usage, latency and USD cost of every store "
                                             "into ../results/tables/cost.csv.")
    ap.parse_args(argv)
    df = build()
    if len(df):
        cols = ["stage", "lang", "track", "condition", "judge", "fmt", "temp", "splits", "n_examples",
                "repeats", "n_api_rows", "prompt_tokens_mean", "completion_tokens_mean", "latency_s_median", USD]
        with pd.option_context("display.width", 220, "display.max_columns", None):
            print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
