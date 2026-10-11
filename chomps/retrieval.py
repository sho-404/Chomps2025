"""Retrieval conditions: the evidence the factuality judge sees (DESIGN.md,
"Retrieval conditions").

Every condition selects from the same text -- the example's own paper cut at
MAX_PAPER_CHARS, exactly what the competition's pseudo-RAG extractor read -- and
the conditions differ only in how the evidence is chosen:

    pseudo_rag        competition span (`context_original`), copied unchanged
    pseudo_rag_rerun  the same gpt-5-mini verbatim-extraction call, re-executed
    bm25_k{K}         top-K chunks by BM25Okapi                      (K = 1, 3, 5)
    dense_k{K}        top-K chunks by text-embedding-3-large cosine  (K = 1, 3, 5)
    full              paper_text[:MAX_PAPER_CHARS], no selection
    none              no evidence (JSON null)

Store: ../results/contexts/{lang}/{split}/{condition}.jsonl, one row
{"index", "context", "meta"} per example, in `data.load_examples` order. A store
is rewritten in full on every run (from the LLM cache, so re-running is free) and
atomically, so a crash never leaves a partial store behind.

Protocol decisions
* Chunks: whitespace tokens of the truncated paper, windows of 100 words with
  stride 50, re-joined by single spaces. The last window is the first one that
  reaches the end of the text, so no chunk is a sub-span of another chunk.
* Query: `question`, not `prompt` (the prompt prepends the title and author
  names, which would pull the paper header into every top-K).
* Ranking: score descending, ties to the lower chunk id (deterministic). The K
  winners are concatenated in document order with "\\n[...]\\n"; overlapping
  neighbours are kept whole, not merged, as in standard chunked RAG. Meta lists
  `chunk_ids` in that document order with their `scores` aligned.
* BM25: one BM25Okapi index per paper (IDF over that paper's chunks), tokenizer
  re.findall(r"\\w+", s.lower()). A question sharing no token with its paper
  scores 0 on every chunk, so the tie rule returns the first K chunks; the
  summary reports these as `zero_overlap` (Spanish questions are matched
  against English papers) instead of hiding them.
* Dense: llm.embed caches every chunk and question by text, so each is embedded
  once; cosine is computed explicitly (unit norm is not assumed). Meta also
  records `embed_tokens` = cl100k_base counts (the embedding model's tokenizer)
  of the query and of all chunks of the paper; the paper part is a one-time
  indexing cost shared by every example of that paper and by every K.
* pseudo_rag_rerun: gpt-5-mini, prompts.retrieval, max_completion_tokens=4096,
  no temperature, reply .strip()-ed, as in {lang}/scripts/
  inject_context_single_block.py. "The exact original prompt" is enforced, not
  assumed: before any condition runs, build() evaluates that script's own
  prompt f-string and create(...) keywords from its syntax tree and raises
  unless our request is byte-identical, so a drifted prompt is never paid for.
  Every paper is now available, so an empty extraction is a failed call: the
  run finishes all calls (they are cached), raises listing the failures, and
  writes no store.
"""
import argparse
import ast
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config, data, llm, prompts

K_VALUES = (1, 3, 5)
# Cheapest first, so that a failing API condition never blocks the free ones.
CONDITIONS = ("pseudo_rag", "none", "full", *(f"bm25_k{k}" for k in K_VALUES),
              *(f"dense_k{k}" for k in K_VALUES), "pseudo_rag_rerun")
CHUNK_WORDS = 100
CHUNK_STRIDE = 50
SEPARATOR = "\n[...]\n"
RERUN_MODEL = config.DEPLOYMENTS["gpt-5-mini"]
EMBED_MODEL = config.DEPLOYMENTS["embedding"]
_RERUN_META = ("model", "prompt_tokens", "completion_tokens", "reasoning_tokens",
               "latency_s", "finish_reason", "cached")


def parse_condition(condition: str) -> tuple:
    """'bm25_k3' -> ('bm25', 3); 'full' -> ('full', None). Unknown names raise."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    method, _, k = condition.partition("_k")
    return (method, int(k)) if k else (condition, None)


# --- store ---------------------------------------------------------------------

def store_path(lang: str, split: str, condition: str) -> Path:
    if lang not in config.LANGS:
        raise ValueError(f"unknown lang {lang!r}; expected one of {config.LANGS}")
    if split not in config.SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {config.SPLITS}")
    parse_condition(condition)
    return config.RESULTS / "contexts" / lang / split / f"{condition}.jsonl"


def write_store(lang: str, split: str, condition: str, rows: list) -> Path:
    """Replace the store atomically: write a process-unique temp file next to
    it, then rename (plain open(), so the file gets the umask's normal mode)."""
    path = store_path(lang, split, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load_rows(lang: str, split: str, condition: str) -> list:
    """All store rows ({"index", "context", "meta"}) in example order. Every line
    must be a JSON object with exactly those keys, a string or integer index, an
    object meta, and a context that is null for `none` and a string for every
    other condition; anything else raises ValueError naming path:line."""
    path = store_path(lang, split, condition)
    if not path.exists():
        raise FileNotFoundError(
            f"no context store at {path}. Build it from code/ with: ../.venv/bin/python -m "
            f"chomps.retrieval --lang {lang} --split {split} --conditions {condition}")
    context_type, context_kind = (type(None), "null") if condition == "none" else (str, "a string")
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            where = f"{path}:{n}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: not valid JSON ({e})") from e
            if not isinstance(row, dict) or set(row) != {"index", "context", "meta"}:
                raise ValueError(f"{where}: malformed row, expected an object with exactly the keys "
                                 f"index, context, meta: {line.strip()[:80]!r}")
            if type(row["index"]) not in (str, int):
                raise ValueError(f"{where}: index must be a string or an integer, got {row['index']!r}")
            if not isinstance(row["meta"], dict):
                raise ValueError(f"{where}: meta must be an object, got {type(row['meta']).__name__}")
            if not isinstance(row["context"], context_type):
                raise ValueError(f"{where}: context must be {context_kind} for condition {condition!r}, "
                                 f"got {type(row['context']).__name__}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    indices = [r["index"] for r in rows]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{path}: duplicate example indices")
    return rows


def load_contexts(lang: str, split: str, condition: str) -> dict:
    """index -> context (str, or None for condition `none`). Raises
    FileNotFoundError naming the command that builds a missing store."""
    return {r["index"]: r["context"] for r in load_rows(lang, split, condition)}


# --- chunking and ranking --------------------------------------------------------

def chunk_paper(paper_text: str) -> list:
    """Windows of CHUNK_WORDS whitespace tokens with stride CHUNK_STRIDE over
    paper_text[:MAX_PAPER_CHARS]; the last window is the first that reaches the end."""
    words = paper_text[:config.MAX_PAPER_CHARS].split()
    if not words:
        raise ValueError("paper text has no words to chunk")
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            return chunks
        start += CHUNK_STRIDE


def bm25_tokens(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def top_k(scores, k: int) -> list:
    """Ids of the k best chunks (score desc, ties -> lower id), in document order."""
    s = np.asarray(scores, dtype=float)
    if k < 1 or s.ndim != 1 or s.size == 0 or not np.isfinite(s).all():
        raise ValueError(f"top_k needs k >= 1 and a non-empty finite score vector (k={k}, shape={s.shape})")
    return sorted(int(i) for i in np.argsort(-s, kind="stable")[:k])


def bm25_scores(examples: list, chunks: dict) -> list:
    """Per example: BM25Okapi score of every chunk of its paper for its question."""
    indexes, out = {}, []
    for ex in examples:
        title = ex["title"]
        if title not in indexes:
            corpus = [bm25_tokens(c) for c in chunks[title]]
            if not any(corpus):
                raise ValueError(f"paper {title!r} has no word tokens; BM25 is undefined")
            indexes[title] = BM25Okapi(corpus)
        query = bm25_tokens(_question(ex))
        if not query:
            raise ValueError(f"{ex['index']}: question has no word tokens: {ex['question']!r}")
        out.append(indexes[title].get_scores(query))
    return out


def dense_scores(examples: list, chunks: dict, workers: int) -> list:
    """Per example: cosine(question, chunk) for every chunk of its paper. One
    llm.embed call for all questions, then one per paper (papers in parallel)."""
    queries = _embed([_question(ex) for ex in examples])
    members = {}
    for i, ex in enumerate(examples):
        members.setdefault(ex["title"], []).append(i)

    def score_paper(title):
        rows = members[title]
        return rows, _unit(queries[rows]) @ _unit(_embed(chunks[title])).T

    out = [None] * len(examples)
    for rows, sims in llm.run_parallel(score_paper, list(members), workers=workers, desc="dense: papers"):
        for i, s in zip(rows, sims):
            out[i] = s
    return out


def _embed(texts: list) -> np.ndarray:
    vecs = llm.embed(texts, deployment=EMBED_MODEL)
    if len(vecs) != len(texts) or any(v is None for v in vecs):
        raise RuntimeError(f"llm.embed returned {len(vecs)} vectors, "
                           f"{sum(v is None for v in vecs)} missing, for {len(texts)} texts")
    m = np.asarray(vecs, dtype=float)
    if m.ndim != 2:
        raise RuntimeError(f"llm.embed returned an array of shape {m.shape}, expected (n, dim)")
    return m


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero-length embedding vector: cosine is undefined")
    return m / norms


@lru_cache(maxsize=None)
def _cl100k():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def embed_tokens(text: str) -> int:
    """Tokens the embedding model bills for `text` (cl100k_base). Special-token
    strings that occur in some papers (e.g. '<|endoftext|>') count as plain text."""
    return len(_cl100k().encode(text, disallowed_special=()))


# --- conditions --------------------------------------------------------------------

def _question(ex: dict) -> str:
    q = ex.get("question")
    if not isinstance(q, str) or not q.strip():
        raise ValueError(f"{ex.get('index')}: empty or missing question")
    return q


def _row(ex: dict, context, meta: dict) -> dict:
    return {"index": ex["index"], "context": context, "meta": meta}


def _full_row(ex: dict, paper: str) -> dict:
    ctx = paper[:config.MAX_PAPER_CHARS]
    return _row(ex, ctx, {"chars": len(ctx), "truncated": len(paper) > config.MAX_PAPER_CHARS,
                          "paper_chars": len(paper)})


def _topk_rows(examples: list, chunks: dict, scores: list, k: int, extra_meta=None) -> list:
    rows = []
    for i, (ex, s) in enumerate(zip(examples, scores)):
        paper_chunks = chunks[ex["title"]]
        ids = top_k(s, k)
        meta = {"chunk_ids": ids, "scores": [round(float(s[j]), 6) for j in ids],
                "n_chunks": len(paper_chunks)}
        if extra_meta is not None:
            meta.update(extra_meta[i])
        rows.append(_row(ex, SEPARATOR.join(paper_chunks[j] for j in ids), meta))
    return rows


def rerun_call(question: str, paper_text: str) -> dict:
    """The request pseudo_rag_rerun sends, keyed like the original create(...) call."""
    return {"model": RERUN_MODEL, "messages": prompts.retrieval(question, paper_text),
            "max_completion_tokens": config.RETRIEVAL_MAX_COMPLETION_TOKENS}


def original_script(lang: str) -> Path:
    """The competition's extraction script (get_context_from_llm is the same in both languages)."""
    return config.CODE_ROOT / lang / "scripts" / "inject_context_single_block.py"


def original_extraction_call(lang: str, question: str, paper_text: str) -> dict:
    """The keyword arguments of client.chat.completions.create(...) in the original
    get_context_from_llm, evaluated for (question, paper_text) from the script's own
    syntax tree: first its `prompt` f-string, then the call's keywords, with the
    script's module-level constants (DEPLOYMENT) in scope and no builtins. Reading the
    script instead of a transcription is what makes the comparison byte for byte."""
    path = original_script(lang)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fn = _one(path, "def get_context_from_llm", [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_context_from_llm"])
    prompt = _one(path, "prompt = f-string", [
        n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and isinstance(n.value, ast.JoinedStr)
        and [ast.unparse(t) for t in n.targets] == ["prompt"]])
    create = _one(path, "client.chat.completions.create(...)", [
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "client.chat.completions.create"])
    env = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
           for t in n.targets if isinstance(t, ast.Name)}
    env.update(question=question, paper_text=paper_text)
    env["prompt"] = _evaluate(prompt, env, path)
    return {k.arg: _evaluate(k.value, env, path) for k in create.keywords}


def _one(path: Path, what: str, nodes: list):
    if len(nodes) != 1:
        raise ValueError(f"{path}: expected exactly one {what}, found {len(nodes)}")
    return nodes[0]


def _evaluate(node: ast.expr, env: dict, path: Path):
    return eval(compile(ast.Expression(body=node), str(path), "eval"), {"__builtins__": {}}, env)


def check_rerun_call(lang: str) -> None:
    """Raise unless rerun_call equals original_extraction_call byte for byte (model,
    system and user messages, max_completion_tokens; neither sends a temperature).
    The probe paper is longer than MAX_PAPER_CHARS, so the character cut is compared too."""
    question, paper = "Which dataset is used?", "0123456789" * (config.MAX_PAPER_CHARS // 10 + 100)
    ours, original = (json.dumps(c, ensure_ascii=False, sort_keys=True) for c in
                      (rerun_call(question, paper), original_extraction_call(lang, question, paper)))
    if ours != original:
        i = next((i for i, (a, b) in enumerate(zip(original, ours)) if a != b), min(len(original), len(ours)))
        raise RuntimeError(
            f"pseudo_rag_rerun would not repeat the competition's extraction call ({original_script(lang)}): "
            f"the requests as JSON ({len(original)} vs {len(ours)} chars) first differ at char {i}: original "
            f"...{original[max(0, i - 40):i + 20]!r}, ours ...{ours[max(0, i - 40):i + 20]!r}. Make "
            f"prompts.retrieval/config reproduce it before running the rerun; no request was sent.")


def rerun_rows(examples: list, papers: dict, workers: int) -> list:
    """Re-execute the competition extraction call for every example (build() has
    already checked rerun_call against the original script)."""
    def one(ex):
        call = rerun_call(_question(ex), papers[ex["title"]])
        # Exactly the checked request; cache_tag is local to the cache (never sent)
        # and gives every example its own extraction sample, as the original
        # per-example loop did (the same question recurs for 6-8 model outputs).
        r = llm.chat(call.pop("model"), call.pop("messages"), cache_tag=str(ex["index"]), **call)
        return _row(ex, r["content"].strip(), {k: r[k] for k in _RERUN_META})

    rows = llm.run_parallel(one, examples, workers=workers, desc="pseudo_rag_rerun")
    empty = [r for r in rows if not r["context"]]
    if empty:
        detail = "; ".join(f"{r['index']} (finish_reason={r['meta']['finish_reason']}, "
                           f"completion_tokens={r['meta']['completion_tokens']}, "
                           f"reasoning_tokens={r['meta']['reasoning_tokens']})" for r in empty[:20])
        raise RuntimeError(
            f"pseudo_rag_rerun: {len(empty)}/{len(rows)} extractions are empty although every paper "
            f"is available: {detail}. No store written; the replies are cached, so re-running "
            f"reproduces them until the protocol (e.g. RETRIEVAL_MAX_COMPLETION_TOKENS) is changed.")
    return rows


def build(lang: str, examples: list, conditions, workers: int = 12):
    """Yield (condition, rows) per requested condition, rows ordered as
    `examples`. Paper texts, chunks and score vectors are computed once and
    shared (bm25_k1/k3/k5 rank once). Nothing is written here. A requested
    pseudo_rag_rerun whose request is not the original's raises before anything runs."""
    if lang not in config.LANGS:
        raise ValueError(f"unknown lang {lang!r}")
    indices = [ex["index"] for ex in examples]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("examples must be non-empty with unique indices")
    conditions = list(dict.fromkeys(conditions))
    for c in conditions:
        parse_condition(c)
    if "pseudo_rag_rerun" in conditions:
        check_rerun_call(lang)
    memo = {}

    def papers():
        if "papers" not in memo:
            memo["papers"] = {t: data.paper_text(lang, t) for t in dict.fromkeys(ex["title"] for ex in examples)}
        return memo["papers"]

    def chunks():
        if "chunks" not in memo:
            memo["chunks"] = {t: chunk_paper(p) for t, p in papers().items()}
        return memo["chunks"]

    def scores(method):
        if method not in memo:
            memo[method] = (bm25_scores(examples, chunks()) if method == "bm25"
                            else dense_scores(examples, chunks(), workers))
        return memo[method]

    def dense_tokens():
        if "dense_tokens" not in memo:
            per_paper = {t: sum(map(embed_tokens, cs)) for t, cs in chunks().items()}
            memo["dense_tokens"] = [{"embed_tokens": {"query": embed_tokens(ex["question"]),
                                                      "paper_chunks": per_paper[ex["title"]]}}
                                    for ex in examples]
        return memo["dense_tokens"]

    for cond in conditions:
        method, k = parse_condition(cond)
        if method == "pseudo_rag":
            rows = [_row(ex, ex["context_original"], {"source": "original"}) for ex in examples]
        elif method == "none":
            rows = [_row(ex, None, {}) for ex in examples]
        elif method == "full":
            rows = [_full_row(ex, papers()[ex["title"]]) for ex in examples]
        elif method == "pseudo_rag_rerun":
            rows = rerun_rows(examples, papers(), workers)
        elif method == "bm25":
            rows = _topk_rows(examples, chunks(), scores("bm25"), k)
        else:
            rows = _topk_rows(examples, chunks(), scores("dense"), k, dense_tokens())
        yield cond, rows


def summary(condition: str, rows: list) -> str:
    """One line: n, null/empty/'not found' counts, mean chars, plus per-method checks."""
    ctx = [r["context"] for r in rows]
    text = [c for c in ctx if c is not None]
    metas = [r["meta"] for r in rows]
    parts = [f"n={len(rows)}", f"null={len(ctx) - len(text)}", f"empty={sum(not c for c in text)}",
             f"not_found={sum('not found' in c.lower() for c in text)}",
             f"mean_chars={np.mean([len(c) for c in text]):.1f}" if text else "mean_chars=-"]
    method, _ = parse_condition(condition)
    if method == "bm25":
        parts.append(f"zero_overlap={sum(max(m['scores']) <= 0 for m in metas)}")
    elif method == "full":
        parts.append(f"truncated={sum(m['truncated'] for m in metas)}")
    elif method == "pseudo_rag_rerun":
        means = "/".join(f"{np.mean([m[t] for m in metas]):.0f}"
                         for t in ("prompt_tokens", "completion_tokens", "reasoning_tokens"))
        parts += [f"finish_length={sum(m['finish_reason'] == 'length' for m in metas)}",
                  f"api_calls={sum(not m['cached'] for m in metas)}",
                  f"mean_tokens(prompt/completion/reasoning)={means}"]
    return f"{condition}: " + " ".join(parts)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m chomps.retrieval",
                                 description="Build the factuality context stores (DESIGN.md).")
    ap.add_argument("--lang", nargs="+", choices=config.LANGS, default=list(config.LANGS))
    ap.add_argument("--split", nargs="+", choices=config.SPLITS, default=list(config.SPLITS))
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS),
                    help="default: all, cheapest first")
    ap.add_argument("--workers", type=int, default=12)
    return ap


def main(argv=None) -> None:
    ap = parser()
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    for lang in dict.fromkeys(args.lang):
        for split in dict.fromkeys(args.split):
            examples = data.load_examples(lang, split)
            for cond, rows in build(lang, examples, args.conditions, args.workers):
                path = write_store(lang, split, cond, rows)
                print(f"[{lang}/{split}] {summary(cond, rows)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
