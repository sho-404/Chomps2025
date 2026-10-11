"""Judge runs (DESIGN.md "Judge runs"): one LLM judge on every example of a split.

Store: ../results/votes/{lang}/{track}/{split}/{condition}__{judge}__{fmt}__t{temp}.jsonl,
one row per (example, repeat), in `data.load_examples` order and then by repeat:
{"index", "repeat", "vote": "y"|"n", "parse_ok", "raw", "prompt_tokens", "completion_tokens",
 "reasoning_tokens", "latency_s", "cached", "model", "finish_reason"}.

Protocol, reproducing english|spanish/scripts/*/approach_*.py except as listed further down:
* fmt=judge sends the original evaluator prompts (prompts.factuality_judge / fluency_judge);
  fmt=ft sends the fine-tuning format (prompts.ft_factuality / ft_fluency) with question=prompt.
* temp "default" sends no temperature (the only setting gpt-5-mini accepts); "0" / "1.0" send 0.0 / 1.0.
* max_completion_tokens=config.JUDGE_MAX_COMPLETION_TOKENS; repeat r is llm.chat(..., repeat=r);
  replies are parsed by prompts.parse_vote (no 'y'/'n' in the reply -> "y" with parse_ok=False).
* Factuality with a retrieved context: an empty context string is voted "y" without a call
  (raw="AUTO_EMPTY_CONTEXT", approach_1/2's `if not context: return "y"`); condition `full` raises
  the context cap from MAX_CONTEXT_CHARS to MAX_PAPER_CHARS. Condition `none` sends context=None
  (prompts.NO_CONTEXT_NOTE) and exists only for fmt=judge: the ft format has no context-free variant.
* Fluency: condition is always `none`; a blank answer is voted "y" without a call
  (raw="AUTO_EMPTY_ANSWER", approach_1/2's `if not model_answer.strip(): return "y"`).

Deliberate differences from the submitted scripts (DESIGN.md's protocol):
* An API error aborts the run; the scripts turned any API error into a "y" vote.
* One completion cap for every call, 16,384 tokens, as in approach_1 and the factuality councils.
  The fluency councils (approach_2/3) capped at 100 (Spanish approach_3: 5), and a reasoning judge
  spends that before it answers: gpt-5-mini on 3 English test items ended every capped call with
  finish_reason="length" and an empty reply (parsed "y"), and voted y/y/n at 16,384. Reruns of the
  fluency councils therefore do not reproduce the submitted gpt-5-mini fluency votes.
* The AUTO rules hold in every run; approach_3 (meta-learner) had neither and sent context ""
  (pseudo_rag: 8 Spanish test, 1 Spanish train) and blank answers (2 English validation) to its judges.
* Examples with identical inputs share one reply per repeat (see Resumability), where the scripts
  sampled each independently: at temp default/1.0 their votes are now perfectly correlated. Examples
  sharing an earlier example's input: English test 13/240 (pseudo_rag 6), validation 11/240, train
  6/108; Spanish train 3/180, validation 3/240, test 0. Every run prints its count.

Resumability: llm.chat caches on (deployment, messages, parameters, repeat), so rerunning the same
arguments makes no calls and rewrites the same votes. Each distinct (messages, repeat) is called
once per store: two concurrent misses on one key would make two calls of which only the last
survives in the cache, so a rerun would not reproduce the first run's votes. `cached` says whether
the run that wrote the file was served from the cache; `latency_s` is the latency measured when the
call was made (llm.py caches it with the reply). AUTO rows made no call: zero tokens and latency,
cached=False, model="", finish_reason=None.

Overwriting: a store is replaced whole (temp file + one rename, permissions from the umask). Its path
does not encode `repeats`, so a run that would drop stored repeats (e.g. --repeats 1 after
--repeats 5) fails before any call unless overwrite=True (--overwrite).
"""
import argparse
import json
import os
import re
from pathlib import Path

from . import config, data, llm, prompts

FMTS = ("judge", "ft")
TEMPS = ("default", "0", "1.0")
TEMPERATURE = {"default": None, "0": 0.0, "1.0": 1.0}
BASE_JUDGES = ("gpt-5-mini", "gpt-4.1-mini", "gpt-4.1-nano")
JUDGES = BASE_JUDGES + tuple(config.FT_DEPLOYMENTS.values())
CONDITIONS = ("pseudo_rag", "pseudo_rag_rerun", "bm25_k1", "bm25_k3", "bm25_k5",
              "dense_k1", "dense_k3", "dense_k5", "full", "none")
AUTO_EMPTY_CONTEXT = "AUTO_EMPTY_CONTEXT"
AUTO_EMPTY_ANSWER = "AUTO_EMPTY_ANSWER"
_JUDGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _one_of(value, allowed, what: str) -> None:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {list(allowed)}, got {value!r}")


# --- the votes store -----------------------------------------------------------

def votes_path(lang: str, track: str, split: str, condition: str, judge: str, fmt: str,
               temp: str) -> Path:
    """Store file of one run. `temp` is the string "default", "0" or "1.0". `judge` may be any
    name without "__" (meta.py stores meta-* systems here too), so a file name splits back into
    condition, judge, fmt and t{temp} on "__"."""
    _one_of(lang, config.LANGS, "lang")
    _one_of(track, config.TRACKS, "track")
    _one_of(split, config.SPLITS, "split")
    _one_of(condition, CONDITIONS, "condition")
    _one_of(fmt, FMTS, "fmt")
    _one_of(temp, TEMPS, "temp")
    if not isinstance(judge, str) or not _JUDGE_NAME.fullmatch(judge) or "__" in judge:
        raise ValueError(f"invalid judge name {judge!r}")
    if track == "fluency" and condition != "none":
        raise ValueError(f"the fluency track has only condition 'none', got {condition!r}")
    return (config.RESULTS / "votes" / lang / track / split
            / f"{condition}__{judge}__{fmt}__t{temp}.jsonl")


def load_vote_rows(path) -> list:
    """All rows of a votes file, in file order. Raises on a missing or empty file, a line that is
    not a JSON object, a row without index/repeat/vote, a vote other than "y"/"n", a repeat that
    is not a non-negative int, or a repeated (index, repeat)."""
    path = Path(path)
    rows, seen = [], set()
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not JSON ({e})") from e
            if not isinstance(row, dict) or not {"index", "repeat", "vote"} <= row.keys():
                raise ValueError(f"{path}:{n}: a row needs index, repeat and vote")
            if row["vote"] not in ("y", "n"):
                raise ValueError(f"{path}:{n}: vote {row['vote']!r} is not 'y'/'n'")
            rep = row["repeat"]
            if not isinstance(rep, int) or isinstance(rep, bool) or rep < 0:
                raise ValueError(f"{path}:{n}: repeat {rep!r} is not a non-negative int")
            key = (row["index"], rep)
            if key in seen:
                raise ValueError(f"{path}:{n}: duplicate row for index {key[0]!r}, repeat {rep}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty votes file")
    return rows


def load_votes(path) -> dict:
    """{repeat: {index: vote}} of a votes file, repeats ascending, indices in file order. Every
    repeat must cover the same examples, so a truncated or mixed file fails here instead of
    skewing a metric. Repeats need not start at 0 (meta.py stores the repeats it was asked for)."""
    out = {}
    for row in load_vote_rows(path):
        out.setdefault(row["repeat"], {})[row["index"]] = row["vote"]
    repeats = sorted(out)
    for rep in repeats[1:]:
        if out[rep].keys() != out[repeats[0]].keys():
            raise ValueError(f"{path}: repeat {rep} covers different examples than repeat {repeats[0]}")
    return {rep: out[rep] for rep in repeats}


def _write_rows(path: Path, rows: list) -> None:
    """Write the whole store through a temp file and one atomic rename, so no reader (and no
    crash) ever leaves a partial store behind. The temp file is a pid-named plain open(), as in
    meta.write_text, so the store gets the umask's permissions (mkstemp would make it 0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _stores_losing_repeats(paths, repeats: int) -> list:
    """[(path, stored repeats)] of the existing stores holding a repeat >= `repeats`, which a run
    with `repeats` would drop. Reads them with load_vote_rows, so a malformed store raises."""
    out = []
    for path in paths:
        if path.exists():
            stored = sorted({row["repeat"] for row in load_vote_rows(path)})
            if stored[-1] >= repeats:
                out.append((path, stored))
    return out


# --- inputs ----------------------------------------------------------------------

def _examples(lang: str, split: str) -> list:
    exs = data.load_examples(lang, split)
    n_idx = len({e["index"] for e in exs})
    if not exs or n_idx != len(exs):
        raise ValueError(f"{lang}/{split}: {len(exs)} examples with {n_idx} distinct indices")
    return exs


def _contexts(lang: str, split: str, condition: str, examples: list) -> dict:
    """index -> context string from `retrieval.load_contexts`; it must cover exactly the examples."""
    from . import retrieval  # lazy: only the context conditions need it (and rank_bm25)
    ctx = retrieval.load_contexts(lang, split, condition)
    want = {e["index"] for e in examples}
    if not isinstance(ctx, dict) or ctx.keys() != want:
        got = set(ctx) if isinstance(ctx, dict) else set()
        raise ValueError(f"contexts {lang}/{split}/{condition}: {len(want - got)} of {len(want)} "
                         f"examples missing, {len(got - want)} unknown indices")
    return ctx


def _messages(lang: str, track: str, condition: str, fmt: str, ex: dict, context):
    """Chat messages for one example, or the AUTO_* marker of an original no-call "y" rule."""
    prompt, answer = ex["prompt"], ex["output_text"]
    if not (isinstance(prompt, str) and isinstance(answer, str)):
        raise TypeError(f"{ex['index']}: prompt and output_text must be strings")
    if track == "fluency":
        if not answer.strip():
            return AUTO_EMPTY_ANSWER
        build = prompts.fluency_judge if fmt == "judge" else prompts.ft_fluency
        return build(lang, prompt, answer)
    if condition == "none":
        if fmt != "judge":
            raise ValueError("fmt 'ft' has no context-free variant: condition 'none' needs fmt 'judge'")
        return prompts.factuality_judge(lang, None, prompt, answer)
    if not isinstance(context, str):
        raise TypeError(f"{ex['index']}: condition {condition!r} needs a context string, "
                        f"got {type(context).__name__}")
    if not context:
        return AUTO_EMPTY_CONTEXT
    if fmt == "ft":
        return prompts.ft_factuality(lang, context, prompt, answer)
    cap = config.MAX_PAPER_CHARS if condition == "full" else config.MAX_CONTEXT_CHARS
    return prompts.factuality_judge(lang, context, prompt, answer, max_chars=cap)


# --- running -----------------------------------------------------------------------

def _row(index, repeat: int, reply: dict) -> dict:
    vote, ok = prompts.parse_vote(reply["content"])
    return {"index": index, "repeat": repeat, "vote": vote, "parse_ok": ok, "raw": reply["content"],
            "prompt_tokens": reply["prompt_tokens"], "completion_tokens": reply["completion_tokens"],
            "reasoning_tokens": reply["reasoning_tokens"], "latency_s": reply["latency_s"],
            "cached": reply["cached"], "model": reply["model"], "finish_reason": reply["finish_reason"]}


def _auto_row(index, repeat: int, marker: str) -> dict:
    return {"index": index, "repeat": repeat, "vote": "y", "parse_ok": True, "raw": marker,
            "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "latency_s": 0.0,
            "cached": False, "model": "", "finish_reason": None}


def _judge(items: list, judge: str, temp: str, repeats: int, workers: int, desc: str):
    """Rows for items [(index, messages or AUTO marker, call key)]. Every example is
    called on its own (llm.chat cache_tag = its index), exactly as the original
    per-example scripts did, so two examples with identical inputs receive
    independent samples rather than one shared reply. Returns (rows, replies)."""
    calls = {}
    for index, msgs, key in items:
        if key is not None:
            for r in range(repeats):
                calls[(index, r)] = msgs
    keys = list(calls)
    api_keys = [key for _, _, key in items if key is not None]
    print(f"[judge] {desc}: {len(items)} examples ({len(api_keys) - len(set(api_keys))} share an "
          f"earlier example's input, sampled independently) x {repeats} repeats, {len(keys)} calls",
          flush=True)
    temperature = TEMPERATURE[temp]
    replies = llm.run_parallel(
        lambda k: llm.chat(judge, calls[k], temperature=temperature,
                           max_completion_tokens=config.JUDGE_MAX_COMPLETION_TOKENS, repeat=k[1],
                           cache_tag=str(k[0])),
        keys, workers=workers, desc=desc)
    reply = dict(zip(keys, replies))
    rows = [_auto_row(index, r, msgs) if key is None else _row(index, r, reply[index, r])
            for index, msgs, key in items for r in range(repeats)]
    return rows, replies


def _summary(path: Path, rows: list, replies: list) -> str:
    """One line per store: vote counts and protocol fallbacks over rows, cost over distinct calls."""
    n_y = sum(row["vote"] == "y" for row in rows)
    n_auto = sum(row["raw"] in (AUTO_EMPTY_CONTEXT, AUTO_EMPTY_ANSWER) for row in rows)
    n_bad = sum(not row["parse_ok"] for row in rows)
    n_new = sum(not rep["cached"] for rep in replies)
    n_cut = sum(rep["finish_reason"] != "stop" for rep in replies)
    return (f"[judge] wrote {path}\n"
            f"        rows={len(rows)} y={n_y} n={len(rows) - n_y} auto={n_auto} parse_fail={n_bad} | "
            f"distinct calls={len(replies)} new={n_new} cached={len(replies) - n_new} "
            f"finish!=stop={n_cut}")


def run(lang: str, track: str, splits: list, conditions: list, judges: list, fmt: str,
        temp: str, repeats: int, workers: int = 12, overwrite: bool = False) -> list:
    """Run every (split, condition, judge) combination and return the store paths written.

    Everything that can be checked offline (arguments, existing stores, examples, contexts
    stores, prompt building) is checked for all combinations before the first call. An existing
    store holding repeats this run would drop is an error unless `overwrite`, which replaces it
    and says so. The combinations then run one after another; each store is written in full as
    soon as its own calls finish, so an error aborts the run without touching the stores of the
    combinations not yet finished, and rerunning the same command redoes only the missing calls."""
    for what, values in (("split", splits), ("condition", conditions), ("judge", judges)):
        if not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values):
            raise ValueError(f"{what}s must be a non-empty list without repeats, got {values!r}")
    if not isinstance(repeats, int) or repeats < 1 or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"repeats and workers must be positive ints, got {repeats}, {workers}")
    if not isinstance(overwrite, bool):
        raise ValueError(f"overwrite must be a bool, got {overwrite!r}")
    for judge in judges:
        _one_of(judge, JUDGES, "judge")
        if judge in config.REASONING_DEPLOYMENTS and temp != "default":
            raise ValueError(f"{judge} accepts no temperature: use temp 'default', not {temp!r}")
    paths = {(s, c, j): votes_path(lang, track, s, c, j, fmt, temp)
             for s in splits for c in conditions for j in judges}
    losing = dict(_stores_losing_repeats(paths.values(), repeats))
    if losing and not overwrite:
        raise ValueError(f"--repeats {repeats} would drop stored repeats from {len(losing)} store(s); "
                         f"pass --overwrite (overwrite=True) to replace them:\n"
                         + "\n".join(f"  {p}: repeats {s}" for p, s in losing.items()))

    inputs = {}
    for split in splits:
        exs = _examples(lang, split)
        for cond in conditions:
            ctx = _contexts(lang, split, cond, exs) if track == "factuality" and cond != "none" else {}
            items = []
            for ex in exs:
                msgs = _messages(lang, track, cond, fmt, ex, ctx.get(ex["index"]))
                key = None if isinstance(msgs, str) else json.dumps(msgs, ensure_ascii=False)
                items.append((ex["index"], msgs, key))
            inputs[split, cond] = items

    for (split, cond, judge), path in paths.items():
        desc = f"{lang}/{track}/{split}/{cond}/{judge}/{fmt}/t{temp}"
        rows, replies = _judge(inputs[split, cond], judge, temp, repeats, workers, desc)
        if path in losing:
            print(f"[judge] OVERWRITE {path}: stored repeats {losing[path]} -> "
                  f"{list(range(repeats))}", flush=True)
        _write_rows(path, rows)
        print(_summary(path, rows, replies), flush=True)
    return list(paths.values())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m chomps.judge",
                                description="Run LLM judges and write the votes store.")
    p.add_argument("--lang", required=True, choices=config.LANGS)
    p.add_argument("--track", required=True, choices=config.TRACKS)
    p.add_argument("--split", required=True, nargs="+", choices=config.SPLITS)
    p.add_argument("--condition", required=True, nargs="+", choices=CONDITIONS)
    p.add_argument("--judge", required=True, nargs="+", choices=JUDGES)
    p.add_argument("--fmt", required=True, choices=FMTS)
    p.add_argument("--temp", required=True, choices=TEMPS)
    p.add_argument("--repeats", required=True, type=int)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--overwrite", action="store_true",
                   help="replace stores holding repeats >= --repeats (default: refuse before any call)")
    a = p.parse_args(argv)
    run(a.lang, a.track, a.split, a.condition, a.judge, a.fmt, a.temp, a.repeats, a.workers,
        overwrite=a.overwrite)


if __name__ == "__main__":
    main()
